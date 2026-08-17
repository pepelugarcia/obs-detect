#include "obs-utils.h"
#include "plugin-support.h"

#include <obs-module.h>

/**
  * @brief Get RGBA from the stage surface
  *
  * @param tf  The filter data
  * @param width  The width of the stage surface (output)
  * @param height  The height of the stage surface (output)
  * @return true  if successful
  * @return false if unsuccessful
*/
bool getRGBAFromStageSurface(filter_data *tf, uint32_t &width, uint32_t &height)
{

	if (!obs_source_enabled(tf->source)) {
		return false;
	}

	/* Capture via the filter target - the same render path stock obs-detect
	   has always used (proven stable on async DeckLink sources). NOTE:
	   obs_source_default_render(parent) must never be used here: async
	   sources have no synchronous render callback and it crashes the
	   graphics thread (c0000005 at NULL). */
	obs_source_t *target = obs_filter_get_target(tf->source);
	if (!target) {
		return false;
	}
	uint32_t srcWidth = obs_source_get_base_width(target);
	uint32_t srcHeight = obs_source_get_base_height(target);
	if (srcWidth == 0 || srcHeight == 0) {
		return false;
	}

	/* Downscale ON THE GPU before the readback. This copy is synchronous and
	   blocks the render thread, and a 4K BGRA frame is ~33MB; at div 3 a
	   3840x2160 source becomes 1280x720 (~3.7MB, 1/9 the bytes). The detector
	   letterboxes to at most 1280x736 anyway, so nothing it could use is lost.
	   Floor of 320x180 so a small source can never collapse the input. */
	int div = tf->readbackDiv < 1 ? 1 : (tf->readbackDiv > 4 ? 4 : tf->readbackDiv);
	width = srcWidth / (uint32_t)div;
	height = srcHeight / (uint32_t)div;
	if (width < 320 || height < 180) {
		width = srcWidth;
		height = srcHeight;
	}

	/* Factors for the three coordinate boundaries in video_tick: the fence and
	   min-area convert IN from source space, the crop values convert OUT to it.
	   Everything between stays in readback space. */
	tf->sourceW = srcWidth;
	tf->sourceH = srcHeight;
	tf->readbackScaleX = static_cast<float>(srcWidth) / static_cast<float>(width);
	tf->readbackScaleY = static_cast<float>(srcHeight) / static_cast<float>(height);

	gs_texrender_reset(tf->texrender);
	if (!gs_texrender_begin(tf->texrender, width, height)) {
		return false;
	}
	struct vec4 background;
	vec4_zero(&background);
	gs_clear(GS_CLEAR_COLOR, &background, 0.0f, 0);
	/* ortho spans the FULL source rect while the render target is width x height
	   - that mismatch is what makes the GPU scale the frame down into it */
	gs_ortho(0.0f, static_cast<float>(srcWidth), 0.0f, static_cast<float>(srcHeight), -100.0f,
		 100.0f);
	gs_blend_state_push();
	gs_blend_function(GS_BLEND_ONE, GS_BLEND_ZERO);
	obs_source_video_render(target);
	gs_blend_state_pop();
	gs_texrender_end(tf->texrender);

	if (tf->stagesurface) {
		uint32_t stagesurf_width = gs_stagesurface_get_width(tf->stagesurface);
		uint32_t stagesurf_height = gs_stagesurface_get_height(tf->stagesurface);
		if (stagesurf_width != width || stagesurf_height != height) {
			gs_stagesurface_destroy(tf->stagesurface);
			tf->stagesurface = nullptr;
		}
	}
	if (!tf->stagesurface) {
		tf->stagesurface = gs_stagesurface_create(width, height, GS_BGRA);
	}
	gs_stage_texture(tf->stagesurface, gs_texrender_get_texture(tf->texrender));
	uint8_t *video_data;
	uint32_t linesize;
	if (!gs_stagesurface_map(tf->stagesurface, &video_data, &linesize)) {
		return false;
	}
	{
		std::lock_guard<std::mutex> lock(tf->inputBGRALock);
		tf->inputBGRA = cv::Mat(height, width, CV_8UC4, video_data, linesize);
		tf->inputFresh = true; // signal video_tick a new frame is ready
	}
	gs_stagesurface_unmap(tf->stagesurface);
	return true;
}

gs_texture_t *blur_image(struct filter_data *tf, uint32_t width, uint32_t height,
			 gs_texture_t *alphaTexture)
{
	gs_texture_t *blurredTexture = gs_texture_create(width, height, GS_BGRA, 1, nullptr, 0);
	gs_copy_texture(blurredTexture, gs_texrender_get_texture(tf->texrender));
	if (tf->kawaseBlurEffect == nullptr) {
		obs_log(LOG_ERROR, "tf->kawaseBlurEffect is null");
		return blurredTexture;
	}
	gs_eparam_t *image = gs_effect_get_param_by_name(tf->kawaseBlurEffect, "image");
	gs_eparam_t *xOffset = gs_effect_get_param_by_name(tf->kawaseBlurEffect, "xOffset");
	gs_eparam_t *yOffset = gs_effect_get_param_by_name(tf->kawaseBlurEffect, "yOffset");
	gs_eparam_t *mask = gs_effect_get_param_by_name(tf->kawaseBlurEffect, "focalmask");

	for (int i = 0; i < (int)tf->maskingBlurRadius; i++) {
		gs_texrender_reset(tf->texrender);
		if (!gs_texrender_begin(tf->texrender, width, height)) {
			obs_log(LOG_INFO, "Could not open background blur texrender!");
			return blurredTexture;
		}

		gs_effect_set_texture(image, blurredTexture);
		if (alphaTexture != nullptr) {
			gs_effect_set_texture(mask, alphaTexture);
		}
		gs_effect_set_float(xOffset, ((float)i + 0.5f) / (float)width);
		gs_effect_set_float(yOffset, ((float)i + 0.5f) / (float)height);

		struct vec4 background;
		vec4_zero(&background);
		gs_clear(GS_CLEAR_COLOR, &background, 0.0f, 0);
		gs_ortho(0.0f, static_cast<float>(width), 0.0f, static_cast<float>(height), -100.0f,
			 100.0f);
		gs_blend_state_push();
		gs_blend_function(GS_BLEND_ONE, GS_BLEND_ZERO);

		while (gs_effect_loop(tf->kawaseBlurEffect,
				      (alphaTexture == nullptr) ? "Draw" : "DrawMaskAware")) {
			gs_draw_sprite(blurredTexture, 0, width, height);
		}
		gs_blend_state_pop();
		gs_texrender_end(tf->texrender);
		gs_copy_texture(blurredTexture, gs_texrender_get_texture(tf->texrender));
	}
	return blurredTexture;
}

gs_texture_t *pixelate_image(struct filter_data *tf, uint32_t width, uint32_t height,
			     gs_texture_t *alphaTexture, float pixelateRadius)
{
	gs_texture_t *blurredTexture = gs_texture_create(width, height, GS_BGRA, 1, nullptr, 0);
	gs_copy_texture(blurredTexture, gs_texrender_get_texture(tf->texrender));
	if (tf->pixelateEffect == nullptr) {
		obs_log(LOG_ERROR, "tf->pixelateEffect is null");
		return blurredTexture;
	}
	gs_eparam_t *image = gs_effect_get_param_by_name(tf->pixelateEffect, "image");
	gs_eparam_t *mask = gs_effect_get_param_by_name(tf->pixelateEffect, "focalmask");
	gs_eparam_t *pixel_size = gs_effect_get_param_by_name(tf->pixelateEffect, "pixel_size");
	gs_eparam_t *tex_size = gs_effect_get_param_by_name(tf->pixelateEffect, "tex_size");

	gs_texrender_reset(tf->texrender);
	if (!gs_texrender_begin(tf->texrender, width, height)) {
		obs_log(LOG_INFO, "Could not open background blur texrender!");
		return blurredTexture;
	}

	gs_effect_set_texture(image, blurredTexture);
	if (alphaTexture != nullptr) {
		gs_effect_set_texture(mask, alphaTexture);
	}
	gs_effect_set_float(pixel_size, pixelateRadius);
	vec2 texsize_vec;
	vec2_set(&texsize_vec, (float)width, (float)height);
	gs_effect_set_vec2(tex_size, &texsize_vec);

	struct vec4 background;
	vec4_zero(&background);
	gs_clear(GS_CLEAR_COLOR, &background, 0.0f, 0);
	gs_ortho(0.0f, static_cast<float>(width), 0.0f, static_cast<float>(height), -100.0f,
		 100.0f);
	gs_blend_state_push();
	gs_blend_function(GS_BLEND_ONE, GS_BLEND_ZERO);

	while (gs_effect_loop(tf->pixelateEffect, "Draw")) {
		gs_draw_sprite(blurredTexture, 0, width, height);
	}
	gs_blend_state_pop();
	gs_texrender_end(tf->texrender);
	gs_copy_texture(blurredTexture, gs_texrender_get_texture(tf->texrender));

	return blurredTexture;
}
