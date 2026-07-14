#include "detect-filter.h"

#include <onnxruntime_cxx_api.h>

#ifdef _WIN32
#include <wchar.h>
#include <windows.h>
#endif // _WIN32

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cstdio>
#include <cmath>
#include <atomic>
#include <numeric>
#include <memory>
#include <exception>
#include <fstream>
#include <new>
#include <mutex>
#include <regex>
#include <thread>

#include <nlohmann/json.hpp>

#include <plugin-support.h>
#include "FilterData.h"
#include "consts.h"
#include "obs-utils/obs-utils.h"
#include "ort-model/utils.hpp"
#include "detect-filter-utils.h"
#include "edgeyolo/edgeyolo_onnxruntime.hpp"
#include "yunet/YuNet.h"

#define EXTERNAL_MODEL_SIZE "!!!EXTERNAL_MODEL!!!"
#define FACE_DETECT_MODEL_SIZE "!!!FACE_DETECT!!!"

struct detect_filter : public filter_data {};

const char *detect_filter_getname(void *unused)
{
	UNUSED_PARAMETER(unused);
	return obs_module_text("Detect");
}

/**                   PROPERTIES                     */

static bool visible_on_bool(obs_properties_t *ppts, obs_data_t *settings, const char *bool_prop,
			    const char *prop_name)
{
	const bool enabled = obs_data_get_bool(settings, bool_prop);
	obs_property_t *p = obs_properties_get(ppts, prop_name);
	obs_property_set_visible(p, enabled);
	return true;
}

static bool enable_advanced_settings(obs_properties_t *ppts, obs_property_t *p,
				     obs_data_t *settings)
{
	const bool enabled = obs_data_get_bool(settings, "advanced");

	for (const char *prop_name :
	     {"threshold", "useGPU", "numThreads", "model_size", "detected_object", "sort_tracking",
	      "max_unseen_frames", "show_unseen_objects", "save_detections_path", "crop_group",
	      "min_size_threshold"}) {
		p = obs_properties_get(ppts, prop_name);
		obs_property_set_visible(p, enabled);
	}

	return true;
}

void set_class_names_on_object_category(obs_property_t *object_category,
					std::vector<std::string> class_names)
{
	std::vector<std::pair<size_t, std::string>> indexed_classes;
	for (size_t i = 0; i < class_names.size(); ++i) {
		const std::string &class_name = class_names[i];
		// capitalize the first letter of the class name
		std::string class_name_cap = class_name;
		class_name_cap[0] = (char)std::toupper((int)class_name_cap[0]);
		indexed_classes.push_back({i, class_name_cap});
	}

	// sort the vector based on the class names
	std::sort(indexed_classes.begin(), indexed_classes.end(),
		  [](const std::pair<size_t, std::string> &a,
		     const std::pair<size_t, std::string> &b) { return a.second < b.second; });

	// clear the object category list
	obs_property_list_clear(object_category);

	// add the sorted classes to the property list
	obs_property_list_add_int(object_category, obs_module_text("All"), -1);

	// add the sorted classes to the property list
	for (const auto &indexed_class : indexed_classes) {
		obs_property_list_add_int(object_category, indexed_class.second.c_str(),
					  (int)indexed_class.first);
	}
}

void read_model_config_json_and_set_class_names(const char *model_file, obs_properties_t *props_,
						obs_data_t *settings, struct detect_filter *tf_)
{
	if (model_file == nullptr || model_file[0] == '\0' || strlen(model_file) == 0) {
		obs_log(LOG_ERROR, "Model file path is empty");
		return;
	}

	// read the '.json' file near the model file to find the class names
	std::string json_file = model_file;
	json_file.replace(json_file.find(".onnx"), 5, ".json");
	std::ifstream file(json_file);
	if (!file.is_open()) {
		obs_data_set_string(settings, "error", "JSON file not found");
		obs_log(LOG_ERROR, "JSON file not found: %s", json_file.c_str());
	} else {
		obs_data_set_string(settings, "error", "");
		// parse the JSON file
		nlohmann::json j;
		file >> j;
		if (j.contains("names")) {
			std::vector<std::string> labels = j["names"];
			set_class_names_on_object_category(
				obs_properties_get(props_, "object_category"), labels);
			tf_->classNames = labels;
		} else {
			obs_data_set_string(settings, "error",
					    "JSON file does not contain 'names' field");
			obs_log(LOG_ERROR, "JSON file does not contain 'names' field");
		}
	}
}

obs_properties_t *detect_filter_properties(void *data)
{
	struct detect_filter *tf = reinterpret_cast<detect_filter *>(data);

	obs_properties_t *props = obs_properties_create();

	obs_properties_add_bool(props, "preview", obs_module_text("Preview"));

	// add dropdown selection for object category selection: "All", or COCO classes
	obs_property_t *object_category =
		obs_properties_add_list(props, "object_category", obs_module_text("ObjectCategory"),
					OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_INT);
	set_class_names_on_object_category(object_category, edgeyolo_cpp::COCO_CLASSES);
	tf->classNames = edgeyolo_cpp::COCO_CLASSES;

	// options group for masking
	obs_properties_t *masking_group = obs_properties_create();
	obs_property_t *masking_group_prop =
		obs_properties_add_group(props, "masking_group", obs_module_text("MaskingGroup"),
					 OBS_GROUP_CHECKABLE, masking_group);

	// add callback to show/hide masking options
	obs_property_set_modified_callback(masking_group_prop, [](obs_properties_t *props_,
								  obs_property_t *,
								  obs_data_t *settings) {
		const bool enabled = obs_data_get_bool(settings, "masking_group");
		obs_property_t *prop = obs_properties_get(props_, "masking_type");
		obs_property_t *masking_color = obs_properties_get(props_, "masking_color");
		obs_property_t *masking_blur_radius =
			obs_properties_get(props_, "masking_blur_radius");
		obs_property_t *masking_dilation =
			obs_properties_get(props_, "dilation_iterations");

		obs_property_set_visible(prop, enabled);
		obs_property_set_visible(masking_color, false);
		obs_property_set_visible(masking_blur_radius, false);
		obs_property_set_visible(masking_dilation, enabled);
		std::string masking_type_value = obs_data_get_string(settings, "masking_type");
		if (masking_type_value == "solid_color") {
			obs_property_set_visible(masking_color, enabled);
		} else if (masking_type_value == "blur" || masking_type_value == "pixelate") {
			obs_property_set_visible(masking_blur_radius, enabled);
		}
		return true;
	});

	// add masking options drop down selection: "None", "Solid color", "Blur", "Transparent"
	obs_property_t *masking_type = obs_properties_add_list(masking_group, "masking_type",
							       obs_module_text("MaskingType"),
							       OBS_COMBO_TYPE_LIST,
							       OBS_COMBO_FORMAT_STRING);
	obs_property_list_add_string(masking_type, obs_module_text("None"), "none");
	obs_property_list_add_string(masking_type, obs_module_text("SolidColor"), "solid_color");
	obs_property_list_add_string(masking_type, obs_module_text("OutputMask"), "output_mask");
	obs_property_list_add_string(masking_type, obs_module_text("Blur"), "blur");
	obs_property_list_add_string(masking_type, obs_module_text("Pixelate"), "pixelate");
	obs_property_list_add_string(masking_type, obs_module_text("Transparent"), "transparent");

	// add color picker for solid color masking
	obs_properties_add_color(masking_group, "masking_color", obs_module_text("MaskingColor"));

	// add slider for blur radius
	obs_properties_add_int_slider(masking_group, "masking_blur_radius",
				      obs_module_text("MaskingBlurRadius"), 1, 30, 1);

	// add callback to show/hide blur radius and color picker
	obs_property_set_modified_callback(masking_type, [](obs_properties_t *props_,
							    obs_property_t *,
							    obs_data_t *settings) {
		std::string masking_type_value = obs_data_get_string(settings, "masking_type");
		obs_property_t *masking_color = obs_properties_get(props_, "masking_color");
		obs_property_t *masking_blur_radius =
			obs_properties_get(props_, "masking_blur_radius");
		obs_property_t *masking_dilation =
			obs_properties_get(props_, "dilation_iterations");
		obs_property_set_visible(masking_color, false);
		obs_property_set_visible(masking_blur_radius, false);
		const bool masking_enabled = obs_data_get_bool(settings, "masking_group");
		obs_property_set_visible(masking_dilation, masking_enabled);

		if (masking_type_value == "solid_color") {
			obs_property_set_visible(masking_color, masking_enabled);
		} else if (masking_type_value == "blur" || masking_type_value == "pixelate") {
			obs_property_set_visible(masking_blur_radius, masking_enabled);
		}
		return true;
	});

	// add slider for dilation iterations
	obs_properties_add_int_slider(masking_group, "dilation_iterations",
				      obs_module_text("DilationIterations"), 0, 20, 1);

	// add options group for tracking and zoom-follow options
	obs_properties_t *tracking_group_props = obs_properties_create();
	obs_property_t *tracking_group = obs_properties_add_group(
		props, "tracking_group", obs_module_text("TrackingZoomFollowGroup"),
		OBS_GROUP_CHECKABLE, tracking_group_props);

	// add callback to show/hide tracking options
	obs_property_set_modified_callback(tracking_group, [](obs_properties_t *props_,
							      obs_property_t *,
							      obs_data_t *settings) {
		const bool enabled = obs_data_get_bool(settings, "tracking_group");
		for (auto prop_name : {"zoom_factor", "zoom_object", "zoom_speed_factor",
				       "zoom_size_speed_factor"}) {
			obs_property_t *prop = obs_properties_get(props_, prop_name);
			obs_property_set_visible(prop, enabled);
		}
		return true;
	});

	// add zoom factor slider
	obs_properties_add_float_slider(tracking_group_props, "zoom_factor",
					obs_module_text("ZoomFactor"), 0.0, 1.0, 0.05);

	// PAN/track speed: how fast the crop re-centers on the actor
	obs_properties_add_float_slider(tracking_group_props, "zoom_speed_factor",
					obs_module_text("ZoomSpeed"), 0.0, 0.1, 0.01);

	// SIZE/zoom speed: how fast the crop tightens/widens (independent of pan)
	obs_properties_add_float_slider(tracking_group_props, "zoom_size_speed_factor",
					obs_module_text("ZoomSizeSpeed"), 0.0, 0.05, 0.001);

	// add object selection for zoom drop down: "Single", "All"
	obs_property_t *zoom_object = obs_properties_add_list(tracking_group_props, "zoom_object",
							      obs_module_text("ZoomObject"),
							      OBS_COMBO_TYPE_LIST,
							      OBS_COMBO_FORMAT_STRING);
	obs_property_list_add_string(zoom_object, obs_module_text("SingleFirst"), "single");
	obs_property_list_add_string(zoom_object, obs_module_text("Biggest"), "biggest");
	obs_property_list_add_string(zoom_object, obs_module_text("Oldest"), "oldest");
	obs_property_list_add_string(zoom_object, obs_module_text("All"), "all");

	obs_property_t *advanced =
		obs_properties_add_bool(props, "advanced", obs_module_text("Advanced"));

	// If advanced is selected show the advanced settings, otherwise hide them
	obs_property_set_modified_callback(advanced, enable_advanced_settings);

	// add a checkable group for crop region settings
	obs_properties_t *crop_group_props = obs_properties_create();
	obs_property_t *crop_group =
		obs_properties_add_group(props, "crop_group", obs_module_text("CropGroup"),
					 OBS_GROUP_CHECKABLE, crop_group_props);

	// add callback to show/hide crop region options
	obs_property_set_modified_callback(crop_group, [](obs_properties_t *props_,
							  obs_property_t *, obs_data_t *settings) {
		const bool enabled = obs_data_get_bool(settings, "crop_group");
		for (auto prop_name : {"crop_left", "crop_right", "crop_top", "crop_bottom"}) {
			obs_property_t *prop = obs_properties_get(props_, prop_name);
			obs_property_set_visible(prop, enabled);
		}
		return true;
	});

	// add crop region settings
	obs_properties_add_int_slider(crop_group_props, "crop_left", obs_module_text("CropLeft"), 0,
				      1000, 1);
	obs_properties_add_int_slider(crop_group_props, "crop_right", obs_module_text("CropRight"),
				      0, 1000, 1);
	obs_properties_add_int_slider(crop_group_props, "crop_top", obs_module_text("CropTop"), 0,
				      1000, 1);
	obs_properties_add_int_slider(crop_group_props, "crop_bottom",
				      obs_module_text("CropBottom"), 0, 1000, 1);

	// add a text input for the currently detected object
	obs_property_t *detected_obj_prop = obs_properties_add_text(
		props, "detected_object", obs_module_text("DetectedObject"), OBS_TEXT_DEFAULT);
	// disable the text input by default
	obs_property_set_enabled(detected_obj_prop, false);

	// add threshold slider
	obs_properties_add_float_slider(props, "threshold", obs_module_text("ConfThreshold"), 0.0,
					1.0, 0.025);

	// add minimal size threshold slider
	obs_properties_add_int_slider(props, "min_size_threshold",
				      obs_module_text("MinSizeThreshold"), 0, 10000, 1);

	// add SORT tracking enabled checkbox
	obs_properties_add_bool(props, "sort_tracking", obs_module_text("SORTTracking"));

	// add parameter for number of missing frames before a track is considered lost
	obs_properties_add_int(props, "max_unseen_frames", obs_module_text("MaxUnseenFrames"), 1,
			       30, 1);

	// add option to show unseen objects
	obs_properties_add_bool(props, "show_unseen_objects", obs_module_text("ShowUnseenObjects"));

	// add file path for saving detections
	obs_properties_add_path(props, "save_detections_path",
				obs_module_text("SaveDetectionsPath"), OBS_PATH_FILE_SAVE,
				"JSON file (*.json);;All files (*.*)", nullptr);

	/* GPU, CPU and performance Props */
	obs_property_t *p_use_gpu =
		obs_properties_add_list(props, "useGPU", obs_module_text("InferenceDevice"),
					OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_STRING);

	obs_property_list_add_string(p_use_gpu, obs_module_text("CPU"), USEGPU_CPU);
#if defined(__linux__) && defined(__x86_64__)
	obs_property_list_add_string(p_use_gpu, obs_module_text("GPUTensorRT"), USEGPU_TENSORRT);
#endif
#if _WIN32
	obs_property_list_add_string(p_use_gpu, obs_module_text("GPUDirectML"), USEGPU_DML);
#endif
#if defined(__APPLE__)
	obs_property_list_add_string(p_use_gpu, obs_module_text("CoreML"), USEGPU_COREML);
#endif

	obs_properties_add_int_slider(props, "numThreads", obs_module_text("NumThreads"), 0, 8, 1);

	// add drop down option for model size: Small, Medium, Large
	obs_property_t *model_size =
		obs_properties_add_list(props, "model_size", obs_module_text("ModelSize"),
					OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_STRING);
	obs_property_list_add_string(model_size, obs_module_text("SmallFast"), "small");
	obs_property_list_add_string(model_size, obs_module_text("Medium"), "medium");
	obs_property_list_add_string(model_size, obs_module_text("LargeSlow"), "large");
	obs_property_list_add_string(model_size, obs_module_text("FaceDetect"),
				     FACE_DETECT_MODEL_SIZE);
	obs_property_list_add_string(model_size, obs_module_text("ExternalModel"),
				     EXTERNAL_MODEL_SIZE);

	// add external model file path
	obs_properties_add_path(props, "external_model_file", obs_module_text("ModelPath"),
				OBS_PATH_FILE, "EdgeYOLO onnx files (*.onnx);;all files (*.*)",
				nullptr);

	// add callback to show/hide the external model file path
	obs_property_set_modified_callback2(
		model_size,
		[](void *data_, obs_properties_t *props_, obs_property_t *p, obs_data_t *settings) {
			UNUSED_PARAMETER(p);
			struct detect_filter *tf_ = reinterpret_cast<detect_filter *>(data_);
			std::string model_size_value = obs_data_get_string(settings, "model_size");
			bool is_external = model_size_value == EXTERNAL_MODEL_SIZE;
			obs_property_t *prop = obs_properties_get(props_, "external_model_file");
			obs_property_set_visible(prop, is_external);
			if (!is_external) {
				if (model_size_value == FACE_DETECT_MODEL_SIZE) {
					// set the class names to COCO classes for face detection model
					set_class_names_on_object_category(
						obs_properties_get(props_, "object_category"),
						yunet::FACE_CLASSES);
					tf_->classNames = yunet::FACE_CLASSES;
				} else {
					// reset the class names to COCO classes for default models
					set_class_names_on_object_category(
						obs_properties_get(props_, "object_category"),
						edgeyolo_cpp::COCO_CLASSES);
					tf_->classNames = edgeyolo_cpp::COCO_CLASSES;
				}
			} else {
				// if the model path is already set - update the class names
				const char *model_file =
					obs_data_get_string(settings, "external_model_file");
				read_model_config_json_and_set_class_names(model_file, props_,
									   settings, tf_);
			}
			return true;
		},
		tf);

	// add callback on the model file path to check if the file exists
	obs_property_set_modified_callback2(
		obs_properties_get(props, "external_model_file"),
		[](void *data_, obs_properties_t *props_, obs_property_t *p, obs_data_t *settings) {
			UNUSED_PARAMETER(p);
			const char *model_size_value = obs_data_get_string(settings, "model_size");
			bool is_external = strcmp(model_size_value, EXTERNAL_MODEL_SIZE) == 0;
			if (!is_external) {
				return true;
			}
			struct detect_filter *tf_ = reinterpret_cast<detect_filter *>(data_);
			const char *model_file =
				obs_data_get_string(settings, "external_model_file");
			read_model_config_json_and_set_class_names(model_file, props_, settings,
								   tf_);
			return true;
		},
		tf);

	// Add a informative text about the plugin
	std::string basic_info =
		std::regex_replace(PLUGIN_INFO_TEMPLATE, std::regex("%1"), PLUGIN_VERSION);
	obs_properties_add_text(props, "info", basic_info.c_str(), OBS_TEXT_INFO);

	UNUSED_PARAMETER(data);
	return props;
}

void detect_filter_defaults(obs_data_t *settings)
{
	obs_data_set_default_bool(settings, "advanced", false);
#if _WIN32
	obs_data_set_default_string(settings, "useGPU", USEGPU_DML);
#elif defined(__APPLE__)
	obs_data_set_default_string(settings, "useGPU", USEGPU_CPU);
#else
	// Linux
	obs_data_set_default_string(settings, "useGPU", USEGPU_CPU);
#endif
	obs_data_set_default_bool(settings, "sort_tracking", false);
	obs_data_set_default_int(settings, "max_unseen_frames", 10);
	obs_data_set_default_bool(settings, "show_unseen_objects", true);
	obs_data_set_default_int(settings, "numThreads", 1);
	obs_data_set_default_bool(settings, "preview", true);
	obs_data_set_default_double(settings, "threshold", 0.5);
	obs_data_set_default_string(settings, "model_size", "small");
	obs_data_set_default_int(settings, "object_category", -1);
	obs_data_set_default_bool(settings, "masking_group", false);
	obs_data_set_default_string(settings, "masking_type", "none");
	obs_data_set_default_string(settings, "masking_color", "#000000");
	obs_data_set_default_int(settings, "masking_blur_radius", 0);
	obs_data_set_default_int(settings, "dilation_iterations", 0);
	obs_data_set_default_bool(settings, "tracking_group", false);
	obs_data_set_default_double(settings, "zoom_factor", 0.0);
	obs_data_set_default_double(settings, "zoom_speed_factor", 0.05);
	// studio default, operator-tuned on the floor 2026-07-10 (with the
	// asymmetric widen-4x behaviour this is the TIGHTEN rate)
	obs_data_set_default_double(settings, "zoom_size_speed_factor", 0.022);
	obs_data_set_default_string(settings, "zoom_object", "single");
	// director lock: SORT track id to follow exclusively (0 = automatic).
	// set from the control panel; deliberately not exposed in the OBS UI
	obs_data_set_default_int(settings, "locked_track_id", 0);
	obs_data_set_default_bool(settings, "lock_auto_relock", true);
	obs_data_set_default_string(settings, "save_detections_path", "");
	obs_data_set_default_bool(settings, "crop_group", false);
	obs_data_set_default_int(settings, "crop_left", 0);
	obs_data_set_default_int(settings, "crop_right", 0);
	obs_data_set_default_int(settings, "crop_top", 0);
	obs_data_set_default_int(settings, "crop_bottom", 0);
}

void detect_filter_update(void *data, obs_data_t *settings)
{
	obs_log(LOG_INFO, "Detect filter update");

	struct detect_filter *tf = reinterpret_cast<detect_filter *>(data);

	tf->isDisabled = true;

	tf->preview = obs_data_get_bool(settings, "preview");
	tf->conf_threshold = (float)obs_data_get_double(settings, "threshold");
	tf->objectCategory = (int)obs_data_get_int(settings, "object_category");
	tf->maskingEnabled = obs_data_get_bool(settings, "masking_group");
	tf->maskingType = obs_data_get_string(settings, "masking_type");
	tf->maskingColor = (int)obs_data_get_int(settings, "masking_color");
	tf->maskingBlurRadius = (int)obs_data_get_int(settings, "masking_blur_radius");
	tf->maskingDilateIterations = (int)obs_data_get_int(settings, "dilation_iterations");
	bool newTrackingEnabled = obs_data_get_bool(settings, "tracking_group");
	tf->zoomFactor = (float)obs_data_get_double(settings, "zoom_factor");
	tf->zoomSpeedFactor = (float)obs_data_get_double(settings, "zoom_speed_factor");
	tf->zoomSizeSpeedFactor = (float)obs_data_get_double(settings, "zoom_size_speed_factor");
	tf->zoomObject = obs_data_get_string(settings, "zoom_object");
	tf->lockedTrackId = obs_data_get_int(settings, "locked_track_id");
	tf->lockAutoRelock = obs_data_get_bool(settings, "lock_auto_relock");
	tf->sortTracking = obs_data_get_bool(settings, "sort_tracking");
	size_t maxUnseenFrames = (size_t)obs_data_get_int(settings, "max_unseen_frames");
	if (tf->tracker.getMaxUnseenFrames() != maxUnseenFrames) {
		tf->tracker.setMaxUnseenFrames(maxUnseenFrames);
	}
	tf->showUnseenObjects = obs_data_get_bool(settings, "show_unseen_objects");
	tf->saveDetectionsPath = obs_data_get_string(settings, "save_detections_path");
	tf->crop_enabled = obs_data_get_bool(settings, "crop_group");
	tf->crop_left = (int)obs_data_get_int(settings, "crop_left");
	tf->crop_right = (int)obs_data_get_int(settings, "crop_right");
	tf->crop_top = (int)obs_data_get_int(settings, "crop_top");
	tf->crop_bottom = (int)obs_data_get_int(settings, "crop_bottom");
	tf->minAreaThreshold = (int)obs_data_get_int(settings, "min_size_threshold");

	// check if tracking state has changed
	if (tf->trackingEnabled != newTrackingEnabled) {
		tf->trackingEnabled = newTrackingEnabled;
		obs_source_t *parent = obs_filter_get_parent(tf->source);
		if (!parent) {
			// during scene load this update runs before the filter is
			// attached to its parent. Returning here would abort the whole
			// update - including model initialization below - leaving the
			// filter completely dead until the user re-saves settings.
			// Defer instead: the video pipeline re-acquires the tracking
			// crop filter once a parent exists.
			obs_log(LOG_WARNING,
				"Parent source not found - deferring tracking filter setup");
		} else if (tf->trackingEnabled) {
			obs_log(LOG_DEBUG, "Tracking enabled");
			// get the parent of the source
			// check if it has a crop/pad filter
			obs_source_t *crop_pad_filter =
				obs_source_get_filter_by_name(parent, "Detect Tracking");
			if (!crop_pad_filter) {
				// create a crop-pad filter
				crop_pad_filter = obs_source_create(
					"crop_filter", "Detect Tracking", nullptr, nullptr);
				// add a crop/pad filter to the source
				// set the parent of the crop/pad filter to the parent of the source
				obs_source_filter_add(parent, crop_pad_filter);
			}
			tf->trackingFilter = crop_pad_filter;
		} else {
			obs_log(LOG_DEBUG, "Tracking disabled");
			// remove the crop/pad filter
			obs_source_t *crop_pad_filter =
				obs_source_get_filter_by_name(parent, "Detect Tracking");
			if (crop_pad_filter) {
				obs_source_filter_remove(parent, crop_pad_filter);
			}
			tf->trackingFilter = nullptr;
		}
	}

	const std::string newUseGpu = obs_data_get_string(settings, "useGPU");
	const uint32_t newNumThreads = (uint32_t)obs_data_get_int(settings, "numThreads");
	const std::string newModelSize = obs_data_get_string(settings, "model_size");

	bool reinitialize = false;
	if (tf->useGPU != newUseGpu || tf->numThreads != newNumThreads ||
	    tf->modelSize != newModelSize) {
		obs_log(LOG_INFO, "Reinitializing model");
		reinitialize = true;

		// lock modelMutex
		std::unique_lock<std::mutex> lock(tf->modelMutex);

		char *modelFilepath_rawPtr = nullptr;
		if (newModelSize == "small") {
			modelFilepath_rawPtr =
				obs_module_file("models/edgeyolo_tiny_lrelu_coco_256x416.onnx");
		} else if (newModelSize == "medium") {
			modelFilepath_rawPtr =
				obs_module_file("models/edgeyolo_tiny_lrelu_coco_480x800.onnx");
		} else if (newModelSize == "large") {
			modelFilepath_rawPtr =
				obs_module_file("models/edgeyolo_tiny_lrelu_coco_736x1280.onnx");
		} else if (newModelSize == FACE_DETECT_MODEL_SIZE) {
			modelFilepath_rawPtr =
				obs_module_file("models/face_detection_yunet_2023mar.onnx");
		} else if (newModelSize == EXTERNAL_MODEL_SIZE) {
			const char *external_model_file =
				obs_data_get_string(settings, "external_model_file");
			if (external_model_file == nullptr || external_model_file[0] == '\0' ||
			    strlen(external_model_file) == 0) {
				obs_log(LOG_ERROR, "External model file path is empty");
				tf->isDisabled = true;
				return;
			}
			modelFilepath_rawPtr = bstrdup(external_model_file);
		} else {
			obs_log(LOG_ERROR, "Invalid model size: %s", newModelSize.c_str());
			tf->isDisabled = true;
			return;
		}

		if (modelFilepath_rawPtr == nullptr) {
			obs_log(LOG_ERROR, "Unable to get model filename from plugin.");
			tf->isDisabled = true;
			return;
		}

#if _WIN32
		int outLength = MultiByteToWideChar(CP_ACP, MB_PRECOMPOSED, modelFilepath_rawPtr,
						    -1, nullptr, 0);
		tf->modelFilepath = std::wstring(outLength, L'\0');
		MultiByteToWideChar(CP_ACP, MB_PRECOMPOSED, modelFilepath_rawPtr, -1,
				    tf->modelFilepath.data(), outLength);
#else
		tf->modelFilepath = std::string(modelFilepath_rawPtr);
#endif
		bfree(modelFilepath_rawPtr);

		// Re-initialize model if it's not already the selected one or switching inference device
		tf->useGPU = newUseGpu;
		tf->numThreads = newNumThreads;
		tf->modelSize = newModelSize;

		// parameters
		int onnxruntime_device_id_ = 0;
		bool onnxruntime_use_parallel_ = true;
		float nms_th_ = 0.45f;
		int num_classes_ = (int)edgeyolo_cpp::COCO_CLASSES.size();
		tf->classNames = edgeyolo_cpp::COCO_CLASSES;

		// If this is an external model - look for the config JSON file
		if (tf->modelSize == EXTERNAL_MODEL_SIZE) {
#ifdef _WIN32
			std::wstring labelsFilepath = tf->modelFilepath;
			labelsFilepath.replace(labelsFilepath.find(L".onnx"), 5, L".json");
#else
			std::string labelsFilepath = tf->modelFilepath;
			labelsFilepath.replace(labelsFilepath.find(".onnx"), 5, ".json");
#endif
			std::ifstream labelsFile(labelsFilepath);
			if (labelsFile.is_open()) {
				// Parse the JSON file
				nlohmann::json j;
				labelsFile >> j;
				if (j.contains("names")) {
					std::vector<std::string> labels = j["names"];
					num_classes_ = (int)labels.size();
					tf->classNames = labels;
				} else {
					obs_log(LOG_ERROR,
						"JSON file does not contain 'labels' field");
					tf->isDisabled = true;
					tf->onnxruntimemodel.reset();
					return;
				}
			} else {
				obs_log(LOG_ERROR, "Failed to open JSON file: %s",
					labelsFilepath.c_str());
				tf->isDisabled = true;
				tf->onnxruntimemodel.reset();
				return;
			}
		} else if (tf->modelSize == FACE_DETECT_MODEL_SIZE) {
			num_classes_ = 1;
			tf->classNames = yunet::FACE_CLASSES;
		}

		// Load model
		try {
			if (tf->onnxruntimemodel) {
				tf->onnxruntimemodel.reset();
			}
			if (tf->modelSize == FACE_DETECT_MODEL_SIZE) {
				tf->onnxruntimemodel = std::make_unique<yunet::YuNetONNX>(
					tf->modelFilepath, tf->numThreads, 50, tf->numThreads,
					tf->useGPU, onnxruntime_device_id_,
					onnxruntime_use_parallel_, nms_th_, tf->conf_threshold);
			} else {
				tf->onnxruntimemodel =
					std::make_unique<edgeyolo_cpp::EdgeYOLOONNXRuntime>(
						tf->modelFilepath, tf->numThreads, num_classes_,
						tf->numThreads, tf->useGPU, onnxruntime_device_id_,
						onnxruntime_use_parallel_, nms_th_,
						tf->conf_threshold);
			}
			// clear error message
			obs_data_set_string(settings, "error", "");
		} catch (const std::exception &e) {
			obs_log(LOG_ERROR, "Failed to load model: %s", e.what());
			// disable filter
			tf->isDisabled = true;
			tf->onnxruntimemodel.reset();
			return;
		}
	}

	// update threshold on edgeyolo
	if (tf->onnxruntimemodel) {
		tf->onnxruntimemodel->setBBoxConfThresh(tf->conf_threshold);
	}

	if (reinitialize) {
		// Log the currently selected options
		obs_log(LOG_INFO, "Detect Filter Options:");
		// name of the source that the filter is attached to
		obs_log(LOG_INFO, "  Source: %s", obs_source_get_name(tf->source));
		obs_log(LOG_INFO, "  Inference Device: %s", tf->useGPU.c_str());
		obs_log(LOG_INFO, "  Num Threads: %d", tf->numThreads);
		obs_log(LOG_INFO, "  Model Size: %s", tf->modelSize.c_str());
		obs_log(LOG_INFO, "  Preview: %s", tf->preview ? "true" : "false");
		obs_log(LOG_INFO, "  Threshold: %.2f", tf->conf_threshold);
		obs_log(LOG_INFO, "  Object Category: %s",
			obs_data_get_string(settings, "object_category"));
		obs_log(LOG_INFO, "  Masking Enabled: %s",
			obs_data_get_bool(settings, "masking_group") ? "true" : "false");
		obs_log(LOG_INFO, "  Masking Type: %s",
			obs_data_get_string(settings, "masking_type"));
		obs_log(LOG_INFO, "  Masking Color: %s",
			obs_data_get_string(settings, "masking_color"));
		obs_log(LOG_INFO, "  Masking Blur Radius: %d",
			obs_data_get_int(settings, "masking_blur_radius"));
		obs_log(LOG_INFO, "  Tracking Enabled: %s",
			obs_data_get_bool(settings, "tracking_group") ? "true" : "false");
		obs_log(LOG_INFO, "  Zoom Factor: %.2f",
			obs_data_get_double(settings, "zoom_factor"));
		obs_log(LOG_INFO, "  Zoom Object: %s",
			obs_data_get_string(settings, "zoom_object"));
		obs_log(LOG_INFO, "  Disabled: %s", tf->isDisabled ? "true" : "false");
#ifdef _WIN32
		obs_log(LOG_INFO, "  Model file path: %ls", tf->modelFilepath.c_str());
#else
		obs_log(LOG_INFO, "  Model file path: %s", tf->modelFilepath.c_str());
#endif
	}

	// enable
	tf->isDisabled = false;
}

void detect_filter_activate(void *data)
{
	obs_log(LOG_INFO, "Detect filter activated");
	struct detect_filter *tf = reinterpret_cast<detect_filter *>(data);
	tf->isDisabled = false;
}

void detect_filter_deactivate(void *data)
{
	obs_log(LOG_INFO, "Detect filter deactivated");
	struct detect_filter *tf = reinterpret_cast<detect_filter *>(data);
	tf->isDisabled = true;
}

/**                   FILTER CORE                     */

void *detect_filter_create(obs_data_t *settings, obs_source_t *source)
{
	obs_log(LOG_INFO, "Detect filter created");
	void *data = bmalloc(sizeof(struct detect_filter));
	struct detect_filter *tf = new (data) detect_filter();

	tf->source = source;
	tf->texrender = gs_texrender_create(GS_BGRA, GS_ZS_NONE);
	tf->lastDetectedObjectId = -1;
	tf->boxHeightHistN = 0;
	tf->boxHeightHistIdx = 0;
	tf->lockedTrackId = 0;
	tf->tracksReportTick = 0;
	tf->lockAutoRelock = true;
	tf->lockLastValid = false;
	tf->lockDeathTicks = 0;
	tf->sizeEnv = 0.0f;
	tf->sizeEnvSmallTicks = 0;

	std::vector<std::tuple<const char *, gs_effect_t **>> effects = {
		{KAWASE_BLUR_EFFECT_PATH, &tf->kawaseBlurEffect},
		{MASKING_EFFECT_PATH, &tf->maskingEffect},
		{PIXELATE_EFFECT_PATH, &tf->pixelateEffect},
	};

	for (auto [effectPath, effect] : effects) {
		char *effectPathPtr = obs_module_file(effectPath);
		if (!effectPathPtr) {
			obs_log(LOG_ERROR, "Failed to get effect path: %s", effectPath);
			tf->isDisabled = true;
			return tf;
		}
		obs_enter_graphics();
		*effect = gs_effect_create_from_file(effectPathPtr, nullptr);
		bfree(effectPathPtr);
		if (!*effect) {
			obs_log(LOG_ERROR, "Failed to load effect: %s", effectPath);
			tf->isDisabled = true;
			return tf;
		}
		obs_leave_graphics();
	}

	detect_filter_update(tf, settings);

	return tf;
}

void detect_filter_destroy(void *data)
{
	obs_log(LOG_INFO, "Detect filter destroyed");

	struct detect_filter *tf = reinterpret_cast<detect_filter *>(data);

	if (tf) {
		tf->isDisabled = true;

		obs_enter_graphics();
		gs_texrender_destroy(tf->texrender);
		if (tf->stagesurface) {
			gs_stagesurface_destroy(tf->stagesurface);
		}
		gs_effect_destroy(tf->kawaseBlurEffect);
		gs_effect_destroy(tf->maskingEffect);
		obs_leave_graphics();
		tf->~detect_filter();
		bfree(tf);
	}
}

void detect_filter_video_tick(void *data, float seconds)
{
	UNUSED_PARAMETER(seconds);

	struct detect_filter *tf = reinterpret_cast<detect_filter *>(data);

	if (tf->isDisabled || !tf->onnxruntimemodel) {
		return;
	}

	if (!obs_source_enabled(tf->source)) {
		return;
	}

	cv::Mat imageBGRA;
	{
		std::unique_lock<std::mutex> lock(tf->inputBGRALock, std::try_to_lock);
		if (!lock.owns_lock()) {
			// No data to process
			return;
		}
		if (tf->inputBGRA.empty()) {
			// No data to process
			return;
		}
		imageBGRA = tf->inputBGRA.clone();
	}

	cv::Mat inferenceFrame;

	// the captured frame is the parent's RAW output (no filters), so the
	// detection region applies directly in source pixels; guard against
	// out-of-range values instead of trusting them blindly
	cv::Rect cropRect(0, 0, imageBGRA.cols, imageBGRA.rows);
	if (tf->crop_enabled) {
		cv::Rect fence(tf->crop_left, tf->crop_top,
			       imageBGRA.cols - tf->crop_left - tf->crop_right,
			       imageBGRA.rows - tf->crop_top - tf->crop_bottom);
		fence &= cv::Rect(0, 0, imageBGRA.cols, imageBGRA.rows);
		if (fence.width > 20 && fence.height > 20) {
			cropRect = fence;
		}
	}
	if (cropRect.width < imageBGRA.cols || cropRect.height < imageBGRA.rows) {
		cv::cvtColor(imageBGRA(cropRect), inferenceFrame, cv::COLOR_BGRA2BGR);
	} else {
		cv::cvtColor(imageBGRA, inferenceFrame, cv::COLOR_BGRA2BGR);
	}

	std::vector<Object> objects;

	try {
		std::unique_lock<std::mutex> lock(tf->modelMutex);
		objects = tf->onnxruntimemodel->inference(inferenceFrame);
	} catch (const Ort::Exception &e) {
		obs_log(LOG_ERROR, "ONNXRuntime Exception: %s", e.what());
	} catch (const std::exception &e) {
		obs_log(LOG_ERROR, "%s", e.what());
	}

	if (tf->crop_enabled) {
		// translate the detected objects to the original frame
		for (Object &obj : objects) {
			obj.rect.x += (float)cropRect.x;
			obj.rect.y += (float)cropRect.y;
		}
	}

	// update the detected object text input
	if (objects.size() > 0) {
		if (tf->lastDetectedObjectId != objects[0].label) {
			tf->lastDetectedObjectId = objects[0].label;
			// get source settings
			obs_data_t *source_settings = obs_source_get_settings(tf->source);
			obs_data_set_string(source_settings, "detected_object",
					    tf->classNames[objects[0].label].c_str());
			// release the source settings
			obs_data_release(source_settings);
		}
	} else {
		if (tf->lastDetectedObjectId != -1) {
			tf->lastDetectedObjectId = -1;
			// get source settings
			obs_data_t *source_settings = obs_source_get_settings(tf->source);
			obs_data_set_string(source_settings, "detected_object", "");
			// release the source settings
			obs_data_release(source_settings);
		}
	}

	if (tf->minAreaThreshold > 0) {
		std::vector<Object> filtered_objects;
		for (const Object &obj : objects) {
			if (obj.rect.area() > (float)tf->minAreaThreshold) {
				filtered_objects.push_back(obj);
			}
		}
		objects = filtered_objects;
	}

	if (tf->objectCategory != -1) {
		std::vector<Object> filtered_objects;
		for (const Object &obj : objects) {
			if (obj.label == tf->objectCategory) {
				filtered_objects.push_back(obj);
			}
		}
		objects = filtered_objects;
	}

	if (tf->sortTracking) {
		objects = tf->tracker.update(objects);
	}

	if (!tf->showUnseenObjects) {
		objects.erase(
			std::remove_if(objects.begin(), objects.end(),
				       [](const Object &obj) { return obj.unseenFrames > 0; }),
			objects.end());
	}

	if (!tf->saveDetectionsPath.empty()) {
		std::ofstream detectionsFile(tf->saveDetectionsPath);
		if (detectionsFile.is_open()) {
			nlohmann::json j;
			for (const Object &obj : objects) {
				nlohmann::json obj_json;
				obj_json["label"] = obj.label;
				obj_json["confidence"] = obj.prob;
				obj_json["rect"] = {{"x", obj.rect.x},
						    {"y", obj.rect.y},
						    {"width", obj.rect.width},
						    {"height", obj.rect.height}};
				obj_json["id"] = obj.id;
				j.push_back(obj_json);
			}
			detectionsFile << j.dump(4);
			detectionsFile.close();
		} else {
			obs_log(LOG_ERROR, "Failed to open file for writing detections: %s",
				tf->saveDetectionsPath.c_str());
		}
	}

	if (tf->preview || tf->maskingEnabled) {
		cv::Mat frame;
		cv::cvtColor(imageBGRA, frame, cv::COLOR_BGRA2BGR);

		if (tf->preview && tf->crop_enabled) {
			// draw the crop rectangle on the frame in a dashed line
			drawDashedRectangle(frame, cropRect, cv::Scalar(0, 255, 0), 5, 8, 15);
		}
		if (tf->preview && objects.size() > 0) {
			draw_objects(frame, objects, tf->classNames);
		}
		if (tf->maskingEnabled) {
			cv::Mat mask = cv::Mat::zeros(frame.size(), CV_8UC1);
			for (const Object &obj : objects) {
				cv::rectangle(mask, obj.rect, cv::Scalar(255), -1);
			}
			std::lock_guard<std::mutex> lock(tf->outputLock);
			mask.copyTo(tf->outputMask);

			if (tf->maskingDilateIterations > 0) {
				cv::Mat dilatedMask;
				cv::dilate(tf->outputMask, dilatedMask, cv::Mat(),
					   cv::Point(-1, -1), tf->maskingDilateIterations);
				dilatedMask.copyTo(tf->outputMask);
			}
		}

		std::lock_guard<std::mutex> lock(tf->outputLock);
		cv::cvtColor(frame, tf->outputPreviewBGRA, cv::COLOR_BGR2BGRA);
	}

	if (tf->trackingEnabled && !tf->trackingFilter) {
		// the startup settings-update can run before this filter is attached
		// to its parent source ("Parent source not found"), which leaves the
		// crop-filter handle unset and tracking silently dead until the user
		// toggles it. Re-acquire the handle here once the parent exists.
		obs_source_t *parent = obs_filter_get_parent(tf->source);
		if (parent) {
			obs_source_t *crop_pad_filter =
				obs_source_get_filter_by_name(parent, "Detect Tracking");
			if (!crop_pad_filter) {
				crop_pad_filter = obs_source_create(
					"crop_filter", "Detect Tracking", nullptr, nullptr);
				obs_source_filter_add(parent, crop_pad_filter);
			}
			tf->trackingFilter = crop_pad_filter;
			obs_log(LOG_INFO, "Tracking crop filter acquired after startup");
		}
	}

	if (tf->trackingEnabled && tf->trackingFilter) {
		// the captured frame is the parent's raw output, so detections are
		// already in raw source pixels - the same space the crop filter
		// operates in
		const int rawW = imageBGRA.cols;
		const int rawH = imageBGRA.rows;

		// select the target object; only currently visible objects may be
		// targeted (unseen ones are Kalman predictions that degenerate
		// without corrections)
		bool found = false;
		cv::Rect2f targetBox;
		// DIRECTOR LOCK: a specific tracker identity picked from the panel
		// wins over every automatic mode, so crowded floors cannot steal the
		// frame. The automatic mode below only runs while the locked identity
		// is not currently visible (occluded or gone).
		// DIRECTOR LOCK v2 - auto-relock. While the locked person is
		// visible, remember where they were. If their track dies entirely
		// (occlusion beyond max_unseen), watch for up to ~4 s for a NEW
		// track appearing near that spot with a similar size and transfer
		// the lock to it. People already visible at the moment of death
		// (the occluder) are never candidates.
		if (tf->lockedTrackId > 0) {
			bool lockedPresent = false;
			for (const Object &obj : objects) {
				if ((int64_t)obj.id + 1 == tf->lockedTrackId) {
					lockedPresent = true;
					if (obj.unseenFrames == 0) {
						tf->lockLastRect = obj.rect;
						tf->lockLastValid = true;
					}
					break;
				}
			}
			if (lockedPresent) {
				tf->lockDeathTicks = 0;
				tf->lockAliveAtDeath.clear();
			} else if (tf->lockLastValid) {
				if (tf->lockDeathTicks == 0) {
					tf->lockAliveAtDeath.clear();
					for (const Object &obj : objects)
						tf->lockAliveAtDeath.push_back(obj.id);
				}
				tf->lockDeathTicks++;
				if (tf->lockAutoRelock && tf->lockDeathTicks <= 240) {
					float cx =
						tf->lockLastRect.x + tf->lockLastRect.width * 0.5f;
					float cy =
						tf->lockLastRect.y + tf->lockLastRect.height * 0.5f;
					float bestDist = (float)rawW * 0.25f;
					int64_t bestId = 0;
					cv::Rect2f bestRect;
					for (const Object &obj : objects) {
						if (obj.unseenFrames > 0)
							continue;
						bool wasAlive = false;
						for (size_t ai = 0;
						     ai < tf->lockAliveAtDeath.size(); ai++) {
							if (tf->lockAliveAtDeath[ai] == obj.id) {
								wasAlive = true;
								break;
							}
						}
						if (wasAlive)
							continue;
						float hr = obj.rect.height /
							   std::max(1.0f, tf->lockLastRect.height);
						if (hr < 0.5f || hr > 2.0f)
							continue;
						float ox = obj.rect.x + obj.rect.width * 0.5f;
						float oy = obj.rect.y + obj.rect.height * 0.5f;
						float d = std::hypot(ox - cx, oy - cy);
						if (d < bestDist) {
							bestDist = d;
							bestId = (int64_t)obj.id + 1;
							bestRect = obj.rect;
						}
					}
					if (bestId > 0) {
						obs_log(LOG_INFO,
							"director lock re-attached %lld -> %lld",
							(long long)tf->lockedTrackId,
							(long long)bestId);
						tf->lockedTrackId = bestId;
						tf->lockLastRect = bestRect;
						tf->lockDeathTicks = 0;
						tf->lockAliveAtDeath.clear();
						obs_data_t *st =
							obs_source_get_settings(tf->source);
						obs_data_set_int(st, "locked_track_id", bestId);
						obs_data_release(st);
					}
				}
			}
		}

		// NOTE: locked_track_id carries SORT id + 1, because SORT ids start
		// at 0 and 0 must keep meaning "automatic" (the very first track of
		// a session would otherwise be unlockable)
		if (tf->lockedTrackId > 0) {
			for (const Object &obj : objects) {
				if ((int64_t)obj.id + 1 == tf->lockedTrackId &&
				    obj.unseenFrames == 0) {
					targetBox = obj.rect;
					found = true;
					break;
				}
			}
		}
		if (!found) {
			if (tf->zoomObject == "single") {
				for (const Object &obj : objects) {
					if (obj.unseenFrames == 0) {
						targetBox = obj.rect;
						found = true;
						break;
					}
				}
			} else if (tf->zoomObject == "biggest") {
				float maxArea = 0;
				for (const Object &obj : objects) {
					if (obj.unseenFrames > 0)
						continue;
					const float area = obj.rect.width * obj.rect.height;
					if (area > maxArea) {
						maxArea = area;
						targetBox = obj.rect;
						found = true;
					}
				}
			} else if (tf->zoomObject == "oldest") {
				uint64_t oldestId = UINT64_MAX;
				for (const Object &obj : objects) {
					if (obj.unseenFrames == 0 && obj.id < oldestId) {
						oldestId = obj.id;
						targetBox = obj.rect;
						found = true;
					}
				}
			} else {
				// bounding box of all currently visible objects
				for (const Object &obj : objects) {
					if (obj.unseenFrames > 0)
						continue;
					if (!found) {
						targetBox = obj.rect;
						found = true;
					} else {
						targetBox |= obj.rect;
					}
				}
			}
		}

		// degenerate/non-finite boxes count as lost
		cv::Rect2f boundingBox;
		if (found) {
			boundingBox = targetBox;
			if (!std::isfinite(boundingBox.x) || !std::isfinite(boundingBox.y) ||
			    !std::isfinite(boundingBox.width) ||
			    !std::isfinite(boundingBox.height) || boundingBox.width < 2.0f ||
			    boundingBox.height < 2.0f) {
				found = false;
			}
		}
		if (!found) {
			boundingBox = cv::Rect2f(0, 0, (float)rawW, (float)rawH);
		}
		bool lostTracking = !found;

		// publish the currently visible tracks and the lock state into the
		// filter's settings a few times a second. The control panel reads
		// them with GetSourceFilter and renders a click-to-lock picker -
		// same reporting channel the "error" key already uses. Coordinates
		// are normalized to the raw frame.
		if (++tf->tracksReportTick >= 15) {
			tf->tracksReportTick = 0;
			std::string tj = "[";
			bool firstTrack = true;
			bool lockVisible = false;
			for (const Object &obj : objects) {
				if (obj.unseenFrames > 0)
					continue;
				if ((int64_t)obj.id + 1 == tf->lockedTrackId)
					lockVisible = true;
				char buf[160];
				snprintf(
					buf, sizeof(buf),
					"%s{\"id\":%llu,\"x\":%.3f,\"y\":%.3f,\"w\":%.3f,\"h\":%.3f}",
					firstTrack ? "" : ",", (unsigned long long)(obj.id + 1),
					obj.rect.x / (float)rawW, obj.rect.y / (float)rawH,
					obj.rect.width / (float)rawW,
					obj.rect.height / (float)rawH);
				firstTrack = false;
				tj += buf;
			}
			tj += "]";
			obs_data_t *st = obs_source_get_settings(tf->source);
			obs_data_set_string(st, "tracks_json", tj.c_str());
			obs_data_set_string(
				st, "lock_status",
				tf->lockedTrackId > 0 ? (lockVisible ? "locked" : "lost") : "off");
			obs_data_release(st);
		}

		// the zooming box maintains the raw frame's aspect ratio, with
		// tf->zoomFactor controlling the buffer around the bounding box
		float frameAspectRatio = (float)rawW / (float)rawH;
		float boxHeight = std::min(boundingBox.height, (float)rawH);
		// median-of-5 on the target height: single-frame detection spikes and
		// brief identity switches must not move the zoom target. History
		// resets while lost, so re-acquire starts fresh (lost path itself
		// keeps the raw full-frame height and is unaffected).
		if (lostTracking) {
			tf->boxHeightHistN = 0;
			tf->boxHeightHistIdx = 0;
		} else {
			tf->boxHeightHist[tf->boxHeightHistIdx] = boxHeight;
			tf->boxHeightHistIdx = (tf->boxHeightHistIdx + 1) % 5;
			if (tf->boxHeightHistN < 5)
				tf->boxHeightHistN++;
			float sortedHeights[5];
			std::copy(tf->boxHeightHist, tf->boxHeightHist + tf->boxHeightHistN,
				  sortedHeights);
			std::sort(sortedHeights, sortedHeights + tf->boxHeightHistN);
			boxHeight = sortedHeights[tf->boxHeightHistN / 2];
		}
		// POSTURE HOLD: crouches and aims must not pump the frame. Track a
		// rising-max envelope of the target height: it rises instantly with
		// the target, but only decays toward a smaller target after the
		// target has stayed smaller for a ~2 s grace period. Posture dips
		// shorter than the grace leave the frame perfectly still; a real
		// retreat starts re-framing right after the grace.
		if (lostTracking) {
			tf->sizeEnv = 0.0f;
			tf->sizeEnvSmallTicks = 0;
		} else {
			if (boxHeight >= tf->sizeEnv) {
				tf->sizeEnv = boxHeight;
				tf->sizeEnvSmallTicks = 0;
			} else {
				tf->sizeEnvSmallTicks++;
				if (tf->sizeEnvSmallTicks > 120)
					tf->sizeEnv += std::max(tf->zoomSizeSpeedFactor, 0.005f) *
						       (boxHeight - tf->sizeEnv);
			}
			boxHeight = tf->sizeEnv;
		}
		float dh = (float)rawH - boxHeight;
		float buffer = dh * (1.0f - tf->zoomFactor);
		float zh = boxHeight + buffer;
		// hard floor: the zoom window can never implode, whatever happens
		zh = std::max(zh, (float)rawH * 0.15f);
		float zw = zh * frameAspectRatio;

		if (tf->trackingRect.width == 0) {
			// initialize the trackingRect centered on the target
			float zx0 = boundingBox.x - (zw - boundingBox.width) / 2.0f;
			float zy0 = boundingBox.y - (zh - boundingBox.height) / 2.0f;
			zw = std::min(zw, (float)rawW);
			zh = std::min(zh, (float)rawH);
			zx0 = std::max(0.0f, std::min(zx0, (float)rawW - zw));
			zy0 = std::max(0.0f, std::min(zy0, (float)rawH - zh));
			tf->trackingRect = cv::Rect2f(zx0, zy0, zw, zh);
		} else {
			// PAN and ZOOM are fully independent axes (camera-operator
			// model), each with its OWN smoothing speed. Coupling them —
			// or switching between size branches (the 0.0.10 deadband) —
			// makes the position target jump when the size target changes,
			// which reads as shake. Track speed drives pan; zoom (size)
			// speed drives resize. Both slow to 0.2x while re-acquiring.
			float posF = tf->zoomSpeedFactor * (lostTracking ? 0.2f : 1.0f);
			// ASYMMETRIC zoom (broadcast auto-framing behaviour): widening
			// (actor grew / came closer — protective) runs at 4x the slider
			// rate; tightening creeps at the slider rate. Pumping cannot
			// sustain when one direction crawls: the window parks near the
			// recent maximum and only slowly creeps back in.
			float sizeInF = tf->zoomSizeSpeedFactor * (lostTracking ? 0.2f : 1.0f);
			float sizeOutF = std::min(sizeInF * 4.0f, 0.06f);

			// SIZE: soft deadband — ignore the first 8% of size error
			// entirely (posture noise), follow anything beyond it at the
			// zoom-speed rate. Continuous in the input: no branch, no dither.
			float sizeErr = zh - tf->trackingRect.height;
			float dead = tf->trackingRect.height * 0.08f;
			if (sizeErr > dead)
				sizeErr -= dead;
			else if (sizeErr < -dead)
				sizeErr += dead;
			else
				sizeErr = 0.0f;
			tf->trackingRect.height += (sizeErr > 0.0f ? sizeOutF : sizeInF) * sizeErr;
			tf->trackingRect.width = tf->trackingRect.height * frameAspectRatio;

			// POSITION: center the CURRENT window on the target's center —
			// the target never depends on the size math, so size noise
			// cannot shake the pan
			float cx = boundingBox.x + boundingBox.width * 0.5f;
			float cy = boundingBox.y + boundingBox.height * 0.5f;
			float tx = cx - tf->trackingRect.width * 0.5f;
			float ty = cy - tf->trackingRect.height * 0.5f;
			// hug frame edges instead of padding out-of-frame black
			tx = std::max(0.0f, std::min(tx, (float)rawW - tf->trackingRect.width));
			ty = std::max(0.0f, std::min(ty, (float)rawH - tf->trackingRect.height));
			tf->trackingRect.x += posF * (tx - tf->trackingRect.x);
			tf->trackingRect.y += posF * (ty - tf->trackingRect.y);
		}

		// clamp the smoothed rect in raw space, with the same implosion
		// floor, so the applied crop is always a sane window
		tf->trackingRect.width = std::max(std::min(tf->trackingRect.width, (float)rawW),
						  (float)rawW * 0.15f);
		tf->trackingRect.height = std::max(std::min(tf->trackingRect.height, (float)rawH),
						   (float)rawH * 0.15f);
		tf->trackingRect.x = std::max(0.0f, std::min(tf->trackingRect.x,
							     (float)rawW - tf->trackingRect.width));
		tf->trackingRect.y = std::max(
			0.0f, std::min(tf->trackingRect.y, (float)rawH - tf->trackingRect.height));

		// apply to the crop/pad filter (values are raw-frame pixels)
		obs_data_t *crop_pad_settings = obs_source_get_settings(tf->trackingFilter);
		obs_data_set_int(crop_pad_settings, "left", (int)tf->trackingRect.x);
		obs_data_set_int(crop_pad_settings, "top", (int)tf->trackingRect.y);
		obs_data_set_int(
			crop_pad_settings, "right",
			(int)((float)rawW - (tf->trackingRect.x + tf->trackingRect.width)));
		obs_data_set_int(
			crop_pad_settings, "bottom",
			(int)((float)rawH - (tf->trackingRect.y + tf->trackingRect.height)));
		obs_source_update(tf->trackingFilter, crop_pad_settings);
		obs_data_release(crop_pad_settings);

		// throttled geometry diagnostic: lets stability be verified from the
		// log with measured numbers instead of assumptions
		static std::atomic<int> diag_counter{0};
		if (++diag_counter % 600 == 0) {
			obs_log(LOG_INFO,
				"[detect-diag] '%s' captured %dx%d %s box %.0fx%.0f rect %.0f,%.0f %.0fx%.0f",
				obs_source_get_name(obs_filter_get_parent(tf->source)), rawW, rawH,
				lostTracking ? "lost" : "live", boundingBox.width,
				boundingBox.height, tf->trackingRect.x, tf->trackingRect.y,
				tf->trackingRect.width, tf->trackingRect.height);
		}
	}
}

void detect_filter_video_render(void *data, gs_effect_t *_effect)
{
	UNUSED_PARAMETER(_effect);

	struct detect_filter *tf = reinterpret_cast<detect_filter *>(data);

	if (tf->isDisabled || !tf->onnxruntimemodel) {
		if (tf->source) {
			obs_source_skip_video_filter(tf->source);
		}
		return;
	}

	uint32_t width, height;
	if (!getRGBAFromStageSurface(tf, width, height)) {
		if (tf->source) {
			obs_source_skip_video_filter(tf->source);
		}
		return;
	}

	// if preview is enabled, render the image
	if (tf->preview || tf->maskingEnabled) {
		cv::Mat outputBGRA, outputMask;
		{
			// lock the outputLock mutex
			std::lock_guard<std::mutex> lock(tf->outputLock);
			if (tf->outputPreviewBGRA.empty()) {
				obs_log(LOG_ERROR, "Preview image is empty");
				if (tf->source) {
					obs_source_skip_video_filter(tf->source);
				}
				return;
			}
			if ((uint32_t)tf->outputPreviewBGRA.cols != width ||
			    (uint32_t)tf->outputPreviewBGRA.rows != height) {
				if (tf->source) {
					obs_source_skip_video_filter(tf->source);
				}
				return;
			}
			outputBGRA = tf->outputPreviewBGRA.clone();
			outputMask = tf->outputMask.clone();
		}

		gs_texture_t *tex = gs_texture_create(width, height, GS_BGRA, 1,
						      (const uint8_t **)&outputBGRA.data, 0);
		gs_texture_t *maskTexture = nullptr;
		std::string technique_name = "Draw";
		gs_eparam_t *imageParam = gs_effect_get_param_by_name(tf->maskingEffect, "image");
		gs_eparam_t *maskParam =
			gs_effect_get_param_by_name(tf->maskingEffect, "focalmask");
		gs_eparam_t *maskColorParam =
			gs_effect_get_param_by_name(tf->maskingEffect, "color");

		if (tf->maskingEnabled) {
			maskTexture = gs_texture_create(width, height, GS_R8, 1,
							(const uint8_t **)&outputMask.data, 0);
			gs_effect_set_texture(maskParam, maskTexture);
			if (tf->maskingType == "output_mask") {
				technique_name = "DrawMask";
			} else if (tf->maskingType == "blur") {
				gs_texture_destroy(tex);
				tex = blur_image(tf, width, height, maskTexture);
			} else if (tf->maskingType == "pixelate") {
				gs_texture_destroy(tex);
				tex = pixelate_image(tf, width, height, maskTexture,
						     (float)tf->maskingBlurRadius);
			} else if (tf->maskingType == "transparent") {
				technique_name = "DrawSolidColor";
				gs_effect_set_color(maskColorParam, 0);
			} else if (tf->maskingType == "solid_color") {
				technique_name = "DrawSolidColor";
				gs_effect_set_color(maskColorParam, tf->maskingColor);
			}
		}

		gs_effect_set_texture(imageParam, tex);

		while (gs_effect_loop(tf->maskingEffect, technique_name.c_str())) {
			gs_draw_sprite(tex, 0, 0, 0);
		}

		gs_texture_destroy(tex);
		gs_texture_destroy(maskTexture);
	} else {
		obs_source_skip_video_filter(tf->source);
	}
	return;
}
