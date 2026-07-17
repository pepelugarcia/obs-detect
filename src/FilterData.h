#ifndef FILTERDATA_H
#define FILTERDATA_H

#include <obs-module.h>
#include "ort-model/ONNXRuntimeModel.h"
#include "sort/Sort.h"

/**
  * @brief The filter_data struct
  *
  * This struct is used to store the base data needed for ORT filters.
  *
*/
struct filter_data {
	std::string useGPU;
	uint32_t numThreads;
	float conf_threshold;
	std::string modelSize;

	int minAreaThreshold;
	int objectCategory;
	bool maskingEnabled;
	std::string maskingType;
	int maskingColor;
	int maskingBlurRadius;
	int maskingDilateIterations;
	bool trackingEnabled;
	float zoomFactor;
	float zoomSpeedFactor;     // PAN/track responsiveness (how fast the crop re-centers)
	float zoomSizeSpeedFactor; // SIZE/zoom responsiveness (how fast the crop resizes)
	float boxHeightHist[5];    // rolling target-height window for the median filter
	int boxHeightHistN;        // valid entries (reset while tracking is lost)
	int boxHeightHistIdx;
	int detectInterval;      // run readback+inference every Nth rendered frame (>=1)
	int renderFrameCount;    // video_render gate counter for frame-skip
	bool inputFresh;         // a captured frame is waiting for video_tick to consume
	int64_t lockedTrackId;   // director lock: SORT track id to follow (0 = automatic)
	int tracksReportTick;    // throttle for publishing tracks_json into settings
	bool lockAutoRelock;     // v2: re-attach a dead lock to the nearest newcomer
	cv::Rect2f lockLastRect; // where the locked person was last seen visible
	bool lockLastValid;
	int lockDeathTicks;                     // ticks since the locked track vanished entirely
	std::vector<uint64_t> lockAliveAtDeath; // ids visible at death = never candidates
	std::string zoomObject;
	obs_source_t *trackingFilter;
	cv::Rect2f trackingRect;
	int lastDetectedObjectId;
	bool sortTracking;
	bool showUnseenObjects;
	std::string saveDetectionsPath;
	bool crop_enabled;
	int crop_left;
	int crop_right;
	int crop_top;
	int crop_bottom;

	// create SORT tracker
	Sort tracker;

	obs_source_t *source;
	gs_texrender_t *texrender;
	gs_stagesurf_t *stagesurface;
	gs_effect_t *kawaseBlurEffect;
	gs_effect_t *maskingEffect;
	gs_effect_t *pixelateEffect;

	cv::Mat inputBGRA;
	cv::Mat outputPreviewBGRA;
	cv::Mat outputMask;

	bool isDisabled;
	bool preview;

	std::mutex inputBGRALock;
	std::mutex outputLock;
	std::mutex modelMutex;

	std::unique_ptr<ONNXRuntimeModel> onnxruntimemodel;
	std::vector<std::string> classNames;

#if _WIN32
	std::wstring modelFilepath;
#else
	std::string modelFilepath;
#endif
};

#endif /* FILTERDATA_H */
