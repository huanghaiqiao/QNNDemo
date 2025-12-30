

#include <jni.h>
#include <memory>
#include <mutex>
#include <string>
#include <android/log.h>

#include "DynamicLoadUtil.hpp"
#include "QnnSampleApp.hpp"
#include "Logger.hpp"
#include "PAL/DynamicLoading.hpp"

#define LOG_TAG "QnnJNI"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

using namespace qnn::tools;
using namespace qnn::tools::sample_app;

static std::mutex g_mutex;

static void* sg_backendHandle{nullptr};
static void* sg_modelHandle{nullptr};
static std::unique_ptr<QnnSampleApp> sg_app{nullptr};
static QnnFunctionPointers sg_qnnFunctionPointers{};
static bool g_ready = false;


extern "C"
JNIEXPORT jint JNICALL
Java_com_mms_lightspotapplication_MainActivity_initQnnSampleApp(
        JNIEnv* env, jobject /*thiz*/, jstring jBasePath) {

    std::lock_guard<std::mutex> lock(g_mutex);

    // ---------- clean old ----------
    if (sg_app) {
        sg_app->freeContext();
        sg_app->freeDevice();
        sg_app.reset();
    }
    if (sg_backendHandle) {
        pal::dynamicloading::dlClose(sg_backendHandle);
        sg_backendHandle = nullptr;
    }
    if (sg_modelHandle) {
        pal::dynamicloading::dlClose(sg_modelHandle);
        sg_modelHandle = nullptr;
    }
    g_ready = false;

    // ---------- base path ----------
    const char* cpath = env->GetStringUTFChars(jBasePath, nullptr);
    std::string basePath(cpath);
    env->ReleaseStringUTFChars(jBasePath, cpath);

    // ---------- paths (你指定的) ----------
//    std::string backEndPath       = basePath + "/files/desensitize/qnn_lib/libQnnCpu.so";
//    std::string systemLibraryPath = basePath + "/files/desensitize/qnn_lib/libQnnSystem.so";
    std::string cachedBinaryPath  = basePath + "/files/desensitize/best_sim.bin";
    std::string inputListPaths    = basePath + "/files/desensitize/input_list.txt";
    std::string qnnLibPath        = basePath + "/files/desensitize/qnn_lib";
    std::string outputPath        = basePath + "/cache/output";

    std::string backEndPath = "libQnnHtp.so";
    std::string systemLibraryPath = "libQnnSystem.so";


    // ---------- env ----------
    setenv("LD_LIBRARY_PATH", qnnLibPath.c_str(), 1);
    setenv("ADSP_LIBRARY_PATH", qnnLibPath.c_str(), 1);

    LOGI("Backend   : %s", backEndPath.c_str());
    LOGI("SystemLib : %s", systemLibraryPath.c_str());
    LOGI("CacheBin  : %s", cachedBinaryPath.c_str());
    LOGI("InputList : %s", inputListPaths.c_str());

    // ---------- logging ----------
    if (!qnn::log::initializeLogging()) {
        LOGE("QNN logging init failed");
        return -1;
    }

    // ---------- load QNN symbols ----------
    bool loadFromCachedBinary = true;

    auto status = dynamicloadutil::getQnnFunctionPointers(
            backEndPath,
            "",                         // modelPath EMPTY (cached binary)
            &sg_qnnFunctionPointers,
            &sg_backendHandle,
            false,
            &sg_modelHandle);

    if (status != dynamicloadutil::StatusCode::SUCCESS) {
        LOGE("getQnnFunctionPointers failed");
        return -2;
    }

    status = dynamicloadutil::getQnnSystemFunctionPointers(
            systemLibraryPath,
            &sg_qnnFunctionPointers);

    if (status != dynamicloadutil::StatusCode::SUCCESS) {
        LOGE("getQnnSystemFunctionPointers failed");
        return -3;
    }

    // ---------- create app ----------
    sg_app.reset(new QnnSampleApp(
            sg_qnnFunctionPointers,
            inputListPaths,
            "",                         // op packages
            sg_backendHandle,
            outputPath,
            false,                      // debug
            iotensor::OutputDataType::FLOAT_ONLY,
            iotensor::InputDataType::FLOAT,
            ProfilingLevel::OFF,
            true,                       // dump outputs
            cachedBinaryPath,
            "",                         // save binary name
            1                           // num inferences
    ));

    if (!sg_app) {
        LOGE("QnnSampleApp create failed");
        return -4;
    }

    // ---------- init sequence (严格照 sample-app) ----------
    if (sg_app->initialize() != StatusCode::SUCCESS) return -10;
    if (sg_app->initializeBackend() != StatusCode::SUCCESS) return -11;

    auto devProp = sg_app->isDevicePropertySupported();
    if (devProp != StatusCode::FAILURE) {
        if (sg_app->createDevice() != StatusCode::SUCCESS) return -12;
    }

    if (sg_app->initializeProfiling() != StatusCode::SUCCESS) return -13;
    if (sg_app->registerOpPackages() != StatusCode::SUCCESS) return -14;

    // ---------- cached binary path ----------
    if (sg_app->createFromBinary() != StatusCode::SUCCESS) return -15;

//    if (sg_app->isFinalizeDeserializedGraphSupported() == StatusCode::SUCCESS) {
//        if (sg_app->finalizeGraphs() != StatusCode::SUCCESS) return -16;
//    }

    g_ready = true;
    LOGI("QNN init SUCCESS");
    return 0;
}


extern "C"
JNIEXPORT jint JNICALL
Java_com_mms_lightspotapplication_MainActivity_runQnnInference(
        JNIEnv*, jobject) {

    std::lock_guard<std::mutex> lock(g_mutex);
    LOGI("sg_backendHandle=%p, sg_modelHandle=%p", sg_backendHandle, sg_modelHandle);

    if (!g_ready || !sg_app) return -1;

//    if (sg_app->createFromBinary() != StatusCode::SUCCESS) return -15;
    if (sg_app->executeGraphs() != StatusCode::SUCCESS) {
        LOGE("executeGraphs failed");
        return -2;
    }
    return 0;
}
