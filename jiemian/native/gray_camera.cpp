// Gray-only callback adapter. Callback buffers belong to the vendor SDK.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <vector>
#include <thread>
#include "MPSizectorS_API.h"
#include "MPSizectorS_Utils.h"
namespace py = pybind11;
using namespace mpapi;
class GrayCamera {
    IMPSizectorS* sensor = nullptr;
    std::mutex mutex;
    std::condition_variable ready;
    std::vector<unsigned char> pixels;
    int width = 0, height = 0;
    bool received = false, streaming = false;
    std::string error;
    static void check(bool ok, const char* message) {
        if (!ok) throw std::runtime_error(message);
    }
    void wait_ready() {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        while (std::chrono::steady_clock::now() < deadline) {
            if (sensor->GetDeviceState() == MPSizectorS_DeviceStateType_StandBy) return;
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        throw std::runtime_error("等待相机 StandBy 超时");
    }
    void callback(MPSizectorS_DataFormatType format, MPSizectorS_DataFrameUndefinedStruct raw) noexcept {
        if (format != MPSizectorS_DataFormatType_Image2D) return;
        std::lock_guard<std::mutex> lock(mutex);
        try {
            const auto frame = MPSizectorS_Utils::ConvertTo2DFrame(raw);
            width = frame.FrameInfo.DataInfo.XPixResolution;
            height = frame.FrameInfo.DataInfo.YPixResolution;
            if (width <= 0 || height <= 0 || !frame.Data) throw std::runtime_error("无效灰度帧");
            pixels.assign(frame.Data, frame.Data + static_cast<size_t>(width) * height);
            received = true;
        } catch (const std::exception& exc) { error = exc.what(); }
        catch (...) { error = "灰度回调异常"; }
        ready.notify_one();
    }
public:
    GrayCamera(int index, float exposure, unsigned char gain) {
        sensor = MPSizectorS_Factory::GetInstance(MPSizectorS_LogMediaType_Off);
        if (!sensor) throw std::runtime_error("无法创建 SDK 实例");
        check(sensor->UpdateDeviceList(), "刷新设备列表失败");
        if (index < 0 || index >= sensor->GetDeviceCount()) throw std::runtime_error("设备序号不存在");
        check(sensor->Open(sensor->GetDeviceInfo(index)), "打开设备失败");
        check(sensor->SetTriggerSource(MPSizectorS_TriggerSourceType_SoftTriggerOnly), "设置触发失败");
        check(sensor->SetHoldState(false), "解除暂停失败");
        sensor->SetAutoReconnect(true);
        wait_ready();
        check(sensor->SetBinningState(false), "设置全分辨率失败");
        check(sensor->SetWorkingMode(MPSizectorS_WorkingModeType_2D_White), "设置 2D 模式失败");
        check(sensor->SetExposureIntensity2D(exposure), "设置灰度曝光失败");
        check(sensor->SetUserGain(gain), "设置增益失败");
        MPSizectorS_ExposureBasicSettingStructType exposure_settings{};
        exposure_settings.ExposureMode = MPSizectorS_ExposureModeType_Manual;
        exposure_settings.ExposureNumber = 1;
        check(sensor->SetExposureBasicSetting(exposure_settings), "设置手动曝光失败");
        wait_ready();
    }
    // Used exclusively in an isolated process that exits with os._exit.
    // Do not destroy the vendor USB instance while its callbacks may be active.
    py::array_t<unsigned char> capture(unsigned timeout_ms) {
        {
            std::lock_guard<std::mutex> lock(mutex);
            received = false;
            error.clear();
        }
        if (!streaming) {
            sensor->SetDataCallBack(&GrayCamera::callback, this);
            check(sensor->SetTriggerSource(MPSizectorS_TriggerSourceType_KeepRunning), "启动 2D 连续采集失败");
            streaming = true;
        }
        std::vector<unsigned char> copied;
        int w, h;
        {
            py::gil_scoped_release release;
            std::unique_lock<std::mutex> lock(mutex);
            if (!ready.wait_for(lock, std::chrono::milliseconds(timeout_ms), [this] { return received || !error.empty(); }))
                throw std::runtime_error("等待灰度回调帧超时");
            if (!error.empty()) throw std::runtime_error(error);
            copied = pixels;
            w = width; h = height;
        }
        py::array_t<unsigned char> result({h, w});
        std::memcpy(result.mutable_data(), copied.data(), copied.size());
        return result;
    }
};
PYBIND11_MODULE(_gray_camera, module) {
    py::class_<GrayCamera>(module, "GrayCamera")
        .def(py::init<int, float, unsigned char>())
        .def("capture", &GrayCamera::capture, py::arg("timeout_ms") = 5000);
}
