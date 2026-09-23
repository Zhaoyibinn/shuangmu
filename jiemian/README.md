# 盛相相机实时重建

在仓库根目录、已安装 `gradio`、`opencv-python`、`open3d` 和盛相 `mpsizector` SDK 的 Python 环境中运行：

```bash
conda activate shenxiang
python jiemian/realtime_reconstruction.py --config config/recon/shenxiang.yaml
```

打开 `http://127.0.0.1:7860`。相机动态库需按 `shenxiang_pysdk/INSTALL_zh.md` 配置。界面可设置设备序号、Binning、工作模式、曝光模式、3D/灰度曝光、增益、曝光次数和自动 HDR 参数。参数在点击“开始”时生效；运行中修改参数，需要先点击“停止”，再点击“开始”，这会创建一个新会话。

“软件设置”可调整每帧普通点云的统计滤波邻居数与标准差倍数、ArUco 板间标定要求的共同观测帧数，以及位姿图配准的体素大小、最大对应距离、ICP 迭代次数、回环连接间隔/权重、连接剪枝阈值和固定节点。默认值来自 `config/recon/shenxiang.yaml`。统计滤波与共同观测帧数在停止后重新开始时生效，并写入新会话的 `session.json`；统计滤波邻居数为 0 时关闭滤波。滤波后的单帧点云同时用于世界点云和位姿图配准。图优化参数在点击“全局图优化”时立即生效，无需重新拍摄，并记录到 `pose_graph_runs.jsonl`。

- **开始**：按界面中当前的拍摄参数启动相机，持续预览灰度图、深度伪彩图和有效深度 mask。参数默认值参考 `examples/continuous_save.py`；预览帧不写盘。
- **拍摄一帧**：保存同一帧的灰度、毫米深度、二值 mask 和点云。首帧也尝试识别 ArUco，作为后续帧统一坐标系的参照；未识别到时仍保存相机坐标系点云。后续有 ArUco 位姿的帧还会保存世界坐标系点云。单帧窗口会选中最新帧；下拉列表可切换当前会话中任一已拍摄帧。全局窗口显示逐帧累积的世界坐标系点云。`captures.jsonl` 记录每次拍摄及位姿是否成功。
- **ArUco 位姿估计**：显示当前选中帧的 `pose_vis`，并列出检测到的板子、每块板上的码 ID/数量、板子是否已注册到全局坐标系以及世界位姿状态。检测到码但尚未完成板间标定时，该帧仍保存单帧点云，不生成 `cloud_world`。
- **全局图优化**：至少有两帧成功识别 ArUco 后，按 `shenxiang.yaml` 中的 `global_registration.pose_graph` 参数单独运行，优化结果显示在全局点云窗口中。
- **停止**：结束相机采集并保存累计位姿与融合点云。

每次点击“开始”都会创建独立的 `realtime_YYYYMMDD_HHMMSS` 子目录。原始采集帧位于 `paths.data_root_path` 下，重建输出位于 `paths.save_dir` 下，均以 `config/recon/shenxiang.yaml` 为准。相机 SDK 运行在独立子进程中，避免设备退出时的底层崩溃影响界面。

如果近处缺少深度，可在界面把 3D 曝光强度调低（允许 0.1–100，默认 100），停止后重新开始拍摄。状态栏会显示 SDK 原始最近深度、SDK mask 判定有效的最近深度，以及低于软件下限的像素数。每次拍摄还会保存未经软件过滤的 `sdk_depth_mm/frame_XXXX.npy` 和 `sdk_mask/frame_XXXX.png`，便于区分相机无深度、SDK mask 无效和软件范围过滤；这两项与显示图像使用同一帧，但保持 SDK 原始分辨率。若整帧没有有效深度，点击拍摄会把原始数据写到 `diagnostics/`，方便检查。
