# 盛相相机实时重建

在仓库根目录、已安装 `gradio`、`opencv-python`、`open3d` 和盛相 `mpsizector` SDK 的 Python 环境中运行：

```bash
conda activate shenxiang
python jiemian/realtime_reconstruction.py --config config/recon/shenxiang.yaml
```

打开 `http://127.0.0.1:7860`。相机动态库需按 `shenxiang_pysdk/INSTALL_zh.md` 配置。界面可设置设备序号、Binning、工作模式、曝光模式、3D/灰度曝光、增益、曝光次数和自动 HDR 参数。参数在点击“开始”时生效；运行中修改参数，需要先点击“停止”，再点击“开始”，这会创建一个新会话。

“软件设置”可调整每帧普通点云的统计滤波邻居数与标准差倍数、ArUco 板间标定要求的共同观测帧数，以及位姿图配准的体素大小、最大对应距离、ICP 迭代次数、回环连接间隔/权重、连接剪枝阈值和固定节点。默认值来自 `config/recon/shenxiang.yaml`。统计滤波与共同观测帧数在停止后重新开始时生效，并写入新会话的 `session.json`；统计滤波邻居数为 0 时关闭滤波。滤波后的单帧点云同时用于世界点云和位姿图配准。图优化参数在点击“全局图优化”时立即生效，无需重新拍摄，并记录到 `pose_graph_runs.jsonl`。

- **开始**：按界面中当前的拍摄参数启动相机，持续预览原图像、深度图和分割后的深度图（两张深度图共用伪彩色标）。参数默认值参考 `examples/continuous_save.py`；预览帧不写盘。
- **拍摄一帧**：保存同一帧的灰度、毫米深度、二值 mask 和点云。首帧也尝试识别 ArUco，作为后续帧统一坐标系的参照；未识别到时仍保存相机坐标系点云。后续有 ArUco 位姿的帧还会保存世界坐标系点云。单帧窗口会选中最新帧；下拉列表可切换当前会话中任一已拍摄帧。全局窗口显示逐帧累积的世界坐标系点云。`captures.jsonl` 记录每次拍摄及位姿是否成功。
- **连续拍摄 / 停止连续拍摄**：先点击“开始”，再点击“连续拍摄”自动保存新帧。默认连续拍摄间隔为 0 秒，当前帧处理完成后不额外等待，有新帧就继续拍摄；也可通过“连续拍摄间隔”设置等待时间（点击“连续拍摄”应用）。没有新帧时等待，不重复保存；未识别到目标时仍保存图像并继续。停止连续拍摄后保持预览；点击“停止”会同时结束连续拍摄和相机采集。写盘等异常会停止连续拍摄并显示原因。
- **ArUco 位姿估计**：显示当前选中帧的 `pose_vis`，并列出检测到的板子、每块板上的码 ID/数量、板子是否已注册到全局坐标系以及世界位姿状态。检测到码但尚未完成板间标定时，该帧仍保存单帧点云，不生成 `cloud_world`。
- **全局图优化**：至少有两帧成功识别 ArUco 后，按 `shenxiang.yaml` 中的 `global_registration.pose_graph` 参数单独运行，优化结果显示在全局点云窗口中。
- **停止**：结束相机采集并保存累计位姿与融合点云。

每次成功开始都会创建独立的 `realtime_YYYYMMDD_HHMMSS_微秒` 子目录，按当前相机分别保存。默认配置下：

| 相机 | 原始采集根目录 | 重建结果根目录 |
| --- | --- | --- |
| 盛相 | `huojian/shengxiang/continuous/` | `depth_outputs/shengxiang/continuous/` |
| RealSense D455 | `huojian/realsense/continuous/` | `depth_outputs/realsense/continuous/` |

根目录由配置中的 `paths.data_root_path` 和 `paths.save_dir` 确定：已有的 `shengxiang`、`shenxiang`、`realsense` 或 `realsense_d455` 目录段替换为当前相机目录（盛相为 `shengxiang`，D455 为 `realsense`），保留后面的子目录；自定义路径没有这些目录段时，在末尾追加当前相机目录。停止后切换相机再开始即使用对应目录，历史会话保留原位置。状态栏的“输出”和 `session.json` 中的 `raw_dir`、`output_dir` 记录实际路径。相机 SDK 运行在独立子进程中，避免设备退出时的底层崩溃影响界面。

如果近处缺少深度，可在界面把 3D 曝光强度调低（允许 0.1–100，默认 100），停止后重新开始拍摄。状态栏会显示 SDK 原始最近深度、SDK mask 判定有效的最近深度，以及低于软件下限的像素数。每次拍摄还会保存未经软件过滤的 `sdk_depth_mm/frame_XXXX.npy` 和 `sdk_mask/frame_XXXX.png`，便于区分相机无深度、SDK mask 无效和软件范围过滤；这两项与显示图像使用同一帧，但保持 SDK 原始分辨率。每次点击“拍摄一帧”都会先保存原始 RGB-D，再进行前景检查和重建。无论 YOLO 是否检测到目标、ArUco 位姿是否解算成功，原始数据均保存在本次会话的 `raw_dir`：彩色图为 `rgb/frame_XXXX.png`（无彩色流时保存 `gray/frame_XXXX.png`），未经过前景分割及软件深度范围过滤的完整深度为 `depth_full/frame_XXXX.png`（uint16，单位毫米）；浮点精度的完整对齐深度仍保存在 `aligned_depth_mm/frame_XXXX.npy`（相机提供时）。`depth/` 保留供重建使用的前景深度。 结果目录 `output_dir`（`session.json` 所在目录）也会同步保存 `gray/frame_XXXX.png`、`depth_full/frame_XXXX.png`，有彩色流时还会保存 `rgb/frame_XXXX.png`；这些图像不依赖识别或重建成功。没有有效前景深度或重建异常时仍计为已拍摄帧、记录到 `captures.jsonl`，界面显示已保存原因，后续帧使用新编号，避免覆盖。
# RealSense D455 实时重建（FoundationStereo + ArUco）

在仓库根目录运行：

```bash
conda activate shenxiang
python jiemian/realtime_reconstruction.py --camera-type realsense_d455
```

也可以在原重建界面的“相机类型”中选择 RealSense D455，停止后点击“开始”生效。设备序号按 D455 序列号排序，从 0 开始。D455 需要 `pyrealsense2` 及仓库 FoundationStereo 所需的 PyTorch、OmegaConf 等依赖，建议使用 CUDA GPU。默认权重是 `jiegouguang/weights/FoundationStereo/23-51-11/model_best_bp2.pth`，同目录必须有 `cfg.yaml`；可通过 `--checkpoint` 或界面指定，`--valid-iters` 设置迭代次数（默认 32）。首次启动包含模型加载，最多等待 180 秒，预览速度由推理耗时决定。

界面的“双目推理模式”可选择 `foundation_stereo` 或 `fast_foundation_stereo`。切换模式自动填入相应默认权重及迭代次数，也可继续修改。Fast 使用序列化权重，不要求旁边有 `cfg.yaml`，需要 CUDA；默认 8 次迭代：

```bash
python jiemian/realtime_reconstruction.py \
  --camera-type realsense_d455 --stereo-method fast_foundation_stereo
```

Fast 默认权重为 `jiegouguang/weights/fast_FoundationStereo/23-36-37/model_best_bp2_serialize.pth`。两种模式共用双目标定、RGB 深度投影、YOLO、重建输出和耗时统计。盛相使用 SDK 深度，双目推理模式参数仅对 RealSense 生效。

采集使用**单台 D455 内置左右红外相机和彩色相机**，IR 1/2 Y8 + color BGR8，固定 1280×720、30 fps，红外散斑投射默认关闭、可选开启，自动曝光。FoundationStereo 在校正后的左右红外图上估计深度；通过设备 IR→RGB 外参、左目校正旋转和 RGB 内参，将深度投影到去畸变后的 RGB 图像，使用 RGB 相机的 Z 深度（毫米），同一像素保留最近表面。重投影造成的孔洞保持为 0，不插值填充背景。不会使用 D455 硬件深度。

ArUco 在完整的去畸变 RGB 图上定位，重建与位姿统一使用 RGB 内参和坐标系。板布局和尺寸来自 `paths.aruco_board_yaml_path`；软件参数统一使用 `config/recon/shenxiang.yaml`，D455 不再使用独立的重建配置文件。D455 未指定 `--config` 时，界面默认的四码双板布局为 `config/aruco_board/main.yaml`（继承 `default.yaml`）：板 0 的码 ID 为 0–3，板 1 为 4–7，码边长 24 mm、相邻格中心距 32 mm，必须与实物一致。盛相专用曝光、增益、Binning 控件对 D455 不生效。

两种相机均默认启用 **YOLO 前景重建**，界面中可统一设置 YOLO 参数，无需勾选开关或传入 `--yolo`。例如：

```bash
python jiemian/realtime_reconstruction.py \
  --camera-type realsense_d455 \
  --yolo-weights jiegouguang/weights/yolo/suobipengguan/last.pt \
  --yolo-imgsz 640 --yolo-conf 0.25 --yolo-device 0
```

`--yolo-imgsz` 是 YOLO 内部推理尺寸，不是相机采集或输出图片尺寸。1280×720 输入可以使用 `--yolo-imgsz 640`，分割 mask 会恢复到原图尺寸。采集固定为 1280×720，如需使用 1280 推理尺寸：

```bash
python jiemian/realtime_reconstruction.py \
  --camera-type realsense_d455 \
  --yolo-imgsz 1280 --yolo-device 0
```

RGB、前景 mask 和对齐深度均按 1280×720 保存，启动时读取 `biaoding/intrinsics_d455_720p_20260526.yml` 和 `biaoding/extrinsics_d455_720p_20260526.yml` 的双目标定。已移除采集分辨率选择及 `--resolution` 参数。增大 `--yolo-imgsz` 会增加推理显存占用与耗时。双目标定使用文件中的 `M1/D1/M2/D2/R/T`，`T` 从毫米转换为米后重新计算校正映射及基线；忽略文件中的占位 `R1/R2/P1/P2/Q`。RGB 标定及 IR→RGB 外参仍从设备读取，这两份文件不包含这些参数。设备必须支持 1280×720、30 fps 的 RGB 与双目流组合；不支持时 SDK 会报告启动失败。

也可在界面的“YOLO 前景分割”中指定权重。`--yolo-device 0` 使用首张 CUDA GPU，`cpu` 使用 CPU，留空自动选择。FoundationStereo 自动使用可用 CUDA。先在完整 RGB 上运行 YOLO，再计算双目深度并对齐 RGB，最后用 RGB 前景 mask 将背景深度置 0；双目输入保留完整图像。实时预览分别展示完整原图、分割前深度、分割后深度；单帧点云、融合及全局图优化使用前景深度。ArUco 图像不受前景 mask 影响。没有检测到目标时输出空 mask 和零深度，点击拍摄会提示，不保存背景点云。

仓库已有三份可加载的分割权重，均为 `segment` 任务，类别为 `{0: huojian}`：

- `jiegouguang/weights/yolo/suobipengguan/last.pt`（默认）
- `huojian/d455_penguan_20260525/video/huojian-600-640.pt`
- `huojian/d455_penguan_20260525/per1/exp-3.pt`

如果替换自己的权重，可复制到 `jiemian/weights/foreground-seg.pt` 或任意仓库路径，通过界面或 `--yolo-weights` 指定；必须是 Ultralytics 分割权重，不能使用仅检测框的模型。多个检测实例的 mask 取并集；现有权重只分割 `huojian`，其他物体需对应训练权重。路径不存在时会提示，不自动下载模型。盛相和 RealSense 均默认启用 YOLO，权重及推理参数继承对应的 `yolo_*` 配置，也可在界面或命令行覆盖。两种相机均在采集进程中执行分割：盛相将灰度图转三通道输入 YOLO，RealSense 使用 RGB。有效性掩码结合 SDK 有效标记和前景区域，主重建流程直接使用前景深度，不再重复分割。ArUco 始终使用完整原图。

同一板型下两块实体板共用世界坐标系：两块板须保持相对固定，先让它们同时入镜，每块板每帧至少识别到 3/4 个码，点击“拍摄一帧”累计至少 3 个共同观测帧（软件设置可改）。完成板间标定后，参考板根据已注册板的可见码总面积自动选择，一块板离开画面后可继续使用另一块板维持同一世界坐标系。预览不累计标定帧。标注图会给两块成功解算的板都绘制坐标轴；`REF` 表示本帧参考板，`registered` / `pending` 表示注册状态。JSON 显示检测码 ID、未配置及重复 ID、各板码数和共同观测帧数。

界面中的“ArUco 标定板类型”可选择四码双板（0–3 / 4–7，码边长 24 mm、中心距 32 mm）或十二码双板（0–11 / 12–23，码边长 27 mm、中心距 36 mm），适用于盛相和 D455。板型选择无需预先拍摄标定帧，在点击“开始”时直接加载对应布局。默认值来自启动配置，界面选择在停止后重新开始时覆盖配置中的板布局，实际使用的路径写入 `session.json`。切换会创建新会话，重新建立世界坐标系和板间标定，已有文件保留；无需更换启动配置。使用错误布局会把两块实物板合并到同一板号并产生错误位姿。

每个 D455 会话保存以下输入（文件名为 `frame_XXXX`）：

| 目录 / 文件 | 内容 |
| --- | --- |
| `rgb/*.png` | 完整去畸变 RGB，与输出深度逐像素对齐 |
| `gray/*.png` | RGB 转灰度 |
| `foreground_mask/*.png` | YOLO 前景掩码，0/255 |
| `depth/*.png` | RGB 对齐、前景筛选及范围过滤后的毫米 uint16 深度，背景为 0 |
| `mask/*.png` | 有效前景深度掩码，0/255 |
| `aligned_depth_mm/*.npy` | RGB 对齐、未做前景筛选的浮点毫米深度 |
| `left_ir/*.png`、`right/*.png` | FoundationStereo 使用的校正后双目图 |
| `stereo_depth_mm/*.npy` | 左红外坐标系的原始浮点毫米深度 |
| `sdk_depth_mm/*.npy`、`sdk_mask/*.png` | 沿用旧目录名，保存范围过滤前的完整 RGB 对齐深度及结合前景的有效性掩码（0 有效、1 无效）；并非硬件深度 |
| `intrinsics.yaml` | 去畸变 RGB 内参，供位姿及重建使用 |

`session.json` 保存设备标定、IR→RGB 外参、双目校正旋转、基线、模型及分割参数。输出点云包含 RGB 颜色。除前景筛选外，拍摄、ArUco 位姿、累计融合及全局图优化操作与原流程相同。

无相机验证双目模式、YOLO 掩码与预览：

```bash
python -m unittest discover -s tests -p test_live_stereo_modes.py -v
```

其他无相机验证：

```bash
python -m unittest discover -s tests -p test_realtime_reconstruction.py -v
```


## RealSense 重建耗时诊断

按原命令启动 `realtime_reconstruction.py` 并选择 RealSense D455，终端会自动输出每帧的阶段耗时，单位为毫秒，按耗时从大到小排列：

- `[RealSense frame N]`：等待相机帧、红外校正、RGB 校正、前景分割、FoundationStereo、毫米深度转换、深度投影到 RGB、mask 和数据打包。
- `[foundation_stereo frame N]` / `[fast_foundation_stereo frame N]`：输入转换/上传/补边、`network_forward` 网络推理、裁边和数据下载、视差转深度、三维点生成（仅普通 FoundationStereo）。它是上一条双目推理阶段的内部细分，不能重复相加。
- `[Receive frame N]`：队列取帧及反序列化、深度统计、重建输入准备。`delivery_age` 是从子进程处理完成到主进程取到帧的时间，包含排队及轮询等待，并非纯传输耗时。
- `[Preview frame N]`：预览回调等待锁、接收和预览图生成。仅在取得新帧时打印；不包含 Gradio 返回后的编码、网络传输和浏览器渲染。
- 点击拍摄后，`[Capture N]` 输出输入准备、重建、ArUco 汇总、原始输入保存、单帧结果保存（含视频及点云累计）、全局点云降采样/写盘和位姿 YAML 保存；`[Reconstruction N]` 细分重建中的 ArUco 估计/绘制、点云生成、滤波、坐标变换等。`[Capture complete N]` 还包含记录日志和界面返回数据准备。拍摄编号从 0 开始，相机序号从 1 开始。

状态栏显示最近一帧的采集处理总耗时、对应处理 FPS 和最慢环节；该 FPS 不代表浏览器显示帧率。成功拍摄的各级耗时保存在会话目录 `captures.jsonl` 的 `timings_ms` 字段；其中 `capture_before_log` 截止写日志前。预览逐帧明细输出到终端。

模型计时在 CUDA 边界同步，包含 GPU 实际执行时间。同步和逐帧日志会引入诊断开销；比较稳定速度时应观察预热后的多帧，首帧可能包含延迟初始化。重点对比 `network_forward`、`project_depth_to_rgb`、`cloud_filter` 和累计全局点云的降采样/写盘，后者可能随拍摄数量增长。

# 纯灰度 ArUco 实时位姿估计

```bash
conda activate shenxiang
python jiemian/realtime_aruco_pose.py --config config/recon/shenxiang.yaml
```

打开 `http://127.0.0.1:7861`。依赖 `gradio`、`opencv-contrib-python`（含 `cv2.aruco`）、`numpy`、`PyYAML`、`plotly` 和盛相 SDK 的头文件/动态库。使用工作区内的纯 2D 回调适配器，避开这款相机在 Linux 上不稳定的 `Snap()` 路径。此入口不加载重建模型或 Open3D。相机使用 `WorkingMode.image_2d_white` 和手动曝光，关闭 Binning，直接拍摄全分辨率 uint8 灰度图；相机进程只向界面传输时间戳与灰度图，不获取或传输深度、mask、点云。启动预览后以 SDK 回调持续采集，连续检测模式按界面定时器获取新的灰度帧，实际帧率取决于曝光、相机传输和检测耗时。

- **启动相机**：连接设备、配置参数并创建新会话，持续显示实时灰度预览；预览不保存、不检测、不建立参考位姿。
- **切换设备**：在“相机类型”中选择“盛相相机”或“RealSense D455”，设置设备序号，然后点击“启动 / 切换相机”。原相机会先停止，新设备创建独立会话并重新建立起始位姿，已有文件保留。D455 序号按已连接 D455 的序列号排序，从 0 开始。盛相沿用原曝光、增益和标定；D455 隐藏盛相曝光/增益控件，使用自动曝光及 1280×720、30 fps 的彩色流转灰度，不启用深度流。切换后的检测、保存和回放操作相同。
- **拍摄起始图像**：查看实时预览并调整板子位置后，单独点击此按钮，保存第 0 帧，同时初始化本帧中每块可见板子的参考位姿。必须先完成这一步，才能进行单帧/连续检测。每块板独立统计，不选择主板，不做板间标定或切换。起始帧未出现的板子在首次成功识别时建立参考，表格显示其“起始帧”。相机须保持固定。
- **连续视频检测**：勾选后连续拍摄、检测、保存和可视化；取消勾选暂停，随后可点击“单帧检测”拍摄一张新图。起始位姿在暂停、漏检、重新出现后均保持不变。启动新会话才会重新建立参考。
- **实时可视化**：图像显示所有检测到的码和各板坐标轴；表格显示各板相对自身起始坐标系的平移、绕 X/Y/Z 的欧拉角、总体旋转角与重投影误差。3D 窗口显示固定相机坐标系中的各板轨迹、起始点和当前坐标轴，可旋转缩放。漏检的板子不沿用上一帧位姿，轨迹在漏检处断开。
- **回放**：暂停连续检测或停止相机后拖动帧滑块，或勾选“自动回放”按原始采集时间播放。操作回放会暂停连续检测。输入历史会话目录并点击“载入历史会话”可在相机未连接时回放；载入成功会停止当前相机。

位姿检测按相机分目录，并加 `_pose` 后缀。默认配置下，盛相保存到 `depth_outputs/shengxiang_pose/continuous/pose_日期时间_微秒/`，RealSense 保存到 `depth_outputs/realsense_pose/continuous/pose_日期时间_微秒/`。配置 `paths.save_dir` 中的相机目录段替换为对应的 `shengxiang_pose` / `realsense_pose`；自定义根目录没有相机目录段时，在末尾追加该目录。可通过 `--output` 指定根目录，例如 `--output /tmp/poses` 会分别保存到 `/tmp/poses/shengxiang_pose/pose_...` 和 `/tmp/poses/realsense_pose/pose_...`。切换相机、运行中清空历史后创建的新会话均使用对应目录，历史文件保留原位置，仍可载入回放。`--port` 修改端口。每帧保存 `gray/frame_XXXXXX.png`、带标注的 `visual/frame_XXXXXX.png`，以及 `poses.jsonl` 中的时间戳、检测模式、各板可见码 ID、起始帧、位姿与误差；`session.json` 保存配置和相机参数。所有图像均保留原始分辨率。

D455 额外依赖 `pyrealsense2`，须安装到运行界面的同一 Python 环境：`python -m pip install pyrealsense2`。D455 直接读取当前彩色流的设备内参，不使用盛相的 `paths.int_yaml_path`；inverse/modified Brown 畸变先校正再检测和保存，保存分辨率仍为 1280×720，相应估计器使用零畸变参数。`session.json` 的 `device` 中保存序列号、实际流内参、原畸变系数和是否校正，板布局仍使用 `paths.aruco_board_yaml_path`。

位姿矩阵 `T_camera_board` 将板坐标变换到固定相机坐标，`T_initial_board = inverse(T_camera_board_initial) @ T_camera_board_current` 将当前板坐标变换到该板起始坐标。平移单位为毫米，角度单位为度。表格采用固定轴 XYZ 欧拉角，即依次绕起始 X、Y、Z 轴旋转，`R = Rz(z) @ Ry(y) @ Rx(x)`；X/Z 范围为 ±180°，Y 范围为 ±90°，在 Y = ±90° 的万向节锁位置约定 Z = 0°。总体旋转角为相对起始姿态的最小等效旋转角（0–180°），不是三轴欧拉角之和、欧拉角向量的长度或累计转角。记录新增 `rotation_euler_xyz_deg` 字段，并保留原旋转向量和总体旋转角；旧会话回放时从相对旋转矩阵计算欧拉角。板布局和长度来自 `paths.aruco_board_yaml_path`；相机内参来自 `paths.int_yaml_path`。当原生图像与标定图像大小不同时，按宽高缩放内参而不缩放检测图；宽高比不一致时拒绝检测，应使用对应的全分辨率标定。

无相机验证几何和记录回放：

```bash
python -m unittest discover -s tests -p test_realtime_aruco_pose.py -v
```

纯灰度回调适配器首次启动时自动编译（需要 C++ 编译器和 pybind11 头文件，支持系统 `/usr/include/pybind11`）。默认 SDK 目录为仓库同级的 `shenxiang_pysdk`；其他目录请先手动构建：

```bash
python jiemian/cameras/build_gray_camera.py --sdk-root /path/to/shenxiang_pysdk
```

适配器只在相机子进程内使用，复制 SDK 回调拥有的灰度缓冲区，不主动释放回调数据；关闭时退出子进程以隔离 SDK USB 线程。编译产物放在 `jiemian` 下，不修改已安装的 SDK。若底层进程中断，状态栏会显示退出码（例如 SIGABRT 对应 -6），而不再显示空白错误。

实时灰度预览与检测/回放窗口独立：启动后即更新实时画面（默认 0.2 秒轮询，实际速度取决于相机和传输），暂停连续检测、单帧检测之间和回放期间仍可查看当前相机画面。停止相机后预览停留在最后一帧。

显示图像在交给 Gradio 编码前按比例缩小至最大宽度 1332 像素（较小图像不放大），同时应用于实时预览、检测标注图和回放图。实时检测直接使用内存中的标注图，不重新读取刚保存的 PNG；历史回放切换到其他帧时才读取文件。检测输入、相机内参计算和落盘图像保持全分辨率。

右侧 3D 窗口会保留手动旋转、平移和缩放后的观察视角，连续检测更新和回放切帧不会恢复默认视角。右侧使用固定 HTML 容器中的持久 Plotly 实例，首次初始化后只通过 `Plotly.react` 更新数据，不重建组件、不反复恢复观察相机。Plotly 脚本从本机 Python 包提供，无需访问外部 CDN。这仅控制观看视角，不改变用于位姿估计的固定相机坐标系。

界面底部的“清空历史”会暂停检测和回放，清空当前轨迹、回放列表和各板起始位姿。相机和实时预览继续运行；重新点击“拍摄起始图像”后可再次检测。已保存的历史会话文件保留，后续记录写入新会话目录。

## RealSense D455 实时物体位姿（YOLO + ICP）

参照 `main_pose.py` 的物体点云位姿流程，固定相机、移动目标，无需 ArUco 板：

```bash
conda activate shenxiang
python jiemian/realtime_realsense_pose.py
```

打开 `http://127.0.0.1:7862`，点击“启动 / 重新启动相机”，再点击“拍摄一帧并估计”或勾选“连续拍摄并估计”。默认使用 Fast FoundationStereo 序列化权重、8 次迭代，GT 为 `depth_outputs/chongjian/video_shu/aruco_fusion/pose_graph_registration/merged_cloud_pose_graph_sam_pca_aligned_clean_section_axis_aligned.ply`。双目及 RGB 对齐复用 D455 相机模块；YOLO、深度范围、体素和 ICP 参数读取 `config/pose/main.yaml`，可用 `--config` 更换配置，`--yolo-weights` 覆盖分割权重。亮度过滤在去畸变 RGB 坐标系中执行。GT 点云必须以毫米为单位。

ICP 始终使用 Open3D 六自由度 point-to-plane 配准。首帧使用质心平移初始化，后续使用上一帧原始 ICP 位姿初始化，不再进行 PCA 方向搜索或回退。`main_pose.py` 共用此流程。旧配置中的 `fallback_fitness` 仅为兼容保留。显示/保存的点云、原始位姿和跟踪初始化均保留六自由度结果，配准点云仍可能绕 GT 主轴转动。

仅在配准后的角度计算中消除 GT-X 主轴自转。假设喷管主轴为 GT-X，先用 `u_i=R_i.T @ [1,0,0]` 提取相机中的主轴方向，再将方向表达在固定的首帧无自转参考系中。该参考系由把首帧主轴对齐 GT-X 的最短摆转确定。按 `u=Rz(yaw) Ry(pitch) X` 输出 `gt_axis_pitch_deg`、`gt_axis_yaw_deg` 两个倾角，以及辅助的 `gt_axis_tilt_deg` 主轴总偏转角；它们不受首帧或当前帧 ICP 中任意 GT-X 自转影响。历史图默认显示 Y、Z、总体三根曲线，可通过“显示曲线”多选框任意组合或全部隐藏。接近反向主轴或方向角极点时存在表示奇异性。

`transform_original_source_to_first` 仍为原始六自由度相对矩阵 `inverse(T_first_camera_to_gt) @ T_current_camera_to_gt`，原始欧拉角和旋转中心分析保留在详细记录中，与新的两个主轴倾角区分。`poses.csv`、`poses.jsonl` 及 `rotation_angles.csv` 保存新角度字段，后者的 `angle_deg` 为主轴总偏转角。`--keep-x-rotation` 跳过去主轴自转角度计算，历史图显示原始相对 Y、Z 及总体旋转角；ICP 在两种模式下均为六自由度。旧会话沿用原来的角度，不会自动重算。

图片分为两行，每行依次为 RGB、分割前的 RGB 对齐深度、分割并过滤后的深度。第一行启动后持续刷新，暂停拍摄时仍预览，预览不保存、不更新 ICP 和首帧参考；第二行仅在拍摄时更新并保留最近捕捉结果。深度图使用相同色标，三张图来自同一帧。连续拍摄按实际推理速度逐帧估计和保存，取消勾选即可暂停。刷新周期至少 0.2 秒，实际速度取决于双目推理、ICP 和保存耗时。

默认保存到 `depth_outputs/realsense_icp_pose/pose_时间戳/`，包含 RGB、校正后双目、前景 mask、分割前后浮点毫米深度（`full_depth_mm/`、`depth_mm/`）、居中及配准点云、位姿 JSON/NPY、`poses.jsonl` 和 `poses.csv`。停止或重新启动时生成 `rotation_angles.csv` 及角度图（需 Matplotlib）。未检测到前景时跳过；异常会暂停连续拍摄并写入失败记录。界面另展示 GT 与配准点云，以及相对位姿和 ICP 质量。

最下方“清除历史与捕捉结果”会暂停连续拍摄，清空第二行、位姿记录和首帧参考，重置 ICP，并创建新会话。相机和第一行实时预览继续运行；下次成功拍摄重新建立参考。已经保存的旧文件保留。相机停止时清除只清空历史显示，下次启动再创建会话。

“历史角度回放”可通过“显示曲线”多选框选择 Y、Z、总体，默认全部显示。新模式的“总体”为去主轴自转后的主轴偏转角，旧会话或保留完整旋转模式沿用原总体旋转角。曲线选择会持续应用于采集刷新、拖动回放、自动回放和切换会话，不会暂停播放或改变回放位置。横轴为原始帧号，拖动滑块选择成功帧，图中虚线和圆圈标记对应角度；勾选自动回放按拍摄时间前进，到末帧自动停止。缺少时间戳的旧记录按每帧 0.2 秒回放。输入历史会话目录可从 `poses.jsonl` 载入角度，不需要历史图像、点云、GT 或连接相机。回放会暂停连续拍摄，只更新角度图，最新捕捉图像和点云保持不变，实时预览继续；再次拍摄切回本次会话的角度历史。清除按钮同时清空角度历史。

图片区域和图表保留固定高度；拍摄、预览和回放均关闭自动滚动及加载遮罩。配准图和角度图使用持久 Plotly 容器原位更新，避免逐帧重建图表导致布局和视角变化。

可使用 `--gt-ply`、`--output`、`--device-index`、`--port` 指定目标、保存目录、设备序号和端口；`--stereo-method foundation_stereo` 切换普通 FoundationStereo（默认 32 次迭代），`--checkpoint`、`--valid-iters` 覆盖对应权重及迭代次数。

实时位姿界面和实时重建界面均提供“开启 RealSense 散斑投射”复选框，默认不勾选。位姿界面点击“启动 / 重新启动相机”应用；重建界面停止后重新开始应用。适用于 FoundationStereo 和 Fast FoundationStereo。也可用 `--emitter` 开启，`--no-emitter` 显式关闭。启动时设置设备的 `emitter_enabled` 并读取确认，实际状态写入 `session.json` 的设备信息（`emitter_requested` / `emitter_enabled`）；不支持开启或设置失败时明确报错。

界面顶部也可通过“深度估计模型”选择 FoundationStereo 或 Fast FoundationStereo（默认）。切换时自动填入对应默认权重和迭代次数（32 / 8），也可修改权重路径和迭代次数。点击“启动 / 重新启动相机”后应用选择，关闭旧相机并创建新会话，旧数据保留。状态栏显示实际启用的模型和迭代次数。

显存较小时可调整“双目推理缩放”（1.0 / 0.75 / 0.5 / 0.25）。本位姿界面普通 Foundation 默认 0.5，Fast 默认 1.0；模型切换同时填入对应缩放值。0.5 使用 640×360 双目输入，视差按宽度比例恢复至 1280×720，并用原始内参计算毫米深度及 RGB 对齐。采集与保存尺寸不变，但缩小推理会降低实际深度细节，恢复尺寸不会补回这些细节。共享相机模块的其他调用仍默认全尺寸推理。

例如在 8GB 显卡上尝试普通 Foundation 的较低显存配置：

```bash
python jiemian/realtime_realsense_pose.py \
  --stereo-method foundation_stereo --valid-iters 16 --inference-scale 0.5
```

普通 Foundation 推理每次都显式禁用梯度，避免 Gradio 工作线程变化时建立计算图；权重先加载到 CPU，再将模型移入 GPU，启用模型的 `low_memory` 选项。停止或切换相机时释放双目和 YOLO 模型引用并清理 CUDA 缓存。预览或拍摄遇到 CUDA 显存不足时停止相机，保留已保存记录，降低缩放或切换 Fast 后重新启动。只降低迭代次数不保证显著降低峰值显存，实际占用仍需在目标显卡上测量。

无相机验证深度过滤、毫米坐标变换、首帧参考和会话保存：

```bash
python -m unittest discover -s tests -p test_realtime_realsense_pose.py -v
```
