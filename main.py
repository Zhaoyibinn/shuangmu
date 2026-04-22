from Recon.Reconstruction import Reconstruction
from Recon.ReconstructionDataset import ReconstructionDataset
from tqdm import tqdm


WORLD_VOXEL_SIZE_MM = 5.0 # 用于全局的体素下采样
ARUCO_LENGTH = 148.0 # ARUCO码的边长

DATA_ROOT_PATH = 'huojian/aurco/d455_0417' # 数据根目录
SAVE_DIR = 'depth_outputs/d455_aruco' # 保存目录

EXT_YAML_PATH = 'biaoding/extrinsics_d455_20250915.yml' # 左右目的外参
INT_YAML_PATH = 'biaoding/intrinsics_d455_20250915.yml' # 左右目的内参
COLOR_EXT_YAML_PATH = 'jiegouguang/color/rgb_calib_zyb_20250121.yaml' # 彩色相机内外参

USE_SAM = True # 目标重建 基于SAM的目标分割
SAM_MODEL_PATH = 'sam3.pt' # SAM3分割的权重
USE_ARUCO = True # 基于ARUCO的位姿估计 需要拍摄的时候放一个ARUCO
STEREO_METHOD = 'fast_foundation_stereo' # 双目深度估计的方法



def main():
    dataset = ReconstructionDataset(
        data_root_path=DATA_ROOT_PATH,
        ext_yaml_path=EXT_YAML_PATH,
        int_yaml_path=INT_YAML_PATH,
        color_ext_yaml_path=COLOR_EXT_YAML_PATH,
    )

    reconstruction = Reconstruction(
        data_root_path=DATA_ROOT_PATH,
        color_ext_yaml_path=COLOR_EXT_YAML_PATH,
        aruco_length=ARUCO_LENGTH,
        world_voxel_size_mm=WORLD_VOXEL_SIZE_MM,
        method=STEREO_METHOD,
        use_sam=USE_SAM,
        use_aruco=USE_ARUCO,
        sam_model_path=SAM_MODEL_PATH,
    )
    reconstruction.prepare_outputs(save_dir=SAVE_DIR)




    for sample in tqdm(dataset, total=len(dataset)):
        result = reconstruction.process_frame(sample)
        reconstruction.save_frame_result(result)

    reconstruction.save_final_result()


if __name__ == '__main__':
    main()