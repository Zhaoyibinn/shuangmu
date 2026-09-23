import argparse

from Recon.Reconstruction import Reconstruction
from Recon.ReconstructionDataset import ReconstructionDataset
from Recon.config_loader import load_config
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='Stereo point-cloud reconstruction')
    parser.add_argument(
        '--config',
        default='config/recon/main.yaml',
        help='Reconstruction YAML; it may only inherit config/recon/default.yaml.',
    )
    parser.add_argument(
        '--global-registration-only',
        action='store_true',
        help=(
            'Override global_registration.only and run only global '
            'registration from point clouds already saved under paths.save_dir.'
        ),
    )
    return parser.parse_args()


def validate_config(config, global_registration_only=False):
    paths = config['paths']
    required_path_keys = ['save_dir']
    if not global_registration_only:
        required_path_keys.append('data_root_path')
        input_mode = config['reconstruction']['input_mode']
        if input_mode == 'stereo':
            required_path_keys.extend(['ext_yaml_path', 'int_yaml_path'])
        elif input_mode == 'direct_depth':
            required_path_keys.append('int_yaml_path')
        else:
            raise ValueError(
                "reconstruction.input_mode must be 'stereo' or 'direct_depth'"
            )
    for key in required_path_keys:
        if not paths[key]:
            raise ValueError('paths.{} must be configured'.format(key))

    segmentation = config['segmentation']
    if segmentation['method'] not in {'sam', 'yolo'}:
        raise ValueError("segmentation.method must be 'sam' or 'yolo'")
    if (
        not global_registration_only
        and segmentation['enabled']
        and segmentation['method'] == 'yolo'
        and not segmentation['yolo_model_path']
    ):
        raise ValueError(
            'segmentation.yolo_model_path is required when YOLO is enabled'
        )

    registration = config['global_registration']
    if registration['method'] not in {'pose_graph', 'ejrgf'}:
        raise ValueError(
            "global_registration.method must be 'pose_graph' or 'ejrgf'"
        )

    brightness_mask = config['reconstruction']['brightness_mask']
    brightness_threshold = int(brightness_mask['threshold'])
    if not 0 <= brightness_threshold <= 255:
        raise ValueError(
            'reconstruction.brightness_mask.threshold must be between 0 and 255'
        )

    depth_range = config['reconstruction']['depth_range_mm']
    min_depth_mm = float(depth_range['min'])
    max_depth_mm = float(depth_range['max'])
    if min_depth_mm <= 0:
        raise ValueError(
            'reconstruction.depth_range_mm.min must be greater than zero'
        )
    if max_depth_mm <= min_depth_mm:
        raise ValueError(
            'reconstruction.depth_range_mm.max must be greater than min'
        )

    if int(config['runtime']['frame_stride']) <= 0:
        raise ValueError('runtime.frame_stride must be greater than zero')


def main():
    args = parse_args()
    config = load_config(args.config)
    global_registration_only = (
        args.global_registration_only
        or bool(config['global_registration']['only'])
    )
    validate_config(
        config,
        global_registration_only=global_registration_only,
    )

    paths = config['paths']
    reconstruction_config = config['reconstruction']
    segmentation_config = config['segmentation']
    registration_config = config['global_registration']
    runtime_config = config['runtime']
    camera_calibration_path = paths['color_ext_yaml_path']
    if reconstruction_config['input_mode'] == 'direct_depth':
        camera_calibration_path = paths['int_yaml_path']

    reconstruction = Reconstruction(
        data_root_path=paths['data_root_path'],
        color_ext_yaml_path=camera_calibration_path,
        aruco_board_yaml_path=paths['aruco_board_yaml_path'],
        aruco_length=reconstruction_config['aruco_length'],
        world_voxel_size_mm=reconstruction_config['world_voxel_size_mm'],
        method=reconstruction_config['stereo_method'],
        use_aruco=reconstruction_config['use_aruco'],
        brightness_mask_enabled=reconstruction_config['brightness_mask']['enabled'],
        brightness_threshold=reconstruction_config['brightness_mask']['threshold'],
        use_sem=segmentation_config['enabled'],
        segmentation_method=segmentation_config['method'],
        sam_model_path=segmentation_config['sam_model_path'],
        yolo_model_path=segmentation_config['yolo_model_path'],
        yolo_conf=segmentation_config['yolo_conf'],
        yolo_imgsz=segmentation_config['yolo_imgsz'],
        yolo_device=segmentation_config['yolo_device'],
        sem_statistical_nb_neighbors=segmentation_config[
            'statistical_nb_neighbors'
        ],
        sem_statistical_std_ratio=segmentation_config[
            'statistical_std_ratio'
        ],
        initialize_processing=not global_registration_only,
        input_mode=reconstruction_config['input_mode'],
    )
    reconstruction.prepare_outputs(save_dir=paths['save_dir'])

    if global_registration_only:
        assert registration_config['enabled'], (
            'global_registration.enabled must be true when '
            'global_registration.only is enabled'
        )
        registration_method = registration_config['method']
        reconstruction.load_global_registration_inputs(
            method=registration_method,
            use_sem=segmentation_config['enabled'],
        )
        reconstruction.run_global_registration(
            method=registration_method,
            use_sem=segmentation_config['enabled'],
            **registration_config[registration_method],
        )
        return

    dataset = ReconstructionDataset(
        data_root_path=paths['data_root_path'],
        ext_yaml_path=paths['ext_yaml_path'],
        int_yaml_path=paths['int_yaml_path'],
        input_mode=reconstruction_config['input_mode'],
        color_ext_yaml_path=paths['color_ext_yaml_path'],
        min_depth_mm=reconstruction_config['depth_range_mm']['min'],
        max_depth_mm=reconstruction_config['depth_range_mm']['max'],
    )

    for sample in tqdm(dataset, total=len(dataset)):
        if sample['idx'] % int(runtime_config['frame_stride']) != 0:
            continue
        result = reconstruction.process_frame(sample)
        reconstruction.save_frame_result(result)

    if registration_config['enabled']:
        registration_method = registration_config['method']
        reconstruction.run_global_registration(
            method=registration_method,
            use_sem=segmentation_config['enabled'],
            **registration_config[registration_method],
        )

    reconstruction.save_final_result()


if __name__ == '__main__':
    main()
