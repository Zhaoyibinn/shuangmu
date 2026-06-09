import argparse

from Recon.Reconstruction import Reconstruction
from Recon.ReconstructionDataset import ReconstructionDataset
from Recon.config_loader import load_config
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='Stereo point-cloud reconstruction')
    parser.add_argument(
        '--config',
        default='config/main.yaml',
        help='Experiment YAML. Its defaults field selects the default YAML.',
    )
    return parser.parse_args()


def validate_config(config):
    paths = config['paths']
    for key in ['data_root_path', 'ext_yaml_path', 'int_yaml_path']:
        if not paths[key]:
            raise ValueError('paths.{} must be configured'.format(key))

    segmentation = config['segmentation']
    if segmentation['method'] not in {'sam', 'yolo'}:
        raise ValueError("segmentation.method must be 'sam' or 'yolo'")
    if (
        segmentation['enabled']
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

    if int(config['runtime']['frame_stride']) <= 0:
        raise ValueError('runtime.frame_stride must be greater than zero')


def main():
    args = parse_args()
    config = load_config(args.config)
    validate_config(config)

    paths = config['paths']
    reconstruction_config = config['reconstruction']
    segmentation_config = config['segmentation']
    registration_config = config['global_registration']
    runtime_config = config['runtime']

    dataset = ReconstructionDataset(
        data_root_path=paths['data_root_path'],
        ext_yaml_path=paths['ext_yaml_path'],
        int_yaml_path=paths['int_yaml_path'],
        color_ext_yaml_path=paths['color_ext_yaml_path'],
    )

    reconstruction = Reconstruction(
        data_root_path=paths['data_root_path'],
        color_ext_yaml_path=paths['color_ext_yaml_path'],
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
    )
    reconstruction.prepare_outputs(save_dir=paths['save_dir'])

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
