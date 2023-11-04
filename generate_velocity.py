import os
import math
import argparse
import numpy as np
from tqdm import tqdm
import multiprocessing
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import points_in_box
from nuscenes.utils.data_io import load_bin_file
from nuscenes.utils.data_classes import LidarPointCloud, LidarSegPointCloud

def generate_scans(nusc: NuScenes, out_dir: str, verbose: bool = False) -> None:
    if not hasattr(nusc, "lidarseg") or len(getattr(nusc, 'lidarseg')) == 0:
        raise RuntimeError(f"No nuscenes-lidarseg annotations found in {nusc.version}")
    name2idx_mapping = nusc.lidarseg_name2idx_mapping  # idx2name_mapping = nusc.lidarseg_idx2name_mapping

    scenes = nusc.scene
    for scene in scenes:
        scan_idx = 0
        sample = nusc.get('sample', scene['first_sample_token'])  # sample['prev'], sample['next']
        lidar_data_tok = sample['data']['LIDAR_TOP']
        lidar_data = nusc.get('sample_data', lidar_data_tok)
        lidar_data_next_tok = lidar_data['next']
        keyframe_dict = {}
        keyframe_dict[scan_idx] = lidar_data['is_key_frame']
        while lidar_data_next_tok != '':
            scan_idx += 1
            lidar_data = nusc.get('sample_data', lidar_data_next_tok)
            lidar_data_next_tok = lidar_data['next']
            keyframe_dict[scan_idx] = lidar_data['is_key_frame']

        keyframe_dict_sorted = {k: v for k, v in sorted(keyframe_dict.items(), key=lambda item: item[1])}
        last_scan_tok = scene['last_sample_token']
        sample_last = nusc.get('sample', last_scan_tok)

def generate_velocity_files(sample, nusc):
    root_dir = nusc.dataroot
    vels_dir = os.path.join(root_dir, "vels", nusc.version)
    os.makedirs(vels_dir, exist_ok=True)

    lidar_tok = sample['data']['LIDAR_TOP']
    lidar_sd = nusc.get('sample_data', lidar_tok)
    lidar_file = os.path.join(root_dir, lidar_sd['filename'])
    points = LidarPointCloud.from_file(lidar_file).points.T  # [num_pts, 4]
    lidarseg_file = os.path.join(root_dir, nusc.get('lidarseg', lidar_tok)['filename'])
    lidarseg_labels = load_bin_file(lidarseg_file, 'lidarseg')  # load lidarseg label
    # lidarseg_pcd = LidarSegPointCloud(lidar_file, lidarseg_file)

    vel_labels = np.full(lidarseg_labels.shape[0] * 3, np.nan)  # per-point velocity is initialized to NaN

    num_boxes = len(sample['anns'])
    num_inconsistent_boxes = 0
    for ann_token in sample['anns']:  # bbox
        ann = nusc.get('sample_annotation', ann_token)
        _, boxes, _ = nusc.get_sample_data(lidar_tok, selected_anntokens=[ann_token], use_flat_vehicle_coordinates=False)
        pts_in_box = points_in_box(boxes[0], points[:, :3].T)
        pts_indices = np.where(pts_in_box)[0]
        assert (pts_indices.shape[0] == ann['num_lidar_pts']), "points_in_box != num_lidar_pts"  # check num pts in bbox
        # box0 = nusc.get_box(ann_token)

        # category inconsistent check
        points_labels = lidarseg_labels[pts_indices]
        uniq_labels = np.unique(points_labels)
        if uniq_labels.shape[0] not in [0, 1]:
            num_inconsistent_boxes += 1

        # read bbox velocity
        boxes[0].velocity = nusc.box_velocity(boxes[0].token)
        box_vel = boxes[0].velocity
        num_inconsistent_pts = 0
        for idx in pts_indices:
            point_label = lidarseg_labels[idx]
            box_label = name2idx_mapping[ann['category_name']]
            if point_label == box_label:
                vel_x = box_vel[0]
                vel_y = box_vel[1]
                vel_z = box_vel[2]
                if math.isnan(vel_x) or math.isnan(vel_y) or math.isnan(vel_z):
                    vel_labels[idx * 3] = np.nan
                    vel_labels[idx * 3 + 1] = np.nan
                    vel_labels[idx * 3 + 2] = np.nan
                else:
                    vel_labels[idx * 3] = vel_x
                    vel_labels[idx * 3 + 1] = vel_y
                    vel_labels[idx * 3 + 2] = vel_z
            else:
                num_inconsistent_pts += 1
    vel_file = os.path.join(vels_dir, lidar_tok + "_vel.bin")
    vel_labels.tofile(vel_file)
    # vels_loaded = np.fromfile(vel_file, dtype=np.float64).reshape(-1, 3)
    return num_inconsistent_boxes, num_boxes

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate nuScenes lidar panaptic gt.')
    parser.add_argument('--root_dir', type=str, default='/home/mars/catkin_ws/src/nuscenes2bag/mini_data',
                        help='Default nuScenes data directory.')
    parser.add_argument('--version', type=str, default='v1.0-mini')
    parser.add_argument('--verbose', type=bool, default=True, help='Whether to print to stdout.')
    parser.add_argument('--out_dir', type=str, default='/home/mars/catkin_ws/src/nuscenes2bag/mini_data/vels')
    args = parser.parse_args()

    print(f'Start velocity ground truths generation... \nArguments: {args}')
    nusc = NuScenes(version=args.version, dataroot=args.root_dir, verbose=args.verbose)
    if not hasattr(nusc, "lidarseg") or len(getattr(nusc, 'lidarseg')) == 0:
        raise RuntimeError(f"No nuscenes-lidarseg annotations found in {nusc.version}")
    name2idx_mapping = nusc.lidarseg_name2idx_mapping  # idx2name_mapping = nusc.lidarseg_idx2name_mapping
    num_samples = len(nusc.sample)
    print(f'There are {num_samples} samples.')

    total_num_inconsistent_boxes = 0
    total_num_boxes = 0
    for sample in tqdm(nusc.sample):
        num_inconsistent_boxes, num_boxes = generate_velocity_files(sample, nusc)
        total_num_inconsistent_boxes += num_inconsistent_boxes
        total_num_boxes += num_boxes

    print(str(total_num_inconsistent_boxes) + " inconsistent boxes out of " + str(total_num_boxes) + " boxes")
    print(f'Velocity ground truths saved at {args.out_dir}. \nFinished velocity ground truth generation.')







