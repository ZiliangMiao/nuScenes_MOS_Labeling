import os
import math
import argparse
import numpy as np
from tqdm import tqdm
import multiprocessing
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import points_in_box
from nuscenes.utils.data_io import load_bin_file
from nuscenes.utils.data_classes import LidarSegPointCloud

def generate_mos_labels(sample, nusc):
    """
    Generate Panoptic nuScenes ground truth labels.
    :param nusc: NuScenes instance.
    :param out_dir: output directory.
    :param verbose: True to print verbose.
    """
    root_dir = nusc.dataroot
    vels_dir = os.path.join(root_dir, "vels", nusc.version)
    mos_dir = os.path.join(root_dir, "mos_labels", nusc.version)
    os.makedirs(mos_dir, exist_ok=True)

    lidar_tok = sample['data']['LIDAR_TOP']
    lidar_data = nusc.get('sample_data', lidar_tok)
    ego_pose_tok = lidar_data['ego_pose_token']
    ego_pose = nusc.get('ego_pose', ego_pose_tok)
    ego_pose_trans = ego_pose['translation']

    # calculate ego_velo by current and the next (or prev) ego_pose
    ego_velo = np.nan
    if lidar_data['prev'] == '':
        lidar_tok_next = lidar_data['next']
        lidar_data_next = nusc.get('sample_data', lidar_tok_next)
        ego_pose_tok_next = lidar_data_next['ego_pose_token']
        ego_pose_next = nusc.get('ego_pose', ego_pose_tok_next)
        ego_pose_next_trans = ego_pose_next['translation']
        delta_x = ego_pose_trans[0] - ego_pose_next_trans[0]
        delta_y = ego_pose_trans[1] - ego_pose_next_trans[1]
        delta_z = ego_pose_trans[2] - ego_pose_next_trans[2]
        delta_t = 0.05
        ego_velo = np.sqrt(np.power(delta_x, 2) + np.power(delta_y, 2) + np.power(delta_z, 2)) / delta_t
    else:
        lidar_tok_prev = lidar_data['prev']
        lidar_data_prev = nusc.get('sample_data', lidar_tok_prev)
        ego_pose_tok_prev = lidar_data_prev['ego_pose_token']
        ego_pose_prev = nusc.get('ego_pose', ego_pose_tok_prev)
        ego_pose_prev_trans = ego_pose_prev['translation']
        delta_x = ego_pose_trans[0] - ego_pose_prev_trans[0]
        delta_y = ego_pose_trans[1] - ego_pose_prev_trans[1]
        delta_z = ego_pose_trans[2] - ego_pose_prev_trans[2]
        delta_t = 0.05
        ego_velo = np.sqrt(np.power(delta_x, 2) + np.power(delta_y, 2) + np.power(delta_z, 2)) / delta_t

    lidar_file = os.path.join(root_dir, lidar_data['filename'])
    lidarseg_file = os.path.join(root_dir, nusc.get('lidarseg', lidar_tok)['filename'])  # read lidarseg label
    lidarseg_pcd = LidarSegPointCloud(lidar_file, lidarseg_file)
    lidarseg_labels = lidarseg_pcd.labels

    vel_file = os.path.join(vels_dir, lidar_tok + "_vel.bin")
    vels_label = np.fromfile(vel_file, dtype=np.float64).reshape(-1, 3)

    num_pts = lidarseg_labels.shape[0]
    mos_label = np.zeros(num_pts, dtype=np.uint8)

    # statistics
    num_ego_mov = 0  # ego vehicle: moving -> 2
    num_ego_sta = 0  # ego vehicle: static -> 1
    num_sta_nan = 0  # static objects: NaN velocity -> 1
    num_sta_inv = 0  # static objects: invalid -> 0
    num_sta_sta = 0  # static objects: static -> 1
    num_mov_nan = 0  # movable objects: NaN velocity -> 0
    num_mov_mov = 0  # movable objects: moving -> 2
    num_mov_sta = 0  # movable objects: static -> 1

    veh_velo_thr_lb = 0.2  # vehicle velocity threshold lower bound
    veh_velo_thr_ub = 0.6  # vehicle velocity threshold upper bound
    hum_velo_thr_lb = 0.10  # human velocity threshold lower bound
    hum_velo_thr_ub = 0.35  # human velocity threshold upper bound

    # 0: unknown; 1: static; 2: moving
    for pt_idx in range(num_pts):
        vel_x = vels_label[pt_idx][0]
        vel_y = vels_label[pt_idx][1]
        vel_z = vels_label[pt_idx][2]
        if math.isnan(vel_x) or math.isnan(vel_y) or math.isnan(vel_z):
            vel = np.nan
        else:
            vel = np.sqrt(np.power(vel_x, 2) + np.power(vel_y, 2) + np.power(vel_z, 2))

        lidarseg_label = lidarseg_labels[pt_idx]
        if lidarseg_label == 31:  # vehicle.ego
            if ego_velo > veh_velo_thr_ub:
                mos_label[pt_idx] = 2  # moving
            elif ego_velo < veh_velo_thr_ub:
                mos_label[pt_idx] = 1  # static
            else:
                mos_label[pt_idx] = 0  # unknown
        elif lidarseg_label in [30, 29, 28, 27, 26, 25, 24, 13]:
            # static.vegetation, static.other, static.manmade
            # flat.driveable_surface, flat.terrain, flat.sidewalk, flat.other
            # static_object.bicycle_rack
            if math.isnan(vel):  # for the static object, the velocity is NaN -> no bbox -> mos = 1
                mos_label[pt_idx] = 1  # static
            elif vel > hum_velo_thr_lb:  # human velocity lower bound
                mos_label[pt_idx] = 0  # lidarseg label is not correct, or the velocity calculation is not correct
            else:
                mos_label[pt_idx] = 1
        elif lidarseg_label in [23, 22, 21, 20, 19, 18, 17, 16, 15, 14]:  # movable vehicle
            if math.isnan(vel):
                mos_label[pt_idx] = 0  # movable class, no velocity -> unknown
            elif vel > veh_velo_thr_ub:
                mos_label[pt_idx] = 2  # moving
            elif vel < veh_velo_thr_lb:
                mos_label[pt_idx] = 1  # static
            else:
                mos_label[pt_idx] = 0  # unknown, between vehicle threshold upper and lower bound
        elif lidarseg_label in [12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]:  # movable human or object
            if math.isnan(vel):
                mos_label[pt_idx] = 0  # movable class, no velocity -> unknown
            elif vel > hum_velo_thr_ub:
                mos_label[pt_idx] = 2  # moving
            elif vel < hum_velo_thr_lb:
                mos_label[pt_idx] = 1  # static
            else:
                mos_label[pt_idx] = 0  # unknown, between human threshold upper and lower bound
        elif lidarseg_label == 0:  # noise
            mos_label[pt_idx] = 0
        else:
            mos_label[pt_idx] = 0
    mos_file = os.path.join(mos_dir, lidar_tok + "_mos.label")
    mos_label.tofile(mos_file)
    # mos_loaded = np.fromfile(mos_file, dtype=np.uint8)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate nuScenes lidar panaptic gt.')
    parser.add_argument('--root_dir', type=str, default='/home/mars/MOS_Projects/nuScenes_MOS_Labeling/data',
                        help='Default nuScenes data directory.')
    parser.add_argument('--version', type=str, default='v1.0-trainval')
    parser.add_argument('--verbose', type=bool, default=True, help='Whether to print to stdout.')
    args = parser.parse_args()

    print(f'Start mos label generation... \nArguments: {args}')
    nusc = NuScenes(version=args.version, dataroot=args.root_dir, verbose=args.verbose)
    if not hasattr(nusc, "lidarseg") or len(getattr(nusc, 'lidarseg')) == 0:
        raise RuntimeError(f"No nuscenes-lidarseg annotations found in {nusc.version}")
    name2idx_mapping = nusc.lidarseg_name2idx_mapping  # idx2name_mapping = nusc.lidarseg_idx2name_mapping
    num_samples = len(nusc.sample)
    print(f'There are {num_samples} samples.')

    for sample in tqdm(nusc.sample):
        generate_mos_labels(sample, nusc)

    # no enough memory
    # multi-processing
    # pool = multiprocessing.Pool(processes=64)
    # for sample in tqdm(nusc.sample):
    #     pool.apply_async(func=generate_mos_labels, args=(sample, nusc,))
    # pool.close()
    # pool.join()
    print('Finished mos label generation.')







