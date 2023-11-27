import os
import math
from multiprocessing.managers import BaseManager
import json
import cv2
import argparse
import numpy as np
from tqdm import tqdm
import multiprocessing
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import points_in_box
from nuscenes.utils.data_io import load_bin_file
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import transform_matrix

def store_calib_pose(sample_data, calib_file):
    calib_token = sample_data['calibrated_sensor_token']
    calib = nusc.get('calibrated_sensor', calib_token)
    calib_mat = transform_matrix(calib['translation'], Quaternion(calib['rotation']))  # from lidar to car
    # save to file
    np.savetxt(calib_file, calib_mat, fmt='%.9f', delimiter=' ',
               newline='\n')  # %11.9f -> total 11 chars, 9 chars digits, float

def store_ego_pose(sample_data, pose_file):
    ego_pose_tok = sample_data['ego_pose_token']
    ego_pose = nusc.get('ego_pose', ego_pose_tok)
    trans_mat = np.array(ego_pose['translation']).reshape(-1, 1)
    rot_mat = Quaternion(ego_pose['rotation']).rotation_matrix
    # R t 0 1
    pose_mat = np.concatenate((rot_mat, trans_mat), axis=1)
    pose_mat = np.concatenate((pose_mat, np.array([[0, 0, 0, 1]])), axis=0)  # all float64
    # save to file
    np.savetxt(pose_file, pose_mat, fmt='%.9f', delimiter=' ',
               newline='\n')  # %11.9f -> total 11 chars, 9 chars digits, float


def store_lidar_intrinsics(root_dir, sample_data, lidar_intrinsics_file):
    num_beams = 32
    fov_up = 90 - 10  # degree (0 - 180)
    fov_down = 90 + 30  # degree
    v_fov = np.absolute(fov_down - fov_up) / 180 * np.pi
    h_fov = np.float32(2 * np.pi)

    azimuth_start_rad = np.float32(0)
    azimuth_end_rad = np.float32(2 * np.pi)
    elevation_start_rad = np.pi  # to be modified
    elevation_end_rad = np.float32(0)  # to be modified

    sample_data_file = os.path.join(root_dir, sample_data['filename'])
    points = LidarPointCloud.from_file(sample_data_file).points.T  # [num_points, 4]
    for point in points:
        x = point[0]
        y = point[1]
        z = point[2]
        r = np.sqrt(np.square(x) + np.square(y) + np.square(z))
        elevation_rad = np.arccos(z / r)

        elevation_start_rad = elevation_rad if (elevation_rad < elevation_start_rad) else elevation_start_rad
        elevation_end_rad = elevation_rad if (elevation_rad > elevation_end_rad) else elevation_end_rad

    scale_factor = 3
    num_azimuth_divisions = scale_factor * 512
    num_elevation_divisions = scale_factor * num_beams

    v_fov = elevation_end_rad - elevation_start_rad

    lidar_intrinsics = [
        [num_azimuth_divisions, num_elevation_divisions, h_fov, v_fov, azimuth_start_rad, azimuth_end_rad,
         elevation_start_rad, elevation_end_rad]]
    # save lidar-intrinsics to txt
    np.savetxt(lidar_intrinsics_file, lidar_intrinsics, fmt='%d %d %.5f %.5f %.5f %.5f %.5f %.5f', newline='\n')
    return lidar_intrinsics


def store_image(root_dir, sample_data, lidar_intrinsics, depth_img, height_img):
    # lidar intrinsics params
    num_azimuth_divisions = lidar_intrinsics[0][0]
    num_elevation_divisions = lidar_intrinsics[0][1]
    h_fov = lidar_intrinsics[0][2]
    v_fov = lidar_intrinsics[0][3]
    azimuth_start_rad = lidar_intrinsics[0][4]
    elevation_start_rad = lidar_intrinsics[0][6]
    # init image mat
    depth_mat = np.zeros((num_elevation_divisions, num_azimuth_divisions), dtype=np.uint16)  # row, col
    height_mat = np.zeros((num_elevation_divisions, num_azimuth_divisions), dtype=np.uint16)

    rad_per_pix_azimuth = h_fov / num_azimuth_divisions
    rad_per_pix_elevation = v_fov / (num_elevation_divisions - 1)

    sample_data_file = os.path.join(root_dir, sample_data['filename'])
    points = LidarPointCloud.from_file(sample_data_file).points.T  # [num_points, 4]

    default_scale_factor = 1000.0  # convert as mm
    default_scale_offset = 10.0  # assume all points are higher than -10.0 m
    for point in points:
        x = point[0]
        y = point[1]
        z = point[2]
        r = np.sqrt(np.square(x) + np.square(y) + np.square(z))
        elevation_rad = np.arccos(z / r)
        azimuth_rad = np.pi - np.arctan2(y, x)  # arctan2: x1 / x2

        row_idx = round((elevation_rad - elevation_start_rad) / rad_per_pix_elevation)
        if (row_idx < 0) or (row_idx > num_elevation_divisions - 1): continue
        col_idx = round((azimuth_rad - azimuth_start_rad) / rad_per_pix_azimuth)
        if (col_idx < 0) or (col_idx > num_azimuth_divisions - 1): continue

        r_uint16 = np.uint16(r * default_scale_factor)
        z_uint16 = np.uint16((z + default_scale_offset) * default_scale_factor)
        depth_mat[row_idx][col_idx] = r_uint16
        height_mat[row_idx][col_idx] = z_uint16

    cv2.imwrite(depth_img, depth_mat)
    cv2.imwrite(height_img, height_mat)

def store_sdt_to_idx_dict(scene_idx, scene, nusc):
    seq_idx = scene_idx
    frame_idx = 0
    sdt_2_idx_dict = {}  # sample data token -> {seq_idx, frame_idx}, only store keyframes

    sample = nusc.get('sample', scene['first_sample_token'])
    sample_data_tok = sample['data']['LIDAR_TOP']
    sample_data = nusc.get('sample_data', sample_data_tok)

    sdt_2_idx_dict[sample_data_tok] = (seq_idx, frame_idx)
    while sample_data['next'] != '':
        frame_idx += 1
        sample_data_tok = sample_data['next']
        sample_data = nusc.get('sample_data', sample_data_tok)
        if not sample_data['is_key_frame']:
            continue
        else:
            sdt_2_idx_dict[sample_data_tok] = (seq_idx, frame_idx)

    dict_dir = "/home/mars/MOS_Projects/nvblox_datasets/nusc/dict"
    dict_file = os.path.join(dict_dir, "seq-" + str(seq_idx).zfill(4) + ".dict.txt")

    with open(dict_file, "w") as fp:
        json.dump(sdt_2_idx_dict, fp)
    print("Save token to index dict to .txt file")


def transfer_to_nvblox(scene_idx, scene, nusc):
    """
    Generate Panoptic nuScenes ground truth labels.
    :param nusc: NuScenes instance.
    :param out_dir: output directory.
    :param verbose: True to print verbose.
    """
    # define index
    seq_idx = scene_idx
    frame_idx = 0
    sdt_2_idx_dict = {}  # sample data token -> {seq_idx, frame_idx}

    # data directory
    root_dir = nusc.dataroot
    nvblox_data_root = "/home/mars/MOS_Projects/nvblox_datasets/nusc"
    os.makedirs(nvblox_data_root, exist_ok=True)
    seq_dir = os.path.join(nvblox_data_root, "seq-" + str(seq_idx).zfill(4))
    os.makedirs(seq_dir, exist_ok=True)
    # create mesh and esdf directory
    mesh_dir = os.path.join(nvblox_data_root, "mesh", "seq-" + str(seq_idx).zfill(4))
    esdf_dir = os.path.join(nvblox_data_root, "esdf", "seq-" + str(seq_idx).zfill(4))
    os.makedirs(mesh_dir, exist_ok=True)
    os.makedirs(esdf_dir, exist_ok=True)

    # process the first sample data
    first_sample = nusc.get('sample', scene['first_sample_token'])
    assert first_sample['prev'] == '', "The first sample is not the first sample data"
    sample_data_tok = first_sample['data']['LIDAR_TOP']
    sample_data = nusc.get('sample_data', sample_data_tok)
    # fill in the dictionary
    sdt_2_idx_dict[sample_data_tok] = (seq_idx, frame_idx)

    # filename of the first sample data
    depth_img = os.path.join(seq_dir, "frame-" + str(frame_idx).zfill(6) + ".depth.png")
    height_img = os.path.join(seq_dir, "frame-" + str(frame_idx).zfill(6) + ".height.png")
    lidar_intr_txt = os.path.join(seq_dir, "frame-" + str(frame_idx).zfill(6) + ".lidar-intrinsics.txt")
    pose_txt = os.path.join(seq_dir, "frame-" + str(frame_idx).zfill(6) + ".pose.txt")
    calib_txt = os.path.join(nvblox_data_root, "calib.txt")

    # store transfered data
    store_calib_pose(sample_data, calib_txt)
    store_ego_pose(sample_data, pose_txt)
    lidar_intrinsics = store_lidar_intrinsics(root_dir, sample_data, lidar_intr_txt)
    store_image(root_dir, sample_data, lidar_intrinsics, depth_img, height_img)

    # while true loop
    while sample_data['next'] != '':
        frame_idx += 1
        # get next sample data
        sample_data_tok = sample_data['next']
        sample_data = nusc.get('sample_data', sample_data_tok)

        # fill in the dictionary
        sdt_2_idx_dict[sample_data_tok] = (seq_idx, frame_idx)

        # file path
        depth_img = os.path.join(seq_dir, "frame-" + str(frame_idx).zfill(6) + ".depth.png")
        height_img = os.path.join(seq_dir, "frame-" + str(frame_idx).zfill(6) + ".height.png")
        lidar_intr_txt = os.path.join(seq_dir, "frame-" + str(frame_idx).zfill(6) + ".lidar-intrinsics.txt")
        pose_txt = os.path.join(seq_dir, "frame-" + str(frame_idx).zfill(6) + ".pose.txt")

        # store transfered data
        store_ego_pose(sample_data, pose_txt)
        lidar_intrinsics = store_lidar_intrinsics(root_dir, sample_data, lidar_intr_txt)
        store_image(root_dir, sample_data, lidar_intrinsics, depth_img, height_img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate nuScenes lidar panaptic gt.')
    parser.add_argument('--root_dir', type=str, default='/home/mars/MOS_Projects/nuScenes_MOS_Labeling/data',
                        help='Default nuScenes data directory.')
    parser.add_argument('--version', type=str, default='v1.0-trainval')
    parser.add_argument('--verbose', type=bool, default=True, help='Whether to print to stdout.')
    args = parser.parse_args()

    print(f'Start transfer to nvblox-format data... \nArguments: {args}')
    nusc = NuScenes(version=args.version, dataroot=args.root_dir, verbose=args.verbose)
    if not hasattr(nusc, "lidarseg") or len(getattr(nusc, 'lidarseg')) == 0:
        raise RuntimeError(f"No nuscenes-lidarseg annotations found in {nusc.version}")
    name2idx_mapping = nusc.lidarseg_name2idx_mapping  # idx2name_mapping = nusc.lidarseg_idx2name_mapping

    num_scene = len(nusc.scene)
    print(f'There are {num_scene} scenes.')
    for scene_idx, scene in tqdm(enumerate(nusc.scene)):
        # store_sdt_to_idx_dict(scene_idx, scene, nusc)
        transfer_to_nvblox(scene_idx, scene, nusc)
