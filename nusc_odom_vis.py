import os
import os.path
import json
import numpy as np
import open3d as o3d
import argparse
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion

o3d_voxel_size = 0.1
o3d_point_size = 2.0

o3d_background_color = (0.859, 0.882, 1.000)  # blue

coolwarm_colormap = {
    0: (59/255, 76/255, 192/255),  # blue
    1: (83/255, 111/255, 220/255),
    2: (111/255, 145/255, 242/255),
    3: (140/255, 174/255, 252/255),
    4: (169/255, 197/255, 252/255),
    5: (197/255, 213/255, 241/255),
    6: (221/255, 221/255, 221/255), # middle point
    7: (238/255, 205/255, 188/255),
    8: (246/255, 182/255, 155/255),
    9: (243/255, 152/255, 122/255),
    10: (230/255, 116/255, 91/255),
    11: (208/255, 73/255, 62/255),
    12: (180/255, 4/255, 38/255)  # red
}

def filter_far_points(points):
    max_dis = 35
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    points_dist = np.sqrt(np.square(x) + np.square(y) + np.square(z))
    valid_points_idx = np.squeeze(np.argwhere(points_dist <= max_dis))
    return valid_points_idx

def get_pcl_in_world_frame(nusc, sample_data_token):
    sample_data = nusc.get('sample_data', sample_data_token)
    pcl_path = os.path.join(nusc.dataroot, sample_data['filename'])
    points_l = LidarPointCloud.from_file(pcl_path).points.T  # [num_points, 4]
    points_l = points_l[:, :3]  # without intensity

    # filter points with distance > max distacne
    valid_points_idx = filter_far_points(points_l)

    # odom pose: from vehicle to world
    pose_token = sample_data['ego_pose_token']
    pose = nusc.get('ego_pose', pose_token)
    T_v_2_w = transform_matrix(pose['translation'], Quaternion(pose['rotation']))

    # calib pose: from lidar to vehicle
    calib_token = sample_data['calibrated_sensor_token']
    calib = nusc.get('calibrated_sensor', calib_token)
    T_l_2_v = transform_matrix(calib['translation'], Quaternion(calib['rotation']))

    # transform point cloud from lidar frame to world frame
    T_l_2_w = T_v_2_w @ T_l_2_v
    points_l_homo = np.hstack([points_l, np.ones((points_l.shape[0], 1))]).T
    points_w = (T_l_2_w @ points_l_homo).T[:, :3]
    return points_w, valid_points_idx

def o3d_vis_pcl(o3d_pcl_list):
    # open3d visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    for pcl in o3d_pcl_list:
        vis.add_geometry(pcl)

    # open3d render options
    opt = vis.get_render_option()
    opt.point_size = o3d_point_size
    opt.background_color = np.asarray(o3d_background_color)
    # open3d view control
    ctrl = vis.get_view_control()

    vis.run()
    vis.destroy_window()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate nuScenes lidar panaptic gt.')
    parser.add_argument('--root_dir', type=str, default='/home/mars/MOS_Projects/nuScenes_MOS_Labeling/mini_data')
    parser.add_argument('--version', type=str, default='v1.0-mini')
    parser.add_argument('--verbose', type=bool, default=True, help='Whether to print to stdout.')
    args = parser.parse_args()
    nusc = NuScenes(version=args.version, dataroot=args.root_dir, verbose=args.verbose)

    scene_idx = 0
    scan_idx_start = 0
    scan_idx_end = 11

    nusc_pcl_tok_list = []
    for curr_scene_idx, scene in enumerate(nusc.scene):
        if curr_scene_idx == scene_idx:
            sample = nusc.get('sample', scene['first_sample_token'])  # sample['prev'], sample['next']
            sample_data_tok = sample['data']['LIDAR_TOP']
            sample_data = nusc.get('sample_data', sample_data_tok)
            scan_idx = 0
            integrate_state = 0
            if scan_idx == scan_idx_start:
                nusc_pcl_tok_list.append(sample_data_tok)
                integrate_state = 1
            if scan_idx == scan_idx_end:
                integrate_state = 0
                break

            while sample_data['next'] != '':
                scan_idx += 1
                sample_data_tok = sample_data['next']
                sample_data = nusc.get('sample_data', sample_data_tok)

                if scan_idx == scan_idx_start:
                    integrate_state = 1

                if integrate_state == 1:
                    nusc_pcl_tok_list.append(sample_data_tok)

                if scan_idx == scan_idx_end:
                    integrate_state = 0
                    break

    o3d_pcl_list = []
    for color_idx, pcl_tok in enumerate(nusc_pcl_tok_list):
        points_w, valid_pts_idx = get_pcl_in_world_frame(nusc, pcl_tok)

        o3d_pcl = o3d.geometry.PointCloud()
        o3d_pcl.points = o3d.utility.Vector3dVector(points_w)
        color = coolwarm_colormap[color_idx]
        colors = np.full(points_w.shape, color)
        o3d_pcl.colors = o3d.utility.Vector3dVector(colors)

        o3d_pcl_list.append(o3d_pcl)

    o3d_vis_pcl(o3d_pcl_list)