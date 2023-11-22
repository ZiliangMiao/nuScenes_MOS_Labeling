import multiprocessing
import os
import os.path
import subprocess
import warnings

import numpy as np
import open3d as o3d
from plyfile import PlyData, PlyElement
import matplotlib.pyplot as plt
import argparse
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion

mos_colormap = {
        0: (0.439, 0.439, 0.439),   # unknown: green-light
        1: (0.859, 0.882, 1.000),   # static: white
        2: (0.071, 0.369, 0.055)     # moving: green-dark
}

# gray colormap: https://colorswall.com/palette/24454
gray_colormap = {
    0: (211/255, 211/255, 211/255),  # Pinball
    1: (190/255, 190/255, 190/255),
    2: (169/255, 169/255, 169/255),
    3: (148/255, 148/255, 148/255),
    4: (127/255, 127/255, 127/255),
    5: (106/255, 106/255, 106/255),
    6: (84/255,  84/255,  84/255),
    7: (63/255,  63/255,  63/255),
    8: (42/255,  42/255,  42/255),
    9: (21/255,  21/255,  21/255)     # Matt Black
}

# magma colormap: https://waldyrious.net/viridis-palette-generator/
magma_colormap = {
    0: (252/255, 253/255, 191/255),  #fcfdbf
    1: (254/255, 202/255, 141/255),  #feca8d
    2: (253/255, 150/255, 104/255),  #fd9668
    3: (241/255, 96/255,  93/255),   #f1605d
    4: (241/255, 96/255,  93/255),   #cd4071
    5: (158/255, 47/255,  127/255),  #9e2f7f
    6: (114/255, 31/255,  129/255),  #721f81
    7: (68/255,  15/255,  118/255),  #440f76
    8: (24/255,  15/255,  61/255),   #180f3d
    9: (0/255,   0/255,   4/255)     #000004
}

coolwarm_colormap = {
    100: (0.859, 0.882, 1.000),  # set nan to backgroud color
    -6: (59/255, 76/255, 192/255),  # blue
    -5: (83/255, 111/255, 220/255),
    -4: (111/255, 145/255, 242/255),
    -3: (140/255, 174/255, 252/255),
    -2: (169/255, 197/255, 252/255),
    -1: (197/255, 213/255, 241/255),
    0: (221/255, 221/255, 221/255), # middle point
    1: (238/255, 205/255, 188/255),
    2: (246/255, 182/255, 155/255),
    3: (243/255, 152/255, 122/255),
    4: (230/255, 116/255, 91/255),
    5: (208/255, 73/255, 62/255),
    6: (180/255, 4/255, 38/255)  # red
}

sdt_2_idx_dict = {
    '3388933b59444c5db71fade0bbfef470': (0, 0),  # seq-0, frame-0
    'bc2cd87d110747cd9849e2b8578b7877': (0, 1),
    '68cb874a16654e909a7b240a608327db': (0, 2),
    '262df5a1529c4dcbb6106cb1a23b8a95': (0, 3),
    '73cd8a8a6c79453f9a7236d550de1e7a': (0, 4),
    '73a245e42075416db6f181debe3b9873': (0, 5),
    'ada43a66cd404ab4a27e3f1e400f8e6e': (0, 6),
    '89db24d0922f4773a63668fb84f82161': (0, 7),
    'b824950a67e94a4eaaeea0bdfc963879': (0, 8),
    'd70b301e89e5422498679514a0547d61': (0, 9),   # no mos labels
    '69b793ec8dc44e2fbd33d8cdd16b5a31': (0, 10),  # has mos labels
    '9d6543c10b024644a00eba4575c8212e': (0, 11),
    '5b3d3ed71f154efe987ced333b98f510': (0, 12),
    '03bf3718e4d44f0c91d73f8161b3d361': (0, 13),
    'a2f38de77cfa46edbf1809fb6d3118e4': (0, 14),
    '276e63c991cf414fb7352546c9c845df': (0, 15),
    '996ef59c52c24b7bb0867462f00837b1': (0, 16),
    'b8a0680730a741e9a66837c2ab2bbd72': (0, 17),
    '0e2597532b814f2a8713aeb54b333cce': (0, 18),
    '9512aac8418f4dcfbe8f72b74061fcff': (0, 19),
    'ec310d318e6a46c08c43311dd898f0a5': (0, 20),
}

idx_2_sdt_dict = {v: k for k, v in sdt_2_idx_dict.items()}

def run_fuse_kitti_shell():
    # os.system
    print(os.system("/home/mars/MOS_Projects/nvblox/nvblox/script/run_fuse_kitti_frames.sh"))

    # subprocess
    # shell_command_list = [
    #     "cd /home/mars/MOS_Projects/nvblox/nvblox/build",
    #
    #     "make -j64  && \\"
    #     "./executables/fuse_kitti \\"
    #     "/home/mars/MOS_Projects/nvblox/nvblox/tests/data/kitti \\"
    #     "--tsdf_integrator_max_integration_distance_m 30.0 \\"
    #     "--tsdf_integrator_truncation_distance_vox 300 \\"
    #     "--num_frames 2 \\"
    #     "--voxel_size 0.2 \\"
    #     "--esdf_frame_subsampling \\"
    #     "--esdf_mode 0 \\"
    #     "--mesh_output_path \\"
    #     "/home/mars/MOS_Projects/nvblox/nvblox/outputs/kitti/kitti_seq_01_sync_mesh_10.ply \\"
    #     "--esdf_output_path \\"
    #     "/home/mars/MOS_Projects/nvblox/nvblox/outputs/kitti/kitti_seq_01_sync_esdf_10.ply \\"
    # ]
    # print(subprocess.call(shell_command_list, shell=True))

def read_esdf(esdf_file):
    plydata = PlyData.read(esdf_file)
    # esdf = plydata.elements[0].data
    # esdf_data = plydata['vertex'].data
    # esdf_array = np.array([[x, y, z, i] for x, y, z, i in esdf_data])
    x = plydata['vertex']['x']
    y = plydata['vertex']['y']
    z = plydata['vertex']['z']
    i = plydata['vertex']['intensity']
    esdf_max = np.max(i)
    esdf_min = np.min(i)
    print('ESDF distance range: ', str(esdf_min), " ", str(esdf_max))
    # get the code from: https://blog.csdn.net/phy12321/article/details/107373073
    # NOTE!!! do not use np.fromfile to read the .ply file
    esdf = np.array([x, y, z, i]).T
    return esdf

def open3d_vis_esdf(esdf):
    esdf_points = esdf[:, :3]
    esdf_dis = esdf[:, -1].reshape((-1, 1))

    esdf_max = np.max(esdf_dis)
    esdf_min = np.min(esdf_dis)
    esdf_dis_norm = 1 - (esdf_dis - esdf_min) / (esdf_max - esdf_min)
    esdf_colors = np.tile(esdf_dis_norm, (1, 3))  # repeat array alone axis-0 and axis-1 respectively

    esdf_pcd = o3d.geometry.PointCloud()
    esdf_pcd.points = o3d.utility.Vector3dVector(esdf_points)
    esdf_pcd.colors = o3d.utility.Vector3dVector(esdf_colors)  # 0: WHITE, 1: BLACK

    print('Displaying voxel grid ...')
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(esdf_pcd, voxel_size=0.09)  # actuall voxel size: 0.1

    # Open3D Vis
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(voxel_grid)

    # Vis Settings (render options & view control)
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.859, 0.882, 1.000])
    ctrl = vis.get_view_control()
    # ctrl.set_front((-0.020588885231781512, 0.10962245064836584, 0.99376003950589564))
    # ctrl.set_lookat((0.066336316011035693, 3.437049303689776, 0.11533577801642712))
    # ctrl.set_up((0.40551647562275533, 0.90945418190674621, -0.091921047703072634))
    # ctrl.set_zoom((0.45999999999999974))

    # Vis Run
    vis.run()
    vis.destroy_window()

def open3d_vis_esdf_res(esdf_res):
    esdf_points = esdf_res[:, :3]
    esdf_dis = esdf_res[:, -1].reshape((-1, 1))

    # process the newly observed voxels (np.nan)
    nan_idx = np.argwhere(np.isnan(esdf_res[:, -1]))
    # transfer nan to zero
    esdf_dis[nan_idx] = 0

    # absolute (to vis esdf residual)
    esdf_dis = np.absolute(esdf_dis)

    esdf_max = np.max(esdf_dis)
    esdf_min = np.min(esdf_dis)
    esdf_dis_norm = 1 - (esdf_dis - esdf_min) / (esdf_max - esdf_min)
    esdf_colors = np.tile(esdf_dis_norm, (1, 3))  # repeat array alone axis-0 and axis-1 respectively
    # esdf_colors[nan_idx] = (0.941, 0.431, 0.478)  # change color of nan to blue
    esdf_colors[nan_idx] = (0.859, 0.882, 1.000)  # transparency: set nan voxel to the background color

    esdf_pcd = o3d.geometry.PointCloud()
    esdf_pcd.points = o3d.utility.Vector3dVector(esdf_points)
    esdf_pcd.colors = o3d.utility.Vector3dVector(
        esdf_colors)  # 0: black, 1: white (voxels with max distance are colored white)

    print('Displaying voxel grid ...')
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(esdf_pcd, voxel_size=0.09)  # actuall voxel size: 0.1

    # Open3D Vis
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(voxel_grid)

    # Vis Settings (render options & view control)
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.859, 0.882, 1.000])
    ctrl = vis.get_view_control()
    # ctrl.set_front((-0.020588885231781512, 0.10962245064836584, 0.99376003950589564))
    # ctrl.set_lookat((0.066336316011035693, 3.437049303689776, 0.11533577801642712))
    # ctrl.set_up((0.40551647562275533, 0.90945418190674621, -0.091921047703072634))
    # ctrl.set_zoom((0.45999999999999974))

    # Vis Run
    vis.run()
    vis.destroy_window()

def open3d_vis_esdf_res_mos(esdf_res, sample_data_token, nusc):
    res = esdf_res[:, -1]
    res_color_idx = np.zeros_like(res, dtype=np.int16)

    # process the newly observed voxels (np.nan)
    nan_idx = np.argwhere(np.isnan(res))
    neg_6_idx = np.argwhere(res <= -1.00)                # (-infinite, -1.00] m
    neg_5_idx = np.argwhere((-1.00 < res) & (res <= -0.50))  # (-1.00, -0.50] m
    neg_4_idx = np.argwhere((-0.50 < res) & (res <= -0.20))  # (-0.50, -0.20] m
    neg_3_idx = np.argwhere((-0.20 < res) & (res <= -0.10))  # (-0.20, -0.10] m
    neg_2_idx = np.argwhere((-0.10 < res) & (res <= -0.05))  # (-0.10, -0.05] m
    neg_1_idx = np.argwhere((-0.05 < res) & (res <   0.00))  # (-0.05, -0.00) m
    mid_0_idx = np.argwhere(res == 0)                    # [0] m
    pos_1_idx = np.argwhere((0.00  < res) & (res < 0.05))    # (0.00, 0.05) m
    pos_2_idx = np.argwhere((0.05 <= res) & (res < 0.10))    # [0.05, 0.10) m
    pos_3_idx = np.argwhere((0.10 <= res) & (res < 0.20))    # [0.10, 0.20) m
    pos_4_idx = np.argwhere((0.20 <= res) & (res < 0.50))    # [0.20, 0.50) m
    pos_5_idx = np.argwhere((0.50 <= res) & (res < 1.00))    # [0.50, 1.00) m
    pos_6_idx = np.argwhere(1.00 <= res)                 # [1.00, infinite) m

    res[nan_idx] = 100
    res[neg_6_idx] = -6
    res[neg_5_idx] = -5
    res[neg_4_idx] = -4
    res[neg_3_idx] = -3
    res[neg_2_idx] = -2
    res[neg_1_idx] = -1
    res[mid_0_idx] = 0
    res[pos_1_idx] = 1
    res[pos_2_idx] = 2
    res[pos_3_idx] = 3
    res[pos_4_idx] = 4
    res[pos_5_idx] = 5
    res[pos_6_idx] = 6

    color_func = np.vectorize(coolwarm_colormap.get)
    esdf_res_color = np.array(color_func(res)).T

    esdf_pcd = o3d.geometry.PointCloud()
    esdf_pcd.points = o3d.utility.Vector3dVector(esdf_res[:, :3])
    esdf_pcd.colors = o3d.utility.Vector3dVector(esdf_res_color)

    print('Displaying voxel grid ...')
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(esdf_pcd, voxel_size=0.1)  # actuall voxel size: 0.1

    # point cloud colored by mos label
    sample_data = nusc.get('sample_data', sample_data_token)
    sample_data_next_token = sample_data['next']
    sample_data_next = nusc.get('sample_data', sample_data_next_token)

    pcl_path = os.path.join(nusc.dataroot, sample_data['filename'])
    points_l = LidarPointCloud.from_file(pcl_path).points.T  # [num_points, 4]

    # transform from current lidar frame to global frame
    pose_token = sample_data['ego_pose_token']
    pose = nusc.get('ego_pose', pose_token)  # vehicle to world
    T_v_2_w = transform_matrix(pose['translation'], Quaternion(pose['rotation']))  # from vehicle to world

    # calib pose
    calib_token = sample_data['calibrated_sensor_token']
    calib = nusc.get('calibrated_sensor', calib_token)  # lidar to vehicle
    T_l_2_v = transform_matrix(calib['translation'], Quaternion(calib['rotation']))  # from lidar to vehicle

    points_l = points_l[:, :3]

    T_l_2_w = T_v_2_w @ T_l_2_v
    points_l_homo = np.hstack([points_l, np.ones((points_l.shape[0], 1))]).T
    points_w = (T_l_2_w @ points_l_homo).T[:, :3]

    # esdf residual: next - curr
    # mos label: next
    mos_labels_file = os.path.join(nusc.dataroot, 'mos_labels', nusc.version, sample_data_token + "_mos.label")
    points_label = np.fromfile(mos_labels_file, dtype=np.uint8)
    # remove static points from the vis point cloud
    unknown_idx = np.argwhere(points_label == 0)
    static_idx = np.argwhere(points_label == 1)
    mov_idx = np.argwhere(points_label == 2)
    vis_points = np.squeeze(np.vstack((points_w[mov_idx], points_w[unknown_idx])))
    vis_labels = np.squeeze(np.vstack((points_label[mov_idx], points_label[unknown_idx])))

    pts = o3d.geometry.PointCloud()
    pts.points = o3d.utility.Vector3dVector(vis_points)
    vfunc = np.vectorize(mos_colormap.get)
    points_color = np.array(vfunc(vis_labels)).T
    pts.colors = o3d.utility.Vector3dVector(points_color)

    # origin
    axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])

    # Open3D Vis
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(voxel_grid)
    vis.add_geometry(pts)
    vis.add_geometry(axis_pcd)

    # Vis Settings (render options & view control)
    opt = vis.get_render_option()
    opt.point_size = 2.0
    opt.background_color = np.asarray([0.859, 0.882, 1.000])
    ctrl = vis.get_view_control()
    # ctrl.set_front((-0.020588885231781512, 0.10962245064836584, 0.99376003950589564))
    # ctrl.set_lookat((0.066336316011035693, 3.437049303689776, 0.11533577801642712))
    # ctrl.set_up((0.40551647562275533, 0.90945418190674621, -0.091921047703072634))
    # ctrl.set_zoom((0.45999999999999974))

    # Vis Run
    vis.run()
    vis.destroy_window()

def esdf_histogram(esdf):
    esdf_dis = esdf[:, -1].reshape((-1, 1))
    # n, bins, patches = plt.hist(x=esdf_dis, bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85)
    n, bins, patches = plt.hist(x=esdf_dis, bins=100, color='#0504aa', alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('ESDF Distance')
    plt.ylabel('Frequency')
    plt.title('ESDF Distance Histogram')
    maxfreq = n.max()
    # Set a clean upper y-axis limit.
    plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    plt.show()

def compute_residual(curr_voxel_idx, curr_voxel, esdf_hist_voxels, esdf_residual):
    row_idx = np.where((esdf_hist_voxels == curr_voxel).all(axis=1))[0]
    if row_idx:  # not empty
        curr_residual = esdf_next[curr_voxel_idx, -1] - esdf_curr[row_idx, -1]
        esdf_residual[curr_voxel_idx] = np.append(curr_voxel, curr_residual)
    else:  # empty
        curr_residual = np.nan
        esdf_residual[curr_voxel_idx] = np.append(curr_voxel, curr_residual)


if __name__ == '__main__':
    # Switch
    run_kitti = False
    load_residual = False
    vis_esdf = False
    vis_esdf_res = True
    vis_esdf_histogram = False

    # Path
    seq_idx = 0
    frame_idx = 10
    sample_data_token = idx_2_sdt_dict[(seq_idx, frame_idx)]
    esdf_dir = "/home/mars/MOS_Projects/nvblox_datasets/nusc/esdf"
    esdf_res_dir = "/home/mars/MOS_Projects/nvblox_datasets/nusc/esdf_res"

    # Shell script
    if run_kitti:
        run_fuse_kitti_shell()
    # truncation_distance_vox   |   max_distance_m (esdf)   |   Integrated TSDF block (frame-0 / frame-1)   |   update blocks (frame-0 / frame-1)
    # 50                        |     10                    |    3255 / 3282                                |   40 / 40
    # 30                        |     10                    |    2146 / 2162                                |   40 / 40
    # 10                        |     10                    |    1325 / 1336                                |   40 / 40
    # 4                         |     20                    |    1148 / 1162                                |   40 / 40
    # 4 (default)               |     10                    |    1148 / 1162                                |   40 / 40
    # 1                         |     10                    |    1086 / 1098                                |   40 / 40

    # load stored esdf residual file
    if load_residual:
        residual_file = os.path.join(esdf_res_dir, "seq-" + str(seq_idx).zfill(4), "frame-" + str(frame_idx).zfill(6) + ".esdf_res.bin")
        esdf_residual = np.fromfile(residual_file, dtype=np.float32).reshape((-1, 4))
        if vis_esdf_histogram:
            esdf_histogram(esdf_residual)
        if vis_esdf_res:
            parser = argparse.ArgumentParser(description='Generate nuScenes lidar panaptic gt.')
            parser.add_argument('--root_dir', type=str, default='/home/mars/MOS_Projects/nuScenes_MOS_Labeling/data')
            parser.add_argument('--version', type=str, default='v1.0-trainval')
            parser.add_argument('--verbose', type=bool, default=True, help='Whether to print to stdout.')
            args = parser.parse_args()
            nusc = NuScenes(version=args.version, dataroot=args.root_dir, verbose=args.verbose)
            # esdf residual with point cloud colored by mos labels
            open3d_vis_esdf_res_mos(esdf_residual, sample_data_token, nusc)

        # not_nan_idx = np.argwhere(np.invert(np.isnan(esdf_residual[:, -1])))
        # residual = esdf_residual[not_nan_idx, -1]
    else:  # compute esdf residual
        # read esdf that integrates 5 frames
        esdf_next = read_esdf(os.path.join(esdf_dir, "seq-" + str(seq_idx).zfill(4), "frame-" + str(frame_idx + 1).zfill(6) + ".esdf.ply"))
        # Vis ESDF in Open3D
        if vis_esdf:
            open3d_vis_esdf(esdf_next)
        # histogram of esdf distances
        if vis_esdf_histogram:
            esdf_histogram(esdf_next)

        esdf_curr = read_esdf(os.path.join(esdf_dir, "seq-" + str(seq_idx).zfill(4), "frame-" + str(frame_idx).zfill(6) + ".esdf.ply"))
        if vis_esdf:
            open3d_vis_esdf(esdf_curr)
        esdf_next_voxels = esdf_next[:, :3]
        esdf_curr_voxels = esdf_curr[:, :3]
        esdf_residual = np.zeros_like(esdf_next)  # dtype=np.float32

        num_nan = 0
        warnings.filterwarnings("ignore")
        for next_voxel_idx, next_voxel in enumerate(esdf_next_voxels):
            row_idx = np.where((esdf_curr_voxels == next_voxel).all(axis=1))[0]
            if row_idx:  # not empty
                residual = esdf_next[next_voxel_idx, -1] - esdf_curr[row_idx, -1]
                esdf_residual[next_voxel_idx] = np.append(next_voxel, residual)
            else:  # empty
                residual = np.nan
                esdf_residual[next_voxel_idx] = np.append(next_voxel, residual)
                num_nan += 1
        print("Num of current voxels: ", str(esdf_next.shape[0]), "; Num of history voxels: ", str(esdf_curr.shape[0]),
              "; Num of voxels with NaN residual: ", str(num_nan))

        residual_file = os.path.join(esdf_res_dir, "seq-" + str(seq_idx).zfill(4), "frame-" + str(frame_idx).zfill(6) + ".esdf_res.bin")
        esdf_residual.tofile(residual_file)
        print("Save ESDF Residual to .bin file: ")
        if vis_esdf_res:
            parser = argparse.ArgumentParser(description='Generate nuScenes lidar panaptic gt.')
            parser.add_argument('--root_dir', type=str, default='/home/mars/MOS_Projects/nuScenes_MOS_Labeling/data')
            parser.add_argument('--version', type=str, default='v1.0-trainval')
            parser.add_argument('--verbose', type=bool, default=True, help='Whether to print to stdout.')
            args = parser.parse_args()
            nusc = NuScenes(version=args.version, dataroot=args.root_dir, verbose=args.verbose)
            # esdf residual with point cloud colored by mos labels
            open3d_vis_esdf_res_mos(esdf_residual, sample_data_token, nusc)

