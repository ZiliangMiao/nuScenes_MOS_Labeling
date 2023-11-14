import multiprocessing
import os
import os.path
import subprocess
import warnings

import numpy as np
import open3d as o3d
from plyfile import PlyData, PlyElement
import matplotlib.pyplot as plt

def run_fuse_kitti_shell():
    # os.system
    print(os.system("/home/mars/MOS_Projects/nvblox/nvblox/script/run_fuse_kitti.sh"))

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

    # process the newly observed voxels (np.nan)
    nan_idx = np.argwhere(np.isnan(esdf[:, -1]))
    # transfer nan to zero
    esdf_dis[nan_idx] = 0

    esdf_max = np.max(esdf_dis)
    esdf_min = np.min(esdf_dis)
    esdf_dis_norm = (esdf_dis - esdf_min) / (esdf_max - esdf_min)
    esdf_colors = np.tile(esdf_dis_norm, (1, 3))  # repeat array alone axis-0 and axis-1 respectively
    esdf_colors[nan_idx] = (0.941, 0.431, 0.478)  # change color of nan to red
    # esdf_colors[nan_idx] = (0.859, 0.882, 1.000)

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
    ctrl.set_front((-0.020588885231781512, 0.10962245064836584, 0.99376003950589564))
    ctrl.set_lookat((0.066336316011035693, 3.437049303689776, 0.11533577801642712))
    ctrl.set_up((0.40551647562275533, 0.90945418190674621, -0.091921047703072634))
    ctrl.set_zoom((0.45999999999999974))

    # Vis Run
    vis.run()
    vis.destroy_window()

def esdf_histogram(esdf):
    esdf_dis = esdf[:, -1].reshape((-1, 1))
    n, bins, patches = plt.hist(x=esdf_dis, bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85)
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
        curr_residual = esdf_curr[curr_voxel_idx, -1] - esdf_hist[row_idx, -1]
        esdf_residual[curr_voxel_idx] = np.append(curr_voxel, curr_residual)
    else:  # empty
        curr_residual = np.nan
        esdf_residual[curr_voxel_idx] = np.append(curr_voxel, curr_residual)


if __name__ == '__main__':
    # Switch
    run_kitti = False
    vis_esdf_hist = False
    vis_esdf_curr = False
    vis_esdf_res = True
    vis_esdf_histogram = False

    # Path
    nvblox_output = "/home/mars/MOS_Projects/nvblox/nvblox/outputs"
    dataset = "kitti"

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

    # read esdf that integrates 5 frames
    esdf_curr = read_esdf(os.path.join(nvblox_output, dataset, "kitti_seq_01_sync_esdf_3.ply"))
    esdf_hist = read_esdf(os.path.join(nvblox_output, dataset, "kitti_seq_01_sync_esdf_2.ply"))
    esdf_curr_voxels = esdf_curr[:, :3]
    esdf_hist_voxels = esdf_hist[:, :3]
    esdf_residual = np.zeros_like(esdf_curr)  # dtype=np.float32

    num_nan = 0
    warnings.filterwarnings("ignore")
    for curr_voxel_idx, curr_voxel in enumerate(esdf_curr_voxels):
        row_idx = np.where((esdf_hist_voxels == curr_voxel).all(axis=1))[0]
        if row_idx:  # not empty
            curr_residual = esdf_curr[curr_voxel_idx, -1] - esdf_hist[row_idx, -1]
            esdf_residual[curr_voxel_idx] = np.append(curr_voxel, curr_residual)
        else:  # empty
            curr_residual = np.nan
            esdf_residual[curr_voxel_idx] = np.append(curr_voxel, curr_residual)
            num_nan += 1
    print("Num of current voxels: ", str(esdf_curr.shape[0]), "; Num of history voxels: ", str(esdf_hist.shape[0]),
          "; Num of voxels with NaN residual: ", str(num_nan))

    residual_file = os.path.join(nvblox_output, dataset, "residual", "kitti_seq_01_esdf_residual_3_2.bin")
    esdf_residual.tofile(residual_file)

    # residual_loaded = np.fromfile(residual_file, dtype=np.float32).reshape((-1, 4))

    # multi-processing: much slower
    # warnings.filterwarnings("ignore")
    # pool = multiprocessing.Pool(processes=64)
    # for curr_voxel_idx, curr_voxel in enumerate(esdf_curr_voxels):
    #     pool.apply_async(func=compute_residual, args=(curr_voxel_idx, curr_voxel, esdf_hist_voxels, esdf_residual,))
    # pool.close()
    # pool.join()

    # histogram of esdf distances
    if vis_esdf_histogram:
        esdf_histogram(esdf_curr)

    # Vis ESDF in Open3D
    if vis_esdf_curr:
        open3d_vis_esdf(esdf_curr)
    elif vis_esdf_hist:
        open3d_vis_esdf(esdf_hist)
    elif vis_esdf_res:
        open3d_vis_esdf(esdf_residual)

