import os.path

import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud

import matplotlib.pyplot as plt

import open3d as o3d

def pcd_split(pcd_ele_angle, bin_edges):
    pcd_beams_list = []
    for i in range(len(bin_edges)-2):
        row_idx = np.where((pcd_ele_angle[:, -1] >= bin_edges[i]) & (pcd_ele_angle[:, -1] < bin_edges[i+1]))
        pcd_beams_list.append(pcd_ele_angle[row_idx][:, 0:3])
    return pcd_beams_list

def beam_random_mask(pcd_beams_list, num):
    # num of beams to be kept
    indices = np.arange(0, 31, step=1)
    np.random.seed(13)
    np.random.shuffle(indices)
    kept_ind = indices[0:num].tolist()
    pcd_beams_kept_list = [pcd_beams_list[idx] for idx in kept_ind]
    pcd_beams_kept = np.concatenate([pcd_beams_list[idx] for idx in kept_ind])
    return pcd_beams_kept_list

def hex_to_rgb(hex):
    r = int(hex[1:3], 16) / 255
    g = int(hex[3:5], 16) / 255
    b = int(hex[5:7], 16) / 255
    rgb = [r, g, b]
    return rgb

if __name__ == '__main__':
    # 浅粉, 红, 紫罗兰, 深紫, 蓝, 道奇蓝, 钢蓝, 深青, 春绿
    # 森林绿, 金, 橙, 巧克力, 橙红, 珊瑚色, 暗灰, 深红, 黑
    rgb_hex_list = ['#DC143C', '#C71585', '#4B0082', '#0000CD', '#1E90FF', '#4682B4', '#008B8B', '#00FF7F',
                    '#228B22', '#FFD700', '#FFA500', '#D2691E', '#FF4500', '#F08080', '#696969', '#8B0000', '#000000', '#FFB6C1']
    rgb_list = []
    for rgb_hex in rgb_hex_list:
        rgb = hex_to_rgb(rgb_hex)
        rgb_list.append(rgb)

    # start_color = (1.0, 0, 0.0)
    # end_color = (0.0, 0.0, 1.0)
    # steps = 16
    # for i in range(steps + 1):
    #     color = [start + i * (end - start) / steps for start, end in zip(start_color, end_color)]
    #     rgb_list.append(color)

    nusc = NuScenes(version='v1.0-mini', dataroot='/home/mars/MOS_Projects/nuScenes_MOS_Labeling/mini_data', verbose=True)
    nusc.list_scenes()

    scene = nusc.scene[1]
    first_sample_tok = scene['first_sample_token']
    sample = nusc.get('sample', first_sample_tok)
    sample_data_tok = sample['data']['LIDAR_TOP']
    sample_data = nusc.get('sample_data', sample_data_tok)

    # point cloud
    pcl_path = os.path.join(nusc.dataroot, sample_data['filename'])
    points_l = LidarPointCloud.from_file(pcl_path).points.T  # [num_points, 4]
    points_l = points_l[:, :3]  # without intensity
    x = points_l[:, 0]
    y = points_l[:, 1]
    z = points_l[:, 2]
    r = np.sqrt(np.square(x) + np.square(y) + np.square(z))
    elevation = 90 - np.degrees(np.arccos(z / r))
    azimuth = np.degrees(np.arctan2(x, y))

    ele_list = []
    azi_list = []
    for point in points_l:
        x = point[0]
        y = point[1]
        z = point[2]
        r = np.sqrt(np.square(x) + np.square(y) + np.square(z))
        if r < 1: continue
        else:
            ele = 90 - np.degrees(np.arccos(z / r))
            ele_list.append(ele)
            azi = np.degrees(np.arctan2(x, y))
            azi_list.append(azi)

    ele = np.array(ele_list)
    azi = np.array(azi_list)
    plt.scatter(azi, ele, s=0.5)
    plt.show()


    # elevation.sort()
    # elevation = elevation.reshape(32, -1)
    # hist, bin_edges = np.histogram(elevation, bins=32, range=(-31, 11), density=False)

    # nusc renderer
    # nusc.render_sample_data(lidar_top_data['token'])

    points_angle = np.concatenate((points_l, azimuth.reshape(-1, 1), elevation.reshape(-1, 1)), axis=1)
    pcd_ele_angle = np.concatenate((bin_pcd, elevation_angle), axis=1)

    num_beam_kept = 24
    pcd_beams_list = pcd_split(pcd_ele_angle, bin_edges)
    pcd_beams_kept_list = beam_random_mask(pcd_beams_list, num_beam_kept)

    o3d_pcd_beam_list = []

    o3d_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(bin_pcd))
    o3d_pcd.paint_uniform_color([0.7, 0.7, 0.7])  # yellow
    o3d_pcd_beam_list.append(o3d_pcd)
    for i in range(len(pcd_beams_kept_list)):
        o3d_pcd_beam = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd_beams_kept_list[i]))
        o3d_pcd_beam.paint_uniform_color(rgb_list[5])
        o3d_pcd_beam_list.append(o3d_pcd_beam)

    # visualization
    o3d.visualization.draw_geometries(o3d_pcd_beam_list, zoom=0.0940000000000000283,  # 0 -> max
                                                    front=[0.29678372412974874, 0.9079246722868356, 0.2959598123808696],
                                                    lookat=[1.1758518052639837, -1.4746038186455057, -2.0579947713357569],
                                                    up=[-0.12965352436494462, -0.26874320434342858, 0.9544459407106172])


    # sample annotation
    test_annotation_token = sample['anns'][18]
    test_annotation_metadata = nusc.get('sample_annotation', test_annotation_token)