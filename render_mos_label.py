import os
import math
import argparse
import numpy as np
from tqdm import tqdm
from enum import IntEnum
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from PIL import Image
from typing import Tuple, List, Iterable
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import points_in_box
from nuscenes.utils.data_io import load_bin_file
from nuscenes.utils.data_classes import LidarSegPointCloud
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud, Box
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility, transform_matrix
from nuscenes.lidarseg.lidarseg_utils import paint_points_label, colormap_to_colors, create_lidarseg_legend, get_labels_in_coloring


def render_ego_centric_map(nusc,
                           sample_data_token: str,
                           axes_limit: float = 40,
                           ax: Axes = None) -> None:
    """
    Render map centered around the associated ego pose.
    :param sample_data_token: Sample_data token.
    :param axes_limit: Axes limit measured in meters.
    :param ax: Axes onto which to render.
    """

    def crop_image(image: np.array,
                   x_px: int,
                   y_px: int,
                   axes_limit_px: int) -> np.array:
        x_min = int(x_px - axes_limit_px)
        x_max = int(x_px + axes_limit_px)
        y_min = int(y_px - axes_limit_px)
        y_max = int(y_px + axes_limit_px)

        cropped_image = image[y_min:y_max, x_min:x_max]

        return cropped_image

    # Get data.
    sd_record = nusc.get('sample_data', sample_data_token)
    sample = nusc.get('sample', sd_record['sample_token'])
    scene = nusc.get('scene', sample['scene_token'])
    log = nusc.get('log', scene['log_token'])
    map_ = nusc.get('map', log['map_token'])
    map_mask = map_['mask']
    pose = nusc.get('ego_pose', sd_record['ego_pose_token'])

    # Retrieve and crop mask.
    pixel_coords = map_mask.to_pixel_coords(pose['translation'][0], pose['translation'][1])
    scaled_limit_px = int(axes_limit * (1.0 / map_mask.resolution))
    mask_raster = map_mask.mask()
    cropped = crop_image(mask_raster, pixel_coords[0], pixel_coords[1], int(scaled_limit_px * math.sqrt(2)))

    # Rotate image.
    ypr_rad = Quaternion(pose['rotation']).yaw_pitch_roll
    yaw_deg = -math.degrees(ypr_rad[0])
    rotated_cropped = np.array(Image.fromarray(cropped).rotate(yaw_deg))

    # Crop image.
    ego_centric_map = crop_image(rotated_cropped,
                                 int(rotated_cropped.shape[1] / 2),
                                 int(rotated_cropped.shape[0] / 2),
                                 scaled_limit_px)

    # Init axes and show image.
    # Set background to white and foreground (semantic prior) to gray.
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(9, 9))
    ego_centric_map[ego_centric_map == map_mask.foreground] = 125
    ego_centric_map[ego_centric_map == map_mask.background] = 255
    ax.imshow(ego_centric_map, extent=[-axes_limit, axes_limit, -axes_limit, axes_limit],
              cmap='gray', vmin=0, vmax=255)

def render_sample_data(nusc,
                       sample_data_token: str,
                       with_anns: bool = True,
                       box_vis_level: BoxVisibility = BoxVisibility.ANY,
                       axes_limit: float = 40,
                       ax: Axes = None,
                       nsweeps: int = 1,
                       out_path: str = None,
                       underlay_map: bool = True,
                       use_flat_vehicle_coordinates: bool = True,
                       show_mos: bool = False,
                       show_lidarseg: bool = False,
                       show_lidarseg_legend: bool = False,
                       filter_lidarseg_labels: List = None,
                       lidarseg_preds_bin_path: str = None,
                       verbose: bool = True,
                       show_panoptic: bool = False) -> None:
    """
    Render sample data onto axis.
    :param sample_data_token: Sample_data token.
    :param with_anns: Whether to draw box annotations.
    :param box_vis_level: If sample_data is an image, this sets required visibility for boxes.
    :param axes_limit: Axes limit for lidar and radar (measured in meters).
    :param ax: Axes onto which to render.
    :param nsweeps: Number of sweeps for lidar and radar.
    :param out_path: Optional path to save the rendered figure to disk.
    :param underlay_map: When set to true, lidar data is plotted onto the map. This can be slow.
    :param use_flat_vehicle_coordinates: Instead of the current sensor's coordinate frame, use ego frame which is
        aligned to z-plane in the world. Note: Previously this method did not use flat vehicle coordinates, which
        can lead to small errors when the vertical axis of the global frame and lidar are not aligned. The new
        setting is more correct and rotates the plot by ~90 degrees.
    :param show_lidarseg: When set to True, the lidar data is colored with the segmentation labels. When set
        to False, the colors of the lidar data represent the distance from the center of the ego vehicle.
    :param show_lidarseg_legend: Whether to display the legend for the lidarseg labels in the frame.
    :param filter_lidarseg_labels: Only show lidar points which belong to the given list of classes. If None
        or the list is empty, all classes will be displayed.
    :param lidarseg_preds_bin_path: A path to the .bin file which contains the user's lidar segmentation
                                    predictions for the sample.
    :param verbose: Whether to display the image after it is rendered.
    :param show_panoptic: When set to True, the lidar data is colored with the panoptic labels. When set
        to False, the colors of the lidar data represent the distance from the center of the ego vehicle.
        If show_lidarseg is True, show_panoptic will be set to False.
    """
    if show_mos:
        show_lidarseg = False
        show_panoptic = False
    if show_lidarseg:
        show_mos = False
        show_panoptic = False
    # define mos colormap, name2idx and idx2map mapping
    mos_name2idx_mapping = {}
    mos_name2idx_mapping['unknown'] = 0
    mos_name2idx_mapping['static'] = 1
    mos_name2idx_mapping['moving'] = 2
    mos_idx2map_mapping = {}
    mos_idx2map_mapping[0] = 'unknown'
    mos_idx2map_mapping[1] = 'static'
    mos_idx2map_mapping[2] = 'moving'
    mos_colormap = {}
    mos_colormap['unknown'] = (65, 105, 225)
    mos_colormap['static'] = (222, 184, 135)
    mos_colormap['moving'] = (255, 61, 99)
    # Get sensor modality.
    sd_record = nusc.get('sample_data', sample_data_token)
    sensor_modality = sd_record['sensor_modality']

    if sensor_modality in ['lidar', 'radar']:
        sample_rec = nusc.get('sample', sd_record['sample_token'])
        chan = sd_record['channel']
        ref_chan = 'LIDAR_TOP'
        ref_sd_token = sample_rec['data'][ref_chan]
        ref_sd_record = nusc.get('sample_data', ref_sd_token)

        if sensor_modality == 'lidar':
            if show_mos or show_lidarseg or show_panoptic:  # 1101
                if show_mos:
                    gt_form = 'mos'
                elif show_lidarseg:
                    gt_from = 'lidarseg'
                else:
                    gt_form = 'panoptic'

                # assert hasattr(nusc, gt_from), f'Error: nuScenes-{gt_from} not installed!' # deleted 1101

                # Ensure that lidar pointcloud is from a keyframe.
                assert sd_record['is_key_frame'], \
                    'Error: Only pointclouds which are keyframes have lidar segmentation labels. Rendering aborted.'

                assert nsweeps == 1, \
                    'Error: Only pointclouds which are keyframes have lidar segmentation labels; nsweeps should ' \
                    'be set to 1.'

                # Load a single lidar point cloud.
                pcl_path = os.path.join(nusc.dataroot, ref_sd_record['filename'])
                pc = LidarPointCloud.from_file(pcl_path)
            else:
                # Get aggregated lidar point cloud in lidar frame.
                pc, times = LidarPointCloud.from_file_multisweep(nusc, sample_rec, chan, ref_chan,
                                                                 nsweeps=nsweeps)
            velocities = None
        else:
            # Get aggregated radar point cloud in reference frame.
            # The point cloud is transformed to the reference frame for visualization purposes.
            pc, times = RadarPointCloud.from_file_multisweep(nusc, sample_rec, chan, ref_chan, nsweeps=nsweeps)

            # Transform radar velocities (x is front, y is left), as these are not transformed when loading the
            # point cloud.
            radar_cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
            ref_cs_record = nusc.get('calibrated_sensor', ref_sd_record['calibrated_sensor_token'])
            velocities = pc.points[8:10, :]  # Compensated velocity
            velocities = np.vstack((velocities, np.zeros(pc.points.shape[1])))
            velocities = np.dot(Quaternion(radar_cs_record['rotation']).rotation_matrix, velocities)
            velocities = np.dot(Quaternion(ref_cs_record['rotation']).rotation_matrix.T, velocities)
            velocities[2, :] = np.zeros(pc.points.shape[1])

        # By default we render the sample_data top down in the sensor frame.
        # This is slightly inaccurate when rendering the map as the sensor frame may not be perfectly upright.
        # Using use_flat_vehicle_coordinates we can render the map in the ego frame instead.
        if use_flat_vehicle_coordinates:
            # Retrieve transformation matrices for reference point cloud.
            cs_record = nusc.get('calibrated_sensor', ref_sd_record['calibrated_sensor_token'])
            pose_record = nusc.get('ego_pose', ref_sd_record['ego_pose_token'])
            ref_to_ego = transform_matrix(translation=cs_record['translation'],
                                          rotation=Quaternion(cs_record["rotation"]))

            # Compute rotation between 3D vehicle pose and "flat" vehicle pose (parallel to global z plane).
            ego_yaw = Quaternion(pose_record['rotation']).yaw_pitch_roll[0]
            rotation_vehicle_flat_from_vehicle = np.dot(
                Quaternion(scalar=np.cos(ego_yaw / 2), vector=[0, 0, np.sin(ego_yaw / 2)]).rotation_matrix,
                Quaternion(pose_record['rotation']).inverse.rotation_matrix)
            vehicle_flat_from_vehicle = np.eye(4)
            vehicle_flat_from_vehicle[:3, :3] = rotation_vehicle_flat_from_vehicle
            viewpoint = np.dot(vehicle_flat_from_vehicle, ref_to_ego)
        else:
            viewpoint = np.eye(4)

        # Init axes.
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(9, 9))

        # Render map if requested.
        if underlay_map:
            assert use_flat_vehicle_coordinates, 'Error: underlay_map requires use_flat_vehicle_coordinates, as ' \
                                                 'otherwise the location does not correspond to the map!'
            render_ego_centric_map(nusc, sample_data_token=sample_data_token, axes_limit=axes_limit, ax=ax)

        # Show point cloud.
        points = view_points(pc.points[:3, :], viewpoint, normalize=False)
        dists = np.sqrt(np.sum(pc.points[:2, :] ** 2, axis=0))
        colors = np.minimum(1, dists / axes_limit / np.sqrt(2))
        if sensor_modality == 'lidar' and (show_mos or show_lidarseg or show_panoptic):
            # Load labels for pointcloud.
            if lidarseg_preds_bin_path:
                sample_token = nusc.get('sample_data', sample_data_token)['sample_token']
                lidarseg_labels_filename = lidarseg_preds_bin_path
                assert os.path.exists(lidarseg_labels_filename), \
                    'Error: Unable to find {} to load the predictions for sample token {} (lidar ' \
                    'sample data token {}) from.'.format(lidarseg_labels_filename, sample_token, sample_data_token)
            else:
                if show_mos:
                    lidarseg_labels_filename = os.path.join(nusc.dataroot, 'mos_labels', nusc.version, sample_data_token + "_mos.label")
                else:
                    # Ensure {lidarseg/panoptic}.json is not empty (e.g. in case of v1.0-test).
                    lidarseg_labels_filename = os.path.join(nusc.dataroot, nusc.get(gt_from, sample_data_token)['filename'])

            if lidarseg_labels_filename:
                # Paint each label in the pointcloud with a RGBA value.
                if show_mos or show_lidarseg or show_panoptic:
                    if show_mos:
                        colors = paint_points_label(lidarseg_labels_filename, filter_lidarseg_labels,
                                                    mos_name2idx_mapping, mos_colormap)
                    elif show_lidarseg:
                        colors = paint_points_label(lidarseg_labels_filename, filter_lidarseg_labels,
                                                    nusc.lidarseg_name2idx_mapping, nusc.colormap)
                    else:
                        colors = paint_panop_points_label(lidarseg_labels_filename, filter_lidarseg_labels,
                                                          nusc.lidarseg_name2idx_mapping, nusc.colormap)

                    if show_lidarseg_legend:

                        # If user does not specify a filter, then set the filter to contain the classes present in
                        # the pointcloud after it has been projected onto the image; this will allow displaying the
                        # legend only for classes which are present in the image (instead of all the classes).
                        if filter_lidarseg_labels is None:
                            if show_mos:
                                color_legend = colormap_to_colors(mos_colormap,
                                                                  mos_name2idx_mapping)
                                filter_lidarseg_labels = get_labels_in_coloring(color_legend, colors)
                            elif show_lidarseg:
                                # Since the labels are stored as class indices, we get the RGB colors from the
                                # colormap in an array where the position of the RGB color corresponds to the index
                                # of the class it represents.
                                color_legend = colormap_to_colors(nusc.colormap,
                                                                  nusc.lidarseg_name2idx_mapping)
                                filter_lidarseg_labels = get_labels_in_coloring(color_legend, colors)
                            else:
                                # Only show legends for stuff categories for panoptic.
                                filter_lidarseg_labels = stuff_cat_ids(len(nusc.lidarseg_name2idx_mapping))

                        if filter_lidarseg_labels and show_panoptic:
                            # Only show legends for filtered stuff categories for panoptic.
                            stuff_labels = set(stuff_cat_ids(len(nusc.lidarseg_name2idx_mapping)))
                            filter_lidarseg_labels = list(stuff_labels.intersection(set(filter_lidarseg_labels)))

                        if show_mos:
                            create_lidarseg_legend(filter_lidarseg_labels, mos_idx2map_mapping, mos_colormap)

                        else:
                            create_lidarseg_legend(filter_lidarseg_labels,
                                               nusc.lidarseg_idx2name_mapping,
                                               nusc.colormap,
                                               loc='upper left',
                                               ncol=1,
                                               bbox_to_anchor=(1.05, 1.0))
            else:
                print('Warning: There are no lidarseg labels in {}. Points will be colored according to distance '
                      'from the ego vehicle instead.'.format(nusc.version))

        point_scale = 0.2 if sensor_modality == 'lidar' else 3.0
        scatter = ax.scatter(points[0, :], points[1, :], c=colors, s=point_scale)

        # Show velocities.
        if sensor_modality == 'radar':
            points_vel = view_points(pc.points[:3, :] + velocities, viewpoint, normalize=False)
            deltas_vel = points_vel - points
            deltas_vel = 6 * deltas_vel  # Arbitrary scaling
            max_delta = 20
            deltas_vel = np.clip(deltas_vel, -max_delta, max_delta)  # Arbitrary clipping
            colors_rgba = scatter.to_rgba(colors)
            for i in range(points.shape[1]):
                ax.arrow(points[0, i], points[1, i], deltas_vel[0, i], deltas_vel[1, i], color=colors_rgba[i])

        # Show ego vehicle.
        ax.plot(0, 0, 'x', color='red')

        # Get boxes in lidar frame.
        _, boxes, _ = nusc.get_sample_data(ref_sd_token, box_vis_level=box_vis_level,
                                                use_flat_vehicle_coordinates=use_flat_vehicle_coordinates)

        # Show boxes.
        if with_anns:
            for box in boxes:
                # c = np.array(nusc.colormap[box.name]) / 255.0
                c = np.array([0, 0, 0])
                box.render(ax, view=np.eye(4), colors=(c, c, c))

        # Limit visible range.
        ax.set_xlim(-axes_limit, axes_limit)
        ax.set_ylim(-axes_limit, axes_limit)
    elif sensor_modality == 'camera':
        # Load boxes and image.
        data_path, boxes, camera_intrinsic = nusc.get_sample_data(sample_data_token,
                                                                       box_vis_level=box_vis_level)
        data = Image.open(data_path)

        # Init axes.
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(9, 16))

        # Show image.
        ax.imshow(data)


        # Show boxes.
        if with_anns:
            for box in boxes:
                c = np.array(nusc.colormap[box.name]) / 255.0
                box.render(ax, view=camera_intrinsic, normalize=True, colors=(c, c, c))

        # Limit visible range.
        ax.set_xlim(0, data.size[0])
        ax.set_ylim(data.size[1], 0)

    else:
        raise ValueError("Error: Unknown sensor modality!")

    ax.axis('off')
    ax.set_title('{} {labels_type}'.format(
        sd_record['channel'], labels_type='(predictions)' if lidarseg_preds_bin_path else ''))
    ax.set_aspect('equal')

    if out_path is not None:
        plt.savefig(out_path, bbox_inches='tight', pad_inches=0, dpi=200)

    if verbose:
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate nuScenes lidar panaptic gt.')
    parser.add_argument('--root_dir', type=str, default='/home/mars/catkin_ws/src/nuscenes2bag/data',
                        help='Default nuScenes data directory.')
    parser.add_argument('--version', type=str, default='v1.0-trainval')
    parser.add_argument('--verbose', type=bool, default=False, help='Whether to print to stdout.')
    args = parser.parse_args()

    print(f'Start rendering... \nArguments: {args}')
    nusc = NuScenes(version=args.version, dataroot=args.root_dir, verbose=args.verbose)
    if not hasattr(nusc, "lidarseg") or len(getattr(nusc, 'lidarseg')) == 0:
        raise RuntimeError(f"No nuscenes-lidarseg annotations found in {nusc.version}")

    for sample in tqdm(nusc.sample):
        lidar_tok = sample['data']['LIDAR_TOP']
        # render_sample_data(nusc, lidar_tok, with_anns=True, show_mos=False, show_lidarseg=True, show_panoptic=False, underlay_map=True)
        render_sample_data(nusc, lidar_tok, with_anns=True, verbose=True, show_lidarseg=True, show_mos=False, underlay_map=False, show_lidarseg_legend=True, out_path='./test.png')
        plt.clf()
        plt.cla()
        plt.close("all")

    print(f'Finished rendering.')







