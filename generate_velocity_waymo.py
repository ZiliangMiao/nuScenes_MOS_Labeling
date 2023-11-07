import os
import math
import argparse
import numpy as np
from tqdm import tqdm

import math
import os
import uuid
import time

from matplotlib import cm
import matplotlib.animation as animation
import matplotlib.pyplot as plt

import numpy as np
import itertools
import os, glob
import tensorflow as tf
from waymo_open_dataset.protos import scenario_pb2
from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import frame_utils


from google.protobuf.json_format import MessageToDict

from google.protobuf import text_format
from waymo_open_dataset.metrics.ops import py_metrics_ops
from waymo_open_dataset.metrics.python import config_util_py as config_util
from waymo_open_dataset.protos import motion_metrics_pb2

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
    inconsistent_bbox_label = np.ones(lidarseg_labels.shape[0], dtype=np.uint8)  # 1: consistent, 0: inconsistent
    inside_bbox_label = np.zeros(lidarseg_labels.shape[0], dtype=np.uint8)  # 1: inside bbox, 0: outside bbox

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
            inside_bbox_label[idx] = 1  # pts inside bbox
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
                inconsistent_bbox_label[idx] = 0  # inconsistent bbox label
                num_inconsistent_pts += 1
    vel_file = os.path.join(vels_dir, lidar_tok + "_vel.bin")
    vel_labels.tofile(vel_file)

    # other checks
    inconsistent_bbox_dir = os.path.join(root_dir, "inconsistent_bbox", nusc.version)
    inside_bbox_dir = os.path.join(root_dir, "inside_bbox", nusc.version)
    os.makedirs(inconsistent_bbox_dir, exist_ok=True)
    os.makedirs(inside_bbox_dir, exist_ok=True)
    inconsistent_bbox_label_file = os.path.join(inconsistent_bbox_dir, lidar_tok + "_inconsistent.bin")
    inside_bbox_label_file = os.path.join(inside_bbox_dir, lidar_tok + "_inside.bin")
    inconsistent_bbox_label.tofile(inconsistent_bbox_label_file)
    inside_bbox_label.tofile(inside_bbox_label_file)

    # vels_loaded = np.fromfile(vel_file, dtype=np.float64).reshape(-1, 3)
    return num_inconsistent_boxes, num_boxes

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate nuScenes lidar panaptic gt.')
    parser.add_argument('--root_dir', type=str, default='/home/mars/MOS_Projects/nuScenes_MOS_Labeling/mini_data',
                        help='Default nuScenes data directory.')
    parser.add_argument('--verbose', type=bool, default=True, help='Whether to print to stdout.')
    args = parser.parse_args()

    raw_data_path = '/media/mars/Disk_1/Datasets/Waymo Motion/scenario/training/'

    raw_data = glob.glob(os.path.join(raw_data_path, '*.tfrecord*'))
    raw_data.sort()

    for data_file in raw_data:
        dataset = tf.data.TFRecordDataset(data_file, compression_type='')

        for idx, data in enumerate(dataset):
            scenario = scenario_pb2.Scenario()
            scenario.ParseFromString(bytearray(data.numpy()))

            # frame = open_dataset.Frame()
            # frame.ParseFromString(bytearray(data.numpy()))  # frame has the timestamp_micros attribute
            # pose = np.array(frame.pose.transform)[0:12].reshape(1, 12)  # pose of frame
            # # load point clouds
            # range_images, camera_projections, _, range_image_top_pose = (
            #     frame_utils.parse_range_image_and_camera_projection(frame))
            # frame.lasers.sort(key=lambda laser: laser.name)
            # points, _ = frame_utils.convert_range_image_to_point_cloud(
            #     frame, range_images, camera_projections,
            #     range_image_top_pose)  # points contains point clouds of both 5 lidars
            # # Points in vehicle frame
            # # scan_points = np.concatenate(points, axis=0)  # register all points of five lidars
            # top_lidar_points = points[0]

            scenario_idx = scenario.scenario_id
            timestamps_seconds = list(scenario.timestamps_seconds)
            current_time_idx = scenario.current_time_index  # index of current timestamp in timestamps_seconds
            ego_track_idx = scenario.sdc_track_index  # ego vehicle id in tracks

            tracks = list(scenario.tracks)  # corresponding to the timestamps_seconds
            num_objs = len(tracks)
            for obj_track in tracks:
                # single object
                obj_id = obj_track.id
                obj_cat = obj_track.object_type  # 0: init, 1: vehicle, 2: pedestrian, 3: cyclist, 4: other
                states = obj_track.states
                num_states = len(states)
                # states of current object in all scans
                for time_idx, state in enumerate(states):
                    time = timestamps_seconds[time_idx]
                    if state.valid:
                        bbox_x = state.center_x
                        bbox_y = state.center_y
                        bbox_z = state.center_z
                        bbox_l = state.length
                        bbox_w = state.width
                        bbox_h = state.height
                        bbox_heading = state.heading
                        box_velox = state.velocity_x
                        box_veloy = state.velocity_y
                        a = 1
                    else:
                        z = state.center_z
                        a = 1








