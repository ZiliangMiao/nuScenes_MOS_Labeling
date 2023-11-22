from cProfile import label
import os
from posixpath import dirname
import shutil
import click
import numpy as np


@click.command()
@click.option(
    "--root_dir",
    "-d",
    type=str,
    help="directory",
    default="/home/mars/catkin_ws/src/nuscenes2bag/mini_data/bags",
)
@click.option(
    "--out_dir",
    "-o",
    type=str,
    help="output directory",
    default="/home/mars/catkin_ws/src/nuscenes2bag/mini_data/sequences",
)
# seq-01
## frame-000000.color.png; frame-000000.depth.png; frame-000000.height.png; frame-000000.lidar-intrinsics.txt; frame-000000.pose.txt
# camera_intrinsics.txt



def main(root_dir, out_dir):
    scenes = []
    for root, dirs, files in os.walk(root_dir, topdown=True):
        for filename in dirs:
            if(filename.isdigit()):
                scenes.append(str(filename).zfill(4))  # filled with zeros to sort correctly
    scenes.sort()
    scenes = [scene.lstrip('0') for scene in scenes]  # delete the filled zeros to get the original name
    for out_scene_idx, org_scene_idx in enumerate(scenes):
        # rename .bag files
        bag_file = os.path.join(root_dir, str(org_scene_idx) + '.bag')
        new_bag_file = os.path.join(root_dir, str(out_scene_idx).zfill(4) + '.bag')
        os.rename(bag_file, new_bag_file)

        scene_dir = os.path.join(root_dir, str(org_scene_idx))
        files_list = os.listdir(scene_dir)
        files_list.sort()

        out_scene_dir = os.path.join(out_dir, str(out_scene_idx).zfill(4))
        lidar_dir = os.path.join(out_scene_dir, "lidar")
        labels_dir = os.path.join(out_scene_dir, "labels_raw")
        vels_dir = os.path.join(out_scene_dir, "vels")
        if not os.path.exists(out_scene_dir):
            os.makedirs(out_scene_dir)
        if not os.path.exists(lidar_dir):
            os.makedirs(lidar_dir)
        if not os.path.exists(labels_dir):
            os.makedirs(labels_dir)
        if not os.path.exists(vels_dir):
            os.makedirs(vels_dir)

        num_scans = int(len(files_list) / 3)
        for scan_idx in range(num_scans):
            # move org files to new sequences directories
            # .bin files
            lidar_file = os.path.join(lidar_dir, str(scan_idx).zfill(6) + '.bin')
            org_lidar_file = os.path.join(scene_dir, files_list[3 * scan_idx])
            shutil.move(org_lidar_file, lidar_file)
            # .label files
            label_file = os.path.join(labels_dir, str(scan_idx).zfill(6) + '.label')
            org_label_file = os.path.join(scene_dir, files_list[3 * scan_idx + 1])
            shutil.move(org_label_file, label_file)
            # .vel files
            vel_file = os.path.join(vels_dir, str(scan_idx).zfill(6) + '.vel')
            org_vel_file = os.path.join(scene_dir, files_list[3 * scan_idx + 2])
            shutil.move(org_vel_file, vel_file)

            # check files
            lidar = np.fromfile(lidar_file, dtype=np.int32).reshape((-1, 4)) & 0xFFFF
            label = np.fromfile(label_file, dtype=np.int32).reshape((-1)) & 0xFFFF
            vel = np.fromfile(vel_file, dtype=np.int32).reshape((-1)) & 0xFFFF
            a = 1

if __name__ == '__main__':
    main()

