import numpy as np
import matplotlib.pyplot as plt
import math

def elevation_trans(alpha, d, delta_h=0.1):
    return np.degrees(np.arctan((d * np.sin(alpha) - delta_h) / (d * np.cos(alpha))))

if __name__ == '__main__':
    # LiDAR 1 - ONCE
    pandar40p_vfov = np.array([-25, -19, -14, -13, -12, -11, -10, -9, -8, -7])
    pandar40p_vfov_mid = np.linspace(-6, 2, 25, endpoint=True)
    pandar40p_vfov = np.concatenate((pandar40p_vfov, pandar40p_vfov_mid, np.array([3, 5, 8, 11, 15])), axis=0)

    # LiDAR 2 - KITTI
    hdl64e_vfov = np.linspace(-24.8, 2, 64, endpoint=True)

    # LiDAR 3 - Argoverse
    vlp32c_vfov = np.array([-25, -15.639, -11.310, -8.843, -7.254, -6.148, -5.333, -4.667])
    vlp32c_vfov_mid = np.linspace(-4.000, 1.667, 18, endpoint=True)
    vlp32c_vfov = np.concatenate((vlp32c_vfov, vlp32c_vfov_mid, np.array([2.333, 3.333, 4.667, 7.000, 10.333, 15.000])), axis=0)

    # LiDAR 4 - Waymo (actually not a uniform distribution)
    waymo_vfov = np.linspace(-17.6, 2.4, 64)

    # LiDAR 5 - nuScenes (not known if it follows a uniform distribution)
    nuscenes_vfov = np.linspace(-30, 10, 32)

    vfov_dist = np.concatenate((pandar40p_vfov, hdl64e_vfov, vlp32c_vfov, waymo_vfov, nuscenes_vfov), axis=0)

    # height transform
    h = 1.7  # height of lidar_uni
    delta_h = 0.1  # height diff between lidar_uni and lidar_1
    alpha_0 = np.radians(25)  # elevation angle of beam of lidar_1
    alpha_1 = np.radians(15)
    alpha_2 = np.radians(5)
    d = np.linspace(1, 10, 100)  # depth in lidar_1
    beta_0 = elevation_trans(alpha_0, d)
    beta_1 = elevation_trans(alpha_1, d)
    beta_2 = elevation_trans(alpha_2, d)

    plt.figure(1)
    plt.plot(d, beta_0 - np.degrees(alpha_0))
    plt.plot(d, beta_1 - np.degrees(alpha_1))
    plt.plot(d, beta_2 - np.degrees(alpha_2))
    plt.legend(labels=['25 degrees', '15 degrees', '5 degrees'])
    plt.xlabel('depth (m)')
    plt.ylabel('eltevaion angle diff (degrees)')
    plt.show()

    # Visualize
    plt.figure(2)
    s1 = plt.scatter(pandar40p_vfov, np.zeros(len(pandar40p_vfov)), c='r', s=1)
    s2 = plt.scatter(hdl64e_vfov, np.zeros(len(hdl64e_vfov))+0.5, c='g', s=1)
    s3 = plt.scatter(vlp32c_vfov, np.zeros(len(vlp32c_vfov))+1, c='b', s=1)
    s4 = plt.scatter(waymo_vfov, np.zeros(len(waymo_vfov))+1.5, c='y', s=1)
    s5 = plt.scatter(nuscenes_vfov, np.zeros(len(nuscenes_vfov))+2, c='k', s=1)
    plt.legend(handles=[s1, s2, s3, s4, s5], labels=['pandar40p', 'hdl64e', 'vlp32c', 'waymo', 'nuscenes'])
    plt.show()


    a = 1
