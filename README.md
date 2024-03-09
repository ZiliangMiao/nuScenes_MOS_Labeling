# nuScenes_MOS_Labeling
## Label nuScenes Moving Objects
### Step0: Download nuScenes Dataset on AutoDL Server
download nuScenes dataset and **lidarseg** labels to local dataset directory: /root/autodl-tmp/data/nuscenes
```
cp /root/autodl-pub/nuScenes/Fulldatasetv1.0/Test/v1.0-test_meta.tar /root/autodl-tmp/data/nuscenes
cp /root/autodl-pub/nuScenes/Fulldatasetv1.0/Test/v1.0-test_blobs.tgz /root/autodl-tmp/data/nuscenes
------------------------------------------------------------------------------------------------------------------
cp /root/autodl-pub/nuScenes/Fulldatasetv1.0/Trainval/v1.0-trainval_meta.tgz /root/autodl-tmp/data/nuscenes
cp /root/autodl-pub/nuScenes/Fulldatasetv1.0/Trainval/v1.0-trainval01_blobs.tgz /root/autodl-tmp/data/nuscenes
cp /root/autodl-pub/nuScenes/Fulldatasetv1.0/Trainval/v1.0-trainval02_blobs.tgz /root/autodl-tmp/data/nuscenes
cp /root/autodl-pub/nuScenes/Fulldatasetv1.0/Trainval/v1.0-trainval03_blobs.tgz /root/autodl-tmp/data/nuscenes
cp /root/autodl-pub/nuScenes/Fulldatasetv1.0/Trainval/v1.0-trainval04_blobs.tgz /root/autodl-tmp/data/nuscenes
cp /root/autodl-pub/nuScenes/Fulldatasetv1.0/Trainval/v1.0-trainval05_blobs.tgz /root/autodl-tmp/data/nuscenes
cp /root/autodl-pub/nuScenes/Fulldatasetv1.0/Trainval/v1.0-trainval06_blobs.tgz /root/autodl-tmp/data/nuscenes
cp /root/autodl-pub/nuScenes/Fulldatasetv1.0/Trainval/v1.0-trainval07_blobs.tgz /root/autodl-tmp/data/nuscenes
cp /root/autodl-pub/nuScenes/Fulldatasetv1.0/Trainval/v1.0-trainval08_blobs.tgz /root/autodl-tmp/data/nuscenes
cp /root/autodl-pub/nuScenes/Fulldatasetv1.0/Trainval/v1.0-trainval09_blobs.tgz /root/autodl-tmp/data/nuscenes
cp /root/autodl-pub/nuScenes/Fulldatasetv1.0/Trainval/v1.0-trainval10_blobs.tgz /root/autodl-tmp/data/nuscenes
```

decompress the .tar files.
```
tar zxvf /root/autodl-tmp/data/nuscenes/v1.0-test/v1.0-test_blobs.tgz -C /root/autodl-tmp/data/nuscenes
tar zxvf /root/autodl-tmp/data/nuscenes/v1.0-test/v1.0-test_meta.tar -C /root/autodl-tmp/data/nuscenes
------------------------------------------------------------------------------------------------------------------
tar zxvf /root/autodl-tmp/data/nuscenes/v1.0-trainval_meta.tgz -C /root/autodl-tmp/data/nuscenes
tar zxvf /root/autodl-tmp/data/nuscenes/v1.0-trainval01_blobs.tgz -C /root/autodl-tmp/data/nuscenes
tar zxvf /root/autodl-tmp/data/nuscenes/v1.0-trainval02_blobs.tgz -C /root/autodl-tmp/data/nuscenes
tar zxvf /root/autodl-tmp/data/nuscenes/v1.0-trainval03_blobs.tgz -C /root/autodl-tmp/data/nuscenes
tar zxvf /root/autodl-tmp/data/nuscenes/v1.0-trainval04_blobs.tgz -C /root/autodl-tmp/data/nuscenes
tar zxvf /root/autodl-tmp/data/nuscenes/v1.0-trainval05_blobs.tgz -C /root/autodl-tmp/data/nuscenes
tar zxvf /root/autodl-tmp/data/nuscenes/v1.0-trainval06_blobs.tgz -C /root/autodl-tmp/data/nuscenes
tar zxvf /root/autodl-tmp/data/nuscenes/v1.0-trainval07_blobs.tgz -C /root/autodl-tmp/data/nuscenes
tar zxvf /root/autodl-tmp/data/nuscenes/v1.0-trainval08_blobs.tgz -C /root/autodl-tmp/data/nuscenes
tar zxvf /root/autodl-tmp/data/nuscenes/v1.0-trainval09_blobs.tgz -C /root/autodl-tmp/data/nuscenes
tar zxvf /root/autodl-tmp/data/nuscenes/v1.0-trainval10_blobs.tgz -C /root/autodl-tmp/data/nuscenes
```

### Step1: Generate Velocities
nuScenes root directory: /autodl-tmp/data/nuscenes

nuScenes version: v1.0-trainval

```
python generate_velocity.py --root_dir '/home/user/Datasets/nuScenes' --version 'v1.0-trainval'
```

this will generate: [nusc_root_dir]/vels/[nusc_version]/[sample_data_token]_vel.bin
### Step2: Generate MOS Labels
```
python generate_mos_label.py --root_dir '/home/user/Datasets/nuScenes' --version 'v1.0-trainval'
```
