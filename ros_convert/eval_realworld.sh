root_dir='/home/kyrie/Desktop/WorkSpace/personal/attack_prediction'


### choose scene ###
dataroot=${root_dir}/ros_convert/data

# # right-clean
# attack_dir=${dataroot}'/scenario_right/clean'
# attack_scenes=('rosbag2_2023_12_05-11_05_28')

# right-moderate_velo
attack_dir=${dataroot}'/scenario_right/moderate_velo'
attack_scenes=(
    'rosbag2_2023_12_05-16_01_33'  # select
)

# # right-high_velo
# attack_dir=${dataroot}'/scenario_right/high_velo'
# benign_scene='rosbag2_2023_12_05-11_05_28'
# attack_scenes=(
#     'rosbag2_2023_12_15-17_15_19'  # select
# )

# # left-clean
# attack_dir=${dataroot}'/scenario_left/clean'
# attack_scenes=('rosbag2_2023_12_05-11_44_31')
### choose scene ###


### init ###
image_topic='/sensing/camera/traffic_light/image_raw' 
# image_topic='/sensing/camera/traffic_light1/image_raw'

source ~/anaconda3/etc/profile.d/conda.sh
### init ###


### eval ###
for attack_scene in "${attack_scenes[@]}"; do
    
    # ### data preprocessing ###
    # # print ros2 bag info
    # source /opt/ros/galactic/setup.bash
    # ros2 bag info ${attack_dir}/${attack_scene}

    # # ros2 to ros1 bag
    # source /opt/ros/noetic/setup.bash
    # rosbags-convert ${attack_dir}/${attack_scene} --dst ${attack_dir}/${attack_scene}/${attack_scene}.bag \
    #     --include-topic $image_topic \
    #     --include-topic /sensing/lidar/top/outlier_filtered/pointcloud \
    #     --include-topic /Sensor_msgs/INS/NavSatFix_BYINS_gnss

    # source /opt/ros/noetic/setup.bash
    # python convert.py --attack_dir ${attack_dir} --attack_scene ${attack_scene} --image_topic $image_topic

    # # create dataset folder 
    # python dataset.py \
    #     --mode 'process' \
    #     --benign_dir ${benign_dir} \
    #     --attack_dir ${attack_dir} \
    #     --benign_scene ${benign_scene} \
    #     --attack_scene ${attack_scene}

    # # # create gt by OpenPCDet
    # conda activate pcdet
    # cd /home/kyrie/Desktop/WorkSpace/lidar/OpenPCDet/tools

    # # cfg_path='cfgs/nuscenes_models/cbgs_pp_multihead.yaml'
    # # ckpt_path='../checkpoints/pp_multihead_nds5823_updated.pth'
    # cfg_path='cfgs/nuscenes_models/cbgs_second_multihead.yaml'
    # ckpt_path='../checkpoints/cbgs_second_multihead_nds6229_updated.pth'
    # demo_data_path=${attack_dir}'/'${attack_scene}
    # python -W ignore infer.py \
    #     --cfg_file $cfg_path --ckpt $ckpt_path --data_path $demo_data_path

    # # create global enu coordinates
    # conda activate py38
    # cd /home/kyrie/Desktop/WorkSpace/datasets/ros_convert
    # python dataset.py --mode 'global' --attack_dir ${attack_dir} --attack_scene ${attack_scene}
    # ### data preprocessing ###


    ### det infer ###
    conda activate pixor_nuscs

    cd ${root_dir}/PIXOR_nuscs/srcs
    python -W ignore eval_realworld.py \
        --data_dir ${attack_dir} --scene_name ${attack_scene} --stage 'infer'
    ### det infer ###


    # ### pred infer ###
    conda activate trajectron++
    cd ${root_dir}/Trajectron-plus-plus/experiments/nuScenes

    # preprocess
    python -W ignore process_data_realworld.py \
        --data_dir ${attack_dir} --scene_name ${attack_scene} \
        --output_dir ${root_dir}/Trajectron-plus-plus/experiments/processed_realworld
    
    # infer
    python -W ignore eval_realworld.py \
        --data_dir ${attack_dir} --scene_name ${attack_scene} --attack_type 'inverse'
    ## pred infer ###

done