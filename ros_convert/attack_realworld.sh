root_dir='/home/kyrie/Desktop/WorkSpace/personal/attack_prediction'

### choose scene ###
dataroot=${root_dir}/ros_convert/data

# # right-clean
# benign_dir=${dataroot}'/scenario_right/clean'
# benign_scenes=(
#     'rosbag2_2023_12_05-11_05_28'
# )

# left-clean
benign_dir=${dataroot}/scenario_left/clean
benign_scenes=(
    'rosbag2_2023_12_05-11_44_31'
)
### choose scene ###


### init ###
image_topic='/sensing/camera/traffic_light/image_raw' 
# image_topic='/sensing/camera/traffic_light1/image_raw'

source ~/anaconda3/etc/profile.d/conda.sh
### init ###


### attack ###
for benign_scene in "${benign_scenes[@]}"; do
    
    # ### data preprocessing ###
    # # print ros2 bag info
    # source /opt/ros/galactic/setup.bash
    # ros2 bag info ${benign_dir}/${benign_scene}

    # # ros2 to ros1 bag
    # source /opt/ros/noetic/setup.bash
    # rosbags-convert ${benign_dir}/${benign_scene} --dst ${benign_dir}/${benign_scene}/${benign_scene}.bag \
    #     --include-topic $image_topic \
    #     --include-topic /sensing/lidar/top/outlier_filtered/pointcloud \
    #     --include-topic /Sensor_msgs/INS/NavSatFix_BYINS_gnss

    # source /opt/ros/noetic/setup.bash
    # python convert.py --attack_dir ${benign_dir} --attack_scene ${benign_scene} --image_topic $image_topic

    # # create dataset folder 
    # python dataset.py \
    #     --mode 'process' \
    #     --benign_dir ${benign_dir} \
    #     --attack_dir ${benign_dir} \
    #     --benign_scene ${benign_scene} \
    #     --attack_scene ${benign_scene}

    # # # create gt by OpenPCDet
    # conda activate pcdet
    # cd ${root_dir}/OpenPCDet/tools

    # cfg_path='cfgs/nuscenes_models/cbgs_second_multihead.yaml'
    # ckpt_path='../checkpoints/cbgs_second_multihead_nds6229_updated.pth'
    # demo_data_path=${benign_dir}'/'${benign_scene}
    # python -W ignore infer.py \
    #     --cfg_file $cfg_path --ckpt $ckpt_path --data_path $demo_data_path

    # # create global enu coordinates
    # conda activate py38
    # cd ${root_dir}/ros_convert
    # python dataset.py --mode 'global' --attack_dir ${benign_dir} --attack_scene ${benign_scene}
    # ### data preprocessing ###


    ### det attack ###
    conda activate pixor_nuscs

    cd ${root_dir}/PIXOR_nuscs/srcs
    python -W ignore attack_realworld.py \
        --data_dir ${benign_dir} --scene_name ${benign_scene}
    ### det infer ###


    ### pred attack ###
    conda activate trajectron++
    cd ${root_dir}/Trajectron-plus-plus/experiments/nuScenes

    # preprocess
    python -W ignore process_data_realworld.py \
        --data_dir ${benign_dir} --scene_name ${benign_scene} \
        --output_dir ${root_dir}/Trajectron-plus-plus/experiments/processed_realworld
    
    # inverse attack
    python -W ignore inverse_realworld.py \
        --data_dir ${benign_dir} --scene_name ${benign_scene}
    
    # matching
    python matching_realworld.py --data_dir ${benign_dir} --scene_name ${benign_scene}
    ### pred attack ###


    ### det refine ###
    conda activate pixor_nuscs
    cd ${root_dir}/PIXOR_nuscs/srcs

    # refine
    python -W ignore refine_realworld.py --data_dir ${benign_dir} --scene_name ${benign_scene}

    # eval
    python -W ignore eval_realworld.py \
        --data_dir ${benign_dir} --scene_name ${benign_scene} --stage 'eval'
    ### det refine ###


    ### pred infer ###
    conda activate trajectron++
    cd ${root_dir}/Trajectron-plus-plus/experiments/nuScenes

    # preprocess
    python -W ignore process_data_realworld.py \
        --data_dir ${benign_dir} --scene_name ${benign_scene} \
        --output_dir ${root_dir}/Trajectron-plus-plus/experiments/processed_realworld
    
    # infer
    python -W ignore eval_realworld.py \
        --data_dir ${benign_dir} --scene_name ${benign_scene} --attack_type 'inverse'
    ### pred infer ###

done
### attack ###