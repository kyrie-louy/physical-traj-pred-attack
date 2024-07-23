import os
import json
import pickle
import torch
import torch.nn as nn
import numpy as np
np.set_printoptions(suppress=True, precision=3,  threshold=2000,  linewidth=150)
import argparse
from torch.multiprocessing import Pool, set_start_method

from loss import CustomLoss
#from model_test import PIXOR
from model import PIXOR
from tqdm import tqdm
from utils import get_model_name, load_config
# from postprocess import filter_pred, compute_matches, compute_ap, compute_iou, convert_format
from utils_nuscs import *
from utils_attack import *

from pyquaternion import Quaternion
from nuscenes.utils.data_classes import Box
from nuscenes.nuscenes import NuScenes
from nuscenes.prediction import PredictHelper


# # debug settings
# import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9501))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass



def build_model(config, device, train=True):
    net = PIXOR(config['geometry'], config['use_bn'])
    loss_fn = CustomLoss(device, config, num_classes=1)

    if torch.cuda.device_count() <= 1:
        config['mGPUs'] = False
    if config['mGPUs']:
        print("using multi gpu")
        net = nn.DataParallel(net)

    net = net.to(device)
    loss_fn = loss_fn.to(device)
    if not train:
        return net, loss_fn

    optimizer = torch.optim.SGD(net.parameters(), lr=config['learning_rate'], momentum=config['momentum'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config['lr_decay_at'], gamma=0.1)

    return net, loss_fn, optimizer, scheduler


def custom_loss(dxyh, dxyh_goal):
    
    dx_pred = dxyh_goal[0, 0]
    dy_pred = dxyh_goal[0, 1]
    if np.abs(dx_pred) > 2 * np.abs(dy_pred):
        weights_x, weights_y = 2.0, 1.0
    elif np.abs(dy_pred) > 2 * np.abs(dx_pred):
        weights_x, weights_y = 1.0, 2.0
    else:
        weights_x, weights_y = 1.0, 1.0

    dx_diff = np.abs(dxyh[0] - dx_pred)
    dy_diff = np.abs(dxyh[1] - dy_pred)
    loss = weights_x * dx_diff + weights_y * dy_diff
    
    return loss


def refine(data_dir, config, device):
    """
    refine detection candidate set using finite difference method
    - input: attack_results/stage1_res/addpts, preds
    - output: attack_results/stage2_res/addpts
    """

    scene_names = os.listdir(data_dir)

    addpts_refined_dict = {}
    loss_list = []

    for s_idx in range(len(scene_names)):
        
        # init
        scene_name = scene_names[s_idx]
        if scene_name != args.scene_name:
            continue
        dataset_dir = os.path.join(data_dir, scene_name, 'dataset')
        curr_work_dir = os.path.join(dataset_dir, 'inverse')

        ### load ###
        # load hyper-parameters
        config_path = os.path.join(dataset_dir, 'config.yaml')
        cfg = cfg_from_yaml_file(config_path)
        cfg = cfg.DET

        match_res_path = os.path.join(curr_work_dir, 'matching_res_det.json')
        with open(match_res_path, 'r') as f:
            match_res = json.load(f)

        # read original added pts
        addpts_center_pool = match_res[scene_name]['addpts_center']  # (n, N_add, 3)
        addpts_center_pool = np.array(addpts_center_pool)  # (n, N_add, 3)

        # read detection goals
        heading_goal = match_res[scene_name]['heading_pred']
        heading_goal = np.array(heading_goal)[:, -1, :]  # (5, 1)
        pos_goal = match_res[scene_name]['pos_pred']
        pos_goal = np.array(pos_goal)[:, -1, :]
        xyh_goal_pool = np.concatenate([pos_goal, heading_goal], axis=1)  # (5, 3)

        dh_goal = match_res[scene_name]['dh_pred']
        dh_goal = np.array(dh_goal)[:, -1, :]  # (1, 1)
        dp_goal = match_res[scene_name]['dp_pred']
        dp_goal = np.array(dp_goal)[:, -1, :] # (1, 2)
        dxyh_goal_pool = np.concatenate([dp_goal, dh_goal], axis=1)  # (1, 3)

        # read detection gt
        heading_gt = np.array(match_res[scene_name]['heading_gt'])  # (5, 1)
        pos_gt = np.array(match_res[scene_name]['pos_gt'])  # (5, 2)

        # get ground truth labels (lidar frame)
        with open(os.path.join(dataset_dir, 'labels_lidar.json'), 'r') as f:
            labels_lidar = json.load(f)

        # read ego vehicle poses
        enu_coords_path = os.path.join(data_dir, scene_name, 'dataset', 'enu_coords.json')
        with open(enu_coords_path, 'r') as f:
            enu_coords = json.load(f)

        ### read ground truth ###
        # get attack frame ids
        frame_ids = list(labels_lidar.keys())
        frame_ids.sort()
        frame_ids = frame_ids[:5]
        curr_frame_id = frame_ids[-1]

        lidar_path = os.path.join(dataset_dir, 'pointclouds', curr_frame_id+'.bin')
        x, y, z, l, w, h, yaw, _, _ = labels_lidar[curr_frame_id]

        ego_x, ego_y, ego_z, ego_heading = enu_coords[curr_frame_id]
        
        bev_corners = np.zeros((4, 2), dtype=np.float32)
        # rear left
        bev_corners[0, 0] = x - l/2 * np.cos(yaw) - w/2 * np.sin(yaw)
        bev_corners[0, 1] = y - l/2 * np.sin(yaw) + w/2 * np.cos(yaw)
        # rear right
        bev_corners[1, 0] = x - l/2 * np.cos(yaw) + w/2 * np.sin(yaw)
        bev_corners[1, 1] = y - l/2 * np.sin(yaw) - w/2 * np.cos(yaw)
        # front right
        bev_corners[2, 0] = x + l/2 * np.cos(yaw) + w/2 * np.sin(yaw)
        bev_corners[2, 1] = y + l/2 * np.sin(yaw) - w/2 * np.cos(yaw)
        # front left
        bev_corners[3, 0] = x + l/2 * np.cos(yaw) - w/2 * np.sin(yaw)
        bev_corners[3, 1] = y + l/2 * np.sin(yaw) + w/2 * np.cos(yaw)
        reg_target = [np.cos(yaw), np.sin(yaw), x, y, w, l]
        geom = config['geometry']['input_shape']
        gt_reg = np.array([[1.0,x,y,w,l,yaw]])
        #print(gt_reg)
        label_list = bev_corners[np.newaxis,:]
        gt3Dboxes = np.array([[w, h, l, y, z, x, yaw]])

        # refine
        addpts_center_refined_pool = np.zeros_like(addpts_center_pool)  # (n, N_add, 3)

        for pool_idx in range(addpts_center_pool.shape[0]):
            
            # init
            addpts_center = addpts_center_pool[pool_idx]  # (N_add, 3)
            init_loss = match_res[scene_name]['dist'][pool_idx]
            dxyh_goal = dxyh_goal_pool[pool_idx:pool_idx+1]  # (3, )

            # apply shift to added points
            addpts_center_shift_pool = apply_center_shift(cfg.N_add, cfg.REFINE.N_shift, cfg.REFINE.d_shift, addpts_center)  # (N_shift, N_add, 3)
            addpts_center_shift_pool = \
                np.concatenate([addpts_center[np.newaxis, ...], addpts_center_shift_pool], axis=0)  # concat original added points
            
            best_loss = init_loss
            best_addpts_center_shift = addpts_center

            if init_loss > 0.3:
                
                # for large init loss -> iteratively refine
                for iter_idx in range(cfg.REFINE.N_shift):
                    
                    # apply shift to added points
                    addpts_center_shift = apply_center_shift(cfg.N_add, 1, cfg.REFINE.d_shift, addpts_center)  # (N_shift, N_add, 3)
                    
                    # inference
                    scores = attack_obj_realworld(addpts_center_shift,net,0,config,geom,lidar_path,label_list,gt3Dboxes,device,cfg)
                    scores = np.array(scores)  # (6, ): s,x,y,w,l,yaw
                    if scores[0] == 0:
                        continue
                    
                    ### convert to global frame ###                
                    # Extract position and heading for target and reference vehicles
                    _, target_x, target_y, target_wid, target_len, target_heading = scores

                    target_x_global = ego_x + (target_x * np.cos(ego_heading) - target_y * np.sin(ego_heading))
                    target_y_global = ego_y + (target_x * np.sin(ego_heading) + target_y * np.cos(ego_heading))

                    # Adjust the heading to the global coordinate system for both target and ego vehicles
                    target_heading_global = ego_heading + target_heading
                    target_heading_global = (target_heading_global + np.pi) % (2 * np.pi) - np.pi

                    # Store target global pose
                    dxyh = np.array([target_x_global-pos_gt[-1, 0], target_y_global-pos_gt[-1, 1], target_heading_global-heading_gt[-1, 0]])
                    ### convert to global frame ###
                    
                    # compute loss
                    loss = custom_loss(dxyh, dxyh_goal)
                    # print(pool_idx, shift_idx, round(loss, 3))

                    # record best added points
                    if loss < best_loss:
                        best_loss = loss
                        best_addpts_center_shift = addpts_center_shift.copy()  # (1, 48)
                    
            else:
                
                for shift_idx in tqdm(range(addpts_center_shift_pool.shape[0])):

                    addpts_center_shift = addpts_center_shift_pool[shift_idx]  # (N_add, 3)

                    # inference
                    scores = attack_obj_realworld(addpts_center_shift,net,0,config,geom,lidar_path,label_list,gt3Dboxes,device,cfg)
                    scores = np.array(scores)  # (6, ): s,x,y,w,l,yaw
                    if scores[0] == 0:
                        continue
                    
                    ### convert to global frame ###                
                    # Extract position and heading for target and reference vehicles
                    _, target_x, target_y, target_wid, target_len, target_heading = scores

                    target_x_global = ego_x + (target_x * np.cos(ego_heading) - target_y * np.sin(ego_heading))
                    target_y_global = ego_y + (target_x * np.sin(ego_heading) + target_y * np.cos(ego_heading))

                    # Adjust the heading to the global coordinate system for both target and ego vehicles
                    target_heading_global = ego_heading + target_heading
                    target_heading_global = (target_heading_global + np.pi) % (2 * np.pi) - np.pi

                    # Store target global pose
                    dxyh = np.array([target_x_global-pos_gt[-1, 0], target_y_global-pos_gt[-1, 1], target_heading_global-heading_gt[-1, 0]])
                    ### convert to global frame ###

                    # compute loss
                    loss = custom_loss(dxyh, dxyh_goal)
                    # print(pool_idx, shift_idx, round(loss, 3))

                    # record best added points
                    if loss < best_loss:
                        best_loss = loss
                        best_addpts_center_shift = addpts_center_shift.copy()  # (1, 48)

            # record best loss
            loss_list.append(best_loss)

            # record best added points for this candidate
            addpts_center_refined_pool[pool_idx] = best_addpts_center_shift 
        
        addpts_center_refined_pool = [addpts_center_refined.tolist() for addpts_center_refined in addpts_center_refined_pool]
        addpts_refined_dict[scene_name] = addpts_center_refined_pool  # match_res format
    
    # save results
    with open(os.path.join(curr_work_dir, 'addpts_center_refined.json'), 'w') as f:
        json.dump(addpts_refined_dict, f)

    # calc average loss
    print('average distance: ', loss_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PIXOR custom implementation')
    parser.add_argument('--data_dir', default='/home/kyrie/Desktop/WorkSpace/lidar/PIXOR_nuscs/data/realworld')
    parser.add_argument('--scene_name', default='scene_3_5kmh')
    args = parser.parse_args()
    
    ### init ###
    # init device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.set_device(0)  # set the default GPU device to use
    else:
        device = torch.device("cpu")
    
    # init config
    config, _, _, _ = load_config('focal_relu')

    # init model
    net, loss_fn = build_model(config, device, train=False)
    net.load_state_dict(torch.load(get_model_name(config), map_location=device))
    net.set_decode(True)
    net.eval()
    ### init ###
    
    refine(args.data_dir, config, device)

