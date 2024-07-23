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


def custom_loss(xyh, xyh_goal, xyh_gt, alpha=1.0, beta=0.2):
    # Ensure xyh and xyh_goal are numpy arrays
    xyh = np.array(xyh).squeeze()[:2]
    xyh_goal = np.array(xyh_goal).squeeze()[:2]
    xyh_gt = np.array(xyh_gt).squeeze()[:2]
    
    # Value Loss
    value_loss = np.abs(xyh - xyh_goal)

    # perturbation sign loss
    perturb_sign_loss = np.sign(xyh - xyh_gt)
    perturb_sign_loss_goal = np.sign(xyh_goal - xyh_gt)
    perturb_sign_loss = np.maximum(0, -perturb_sign_loss * perturb_sign_loss_goal)
    
    # Total Loss
    # total_loss = alpha * np.sum(sign_loss) + beta * np.sum(value_loss)
    total_loss = alpha * np.sum(value_loss) + beta * np.sum(perturb_sign_loss)
    # print(total_loss, alpha * np.sum(value_loss), beta * np.sum(perturb_sign_loss))
    # raise ValueError()
    
    return total_loss


def get_gt_boxes3d_lidar(label_path):
    
    with open(label_path, 'rb') as f:
        gt_box3d_lidar = pickle.load(f)

    x, y, z = gt_box3d_lidar.center
    w, l, h = gt_box3d_lidar.wlh
    yaw = gt_box3d_lidar.orientation.yaw_pitch_roll[0]

    return x, y, z, l, w, h, yaw


def infer(data_dir, config, device):

    # init
    scene_names = os.listdir(data_dir)

    eval_res = {}

    for s_idx in range(len(scene_names)):
        
        # init
        scene_name = scene_names[s_idx]
        if scene_name != args.scene_name:
            continue
        
        # list all attack datasets
        scene_dir = os.path.join(data_dir, scene_name)
        dataset_dirs = os.listdir(scene_dir)
        dataset_dirs = [dataset for dataset in dataset_dirs if dataset.startswith('dataset')]
        dataset_dirs.sort()

        for dataset_dir in dataset_dirs:
            
            dataset_dir = os.path.join(data_dir, scene_name, dataset_dir)

            # load hyper-parameters
            config_path = os.path.join(dataset_dir, 'config.yaml')
            cfg = cfg_from_yaml_file(config_path)
            cfg = cfg.DET

            # get ground truth labels (lidar frame)
            with open(os.path.join(dataset_dir, 'labels_lidar.json'), 'r') as f:
                labels_lidar = json.load(f)

            # get attack frame ids
            frame_ids = list(labels_lidar.keys())
            frame_ids.sort()
            frame_ids = frame_ids[:5]

            # get lidar paths
            lidar_paths = [os.path.join(dataset_dir, 'pointclouds', f_id+'.bin') for f_id in frame_ids]

            # read ego vehicle poses
            enu_coords_path = os.path.join(dataset_dir, 'enu_coords.json')
            with open(enu_coords_path, 'r') as f:
                enu_coords = json.load(f)

            pos_pool = []
            heading_pool = []
            for pool_idx in range(1):
                
                pos = []
                heading = []
                for f_idx in range(len(frame_ids)):

                    # if f_idx == 3:
                    #     raise ValueError()
                    
                    frame_id = frame_ids[f_idx]

                    lidar_path = lidar_paths[f_idx]
                    x, y, z, l, w, h, yaw, _, _ = labels_lidar[frame_id]
                    ego_x, ego_y, ego_z, ego_heading = enu_coords[frame_id]
                    
                    ### get ground truth ###
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
                    # reg_target = [np.cos(yaw), np.sin(yaw), x, y, w, l]
                    geom = config['geometry']['input_shape']
                    # gt_reg = np.array([[1.0,x,y,w,l,yaw]])
                    #print(gt_reg)
                    label_list = bev_corners[np.newaxis,:]
                    gt3Dboxes = np.array([[w, h, l, y, z, x, yaw]])
                            
                    ### attack ###
                    addpts_center_refined = np.empty((0, 3))
                    score = attack_obj_realworld(addpts_center_refined,net,0,config,geom,lidar_path,label_list,gt3Dboxes,device,cfg)
                    ### attack ###

                    ### convert to global frame ###                
                    # Extract position and heading for target and reference vehicles
                    _, target_x, target_y, target_wid, target_len, target_heading = score
                    print(score)

                    target_x_global = ego_x + (target_x * np.cos(ego_heading) - target_y * np.sin(ego_heading))
                    target_y_global = ego_y + (target_x * np.sin(ego_heading) + target_y * np.cos(ego_heading))

                    # Adjust the heading to the global coordinate system for both target and ego vehicles
                    target_heading_global = ego_heading + target_heading
                    target_heading_global = (target_heading_global + np.pi) % (2 * np.pi) - np.pi

                    # Store target global pose
                    pos.append([target_x_global, target_y_global])
                    # print(pos[-1])
                    heading.append([target_heading_global])
                    ### convert to global frame ###

                pos_pool.append(pos)
                heading_pool.append(heading)

            eval_res[scene_name] = {
                'pos': pos_pool,
                'heading': heading_pool,
            }

            with open(os.path.join(dataset_dir, 'infer_det.json'), 'w') as f:
                json.dump(eval_res, f)


def eval(data_dir, config, device):

    # init
    scene_names = os.listdir(data_dir)

    eval_res = {}

    for s_idx in range(len(scene_names)):
        
        # init
        scene_name = scene_names[s_idx]
        if scene_name != args.scene_name:
            continue
        dataset_dir = os.path.join(data_dir, scene_name, 'dataset')
        curr_work_dir = os.path.join(dataset_dir, 'inverse')

        # load hyper-parameters
        config_path = os.path.join(dataset_dir, 'config.yaml')
        cfg = cfg_from_yaml_file(config_path)
        cfg = cfg.DET

        # get ground truth labels (lidar frame)
        with open(os.path.join(dataset_dir, 'labels_lidar.json'), 'r') as f:
            labels_lidar = json.load(f)

        # get attack frame ids
        frame_ids = list(labels_lidar.keys())
        frame_ids.sort()
        frame_ids = frame_ids[:5]

        # get lidar paths
        lidar_paths = [os.path.join(dataset_dir, 'pointclouds', f_id+'.bin') for f_id in frame_ids]

        # read ego vehicle poses
        enu_coords_path = os.path.join(dataset_dir, 'enu_coords.json')
        with open(enu_coords_path, 'r') as f:
            enu_coords = json.load(f)

        # load refined added points
        with open(os.path.join(curr_work_dir, 'addpts_center_refined.json'), 'r') as f:
            addpts_center_refined_pool = json.load(f)[scene_name]  # (cluster_num, N_add, 3)

        pos_pool = []
        heading_pool = []
        for pool_idx in range(len(addpts_center_refined_pool)):
            
            pos = []
            heading = []
            for f_idx in range(len(frame_ids)):
                
                frame_id = frame_ids[f_idx]

                lidar_path = lidar_paths[f_idx]
                x, y, z, l, w, h, yaw, _, _ = labels_lidar[frame_id]
                ego_x, ego_y, ego_z, ego_heading = enu_coords[frame_id]
                
                ### get ground truth ###
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
                # reg_target = [np.cos(yaw), np.sin(yaw), x, y, w, l]
                geom = config['geometry']['input_shape']
                # gt_reg = np.array([[1.0,x,y,w,l,yaw]])
                #print(gt_reg)
                label_list = bev_corners[np.newaxis,:]
                gt3Dboxes = np.array([[w, h, l, y, z, x, yaw]])
                        
                ### attack ###
                addpts_center_refined = addpts_center_refined_pool[pool_idx]  # (N_add, 3)
                addpts_center_refined = np.array(addpts_center_refined)
                score = attack_obj_realworld(addpts_center_refined,net,0,config,geom,lidar_path,label_list,gt3Dboxes,device,cfg)
                ### attack ###

                ### convert to global frame ###                
                # Extract position and heading for target and reference vehicles
                _, target_x, target_y, target_wid, target_len, target_heading = score

                target_x_global = ego_x + (target_x * np.cos(ego_heading) - target_y * np.sin(ego_heading))
                target_y_global = ego_y + (target_x * np.sin(ego_heading) + target_y * np.cos(ego_heading))

                # Adjust the heading to the global coordinate system for both target and ego vehicles
                target_heading_global = ego_heading + target_heading
                target_heading_global = (target_heading_global + np.pi) % (2 * np.pi) - np.pi

                # Store target global pose
                pos.append([target_x_global, target_y_global])
                heading.append([target_heading_global])
                ### convert to global frame ###

            pos_pool.append(pos)
            heading_pool.append(heading)

        eval_res[scene_name] = {
            'pos': pos_pool,
            'heading': heading_pool,
        }

    with open(os.path.join(dataset_dir, 'infer_det.json'), 'w') as f:
        json.dump(eval_res, f)
    print('Eval results saved to', os.path.join(dataset_dir, 'infer_det.json'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PIXOR custom implementation')
    parser.add_argument('--data_dir', default='/mnt/data/attack_pred_foxconn/bag1129')
    parser.add_argument('--scene_name', default='scene_3_5kmh')
    parser.add_argument('--stage', default='prepare')
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

    if args.stage == 'eval':
        eval(args.data_dir, config, device)
    elif args.stage == 'infer':
        infer(args.data_dir, config, device)

