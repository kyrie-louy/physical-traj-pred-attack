import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import time
import ctypes
import json
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

from model import PIXOR
from loss import CustomLoss

from utils_attack import *
from utils_nuscs import *
from utils import get_bev
from postprocess import compute_iou, convert_format,non_max_suppression
from utils import get_model_name, load_config


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


def attack(data_dir, net, config, device):
    
    # init
    scene_names = os.listdir(data_dir)

    attack_dets = {}
    attack_addpts_center = {}
    
    for s_idx in range(len(scene_names)):
        
        # init
        scene_name = scene_names[s_idx]
        if scene_name != args.scene_name:
            continue
        dataset_dir = os.path.join(data_dir, scene_name, 'dataset')

        # load hyper-parameters in config file
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

        # get random added pts
        addpts_center_pool = get_adv_cls_center_realworld(cfg.ATTACK.N_iter, cfg.N_add, cfg.origin)  # (*, N_add, 3)

        # loop over frames to attack
        for f_idx in range(len(frame_ids)):
            
            frame_id = frame_ids[f_idx]

            lidar_path = lidar_paths[f_idx]
            x, y, z, l, w, h, yaw, _, _ = labels_lidar[frame_id]

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
            ### get ground truth ###
            
            ### attack ###
            scores = []  # (sample_num, 6)
            for addpts_center in tqdm(addpts_center_pool):
                scores.append(attack_obj_realworld(addpts_center,net,0,config,geom,lidar_path,label_list,gt3Dboxes,device,cfg))

            pred = []
            for score in scores:
                temp_s, temp_x, temp_y, temp_w, temp_l, temp_yaw = score
                pred.append([float(temp_x), float(temp_y), z, float(temp_l), float(temp_w), h, float(temp_yaw)])
            ### attack ###

            ### record detection results ###
            attack_dets[frame_id] = pred
            ### record detection results ###

        # filter nonempty detections
        attack_det_seq = np.zeros((int(cfg.ATTACK.N_iter), 5, 1))  # store scores
        for frame_idx, frame_id in enumerate(frame_ids):
            for box_idx, box_pred in enumerate(attack_dets[frame_id]):
                attack_det_seq[box_idx, frame_idx, 0] = box_pred[0]

        nonempty_idxes = np.where((attack_det_seq[:, :, 0] != 0).all(axis=1))[0]
        # print('nonempty_idxes: ', nonempty_idxes.shape)

        # update attack det
        for frame_idx, frame_id in enumerate(frame_ids):
            attack_dets[frame_id] = [attack_dets[frame_id][i] for i in nonempty_idxes]

        # record added points
        attack_addpts_center[frame_ids[-1]] = addpts_center_pool[nonempty_idxes]


    ### save results ###
    # save addpts
    attack_addpts_center_path = os.path.join(dataset_dir, 'attack_addpts_center.pkl')
    print('Writing addpts to: %s' % attack_addpts_center_path)
    with open(attack_addpts_center_path, 'wb') as f:
        pickle.dump(attack_addpts_center, f)

    # save attack_det.json
    # Dummy meta data, please adjust accordingly.
    meta = {
        'use_camera': False,
        'use_lidar': True,
        'use_radar': False,
        'use_map': False,
        'use_external': False,
    }
    submission = {
        'meta': meta,
        'results': attack_dets
    }
    attack_det_path = os.path.join(dataset_dir, 'attack_det_lidar.json')
    print('Writing submission to: %s' % attack_det_path)
    with open(attack_det_path, 'w') as f:
        json.dump(submission, f, indent=2)
    # save results ###


def attack_to_global(data_dir, scene_name):

    dataset_dir = os.path.join(data_dir, scene_name, 'dataset')

    # read attack results
    attack_det_path = os.path.join(dataset_dir, 'attack_det_lidar.json')
    with open(attack_det_path, 'r') as f:
        attack_dets_lidar = json.load(f)["results"]

    # read ego vehicle poses
    enu_coords_path = os.path.join(dataset_dir, 'enu_coords.json')
    with open(enu_coords_path, 'r') as f:
        enu_coords = json.load(f)

    # load labels_global
    labels_global_path = os.path.join(dataset_dir, 'labels_global.json')
    with open(labels_global_path, 'r') as f:
        labels_global = json.load(f)

    # create adv det distribution
    adv_det_dir = os.path.join(dataset_dir, 'attack_det_distrib')
    if not os.path.exists(adv_det_dir):
        os.makedirs(adv_det_dir)

    # Extract the current frame to set the global origin and orientation
    frame_ids = list(enu_coords.keys())
    frame_ids.sort()
    curr_frame_id = frame_ids[4]
    # print('current frame: ', curr_frame_id)

    # lidar to global
    attack_det_global = {}
    for frame_id, data in attack_dets_lidar.items():

        attack_det_global[frame_id] = []
        ego_x, ego_y, ego_z, ego_heading = enu_coords[frame_id]

        for box in data:

            ### transform target vehicle ###
            # Extract position and heading for target and reference vehicles
            target_x, target_y, target_z, target_len, target_wid, target_ht, target_heading = box
            
            target_x_global = ego_x + (target_x * np.cos(ego_heading) - target_y * np.sin(ego_heading))
            target_y_global = ego_y + (target_x * np.sin(ego_heading) + target_y * np.cos(ego_heading))

            # Adjust the heading to the global coordinate system for both target and ego vehicles
            target_heading_global = ego_heading + target_heading
            target_heading_global = (target_heading_global + np.pi) % (2 * np.pi) - np.pi

            # Store target global pose
            attack_det_global[frame_id].append([target_x_global, target_y_global, target_heading_global])
            ### target vehicle ###

        ### plot attack det distribution ###
        # get gt label
        gt_label = labels_global[frame_id]['target']
        x_gt = gt_label[0]
        y_gt = gt_label[1]
        heading_gt = gt_label[-1]

        # get attack det
        attack_det_np = np.array(attack_det_global[frame_id])
        
        # visualize x, y, heading using histogram
        x = attack_det_np[:, 0] - x_gt
        y = attack_det_np[:, 1] - y_gt
        heading = attack_det_np[:, 2] - heading_gt

        plt.figure(figsize=(20, 5))
        plt.subplot(1, 3, 1)
        plt.hist(x, bins=100)
        plt.title('delta x')
        plt.subplot(1, 3, 2)
        plt.hist(y, bins=100)
        plt.title('delta y')
        plt.subplot(1, 3, 3)
        plt.hist(heading, bins=100)
        plt.title('delta heading')
        plt.savefig(os.path.join(adv_det_dir, f'{frame_id}.png'))
        ### plot attack det distribution ###

    # save attack_det_global.json
    # Dummy meta data, please adjust accordingly.
    meta = {
        'use_camera': False,
        'use_lidar': True,
        'use_radar': False,
        'use_map': False,
        'use_external': False,
    }
    submission = {
        'meta': meta,
        'results': attack_det_global
    }
    attack_det_path = os.path.join(dataset_dir, 'attack_det_global.json')
    print('Writing submission to: %s' % attack_det_path)
    with open(attack_det_path, 'w') as f:
        json.dump(submission, f, indent=2)


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

    # attack
    attack(args.data_dir, net, config, device)

    # attack to global
    attack_to_global(args.data_dir, args.scene_name)

    # # inference
    # pc_dir = os.path.join(data_dir, 'scene_3_5kmh', 'dataset', 'pointclouds')
    # pc_filenames = os.listdir(pc_dir)
    # pc_filenames = [filename for filename in pc_filenames if filename.endswith('.bin')]
    # pc_filenames.sort()

    # for filename in pc_filenames:
    #     print(filename)
        
    #     infer(net, config, os.path.join(pc_dir, filename), device)