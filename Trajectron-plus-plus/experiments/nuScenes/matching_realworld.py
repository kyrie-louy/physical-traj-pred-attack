import os
import torch
import dill
import pickle
import argparse
import numpy as np
from tqdm import tqdm

from utils_attack import *
from utils_eval import *
from utils_vis import *

torch.set_printoptions(precision=4, sci_mode=False)
np.set_printoptions(suppress=True, precision=3)
import warnings
warnings.filterwarnings('ignore')


# # debug settings
# import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9501))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass


def weighted_distance(dp_det, dp_pred):
    # p1 and p2 are tensors with shape (n, 2)
    # Extract delta x and delta y components
    # Calculate differences in delta x and delta y separately
    x_diff = torch.abs(dp_det[:, [0]] - dp_pred[:, [0]])
    y_diff = torch.abs(dp_det[:, [1]] - dp_pred[:, [1]])

    # Determine which component is larger for each pair and assign weights
    dx_pred, dy_pred = dp_pred[0, :]
    if dx_pred.abs() > 2 * dy_pred.abs():
        weights_x, weights_y = 2.0, 1.0
    elif dy_pred.abs() > 2 * dx_pred.abs():
        weights_x, weights_y = 1.0, 2.0
    else:
        weights_x, weights_y = 1.0, 1.0

    # Compute weighted L1 distance
    distances_p = weights_x * x_diff + weights_y * y_diff

    return distances_p
    

def cand_set_matching():

    matching_res_pred = {}
    matching_res_det = {}

    for scene in eval_scenes:
        
        scene_name = scene.name
        if scene_name != args.scene_name:
            continue
        x_min, y_min, _, _ = scene.patch  # scene <-> global

        ts_curr = 4
        timestep = np.array([ts_curr])
        
        target_inst_token = 'target'
        dataset_dir = os.path.join(args.data_dir, scene_name, 'dataset')
        curr_work_dir = os.path.join(dataset_dir, 'inverse')
        if not os.path.exists(curr_work_dir):
            os.makedirs(curr_work_dir)

        # load
        # load attacked detection candidate set
        with open(os.path.join(dataset_dir, 'attack_det_global.json'), 'rb') as f:
            attack_det_dict = json.load(f)['results']

        # load dh_set_det and dp_set_det
        with open(os.path.join(curr_work_dir, 'dh_set_det.pkl'), 'rb') as f:
            dh_set_det_dict = pickle.load(f)
        with open(os.path.join(curr_work_dir, 'dp_set_det.pkl'), 'rb') as f:
            dp_set_det_dict = pickle.load(f)

        # load non-empty addpts center
        with open(os.path.join(dataset_dir, 'attack_addpts_center.pkl'), 'rb') as f:
            addpts_center_dict = pickle.load(f)  # (*, 48)
        # # load non-empty addpts
        # with open(os.path.join(dataset_dir, 'attack_addpts.pkl'), 'rb') as f:
        #     addpts_det_dict = pickle.load(f)  # (*, 48)

        # load prediction candidate set
        with open(os.path.join(curr_work_dir, 'inverse_res.json'), 'r') as f:
            pred_set_dict = json.load(f)

        # load frame_ids
        frame_ids = list(attack_det_dict.keys())
        frame_ids.sort()
        curr_frame_id = frame_ids[ts_curr]

        # load detection perturbation set for current scenario
        dh_set_det = torch.tensor(dh_set_det_dict[scene_name]).cuda()
        dp_set_det = torch.tensor(dp_set_det_dict[scene_name]).cuda()
        
        # load detection addpts for current scenario
        addpts_centers = addpts_center_dict[curr_frame_id]  # (N_iter, N_add, 3)

        # load prediction candidate set for current scenario
        dh_set_pred = torch.tensor(pred_set_dict[scene_name]['dh']).cuda()
        dp_set_pred = torch.tensor(pred_set_dict[scene_name]['dp']).cuda()

        # load ground truth
        for i, n in enumerate(scene.nodes):
            if n.id == target_inst_token:
                ts_range = np.array([ts_curr-4, ts_curr])
                box_p_gt = n.get(tr_scene=ts_range, state=eval_stg_nm.state['VEHICLE'])[:, :2] + np.array([x_min, y_min]) # (5, 2)
                box_h_gt = n.get(tr_scene=ts_range, state=eval_stg_nm.state['VEHICLE'])[:, 6:7]  # (5, 1)

                box_p_gt = torch.tensor(box_p_gt).float().cuda()  # double -> float
                box_h_gt = torch.tensor(box_h_gt).float().cuda()  # double -> float

                break
        
        # Hungarian matching
        dh_match_pred_list = []
        dp_match_pred_list = []
        dh_match_det_list = []
        dp_match_det_list = []

        dist_list = []
        det_index_list = []

        for i in range(dh_set_pred.shape[0]):
            
            # # cost function 1
            # # distances_h = torch.cdist(dh_set_det[:, -1, :], dh_set_pred[[i], -1, :], p=1)
            # distances_p = torch.cdist(dp_set_det[:, -1, :], dp_set_pred[[i], -1, :], p=1)
            # distances = distances_p

            # # cost function 2
            distances = weighted_distance(dp_set_det[:, -1, :], dp_set_pred[[i], -1, :])

            # calc min cost
            sort_idx = torch.argsort(distances.squeeze())
            dist_list.append(distances[sort_idx[0]].item()) 
        
            closest_index = torch.argmin(distances)
            det_set_index = closest_index // distances.shape[1]
            pred_set_index = closest_index % distances.shape[1]

            # record
            det_index_list.append(det_set_index.item())

            dh_match_pred = dh_set_pred[i, :, :]
            dp_match_pred = dp_set_pred[i, :, :]
            
            dh_match_det = dh_set_det[det_set_index, :, :]
            dp_match_det = dp_set_det[det_set_index, :, :]
            
            dh_match_pred_list.append(dh_match_pred)
            dp_match_pred_list.append(dp_match_pred)
            dh_match_det_list.append(dh_match_det)
            dp_match_det_list.append(dp_match_det)
            
        # get indexes of top 3 smallest distances
        top_k = 3
        top_k_idx = np.argsort(dist_list)[:top_k]
        
        dist_list = [dist_list[idx] for idx in top_k_idx]
        dh_match_pred_list = [dh_match_pred_list[idx] for idx in top_k_idx]
        dp_match_pred_list = [dp_match_pred_list[idx] for idx in top_k_idx]
        dh_match_det_list = [dh_match_det_list[idx] for idx in top_k_idx]
        dp_match_det_list = [dp_match_det_list[idx] for idx in top_k_idx]
        det_index_list = [det_index_list[idx] for idx in top_k_idx]
        
        # save result for evaluation
        heading_pred = [dh+box_h_gt for dh in dh_match_pred_list]
        pos_pred = [dp+box_p_gt for dp in dp_match_pred_list]
        matching_res_pred[scene_name] = {
            'dist': dist_list,
            'dh': out_format(dh_match_pred_list),
            'dp': out_format(dp_match_pred_list),
            'heading': out_format(heading_pred),
            'pos': out_format(pos_pred),
        }

        heading_det = [dh+box_h_gt for dh in dh_match_det_list]
        pos_det = [dp+box_p_gt for dp in dp_match_det_list]
        addpts_centers_det = addpts_centers[det_index_list, :, :]  # (*, N_add, 3)

        matching_res_det[scene_name] = {
            'dist': dist_list,
            'dh': out_format(dh_match_det_list),
            'dp': out_format(dp_match_det_list),
            'heading': out_format(heading_det),
            'pos': out_format(pos_det),
            'dh_pred': out_format(dh_match_pred_list),
            'dp_pred': out_format(dp_match_pred_list),
            'heading_pred': out_format(heading_pred),
            'pos_pred': out_format(pos_pred),
            'heading_gt': out_format(box_h_gt),
            'pos_gt': out_format(box_p_gt),
            'addpts_center': out_format(addpts_centers_det),
        }
        ### match detection ###
        
        ### save matching_res ###
        with open(os.path.join(curr_work_dir, 'matching_res_det.json'), 'w') as f:
            json.dump(matching_res_det, f)
        with open(os.path.join(curr_work_dir, 'matching_res_pred.json'), 'w') as f:
            json.dump(matching_res_pred, f)
        

        # calculate average distance
        dist_list = []
        for scene_name in matching_res_det.keys():
            dists = matching_res_det[scene_name]['dist']
            for dist in dists:
                dist_list.append(dist)

        print('distances: ', dist_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='')
    parser.add_argument('--scene_name', type=str, default=None)
    args = parser.parse_args()

    ### init ###
    ph = 6  # some scenes have less than 12 future timesteps!!!
    log_dir = './models'
    model_name = 'int_ee'  #'int_ee_me'

    # load nuScenes data
    with open('../processed_realworld/attack.pkl', 'rb') as f:
        eval_env = dill.load(f, encoding='latin1')
    eval_scenes = eval_env.scenes

    # load model
    device = 'cuda:0'
    model_dir = os.path.join(log_dir, model_name)
    eval_stg_nm, hyp = load_model(model_dir, eval_env, ts=20, device=device)
    ### init ###
    
    cand_set_matching()