import os
import torch
import dill
import time
import pickle
import argparse
import numpy as np
from tqdm import tqdm
import sys
sys.path.append('./devkit/python-sdk/')
from nuscenes.map_expansion.map_api import NuScenesMap

from utils_attack import *
from utils_eval import *
from utils_vis import *
from utils_collision import *
from utils_realworld import *

torch.set_printoptions(precision=4, sci_mode=False)
np.set_printoptions(suppress=True, precision=3)
import warnings
warnings.filterwarnings('ignore')


def eval():

    # list all datasets
    scene_dir = os.path.join(args.data_dir, args.scene_name)
    dataset_dirs = os.listdir(scene_dir)
    dataset_dirs = [dataset for dataset in dataset_dirs if dataset.startswith('dataset')]
    dataset_dirs.sort()

    for scene_idx, scene in enumerate(eval_scenes):

        scene_name = scene.name
        if scene_name != args.scene_name:
            continue

        x_min, y_min, _, _ = scene.patch  # scene <-> global

        ts_curr = 4
        timestep = np.array([ts_curr])
        
        target_inst_token = 'target'
        dataset_dir = os.path.join(args.data_dir, scene_name, dataset_dirs[scene_idx])
        curr_work_dir = os.path.join(dataset_dir, 'inverse')
        if not os.path.exists(curr_work_dir):
            os.makedirs(curr_work_dir)

        # get target and ego vehicle info (dim and rot)
        for i, n in enumerate(scene.nodes):
            if n.id == target_inst_token:
                target_length, target_width = n.length, n.width
                target_cur_trans = n.get(tr_scene=timestep, state=eval_stg_nm.state['VEHICLE'])[0, :2]
                # rot not equal to the original value, need to investigate
                target_cur_rot = n.get(tr_scene=timestep, state=eval_stg_nm.state['VEHICLE'])[0, 6]# 
            elif n.id == 'ego':
                ego_length, ego_width = n.length, n.width
                ego_cur_rot = n.get(tr_scene=timestep, state=eval_stg_nm.state['VEHICLE'])[0, 6]

        # load ground truth
        for i, n in enumerate(scene.nodes):
            if n.id == target_inst_token:
                ts_range = np.array([ts_curr-4, ts_curr])
                box_p_gt = n.get(tr_scene=ts_range, state=eval_stg_nm.state['VEHICLE'])[:, :2] + np.array([x_min, y_min]) # (5, 2)
                box_h_gt = n.get(tr_scene=ts_range, state=eval_stg_nm.state['VEHICLE'])[:, 6:7]  # (5, 1)

                box_p_gt = torch.tensor(box_p_gt).float().cuda()  # double -> float
                box_h_gt = torch.tensor(box_h_gt).float().cuda()  # double -> float

                break

        ### inference wo attack ###
        with torch.no_grad():
            _, _, future_traces_noatk, predict_trajs_noatk = eval_stg_nm.run(scene,
                                                timestep,
                                                ph,
                                                min_future_timesteps=ph,
                                                num_samples=1,
                                                z_mode=True,
                                                gmm_mode=True,
                                                full_dist=False,
                                                all_z_sep=False)
            egoplan_clean_scene = torch.tensor(plan(predict_trajs_noatk, scene, target_inst_token, ts_curr)).cuda()
            egoplan_gt_scene = torch.tensor(future_traces_noatk['ego'][:, :2]).cuda()
        ### inference wo attack ###


        ### hyperparameters setting based on det ###
        if args.attack_type == 'inverse':
            attack_res_path = os.path.join(dataset_dir,'infer_det.json')
        elif args.attack_type == 'forward':
            attack_res_path = os.path.join(dataset_dir,'forward_res.json')  
        else:
            raise ValueError('attack_type should be inverse or forward')
        
        with open(attack_res_path, 'rb') as f:
            attack_res = json.load(f)
        heading_set = attack_res[scene_name]['heading']
        pos_set = attack_res[scene_name]['pos']


        # select best candidate
        if isinstance(heading_set, list):
            
            pi_ade_list = []
            heading_cand_list = []
            pos_cand_list = []
            for i in range(len(heading_set)):

                det_cand = {
                    'obj_id': target_inst_token, 
                    'duration': range(-5, 0),
                    'delta_heading': torch.tensor(heading_set[i]).cuda(),
                    'delta_position': torch.tensor(pos_set[i]).cuda()
                }

                # EKF tracker
                det_cand_copy = {
                    'obj_id': target_inst_token,
                    'duration': [-5, -4, -3, -2, -1],
                }
                det_cand_copy['delta_heading'] = det_cand['delta_heading'].clone()
                det_cand_copy['delta_position'] = det_cand['delta_position'].clone()
                det_cand_filtered = ekf_tracker(det_cand_copy)

                # # smooth
                curvature, pl, _ = trajectory_curvature(det_cand_filtered['delta_position'].detach().cpu().numpy())
                if pl < 1.:  # vehicle is "not" moving
                    # overwrite det_cand_filtered['delta_position'] (with shape (5, 2))
                    # by repeat det_cand_filtered['delta_position'][0]
                    det_cand_filtered['delta_position'][1:, :] = det_cand_filtered['delta_position'][0]
                    det_cand_filtered['delta_heading'][1:, :] = det_cand_filtered['delta_heading'][0]
                
                with torch.no_grad():
                    _, observe_trajs_cand, _, predict_trajs_cand = eval_stg_nm.eval(
                                                    scene,
                                                    timestep,
                                                    ph,
                                                    min_future_timesteps=ph,
                                                    num_samples=1,
                                                    z_mode=True,
                                                    gmm_mode=True,
                                                    full_dist=False,
                                                    all_z_sep=False,
                                                    det_cand=det_cand_filtered)

                # plan
                pi_ade_list.append(ade(predict_trajs_cand[target_inst_token], egoplan_gt_scene))
                # pi_ade_list.append(ade(predict_trajs_cand[target_inst_token], egoplan_clean_scene))

            # get index of top 3 candidates with min ade
            pi_ade_list = torch.stack(pi_ade_list)
            pi_ade_min_idx = torch.argsort(pi_ade_list)[:1]
            heading_cand_list = [heading_set[i] for i in pi_ade_min_idx]
            pos_cand_list = [pos_set[i] for i in pi_ade_min_idx]
            pi_ade_list = [pi_ade_list[i] for i in pi_ade_min_idx]
            

        ### Evalution ###
        for i in range(len(heading_cand_list)):
            det_cand = {
                'obj_id': target_inst_token, 
                'duration': range(-5, 0),
                'delta_heading': torch.tensor(heading_cand_list[i]).cuda(),
                'delta_position': torch.tensor(pos_cand_list[i]).cuda()
            }

            # EKF tracker
            det_cand_copy = {
                'obj_id': target_inst_token,
                'duration': [-5, -4, -3, -2, -1],
            }
            det_cand_copy['delta_heading'] = det_cand['delta_heading'].clone()
            det_cand_copy['delta_position'] = det_cand['delta_position'].clone()
            det_cand_filtered = ekf_tracker(det_cand_copy)
            perturbed_adv_his_traj = det_cand_filtered['delta_position'].clone().detach().cpu().numpy()

            # smooth
            curvature, pl, _ = trajectory_curvature(det_cand_filtered['delta_position'].detach().cpu().numpy())
            if pl < 1.:  # vehicle is "not" moving
                # overwrite det_cand_filtered['delta_position'] (with shape (5, 2))
                # by repeat det_cand_filtered['delta_position'][0]
                # print('path length below threshold: ', pl)   
                det_cand_filtered['delta_position'][1:, :] = det_cand_filtered['delta_position'][0]
                det_cand_filtered['delta_heading'][1:, :] = det_cand_filtered['delta_heading'][0]
            
            with torch.no_grad():
                _, observe_trajs_cand, _, predict_trajs_cand = eval_stg_nm.eval(
                                                scene,
                                                timestep,
                                                ph,
                                                min_future_timesteps=ph,
                                                num_samples=1,
                                                z_mode=True,
                                                gmm_mode=True,
                                                full_dist=False,
                                                all_z_sep=False,
                                                det_cand=det_cand_filtered)

            # plan
            egoplan_attack_scene = torch.tensor(plan(predict_trajs_cand, scene, target_inst_token, ts_curr)).cuda()
            
            # visualization
            vic_his_traj = observe_trajs_cand['ego'][:, :].detach().cpu().numpy()
            vic_plan_traj = egoplan_attack_scene.detach().cpu().numpy()
            adv_his_traj = observe_trajs_cand['target'][:, :2].detach().cpu().numpy()
            adv_pred_traj = predict_trajs_cand[target_inst_token].detach().cpu().numpy()

            if args.attack_type == 'inverse':
                output_path = os.path.join(scene_dir, 'vis_{}.pdf'.format(dataset_dirs[scene_idx]))
                # output_path = os.path.join(scene_dir, 'vis_{}.png'.format(dataset_dirs[scene_idx]))
            elif args.attack_type == 'forward':
                output_path = os.path.join(dataset_dir, 'forward_vis.png')
            print(output_path)
            plot_realworld(vic_his_traj, vic_plan_traj, ego_cur_rot, 
                           perturbed_adv_his_traj, adv_pred_traj, target_cur_trans[0], target_cur_trans[1],
                           output_path)
            ### predict and plan ###       
            
            ### calc evaluate metrics ###
            # check collision
            is_collision = False
            tar_traj = predict_trajs_cand[target_inst_token].detach().cpu().numpy()
            vic_traj = egoplan_gt_scene.detach().cpu().numpy()

            tar_yaws = approximate_yaw(tar_traj, target_cur_rot, thre=0.2)
            vic_yaws = approximate_yaw(vic_traj, ego_cur_rot, thre=0.2)
            tar_yaws = np.concatenate(([tar_yaws[0]], tar_yaws))
            vic_yaws = np.concatenate(([vic_yaws[0]], vic_yaws))

            for i in range(tar_traj.shape[0]):
                is_collision = check_collision_for_point_in_path( \
                    tar_traj[i, :], np.array([target_width, target_length]), tar_yaws[i], \
                    vic_traj[i, :], np.array([ego_width, ego_length]), vic_yaws[i])
                
                if is_collision:
                    print('dataset {}: collision!'.format(str(scene_idx+1)))
                    break

            # ade between target prediction and ego plan
            pi_ade_cand = ade(predict_trajs_cand[target_inst_token], egoplan_gt_scene)
            # pi_ade_cand = ade(predict_trajs_cand[target_inst_token], egoplan_clean_scene)

            # ade between ego plan before and after attack
            ade_ego2ego_cand = ade(egoplan_attack_scene, egoplan_clean_scene)
            # ade_ego2ego_cand = ade(egoplan_clean_scene, egoplan_clean_scene)
            # ade_ego2ego_cand = ade(egoplan_attack_scene, egoplan_gt_scene)
        
            # average lateral/longtitude deviation (6, ) for 5 frames
            # deviation: l2 distance between ego plan and ground truth
            lat_dev = horizonal_distance(observe_trajs_cand['ego'][:, :2], egoplan_attack_scene, egoplan_clean_scene)
            long_dev = vertical_distance(observe_trajs_cand['ego'][:, :2], egoplan_attack_scene, egoplan_clean_scene)
            # print(torch.mean(long_dev), torch.mean(lat_dev))

            # lateral/longtitude acceleration (9, )
            # diff between acc before and after current frame: acc[4] - acc[3]
            # diff > 0: acc; < 0: dec
            long_acc_atk, lat_acc_atk = get_acceleration(observe_trajs_cand['ego'][-5:, :2], egoplan_attack_scene)
            long_jerk_atk = (long_acc_atk[4] - long_acc_atk[3]) / 0.5
            lat_jerk_atk = (lat_acc_atk[4] - lat_acc_atk[3]) / 0.5
            ### eval best candidate ###

            # save results to csv
            csv_path = os.path.join(args.data_dir, '../res.csv')
            scene_type = args.data_dir.split('/')[-1]
            dataset_name = dataset_dirs[scene_idx]
            output_to_csv_realworld(csv_path, scene_type, scene_name, dataset_name, 
                                    is_collision, pi_ade_cand.item(), ade_ego2ego_cand.item(), 
                                    torch.mean(lat_dev).item(), torch.mean(long_dev).item(), lat_jerk_atk, long_jerk_atk)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='')
    parser.add_argument('--attack_type', type=str, default='')
    parser.add_argument('--scene_name', type=str, default='')
    args = parser.parse_args()

    ### hyperparameters ###
    ph = 6  # some scenes have less than 12 future timesteps!!!
    log_dir = './models'
    model_name = 'int_ee'  #'int_ee_me'
    ### hyperparameters ###

    ### load ###
    # load nuScenes data
    with open('../processed_realworld/attack.pkl', 'rb') as f:
        eval_env = dill.load(f, encoding='latin1')
    eval_scenes = eval_env.scenes

    # load model
    device = 'cuda:0'
    model_dir = os.path.join(log_dir, model_name)
    eval_stg_nm, hyp = load_model(model_dir, eval_env, ts=20, device=device)
    ### load ###
    
    eval()