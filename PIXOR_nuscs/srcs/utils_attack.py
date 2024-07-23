import os
import torch
import numpy as np
import time
import ctypes
import yaml
from easydict import EasyDict

import cv2
from utils import get_bev
from utils_nuscs import trans_v2world_nuscs, trans_v2world_realworld
from postprocess import compute_iou, convert_format,non_max_suppression


def merge_new_config(config, new_config):

    for key, val in new_config.items():
        if not isinstance(val, dict):
            config[key] = val
            continue
        if key not in config:
            config[key] = EasyDict()
        merge_new_config(config[key], val)

    return config


def cfg_from_yaml_file(cfg_file):

    config = EasyDict()

    with open(cfg_file, 'r') as f:
        try:
            new_config = yaml.safe_load(f, Loader=yaml.FullLoader)
        except:
            new_config = yaml.safe_load(f)

        merge_new_config(config=config, new_config=new_config)

    return config


### generate adv cls ###
def addpts_to_center(N_iter, N_add, Npts_cls, added_points_pool):

    added_points_pool = added_points_pool.reshape((N_iter*N_add, Npts_cls, 4))

    return np.mean(added_points_pool, axis=1)[:, :-1]  # (3, 3)


def center_to_addpts(N_iter, N_add, Npts_cls, size_cls, added_points_center):
    # add_points_center: (3, 3)
    
    xs = np.random.rand(N_iter*N_add,Npts_cls,1)*size_cls-size_cls/2.0+added_points_center[:,np.newaxis,0:1]
    ys = np.random.rand(N_iter*N_add,Npts_cls,1)*size_cls-size_cls/2.0+added_points_center[:,np.newaxis,1:2]
    zs = np.random.rand(N_iter*N_add,Npts_cls,1)*size_cls-size_cls/2.0+added_points_center[:,np.newaxis,2:3]
    rs = np.random.rand(N_iter*N_add,Npts_cls,1)*0.3+0.4  # reflectance

    added_points = np.concatenate([xs,ys,zs,rs],axis=2) # (3, 4, 4)
    added_points = added_points.reshape((N_iter, N_add*Npts_cls*4))

    return added_points



def get_adv_pts(N_iter,N_add):
    xs = np.random.rand(N_iter,N_add,1)*4.0-2.0
    ys = np.random.rand(N_iter,N_add,1)*4.0-2.0
    zs = np.random.rand(N_iter,N_add,1)*0.6+0.0
    rs = np.random.rand(N_iter,N_add,1)*0.3+0.4
    added_points_pool = np.concatenate([xs,ys,zs,rs],axis=2)
    added_points_pool = np.reshape(added_points_pool, (N_iter,N_add*4))
    #print(added_points_pool.shape)
    return added_points_pool


def get_adv_cls(N_iter, N_add, Npts_cls, size_cls = 0.2):
    # N_iter, N_add, Npts_cls: 100, 3, 4
    
    # cluster centers (shape: N_iter, N_add, 1)
    xs_cls = np.random.rand(N_iter,N_add,1)*4.0-2.0  # [-2, 2]
    ys_cls = np.random.rand(N_iter,N_add,1)*4.0-2.0  # [-2, 2]
    zs_cls = np.random.rand(N_iter,N_add,1)*0.8+0.0  # [0, 0.8]
    #rs = np.random.rand(N_iter,N_add,1)*0.3+0.4
    
    adv_cls_list = np.concatenate([xs_cls, ys_cls, zs_cls], axis=2)
    adv_cls_list = np.reshape(adv_cls_list, (N_iter*N_add,3))  # (3, 3)

    # generate points within each cluster (a cube of size 0.2)
    xs = np.random.rand(N_iter*N_add,Npts_cls,1)*size_cls-size_cls/2.0+adv_cls_list[:,np.newaxis,0:1]
    ys = np.random.rand(N_iter*N_add,Npts_cls,1)*size_cls-size_cls/2.0+adv_cls_list[:,np.newaxis,1:2]
    zs = np.random.rand(N_iter*N_add,Npts_cls,1)*size_cls-size_cls/2.0+adv_cls_list[:,np.newaxis,2:3]
    rs = np.random.rand(N_iter*N_add,Npts_cls,1)*0.3+0.4  # reflectance

    added_points_pool = np.concatenate([xs,ys,zs,rs],axis=2) # 1*3, 4, 4
    added_points_pool = added_points_pool.reshape((N_iter,N_add*Npts_cls*4))
    
    return added_points_pool



def get_adv_cls_realworld(N_iter, N_add, Npts_cls, length=0.2, width=0.3, height=0.42):
    # N_iter, N_add, Npts_cls: 100, 3, 4
    
    # Cluster centers (shape: N_iter, N_add, 1)
    # xs_cls in [0, 4.7] 
    xs_cls = np.random.rand(N_iter, N_add, 1) * 4.7 + 0.0
    # ys_cls in [-1.95, -0.95] or [0.95, 1.95]
    # ys_cls in [-2, 2]
    ys_cls = np.random.rand(N_iter, N_add, 1) * 4.0 - 2.0
    # zs_cls in [0, 2.0]
    zs_cls = np.random.rand(N_iter, N_add, 1) * 0.4 + 0.0

    adv_cls_list = np.concatenate([xs_cls, ys_cls, zs_cls], axis=2)
    adv_cls_list = np.reshape(adv_cls_list, (N_iter * N_add, 3))  # (500*1, 3)

    # Generate points within each cluster (a cuboid)
    # x <-> length, y <-> width, z <-> height
    xs = np.random.rand(N_iter * N_add, Npts_cls, 1) * length - length / 2.0 + adv_cls_list[:, np.newaxis, 0:1]
    ys = np.random.rand(N_iter * N_add, Npts_cls, 1) * width - width / 2.0 + adv_cls_list[:, np.newaxis, 1:2]
    zs = np.random.rand(N_iter * N_add, Npts_cls, 1) * height - height / 2.0 + adv_cls_list[:, np.newaxis, 2:3]
    rs = np.random.rand(N_iter * N_add, Npts_cls, 1) * 0.3 + 0.4  # Reflectance

    added_points_pool = np.concatenate([xs, ys, zs, rs], axis=2)
    added_points_pool = added_points_pool.reshape((N_iter, N_add * Npts_cls * 4))
    
    return added_points_pool


def get_adv_cls_center_realworld(N_iter, N_add, origin):
    # N_iter, N_add, Npts_cls: 100, 3, 4
    
    ### init cluster centers ###
    xs_cls = np.zeros((N_iter, N_add, 1))
    ys_cls = np.zeros((N_iter, N_add, 1))
    zs_cls = np.zeros((N_iter, N_add, 1))
    regions = np.random.rand(*xs_cls.shape) > 0.5

    # random init in pre-defined regions
    # tesla model 3: length=4.7, width=1.9, height=1.4
    if origin == 'rear_left' or origin == 'front_left':
        # for region 0 (left), x in [0, 5.2], y in [0.1, 0.3], z in [1.2, 2]
        xs_cls_r1 = np.random.rand(*xs_cls.shape) * 5.2 + 0.0
        ys_cls_r1 = np.random.rand(*xs_cls.shape) * 0.2 + 0.1
        zs_cls_r1 = np.random.rand(*xs_cls.shape) * 0.8 + 1.2
        # # for region 1 (behind), x in [-0.5, -0.1], y in [-1.9, 0.3], z in [1.2, 2]
        xs_cls_r2 = np.random.rand(*xs_cls.shape) * 0.4 - 0.5
        ys_cls_r2 = np.random.rand(*xs_cls.shape) * 2.2 - 1.9
        zs_cls_r2 = np.random.rand(*xs_cls.shape) * 0.8 + 1.2
    elif origin == 'rear_right' or origin == 'front_right':
        # for region 0 (left), x in [0, 5.2], y in [-0.3, -0.1], z in [1.2, 2]
        xs_cls_r1 = np.random.rand(*xs_cls.shape) * 5.2 + 0.0
        ys_cls_r1 = np.random.rand(*xs_cls.shape) * 0.2 - 0.3
        zs_cls_r1 = np.random.rand(*xs_cls.shape) * 0.8 + 1.2
        # # for region 1 (behind), x in [-0.5, -0.1], y in [-0.3, 1.9], z in [1.2, 2]
        xs_cls_r2 = np.random.rand(*xs_cls.shape) * 0.4 - 0.5
        ys_cls_r2 = np.random.rand(*xs_cls.shape) * 2.2 - 0.3
        zs_cls_r2 = np.random.rand(*xs_cls.shape) * 0.8 + 1.2
    else:
        raise ValueError('Wrong origin: {}'.format(origin))

    xs_cls = np.where(regions, xs_cls_r1, xs_cls_r2)
    ys_cls = np.where(regions, ys_cls_r1, ys_cls_r2)
    zs_cls = np.where(regions, zs_cls_r1, zs_cls_r2)
    ### Cluster centers (origin: bottom left corner) ###

    # # Cluster centers (bottom top center)
    # xs_cls = np.random.rand(N_iter, N_add, 1) * 4.7 + 0.0
    # ys_cls = np.random.rand(N_iter, N_add, 1) * 4.0 - 2.0
    # zs_cls = np.random.rand(N_iter, N_add, 1) * 0.8 - 0.4

    adv_cls_center_pool = np.concatenate([xs_cls, ys_cls, zs_cls], axis=2)
    adv_cls_center_pool = np.reshape(adv_cls_center_pool, (N_iter, N_add, 3))  # (500, 1/2/3, 3)
    
    return adv_cls_center_pool


def get_adv_one_cls_shift(N_iter,N_add,Npts_cls,N_shift,d_shift,cls_idxes=[0,1,2]):
    
    size_cls = 0.2

    # check whether there exist file in './added_points_pool.npy'
    if os.path.exists('added_points_pool.npy'):
        print('read from added_points_pool.npy')
        with open('added_points_pool.npy', 'rb') as f:
            added_points_pool = np.load(f)

        xs = added_points_pool[:, :, [0]]
        ys = added_points_pool[:, :, [1]]
        zs = added_points_pool[:, :, [2]]
        rs = added_points_pool[:, :, [3]]

        added_points_pool = added_points_pool.reshape((N_iter,N_add*Npts_cls*4))
    else:
        xs_cls = np.random.rand(N_iter,N_add,1)*4.0-2.0
        ys_cls = np.random.rand(N_iter,N_add,1)*4.0-2.0
        zs_cls = np.random.rand(N_iter,N_add,1)*0.8+0.0

        adv_cls_list = np.concatenate([xs_cls,ys_cls,zs_cls],axis=2)
        adv_cls_list = np.reshape(adv_cls_list, (N_iter*N_add,3))  # (3, 3)
        
        # (3, 4, 1) + (3, 1, 1) + (3, 1, 1)
        xs = np.random.rand(N_iter*N_add,Npts_cls,1)*size_cls-size_cls/2.0+adv_cls_list[:,np.newaxis,0:1]
        ys = np.random.rand(N_iter*N_add,Npts_cls,1)*size_cls-size_cls/2.0+adv_cls_list[:,np.newaxis,1:2]
        zs = np.random.rand(N_iter*N_add,Npts_cls,1)*size_cls-size_cls/2.0+adv_cls_list[:,np.newaxis,2:3]
        rs = np.random.rand(N_iter*N_add,Npts_cls,1)*0.3+0.4

        added_points_pool = np.concatenate([xs,ys,zs,rs],axis=2) # 1*3, 4, 4
        with open('added_points_pool.npy', 'wb') as f:
            np.save(f, added_points_pool)
        added_points_pool = added_points_pool.reshape((N_iter,N_add*Npts_cls*4))

    # shift
    x_shift = np.random.rand(N_shift, len(cls_idxes), 1, 1)*2.0*d_shift-d_shift  # (30, 3, 1)
    y_shift = np.random.rand(N_shift, len(cls_idxes), 1, 1)*2.0*d_shift-d_shift
    z_shift = np.random.rand(N_shift, len(cls_idxes), 1, 1)*2.0*d_shift-d_shift

    added_points_pool_shift = [added_points_pool]
    for shift_i in range(N_shift):

        # apply shift
        # (1, 4, 1) += (1, 1, 1)
        xs[cls_idxes, :] += x_shift[shift_i] 
        ys[cls_idxes, :] += y_shift[shift_i]
        zs[cls_idxes, :] += z_shift[shift_i]
        
        added_points = np.concatenate([xs,ys,zs,rs],axis=2) #N_iter*N_add,Npts_cls,4
        added_points = added_points.reshape((N_iter,N_add*Npts_cls*4))
        added_points_pool_shift.append(added_points)
    
    added_points_pool_shift = np.array(added_points_pool_shift) #N_shift,N_iter,N_add*Npts_cls*4
    return added_points_pool_shift


def get_adv_cls_shift(N_iter, N_add, Npts_cls, N_shift, d_shift, size_cls=0.2):
    xs_cls = np.random.rand(N_iter,N_add,1)*4.0-2.0
    ys_cls = np.random.rand(N_iter,N_add,1)*4.0-2.0
    zs_cls = np.random.rand(N_iter,N_add,1)*0.8+0.0
    #xs_cls = np.array([[[1.0],[-1.0],[0.0]]])
    #ys_cls = np.array([[[1.0],[-1.0],[0.0]]])
    #zs_cls = np.array([[[0.0],[0.0],[0.0]]])
    #print(zs_cls.shape)
    #rs = np.random.rand(N_iter,N_add,1)*0.3+0.4
    adv_cls_list = np.concatenate([xs_cls,ys_cls,zs_cls],axis=2)
    adv_cls_list = np.reshape(adv_cls_list, (N_iter*N_add,3))

    x_shift = np.random.rand(N_shift,N_iter*N_add,1,1)*2.0*d_shift-d_shift
    y_shift = np.random.rand(N_shift,N_iter*N_add,1,1)*2.0*d_shift-d_shift
    z_shift = np.random.rand(N_shift,N_iter*N_add,1,1)*2.0*d_shift-d_shift

    added_points_pool_shift = []
    for shift_i in range(N_shift):
        xs = np.random.rand(N_iter*N_add,Npts_cls,1)*size_cls-size_cls/2.0+adv_cls_list[:,np.newaxis,0:1]+x_shift[shift_i]
        ys = np.random.rand(N_iter*N_add,Npts_cls,1)*size_cls-size_cls/2.0+adv_cls_list[:,np.newaxis,1:2]+y_shift[shift_i]
        zs = np.random.rand(N_iter*N_add,Npts_cls,1)*size_cls-size_cls/2.0+adv_cls_list[:,np.newaxis,2:3]+z_shift[shift_i]
        rs = np.random.rand(N_iter*N_add,Npts_cls,1)*0.3+0.4
        added_points_pool = np.concatenate([xs,ys,zs,rs],axis=2) #N_iter*N_add,Npts_cls,4
        added_points_pool = added_points_pool.reshape((N_iter,N_add*Npts_cls*4))
        added_points_pool_shift.append(added_points_pool)
    added_points_pool_shift = np.array(added_points_pool_shift) #N_shift,N_iter,N_add*Npts_cls*4
    return added_points_pool_shift


def apply_shift(N_iter, N_add, Npts_cls, N_shift, d_shift, added_points, size_cls=0.2):
    # added_points: (1, 48=3*4*4)

    # get center
    added_points_center = addpts_to_center(N_iter, N_add, Npts_cls, added_points)  # (3, 3)
    xs = added_points_center[:, [0]]  # (3, 1)
    ys = added_points_center[:, [1]]
    zs = added_points_center[:, [2]]
    
    # apply shift
    added_points_shift_pool = []
    for shift_i in range(N_shift):
        # init shift
        x_shift = np.random.rand(N_iter*N_add, 1)*2.0*d_shift-d_shift  # (3, 1, 1)
        y_shift = np.random.rand(N_iter*N_add, 1)*2.0*d_shift-d_shift
        z_shift = np.random.rand(N_iter*N_add, 1)*2.0*d_shift-d_shift

        # apply shift on center
        xs_shift = xs + x_shift
        ys_shift = ys + y_shift
        zs_shift = zs + z_shift
        added_points_center_shift = np.concatenate([xs_shift,ys_shift,zs_shift],axis=1) # (3, 3)

        # center to addpts
        added_points_shift_pool.append(center_to_addpts(N_iter, N_add, Npts_cls, size_cls, added_points_center_shift))

    return np.array(added_points_shift_pool).reshape(N_shift, -1)  # (N_shift, 48)


def apply_center_shift(N_add, N_shift, d_shift, addpts_center):
    # addpts_center: (N_add, 3)

    # init
    xs = addpts_center[:, [0]]  # (N_add, 1)
    ys = addpts_center[:, [1]]
    zs = addpts_center[:, [2]]
    
    # apply shift
    addpts_center_shift_pool = []
    for shift_i in range(N_shift):
        # init shift
        x_shift = np.random.rand(N_add, 1)*2.0*d_shift-d_shift  # (N_add, 1)
        y_shift = np.random.rand(N_add, 1)*2.0*d_shift-d_shift
        z_shift = np.random.rand(N_add, 1)*2.0*d_shift-d_shift

        # apply shift on center
        xs_shift = xs + x_shift
        ys_shift = ys + y_shift
        zs_shift = zs + z_shift
        addpts_center_shift_pool.append(np.concatenate([xs_shift,ys_shift,zs_shift],axis=1)) # (N_add, 3)

    return np.array(addpts_center_shift_pool)# (N_shift, N_add, 3)


def apply_loc_error(N_iter, N_add, Npts_cls, d_shift, added_points):
    # added_points: (1, 48=3*4*4)

    # get clusters
    added_point_clusters = added_points.reshape(N_add, Npts_cls, -1)  # (3, 4, 4)
    
    # # shift all clusters
    # for cluster_i in range(added_point_clusters.shape[0]):
    for i in range(2):

        # random shift a cluster
        cluster_i = np.random.randint(N_add)
        # init shift
        shift = np.random.rand(1, 3) * 2.0 * d_shift - d_shift  # x, y, z shift

        # apply shift
        added_point_clusters[cluster_i, :, :3] += shift

    return added_point_clusters.reshape(1, -1)  # (1, 48)


def shift_cluster_size(N_iter, N_add, Npts_cls, added_points, size_cls):
    # added_points: (1, 48=3*4*4)
    # Npts_cls: 4 <- 0.2 (0.04*100), 9 <- 0.3 (0.09*100), 16 <- 0.4 (0.16*100)

    # get centers
    added_points_center = addpts_to_center(N_iter, N_add, Npts_cls, added_points)  # (3, 3)

    # define new Npts_cls
    new_Npts_cls = int(size_cls**2 * 100)
    
    return center_to_addpts(N_iter, N_add, new_Npts_cls, size_cls, added_points_center)


# def apply_shift(N_iter,N_add,Npts_cls,shift_xyz,added_points_pool):
    
#     # parse added_points_pool
#     added_points_pool = added_points_pool.reshape((N_iter*N_add, Npts_cls, 4))  # (1*3, 4, 4)
#     xs = added_points_pool[:, :, [0]]  # (3, 4, 1)
#     ys = added_points_pool[:, :, [1]]
#     zs = added_points_pool[:, :, [2]]
#     rs = added_points_pool[:, :, [3]]
#     added_points_pool = added_points_pool.reshape((N_iter,N_add*Npts_cls*4))  # 1, 48

#     # init shift
#     # x_shift = np.random.rand(N_iter*N_add, 1, 1)*2.0*d_shift-d_shift  # (3, 1, 1)
#     # y_shift = np.random.rand(N_iter*N_add, 1, 1)*2.0*d_shift-d_shift
#     # z_shift = np.random.rand(N_iter*N_add, 1, 1)*2.0*d_shift-d_shift
#     x_shift = np.full((N_iter*N_add, 1, 1), shift_xyz[0])
#     y_shift = np.full((N_iter*N_add, 1, 1), shift_xyz[1])
#     z_shift = np.full((N_iter*N_add, 1, 1), shift_xyz[2])

#     # apply shift
#     # (3, 4, 1) += (3, 1, 1)
#     xs += x_shift
#     ys += y_shift
#     zs += z_shift
    
#     added_points_pool_shift = np.concatenate([xs,ys,zs,rs],axis=2) #N_iter*N_add,Npts_cls,4
#     added_points_pool_shift = added_points_pool_shift.reshape((N_iter,N_add*Npts_cls*4))  #N_iter,N_add*Npts_cls*4
    
#     return added_points_pool_shift


# def apply_shift(N_iter,N_add,Npts_cls,shift_xyz,added_points_center):
    
#     # parse added_points_center (3, 3)
#     xs = added_points_center[:, [0]]  # (3, 1)
#     ys = added_points_center[:, [1]]
#     zs = added_points_center[:, [2]]

#     # parse and apply shift_xyz (3, 1)
#     xs_shift = xs + np.full((N_iter*N_add, 1), shift_xyz[0])
#     ys_shift = ys + np.full((N_iter*N_add, 1), shift_xyz[1])
#     zs_shift = zs + np.full((N_iter*N_add, 1), shift_xyz[2])
#     added_points_center_shift = np.concatenate([xs_shift,ys_shift,zs_shift],axis=1) # (3, 3)

#     # generate points within each cluster (a cube of size 0.2)
#     size_cls = 0.2
#     xs_shift = np.random.rand(N_iter*N_add,Npts_cls,1)*size_cls-size_cls/2.0 + xs_shift[:,np.newaxis,:]
#     ys_shift = np.random.rand(N_iter*N_add,Npts_cls,1)*size_cls-size_cls/2.0 + ys_shift[:,np.newaxis,:]
#     zs_shift = np.random.rand(N_iter*N_add,Npts_cls,1)*size_cls-size_cls/2.0 + zs_shift[:,np.newaxis,:]
#     rs_shift = np.random.rand(N_iter*N_add,Npts_cls,1)*0.3+0.4  # reflectance

#     added_points_shift = np.concatenate([xs_shift, ys_shift, zs_shift, rs_shift],axis=2) # 1*3, 4, 4
#     added_points_shift = added_points_shift.reshape((N_iter,N_add*Npts_cls*4))
    
#     return added_points_shift, added_points_center_shift
### generate adv cls ###


### detection model inference ###
def attack_obj_nuscs(added_points,net,vehicle_idx,config,geom,filename,label_list,gt3Dboxes,device,N_add):
    with torch.no_grad():
        # read lidar and add points -> bev view
        c_name = bytes(filename, 'utf-8')
        scan = np.zeros(geom, dtype=np.float32)
        c_data = ctypes.c_void_p(scan.ctypes.data)  
        added_points_t = trans_v2world_nuscs(added_points,gt3Dboxes,vehicle_idx,N_add)
        add = np.asarray(added_points_t,dtype=np.float32)  # (48,)


        # ## rebuttal ###
        # generate additional two point clusters to simulate vehicle or pedestrians
        # point cluster of ped 1
        # moving_ped_1 = get_adv_cls(1, 1, 20, size_cls=0.5)  # (1, 160)
        moving_ped_1 = np.fromfile('/home/kyrie/Desktop/WorkSpace/lidar/OpenPCDet/data/kitti/gt_database/003018_Pedestrian_3.bin', dtype=np.float32).reshape(-1, 4)
        moving_ped_2 = moving_ped_1.copy()
        # random sample moving_ped_1 position: x within [5, 10], y with [-2, -4]
        moving_ped_1_x = np.random.rand(1) * 5 + 5
        moving_ped_1_y = np.random.rand(1) * 2 - 4
        moving_ped_1[:, 0] += moving_ped_1_x
        moving_ped_1[:, 1] += moving_ped_1_y
        # moving_ped_1_3dboxes = np.array([[ 0.6,  1.6,  0.6,  moving_ped_1_y, 0.8,  moving_ped_1_x,  0]])
        # addpts_moving_ped_1 = trans_v2world_nuscs(moving_ped_1, moving_ped_1_3dboxes, 0, 1*20)
        addpts_moving_ped_1 = np.array(moving_ped_1, dtype=np.float32).reshape(-1,)

        # # point cluster of ped 2
        # # moving_ped_2 = get_adv_cls(1, 1, 20, size_cls=0.5)  # (1, 160)
        # # random sample moving_ped_2 position: x within [5, 10], y with [2, 4]
        # moving_ped_2_x = np.random.rand(1) * 5 + 5
        # moving_ped_2_y = np.random.rand(1) * 2 + 2
        # moving_ped_2[:, 0] += moving_ped_2_x
        # moving_ped_2[:, 1] += moving_ped_2_y
        # # moving_ped_2_3dboxes = np.array([[ 0.6,  1.6,  0.6,  moving_ped_2_y, 0.8,  moving_ped_2_x,  0]])
        # # addpts_moving_ped_2 = trans_v2world_nuscs(moving_ped_2, moving_ped_2_3dboxes, 0, 1*20)
        # addpts_moving_ped_2 = np.array(moving_ped_2, dtype=np.float32).reshape(-1,)
        
        add = np.concatenate([add, addpts_moving_ped_1], axis=0)
        # # save add as .bin file for debug
        # add.tofile('./add.bin')
        # raise ValueError()
        ### rebuttal ###
        

        c_add = ctypes.c_void_p(add.ctypes.data) 
        ctypes.cdll.LoadLibrary('preprocess_nuscs/LidarPreprocess_nuscs.so'). \
                    createTopViewMaps(c_data, c_name, c_add, ctypes.c_int(int(len(add)/4)))
        scan = torch.from_numpy(scan)
        scan = scan.permute(2, 0, 1)
        input = scan
        input = input.to(device)
        
        # inference
        # pred_start_time = time.time()
        pred = net(input.unsqueeze(0))
        # pred_end_time = time.time()
        # print("pred time: ", pred_end_time - pred_start_time)  # 0.7s
        # raise ValueError()
        pred.squeeze_(0)
        #target_mean = [0.008, 0.001, 0.202, 0.2, 0.43, 1.368]
        #target_std_dev = [0.866, 0.5, 0.954, 0.668, 0.09, 0.111]
        
        # Filter Predictions (score)
        cls_pred = pred[0, ...]
        activation = cls_pred > 0.3  #0.5
        num_boxes = int(activation.sum())
        corners = torch.zeros((num_boxes, 8))
        if num_boxes==0:
            return np.zeros((6,))

        for i in range(7, 15):
            corners[:, i - 7] = torch.masked_select(pred[i, ...], activation)
        corners = corners.view(-1, 4, 2)
        scores = torch.masked_select(cls_pred, activation).cpu()
        cos_t = torch.masked_select(pred[1, ...], activation).cpu()# * target_std_dev[0] + target_mean[0]
        sin_t = torch.masked_select(pred[2, ...], activation).cpu()# * target_std_dev[1] + target_mean[1]
        theta = torch.atan2(sin_t, cos_t)
        bbox_w = torch.masked_select(pred[5, ...], activation).cpu().exp()# * target_std_dev[4] + target_mean[4]).exp()
        bbox_l = torch.masked_select(pred[6, ...], activation).cpu().exp()# * target_std_dev[5] + target_mean[5]).exp()
        center = corners.mean(dim=1)

        # NMS
        # nms_start_time = time.time()
        selected_ids = non_max_suppression(corners.numpy(), scores.numpy(), config['nms_iou_threshold'])
        corners = corners[selected_ids].numpy()
        scores = scores[selected_ids].numpy()
        theta = theta[selected_ids]
        bbox_w = bbox_w[selected_ids]
        bbox_l = bbox_l[selected_ids]
        center = center[selected_ids]
        regs = torch.stack([center[:,0],center[:,1],bbox_w,bbox_l,theta],dim=-1).numpy()
        # nms_end_time = time.time()
        # print('nms time: ', nms_end_time - nms_start_time)
        #print(regs)
        #corners, scores = filter_pred(config, pred)

    # find pred that best match with the gt
    # iou_start_time = time.time()
    gt_boxes = np.array(label_list)
    gtbox_poly = convert_format(gt_boxes)[vehicle_idx]
    corners_poly = convert_format(corners)
    ious = compute_iou(gtbox_poly, corners_poly)   #ious for each predicted boxes of the id-th gt box
    bestmatch_id = np.argsort(ious)[-1] #largest iou
    IOU = ious[bestmatch_id]
    if IOU> 0.:  #0.01:
        out = np.concatenate([scores[bestmatch_id:bestmatch_id+1],regs[bestmatch_id]],axis=0)
    else:
        out = np.zeros((6,))
    # iou_end_time = time.time()
    # print('iou time: ', iou_end_time - iou_start_time)
    # raise ValueError()

    # input_np = input.cpu().permute(1, 2, 0).numpy()
    # # pred_image_start_time = time.time()
    # pred_image = get_bev(input_np, corners)
    # pred_image_gt = get_bev(input_np, np.expand_dims(gt_boxes[vehicle_idx], axis=0))
    # # pred_image_end_time = time.time()
    # # print("pred image time: ", pred_image_end_time - pred_image_start_time)
    # # cv2.imwrite("attack_results/pred_image.png",pred_image) 
    # cv2.imwrite("pred_image.png",pred_image) 
    # cv2.imwrite("pred_image_gt.png",pred_image_gt) 
    return out


def attack_obj_realworld(addpts_center,net,vehicle_idx,config,geom,filename,label_list,gt3Dboxes,device,cfg):
    # addpts_center: (N_add, 3)
    
    with torch.no_grad():
    
        # read lidar and add points -> bev view
        c_name = bytes(filename, 'utf-8')
        scan = np.zeros(geom, dtype=np.float32)
        c_data = ctypes.c_void_p(scan.ctypes.data)  
        added_points_t = trans_v2world_realworld(addpts_center,gt3Dboxes,vehicle_idx,cfg.N_add, cfg.origin)

        add = np.asarray(added_points_t,dtype=np.float32)  # (48,)
        c_add = ctypes.c_void_p(add.ctypes.data) 
        
        output_pc_fn = os.path.join('attack_results/output_pc', filename.split('/')[-1].split('.')[0] + '.txt')
        c_output_pc_fn = bytes(output_pc_fn, 'utf-8')
        ctypes.cdll.LoadLibrary('preprocess_nuscs/LidarPreprocess_realworld.so'). \
                    createTopViewMaps(c_data, c_name, c_add, ctypes.c_int(int(len(add)/4)), c_output_pc_fn)
        scan = torch.from_numpy(scan)
        scan = scan.permute(2, 0, 1)
        input = scan
        input = input.to(device)
        
        # inference
        pred = net(input.unsqueeze(0))  # pred time: ~0.7s
        pred.squeeze_(0)
        #target_mean = [0.008, 0.001, 0.202, 0.2, 0.43, 1.368]
        #target_std_dev = [0.866, 0.5, 0.954, 0.668, 0.09, 0.111]
        
        # Filter Predictions (score)
        cls_pred = pred[0, ...]
        activation = cls_pred > 0.1  #0.5
        num_boxes = int(activation.sum())
        corners = torch.zeros((num_boxes, 8))
        if num_boxes==0:
            return np.zeros((6,))

        for i in range(7, 15):
            corners[:, i - 7] = torch.masked_select(pred[i, ...], activation)
        corners = corners.view(-1, 4, 2)
        scores = torch.masked_select(cls_pred, activation).cpu()
        cos_t = torch.masked_select(pred[1, ...], activation).cpu()# * target_std_dev[0] + target_mean[0]
        sin_t = torch.masked_select(pred[2, ...], activation).cpu()# * target_std_dev[1] + target_mean[1]
        theta = torch.atan2(sin_t, cos_t)
        bbox_w = torch.masked_select(pred[5, ...], activation).cpu().exp()# * target_std_dev[4] + target_mean[4]).exp()
        bbox_l = torch.masked_select(pred[6, ...], activation).cpu().exp()# * target_std_dev[5] + target_mean[5]).exp()
        center = corners.mean(dim=1)

        # NMS
        selected_ids = non_max_suppression(corners.numpy(), scores.numpy(), config['nms_iou_threshold'])
        corners = corners[selected_ids].numpy()
        scores = scores[selected_ids].numpy()
        theta = theta[selected_ids]
        bbox_w = bbox_w[selected_ids]
        bbox_l = bbox_l[selected_ids]
        center = center[selected_ids]
        regs = torch.stack([center[:,0],center[:,1],bbox_w,bbox_l,theta],dim=-1).numpy()
        #print(regs)
        #corners, scores = filter_pred(config, pred)

    # find pred that best match with the gt
    gt_boxes = np.array(label_list)
    gtbox_poly = convert_format(gt_boxes)[vehicle_idx]
    corners_poly = convert_format(corners)
    ious = compute_iou(gtbox_poly, corners_poly)   #ious for each predicted boxes of the id-th gt box
    bestmatch_id = np.argsort(ious)[-1] #largest iou
    IOU = ious[bestmatch_id]
    if IOU> 0.01:
        out = np.concatenate([scores[bestmatch_id:bestmatch_id+1],regs[bestmatch_id]],axis=0)
    else:
        out = np.zeros((6,))
    
    # # plot
    # input_np = input.cpu().permute(1, 2, 0).numpy()
    # pred_image = get_bev(input_np, corners)
    # pred_image_fn = os.path.join('attack_results/pred_image', filename.split('/')[-1].split('.')[0] + '.png')
    # cv2.imwrite(pred_image_fn,pred_image)
    # pred_image_gt = get_bev(input_np, np.expand_dims(gt_boxes[vehicle_idx], axis=0))
    # cv2.imwrite("pred_image_gt.png",pred_image_gt) 
    
    return out
### detection model inference ###


### second stage refinement ###
def box_to_sample_result(sample_token, box, attribute_name: str = ''):
    # Prepare data
    translation = box.center
    size = box.wlh
    rotation = box.orientation.q
    velocity = box.velocity
    detection_name = box.name
    detection_score = box.score

    # Create result dict
    sample_result = dict()
    sample_result['sample_token'] = sample_token
    sample_result['translation'] = translation.tolist()
    sample_result['size'] = size.tolist()
    sample_result['rotation'] = rotation.tolist()
    sample_result['velocity'] = velocity.tolist()[:2]  # Only need vx, vy.
    sample_result['detection_name'] = detection_name
    sample_result['detection_score'] = detection_score
    sample_result['attribute_name'] = attribute_name

    return sample_result
### second stage refinement ###


### output format ###
def out_format(data):
    # Base case: if data is already a list
    if isinstance(data, list):
        return [out_format(item) for item in data]

    if isinstance(data, torch.Tensor):  # If data is a PyTorch tensor
        return data.tolist()
    
    if isinstance(data, np.ndarray):  # If data is a NumPy array
        return data.tolist()
    
    return data
### output format ###
