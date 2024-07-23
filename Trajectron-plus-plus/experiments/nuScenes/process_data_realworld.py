import os
import sys
import dill
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

from pyquaternion import Quaternion
from kalman_filter import NonlinearKinematicBicycle
from sklearn.model_selection import train_test_split

nu_path = './devkit/python-sdk/'
sys.path.append(nu_path)
sys.path.append("../../trajectron")
from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.utils.splits import create_splits_scenes
from environment import Environment, Scene, Node, GeometricMap, derivative_of


FREQUENCY = 2
dt = 1 / FREQUENCY
data_columns_vehicle = pd.MultiIndex.from_product([['position', 'velocity', 'acceleration', 'heading'], ['x', 'y']])
data_columns_vehicle = data_columns_vehicle.append(pd.MultiIndex.from_tuples([('heading', '°'), ('heading', 'd°')]))
data_columns_vehicle = data_columns_vehicle.append(pd.MultiIndex.from_product([['velocity', 'acceleration'], ['norm']]))

data_columns_pedestrian = pd.MultiIndex.from_product([['position', 'velocity', 'acceleration'], ['x', 'y']])


standardization = {
    'PEDESTRIAN': {
        'position': {
            'x': {'mean': 0, 'std': 1},
            'y': {'mean': 0, 'std': 1}
        },
        'velocity': {
            'x': {'mean': 0, 'std': 2},
            'y': {'mean': 0, 'std': 2}
        },
        'acceleration': {
            'x': {'mean': 0, 'std': 1},
            'y': {'mean': 0, 'std': 1}
        }
    },
    'VEHICLE': {
        'position': {
            'x': {'mean': 0, 'std': 80},
            'y': {'mean': 0, 'std': 80}
        },
        'velocity': {
            'x': {'mean': 0, 'std': 15},
            'y': {'mean': 0, 'std': 15},
            'norm': {'mean': 0, 'std': 15}
        },
        'acceleration': {
            'x': {'mean': 0, 'std': 4},
            'y': {'mean': 0, 'std': 4},
            'norm': {'mean': 0, 'std': 4}
        },
        'heading': {
            'x': {'mean': 0, 'std': 1},
            'y': {'mean': 0, 'std': 1},
            '°': {'mean': 0, 'std': np.pi},
            'd°': {'mean': 0, 'std': 1}
        }
    }
}


def process_scene(env, data_path, scene_name):

    data = pd.DataFrame(columns=['frame_id',
                                 'type',
                                 'node_id',
                                 'robot',
                                 'x', 'y', 'z',
                                 'length',
                                 'width',
                                 'height',
                                 'heading'])
    
    # read data.json
    with open(data_path, 'r') as f:
        scene_data = json.load(f)
    frame_ids = list(scene_data.keys())

    our_category = env.NodeType.VEHICLE

    for frame_idx in range(len(frame_ids)):
        
        frame_id = frame_ids[frame_idx]
        scene_data_target = scene_data[frame_id]['target']
        scene_data_ego = scene_data[frame_id]['ego']

        # Target Vehicle
        data_point = pd.Series({'frame_id': frame_idx,
                                'type': our_category,
                                'node_id': 'target',
                                'robot': False,
                                'x': scene_data_target[0],
                                'y': scene_data_target[1],
                                'z': scene_data_target[2],
                                'length': scene_data_target[3],
                                'width': scene_data_target[4],
                                'height': scene_data_target[5],
                                'heading': scene_data_target[6]})
        data = data.append(data_point, ignore_index=True)

        # Ego Vehicle
        data_point = pd.Series({'frame_id': frame_idx,
                                'type': our_category,
                                'node_id': 'ego',
                                'robot': True,
                                'x': scene_data_ego[0],
                                'y': scene_data_ego[1],
                                'z': scene_data_ego[2],
                                'length': 4,
                                'width': 1.7,
                                'height': 1.5,
                                'heading': scene_data_ego[6],
                                'orientation': None})
        data = data.append(data_point, ignore_index=True)

        frame_idx += 1

    if len(data.index) == 0:
        return None

    data.sort_values('frame_id', inplace=True)
    max_timesteps = data['frame_id'].max()

    # get dataframe data with node_id equal to ego
    vis_data = data[data['node_id'] == 'ego']
    vis_x_min = np.round(vis_data['x'].min() - 25)
    vis_x_max = np.round(vis_data['x'].max() + 25)
    vis_y_min = np.round(vis_data['y'].min() - 25)
    vis_y_max = np.round(vis_data['y'].max() + 25)
    vis_patch = (vis_x_min, vis_y_min, vis_x_max, vis_y_max)

    x_min = 0.
    x_max = 0.
    y_min = 0.
    y_max = 0.
    # x_min = np.round(data['x'].min() - 50)
    # x_max = np.round(data['x'].max() + 50)
    # y_min = np.round(data['y'].min() - 50)
    # y_max = np.round(data['y'].max() + 50)
    patch = (x_min, y_min, x_max, y_max)

    data['x'] = data['x'] - x_min
    data['y'] = data['y'] - y_min

    scene = Scene(timesteps=max_timesteps + 1, dt=dt, name=str(scene_name), patch=patch, vis_patch=vis_patch)
    # scene = Scene(timesteps=max_timesteps + 1, dt=dt, name=str(scene_id), patch=(x_min, y_min, x_max, y_max))

    # Generate Maps
    scene.map = None

    for node_id in pd.unique(data['node_id']):
        node_frequency_multiplier = 1
        node_df = data[data['node_id'] == node_id]

        if node_df['x'].shape[0] < 2:
            continue

        node_values = node_df[['x', 'y']].values
        x = node_values[:, 0]
        y = node_values[:, 1]
        heading = node_df['heading'].values

        vx = derivative_of(x, scene.dt)
        vy = derivative_of(y, scene.dt)
        ax = derivative_of(vx, scene.dt)
        ay = derivative_of(vy, scene.dt)

        if node_df.iloc[0]['type'] == env.NodeType.VEHICLE:
            v = np.stack((vx, vy), axis=-1)
            v_norm = np.linalg.norm(np.stack((vx, vy), axis=-1), axis=-1, keepdims=True)
            heading_v = np.divide(v, v_norm, out=np.zeros_like(v), where=(v_norm > 1.))
            heading_x = heading_v[:, 0]
            heading_y = heading_v[:, 1]

            data_dict = {('position', 'x'): x,
                         ('position', 'y'): y,
                         ('velocity', 'x'): vx,
                         ('velocity', 'y'): vy,
                         ('velocity', 'norm'): np.linalg.norm(np.stack((vx, vy), axis=-1), axis=-1),
                         ('acceleration', 'x'): ax,
                         ('acceleration', 'y'): ay,
                         ('acceleration', 'norm'): np.linalg.norm(np.stack((ax, ay), axis=-1), axis=-1),
                         ('heading', 'x'): heading_x,
                         ('heading', 'y'): heading_y,
                         ('heading', '°'): heading,
                         ('heading', 'd°'): derivative_of(heading, dt, radian=True)}
            node_data = pd.DataFrame(data_dict, columns=data_columns_vehicle)
        else:
            data_dict = {('position', 'x'): x,
                         ('position', 'y'): y,
                         ('velocity', 'x'): vx,
                         ('velocity', 'y'): vy,
                         ('acceleration', 'x'): ax,
                         ('acceleration', 'y'): ay}
            node_data = pd.DataFrame(data_dict, columns=data_columns_pedestrian)

        node = Node(node_type=node_df.iloc[0]['type'], node_id=node_id, data=node_data, length=node_df.iloc[0]['length'], \
                    width=node_df.iloc[0]['width'], frequency_multiplier=node_frequency_multiplier)
        node.first_timestep = node_df['frame_id'].iloc[0]
        if node_df.iloc[0]['robot'] == True:
            node.is_robot = True
            scene.robot = node

        scene.nodes.append(node)
    
    return scene


def process_data(data_dir, output_dir):

    # create env
    env = Environment(node_type_list=['VEHICLE', 'PEDESTRIAN'], standardization=standardization)
    attention_radius = dict()
    attention_radius[(env.NodeType.PEDESTRIAN, env.NodeType.PEDESTRIAN)] = 10.0
    attention_radius[(env.NodeType.PEDESTRIAN, env.NodeType.VEHICLE)] = 20.0
    attention_radius[(env.NodeType.VEHICLE, env.NodeType.PEDESTRIAN)] = 20.0
    attention_radius[(env.NodeType.VEHICLE, env.NodeType.VEHICLE)] = 30.0

    env.attention_radius = attention_radius
    env.robot_type = env.NodeType.VEHICLE

    # read data folder
    scene_names = os.listdir(data_dir)
    
    scenes = []
    for scene_idx in range(len(scene_names)):

        if scene_names[scene_idx] != args.scene_name:
            continue
        
        # loop datasets
        scene_dir = os.path.join(data_dir, scene_names[scene_idx])
        dataset_dirs = os.listdir(scene_dir)
        dataset_dirs = [dataset for dataset in dataset_dirs if dataset.startswith('dataset')]
        dataset_dirs.sort()

        for dataset_dir in dataset_dirs:
            
            scene_data_path = os.path.join(data_dir, scene_names[scene_idx], dataset_dir, 'labels_global.json')
            scene = process_scene(env, scene_data_path, scene_names[scene_idx])

            if scene is not None:
                scenes.append(scene)

    print(f'Processed {len(scenes):.2f} scenes')

    env.scenes = scenes

    if len(scenes) > 0:

        data_dict_path = os.path.join(output_dir, 'attack.pkl')
        with open(data_dict_path, 'wb') as f:
            dill.dump(env, f, protocol=dill.HIGHEST_PROTOCOL)
        print('Saved Environment!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--scene_name', type=str)
    parser.add_argument('--output_dir', type=str, default='../processed_realworld/')
    args = parser.parse_args()

    # init
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # pre-process
    process_data(args.data_dir, args.output_dir)
