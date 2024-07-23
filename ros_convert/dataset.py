import os
import json
import math
import shutil
import argparse
import numpy as np
from PIL import Image
import pymap3d as pm
from datetime import datetime


### utils ###
def frame2int(frame):
    if frame.endswith('.png') or frame.endswith('.bin'):
        frame = frame.split('.')[0]
    frames = frame.split('-')

    frame_int = ''
    for f in frames:
        frame_int += f

    return int(frame_int)


def int2frame(frame_int):
    frame = str(frame_int)
    frame = frame[:2] + '-' + frame[2:4] + '-' + frame[4:6] + '-' + frame[6:]
    
    return frame

def int2datetime(frame_int):
    format_str = "%Y-%m-%d %H:%M:%S"
    str_time = "2023-12-04 {}:{}:{}".format(frame_int[0:2], frame_int[2:4], frame_int[4:6])
    
    datetime

# find the frame in pointclouds that cloest to the curr_frame
def find_closest_frame(curr_frame, pointclouds):

    closest_frame = pointclouds[0]
    for frame in pointclouds:
        if abs(frame - curr_frame) < abs(closest_frame - curr_frame):
            closest_frame = frame

    # # first find frames with same second
    # curr_frame_candidates = []
    # for frame in pointclouds:
    #     if int(frame/1000) == int(curr_frame/1000):
    #         curr_frame_candidates.append(frame)
    
    # # then find the frame with closest nano seocnd (last 3 digits)
    # closest_frame = curr_frame_candidates[0]
    # for frame in curr_frame_candidates:
    #     if abs(frame - curr_frame) < abs(closest_frame - curr_frame):
    #         closest_frame = frame
            
    return closest_frame


# visualize the pointcloud in bev
def scale_to_255(a, min, max, dtype=np.uint8):
    """ Scales an array of values from specified min, max range to 0-255
        Optionally specify the data type of the output (default is uint8)
    """
    return (((a - min) / float(max - min)) * 255).astype(dtype)


def point_cloud_2_birdseye(points,
                           res=0.05,
                           side_range=(-15., 15.),  # left-most to right-most
                           fwd_range = (-10., 20.), # back-most to forward-most
                           height_range=(-2., 2.),  # bottom-most to upper-most
                           ):
    """ Creates an 2D birds eye view representation of the point cloud data.

    Args:
        points:     (numpy array)
                    N rows of points data
                    Each point should be specified by at least 3 elements x,y,z
        res:        (float)
                    Desired resolution in metres to use. Each output pixel will
                    represent an square region res x res in size.
        side_range: (tuple of two floats)
                    (-left, right) in metres
                    left and right limits of rectangle to look at.
        fwd_range:  (tuple of two floats)
                    (-behind, front) in metres
                    back and front limits of rectangle to look at.
        height_range: (tuple of two floats)
                    (min, max) heights (in metres) relative to the origin.
                    All height values will be clipped to this min and max value,
                    such that anything below min will be truncated to min, and
                    the same for values above max.
    Returns:
        2D numpy array representing an image of the birds eye view.
    """
    # EXTRACT THE POINTS FOR EACH AXIS
    x_points = points[:, 0]
    y_points = points[:, 1]
    z_points = points[:, 2]

    # FILTER - To return only indices of points within desired cube
    # Three filters for: Front-to-back, side-to-side, and height ranges
    # Note left side is positive y axis in LIDAR coordinates
    f_filt = np.logical_and((x_points > fwd_range[0]), (x_points < fwd_range[1]))
    s_filt = np.logical_and((y_points > -side_range[1]), (y_points < -side_range[0]))
    filter = np.logical_and(f_filt, s_filt)
    indices = np.argwhere(filter).flatten()

    # KEEPERS
    x_points = x_points[indices]
    y_points = y_points[indices]
    z_points = z_points[indices]

    # CONVERT TO PIXEL POSITION VALUES - Based on resolution
    x_img = (-y_points / res).astype(np.int32)  # x axis is -y in LIDAR
    y_img = (-x_points / res).astype(np.int32)  # y axis is -x in LIDAR

    # SHIFT PIXELS TO HAVE MINIMUM BE (0,0)
    # floor & ceil used to prevent anything being rounded to below 0 after shift
    x_img -= int(np.floor(side_range[0] / res))
    y_img += int(np.ceil(fwd_range[1] / res))

    # CLIP HEIGHT VALUES - to between min and max heights
    pixel_values = np.clip(a=z_points,
                           a_min=height_range[0],
                           a_max=height_range[1])

    # RESCALE THE HEIGHT VALUES - to be between the range 0-255
    pixel_values = scale_to_255(pixel_values,
                                min=height_range[0],
                                max=height_range[1])

    # INITIALIZE EMPTY ARRAY - of the dimensions we want
    x_max = 1 + int((side_range[1] - side_range[0]) / res)
    y_max = 1 + int((fwd_range[1] - fwd_range[0]) / res)
    im = np.zeros([y_max, x_max], dtype=np.uint8)

    # FILL PIXEL VALUES IN IMAGE ARRAY
    im[y_img, x_img] = pixel_values

    return im


def calculate_heading(x1, y1, x2, y2):
    """
    Calculate the heading angle in radians from point (x_i, y_i) to point (x_i_plus_1, y_i_plus_1).

    :param x_i: x-coordinate of the current point
    :param y_i: y-coordinate of the current point
    :param x_i_plus_1: x-coordinate of the next point
    :param y_i_plus_1: y-coordinate of the next point
    :return: heading angle in radians
    """
    dx = x2 - x1
    dy = y2 - y1
    heading = math.atan2(dy, dx)
    return heading


def match():
    
    ### read benign info ###
    # get frame ids
    with open(os.path.join(benign_dataset_dir, 'labels_lidar.json'), 'r') as f:
        labels_lidar = json.load(f)
    frame_ids = list(labels_lidar.keys())
    frame_ids.sort()
    curr_frame_id = frame_ids[4]

    # get ego coords
    with open(os.path.join(benign_scene_dir, 'enu_coords.json')) as f:
        benign_ego_coords = json.load(f)
    benign_ego_coords_frames = list(benign_ego_coords.keys())
    benign_ego_coords_frames.sort()
    benign_ego_coords_frames = [frame2int(frame) for frame in benign_ego_coords_frames]

    # get curr ego coords
    benign_ego_coords_curr_frame = find_closest_frame(frame2int(curr_frame_id), benign_ego_coords_frames)
    benign_ego_coords_curr_frame = int2frame(benign_ego_coords_curr_frame)
    benign_ego_coords_curr = benign_ego_coords[benign_ego_coords_curr_frame]
    ### read benign info ###

    ### read benign info ###
    with open(os.path.join(attack_scene_dir, 'enu_coords.json')) as f:
        attack_ego_coords = json.load(f)
    ### read benign info ###

    # find the closest frame in attack scenes
    closest_dist = 999.
    closest_ego_coords = None
    closest_frame_id = None
    for frame_id, coords in attack_ego_coords.items():
        dist = np.linalg.norm(np.array(coords[:2]) - np.array(benign_ego_coords_curr[:2]))
        if dist <= closest_dist:
            closest_dist = dist
            closest_ego_coords = coords
            closest_frame_id = frame_id

    # find frame id list that have the same ego coords
    matched_frame_ids = []
    for frame_id, coords in attack_ego_coords.items():
        if coords == closest_ego_coords:
            matched_frame_ids.append(frame_id)
    # print('matched frame ids: {}'.format(matched_frame_ids))

    # find the smallest attack point cloud frame ids that fall into the matched frame ids
    # cause in later frames, ego vehicle may move
    frame_ids_pc = os.listdir(os.path.join(attack_scene_dir, 'pointclouds'))
    frame_ids_pc = [frame2int(frame) for frame in frame_ids_pc]
    frame_ids_pc.sort()
    matched_frame_id_pc = None
    for frame_id in frame_ids_pc:
        if frame_id >= frame2int(matched_frame_ids[0]) and frame_id <= frame2int(matched_frame_ids[-1]):
            matched_frame_id_pc = frame_id
            break
    matched_frame_id_pc = int2frame(matched_frame_id_pc)
    
    # print('matched frame ids pc: {}'.format(matched_frame_id_pc))
    return matched_frame_id_pc
### utils ###


def process_data(curr_frame_pc):

    ### init ###
    frame_interval = 500  # 0.5s
    his_frame_num = 4
    fut_frame_num = 6
    use_image = True

    # read all image frame ids
    images_dir = os.path.join(args.attack_dir, args.attack_scene, 'images')
    if not os.path.exists(images_dir):
        use_image = False
    if use_image:
        images = os.listdir(images_dir)
        images.sort()
        images = [frame2int(image) for image in images]

    # read all pointcloud frame ids
    pointclouds_dir = os.path.join(args.attack_dir, args.attack_scene, 'pointclouds')
    pointclouds = os.listdir(pointclouds_dir)
    pointclouds.sort()
    pointclouds = [frame2int(pointcloud) for pointcloud in pointclouds]
    
    # read all gps frame ids
    gps_lla_dir = os.path.join(args.attack_dir, args.attack_scene, 'gps_lla.json')
    with open(gps_lla_dir, 'r') as f:
        gps_llas = json.load(f)
    llas = list(gps_llas.keys())
    llas.sort()
    llas = [frame2int(lla) for lla in llas]
    ### init ###

    # obtained current frame list as the center frame
    curr_frame_pc_list = [frame2int(curr_frame_pc)]
    curr_frame_pc_idx = pointclouds.index(frame2int(curr_frame_pc))
    for i in range(curr_frame_pc_idx-1, curr_frame_pc_idx-3, -1):
        curr_frame_pc_list.append(pointclouds[i])
    for i in range(curr_frame_pc_idx+1, curr_frame_pc_idx+3):
        curr_frame_pc_list.append(pointclouds[i])
    curr_frame_pc_list.sort()

    for curr_frame_pc in curr_frame_pc_list:
        curr_frame_lla = find_closest_frame(curr_frame_pc, llas)
        if use_image:
            curr_frame_img = find_closest_frame(curr_frame_pc, images)
        print('current frame: ', int2frame(curr_frame_pc))
        
        # create output dir
        attack_dataset_dir = os.path.join(attack_scene_dir, 'dataset_{}'.format(int2frame(curr_frame_pc)))
        if not os.path.exists(attack_dataset_dir):
            os.makedirs(attack_dataset_dir)

        for subfolder in ['images', 'pointclouds', 'pointclouds_vis']:
            subfolder_dir = os.path.join(attack_dataset_dir, subfolder)
            if not os.path.exists(subfolder_dir):
                os.makedirs(subfolder_dir)

        # copy config file
        shutil.copy('/mnt/data/attack_pred_foxconn/config_template.yaml',
                    os.path.join(attack_dataset_dir, 'config.yaml'))

        frames_img = []
        frames_pc = []
        frames_lla = []

        # record the history frames
        for i in range(1, his_frame_num + 1):

            # point cloud
            temp_frame_pc = curr_frame_pc - i * frame_interval
            seconds = int(str(temp_frame_pc)[4:6])
            if seconds > 59:
                temp_frame_pc -= 40000
            temp_frame_pc = find_closest_frame(temp_frame_pc, pointclouds)
            frames_pc.append(temp_frame_pc)

            temp_frame_lla = find_closest_frame(temp_frame_pc, llas)
            frames_lla.append(temp_frame_lla)

            # corresponding image
            if use_image:
                temp_frame_img = find_closest_frame(temp_frame_pc, images)
                frames_img.append(temp_frame_img)
                # print(int2frame(temp_frame_img))

        frames_pc = [int2frame(frame) for frame in frames_pc]
        frames_pc.reverse()
        frames_pc.append(int2frame(curr_frame_pc))

        frames_lla = [int2frame(frame) for frame in frames_lla]
        frames_lla.reverse()
        frames_lla.append(int2frame(curr_frame_lla))

        if use_image:
            frames_img = [int2frame(frame) for frame in frames_img]
            frames_img.reverse()
            frames_img.append(int2frame(curr_frame_img))

        # record the future frames
        for i in range(1, fut_frame_num + 1):

            # point cloud
            temp_frame_pc = curr_frame_pc + i * frame_interval
            seconds = int(str(temp_frame_pc)[4:6])
            if seconds > 59:
                temp_frame_pc += 40000

            temp_frame_pc = find_closest_frame(temp_frame_pc, pointclouds)
            frames_pc.append(int2frame(temp_frame_pc))

            # corresponding gps
            temp_frame_lla = find_closest_frame(temp_frame_pc, llas)
            frames_lla.append(int2frame(temp_frame_lla))

            # corresponding image
            if use_image:
                temp_frame_img = find_closest_frame(temp_frame_pc, images)
                frames_img.append(int2frame(temp_frame_img))
                # print(int2frame(temp_frame_img))

        # construct dataset folder
        # set enu coordinates origin
        enu_coords = {}
        lat0, lon0, alt0 = gps_llas[int2frame(llas[0])]
        frames_lla_full = list(gps_llas.keys())
        
        # loop
        for i in range(len(frames_pc)):
            # point cloud
            # pointcloud = np.fromfile(os.path.join(pointclouds_dir, frames_pc[i] + '.bin'), dtype=np.float32).reshape(-1, 3)
            pointcloud = np.fromfile(os.path.join(pointclouds_dir, frames_pc[i] + '.bin'), dtype=np.float32).reshape(-1, 4)
            pointcloud = pointcloud[:, :3]
            # add intensity
            intensity = np.random.rand(pointcloud.shape[0], 1) * 0.3 + 0.4
            pointcloud = np.concatenate((pointcloud, intensity), axis=1)
            # copy point cloud file to dataset
            shutil.copy(os.path.join(pointclouds_dir, frames_pc[i] + '.bin'), 
                        os.path.join(attack_dataset_dir, 'pointclouds', frames_pc[i] + '.bin'))

            # lidar birdseye
            pc_bev_img = point_cloud_2_birdseye(pointcloud)
            im2 = Image.fromarray(pc_bev_img)
            im2.save(os.path.join(attack_dataset_dir, 'pointclouds_vis', frames_pc[i] + '.png'))

            # save point cloud to .bin file
            pointcloud = pointcloud.reshape(-1,)
            with open(os.path.join(attack_dataset_dir, 'pointclouds', frames_pc[i] + '.bin'), 'wb') as f:
                f.write(pointcloud.astype(np.float32).tobytes())

            # image
            if use_image:
                shutil.copy(os.path.join(images_dir, frames_img[i] + '.png'),
                            os.path.join(attack_dataset_dir, 'images', frames_img[i] + '.png'))
            
            # gps
            lat, lon, alt = gps_llas[frames_lla[i]]
            e, n, u = pm.geodetic2enu(lat, lon, alt, lat0, lon0, alt0)
            enu_coords[frames_pc[i]] = [e, n, u]

            # ego heading v2
            curr_idx = frames_lla_full.index(frames_lla[i])
            for idx in range(curr_idx - 1, 0, -1):
                temp_idx = frames_lla_full[idx]
                if gps_llas[temp_idx] != gps_llas[frames_lla[i]]:
                    prev_lat, prev_lon, prev_alt = gps_llas[temp_idx]
                    prev_e, prev_n, prev_u = pm.geodetic2enu(prev_lat, prev_lon, prev_alt, lat0, lon0, alt0)
                    break
            
            heading = calculate_heading(x1=prev_e, y1=prev_n, x2=e, y2=n)
            enu_coords[frames_pc[i]].append(heading)
            
            # # heading v1
            # if i != 0:
            #     heading = calculate_heading(x1=enu_coords[frames_pc[i-1]][0], y1=enu_coords[frames_pc[i-1]][1],
            #                 x2=enu_coords[frames_pc[i]][0], y2=enu_coords[frames_pc[i]][1])
            #     enu_coords[frames_pc[i-1]].append(heading)
            # # add heading to the last frame
            # if i == len(frames_pc) - 1:
            #     enu_coords[frames_pc[-1]].append(heading)

        # save enu coordinates
        enu_coords_path = os.path.join(attack_dataset_dir, 'enu_coords.json')
        with open(enu_coords_path, 'w') as f:
            json.dump(enu_coords, f)

        print('Saved {} frames: {}'.format(len(frames_pc), frames_pc))


def labels_to_global():

    # read all folders start with "dataset_" within attack_scene folder
    dataset_folders = os.listdir(attack_scene_dir)
    dataset_folders = [folder for folder in dataset_folders if folder.startswith('dataset_')]
    dataset_folders.sort()

    for attack_dataset_dir in dataset_folders:
        # read labels
        labels_path = os.path.join(attack_scene_dir, attack_dataset_dir, 'labels_lidar.json')
        with open(labels_path, 'r') as f:
            labels_lidar = json.load(f)

        # read enu coordinates
        enu_coords_path = os.path.join(attack_scene_dir, attack_dataset_dir, 'enu_coords.json')
        with open(enu_coords_path, 'r') as f:
            enu_coords = json.load(f)

        # get frame_ids
        frame_ids = list(labels_lidar.keys())
        frame_ids.sort()
        curr_frame_id = frame_ids[4]
        print('current frame: ', curr_frame_id)

        # Extract the current frame to set the global origin and orientation
        labels_global = {}

        for frame_id, lidar_data in labels_lidar.items():

            labels_global[frame_id] = {}

            # Extract position and heading for target and reference vehicles
            target_x, target_y, target_z, target_len, target_wid, target_ht, target_heading, _, _ = lidar_data
            ego_x, ego_y, ego_z, ego_heading = enu_coords[frame_id]

            ### ego vehicle ###
            # Store ego vehicle global pose (z need to be adjusted)
            labels_global[frame_id]['ego'] = [ego_x, ego_y, 0, 
                                            4, 1.7, 1.5, ego_heading]
            ### ego vehicle ###

            ### target vehicle ###
            target_x_global = ego_x + (target_x * np.cos(ego_heading) - target_y * np.sin(ego_heading))
            target_y_global = ego_y + (target_x * np.sin(ego_heading) + target_y * np.cos(ego_heading))

            # Adjust the heading to the global coordinate system for both target and ego vehicles
            target_heading_global = ego_heading + target_heading
            target_heading_global = (target_heading_global + np.pi) % (2 * np.pi) - np.pi

            # Store target global pose
            labels_global[frame_id]['target'] = [target_x_global, target_y_global, target_z, 
                                                target_len, target_wid, target_ht, target_heading_global]
            ### target vehicle ###

        # save global poses
        labels_global_path = os.path.join(attack_scene_dir, attack_dataset_dir, 'labels_global.json')
        with open(labels_global_path, 'w') as f:
            json.dump(labels_global, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='process')
    parser.add_argument('--benign_dir', type=str, default='bag2023.9.21')
    parser.add_argument('--attack_dir', type=str, default='bag2023.9.21')
    parser.add_argument('--benign_scene', type=str, default='scene_3_5kmh')
    parser.add_argument('--attack_scene', type=str, default='scene_3_5kmh')
    parser.add_argument('--curr_frame_img', type=str)  # find through images folder
    parser.add_argument('--curr_frame_pc', type=str)  # find through images folder
    parser.add_argument('--curr_frame_enu', type=str)  # find through images folder
    # parser.add_argument('--curr_frame_id', type=str, default='15-13-03-095')  # find through images folder
    args = parser.parse_args()

    # init paths
    benign_scene_dir = os.path.join(args.benign_dir, args.benign_scene)
    benign_dataset_dir = os.path.join(benign_scene_dir, 'dataset')

    attack_scene_dir = os.path.join(args.attack_dir, args.attack_scene)

    # launch
    if args.mode == 'process':
        curr_frame_pc = match()
        process_data(curr_frame_pc)
    elif args.mode == 'global':
        labels_to_global()
    else:
        raise ValueError('Invalid mode: {}'.format(args.mode))



