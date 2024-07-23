import os
import cv2
import json
import argparse
import numpy as np
import pymap3d as pm
from tqdm import tqdm
from datetime import datetime

import rosbag
from sensor_msgs import point_cloud2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


def timestamp_to_filename(timestamp, precision=3):

    timeStr = "%.{}f".format(precision) % timestamp
    dateTimeStr = int(timeStr[0:10])
    dotTimeStr = timeStr[11:]

    localTimeStr = datetime.fromtimestamp(dateTimeStr)
    localTimeStr = str(localTimeStr)
    localTimeStr = localTimeStr.split(' ')[1]
    localTimeStr = localTimeStr.replace(':', '-')
    
    filename = localTimeStr + '-' + dotTimeStr

    return filename


def export_images(bag_file, output_dir, image_topic):
    """Extract a folder of images from a rosbag.
    """

    print("Extract images from %s on topic %s into %s" % (bag_file,
                                                          image_topic, output_dir))

    bag = rosbag.Bag(bag_file, "r")
    bridge = CvBridge()
    count = 0
    for topic, msg, t in tqdm(bag.read_messages(topics=[image_topic])):

        cv_img = bridge.imgmsg_to_cv2(msg, "bgr8")

        # save
        filename = timestamp_to_filename(msg.header.stamp.to_sec())
        cv2.imwrite(os.path.join(output_dir, "{}.png".format(filename)), cv_img)
        # print("Wrote image {}".format(filename))

        count += 1
        # break

    bag.close()

    return


def export_pointcloud(bag_file, output_dir, topic_name):
    
    print("Extract pointclouds from %s on topic %s into %s" % (bag_file,
                                                        topic_name, output_dir))
    
    bag = rosbag.Bag(bag_file, 'r')
    cnt = 0
    
    # read
    for topic, msg, t in tqdm(bag.read_messages(topics=[topic_name])):

        # if cnt > 2:
        #     raise ValueError()

        # print(t)
        # gen = point_cloud2.read_points(msg, skip_nans=True, field_names=("x", "y", "z", "intensity"))
        gen = point_cloud2.read_points(msg, skip_nans=True, field_names=("x", "y", "z", "intensity"))
        point_cloud = np.array(list(gen))  # shape (*, 3)
        # yield point_cloud

        # save
        filename = timestamp_to_filename(msg.header.stamp.to_sec())
        point_cloud.astype(np.float32).tofile(os.path.join(output_dir, '{}.bin'.format(filename)))
        # print(f'Saved {filename}')
        cnt += 1

    bag.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--attack_dir', default='bag2023.9.21')
    parser.add_argument('--attack_scene', type=str)
    parser.add_argument('--image_topic', type=str, default=None)
    args = parser.parse_args()
    
    target_scene = args.attack_scene
    bag_path = os.path.join(args.attack_dir, target_scene, '{}.bag'.format(target_scene))
    
    ### export images ###
    if args.image_topic is not None:
        output_path = os.path.join(args.attack_dir, target_scene, 'images')
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        export_images(bag_path, output_path, args.image_topic)
    ### export images ###

    ### export point clouds ###
    output_dir = os.path.join(args.attack_dir, target_scene, 'pointclouds')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # export_pointcloud(bag_path, output_dir, '/sensing/lidar/top/pointcloud_raw_ex')
    export_pointcloud(bag_path, output_dir, '/sensing/lidar/top/outlier_filtered/pointcloud')
    ### export point clouds ###

    ### export gps lla ###
    bag = rosbag.Bag(bag_path, 'r')
    gps_topic = '/Sensor_msgs/INS/NavSatFix_BYINS_gnss'

    gps_lla = {}
    for topic, msg, t in bag.read_messages(topics=[gps_topic]):

        frame_id = timestamp_to_filename(msg.header.stamp.to_sec())
        gps_lla[frame_id] = [msg.latitude, msg.longitude, msg.altitude]

    with open(os.path.join(args.attack_dir, target_scene, 'gps_lla.json'), 'w') as f:
        json.dump(gps_lla, f)
    ### export gps lla ###

    # export complete enu coordinates to match current frame in benign and attack scenarios
    with open(os.path.join(args.attack_dir, target_scene, 'gps_lla.json'), 'r') as f:
        gps_lla = json.load(f)

    enu_coords = {}
    # lat0, lon0, alt0 = gps_lla['14-53-34-783']
    for cnt, (frame_id, lla) in enumerate(gps_lla.items()):
        
        if cnt == 0:
            lat0, lon0, alt0 = lla

        lat, lon, alt = lla
        e, n, u = pm.geodetic2enu(lat, lon, alt, lat0, lon0, alt0)
        enu_coords[frame_id] = [e, n, u]
    
    with open(os.path.join(args.attack_dir, target_scene, 'enu_coords.json'), 'w') as f:
        json.dump(enu_coords, f)
    

