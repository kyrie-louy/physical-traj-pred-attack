import os
import csv
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from scipy.ndimage import rotate
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# Suppress warnings from matplotlib
import logging
logging.getLogger('matplotlib').setLevel(logging.ERROR)

robot = plt.imread('icons/Car TOP_VIEW 80CBE5.png')  # light blue
# robot = plt.imread('icons/Car TOP_VIEW 375397.png')
# adversary = plt.imread('icons/Car TOP_VIEW F05F78.png')
# adversary = plt.imread('icons/Car TOP_VIEW ROBOT.png')
adversary = plt.imread('icons/Car TOP_VIEW ABCB51.png')  # green
# adversary = plt.imread('icons/Car TOP_VIEW C8B0B0.png')  # brown


def rotate_trajectory(trajectory, angle):
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                [np.sin(angle), np.cos(angle)]])
    return np.dot(trajectory, rotation_matrix.T)


def add_alpha_channel(img, alpha=0.5):
    """ Add an alpha channel to the image with the given transparency. """
    if img.shape[2] == 3:  # Check if the image has 3 channels (RGB)
        h, w = img.shape[:2]
        new_img = np.zeros((h, w, 4), dtype=img.dtype)  # Create a new image with 4 channels (RGBA)
        new_img[:, :, :3] = img  # Copy the original image
        new_img[:, :, 3] = int(alpha * 255)  # Set the alpha channel
        return new_img
    elif img.shape[2] == 4:  # Check if the image has 4 channels (RGBA)
        img[:, :, 3] = int(alpha * 255)  # Modify the alpha channel
    
    return img  # Return the original image if it already has an alpha channel



def plot_realworld(vic_his_traj, vic_plan_traj, vic_cur_rot, 
                   adv_his_traj, adv_pred_traj, adv_cur_x, adv_cur_y,
                   output_path):
    
    # !!! need to specify vic_cur_rot same as the clean one
    # print(vic_cur_rot)
    if 'right' in output_path:
        vic_cur_rot = -0.476 # right clean
    elif 'left' in output_path:
        vic_cur_rot = -0.479  # left clean/attack
        # vic_cur_rot = -0.476  # left clean/attack
    else:
        raise ValueError('output_path must contain "right" or "left"')
    vehicle_size = 0.025

    # connect history and future trajectories
    vic_plan_traj = np.concatenate([vic_his_traj[-1:, :2], vic_plan_traj], axis=0)
    adv_pred_traj = np.concatenate([adv_his_traj[-1:, :2], adv_pred_traj], axis=0)
    
    # Rotate all trajectories so that victim vehicle's movement is parallel to y-axis
    rotation_angle = np.pi/2 - vic_cur_rot
    vic_his_traj = rotate_trajectory(vic_his_traj[:, :2], rotation_angle)
    vic_plan_traj = rotate_trajectory(vic_plan_traj[:, :2], rotation_angle)
    adv_his_traj = rotate_trajectory(adv_his_traj[:, :2], rotation_angle)
    adv_pred_traj = rotate_trajectory(adv_pred_traj[:, :2], rotation_angle)
    adv_cur_x, adv_cur_y = rotate_trajectory(np.array([[adv_cur_x, adv_cur_y]]), rotation_angle)[0]

    # additional for left/right direction
    direction_shift = 0.93
    if 'right_direction' in output_path:
        vic_his_traj[:, 0] += direction_shift
        vic_plan_traj[:, 0] += direction_shift
        adv_his_traj[:, 0] += direction_shift
        adv_pred_traj[:, 0] += direction_shift
        adv_cur_x += direction_shift
    elif 'left_direction' in output_path:
        vic_his_traj[:, 0] -= direction_shift
        vic_plan_traj[:, 0] -= direction_shift
        adv_his_traj[:, 0] -= direction_shift
        adv_pred_traj[:, 0] -= direction_shift
        adv_cur_x -= direction_shift
    
    # Plotting
    fig, ax = plt.subplots()

    ### Plot vehicles ###
    # Plot adversarial vehicle
    adv_img = rotate(adversary, 90, reshape=True) # 90 degrees for y-axis alignment
    oi_adv = OffsetImage(adv_img, zoom=vehicle_size) # half size
    # veh_box_adv = AnnotationBbox(oi_adv, (adv_his_traj[-1, 0], adv_his_traj[-1, 1]), frameon=False)
    veh_box_adv = AnnotationBbox(oi_adv, (adv_cur_x, adv_cur_y), frameon=False)
    veh_box_adv.zorder = 10  # Lower z-order
    ax.add_artist(veh_box_adv)

    # Plot victim vehicle
    vic_img = rotate(robot, 90, reshape=True) # 90 degrees for y-axis alignment
    oi_vic = OffsetImage(vic_img, zoom=vehicle_size)
    veh_box_vic = AnnotationBbox(oi_vic, (vic_his_traj[-1, 0], vic_his_traj[-1, 1]), frameon=False)
    veh_box_vic.zorder = 10  # Higher z-order
    ax.add_artist(veh_box_vic)
    ### Plot vehicles ###

    ax.text(adv_cur_x+1.5, adv_cur_y, 'Adversarial vehicle', fontsize=20, color='black', ha='center', va='center', zorder=20, path_effects=[pe.withStroke(linewidth=2, foreground='white')])
    ax.text(vic_his_traj[-1, 0]-1, vic_his_traj[-1, 1], 'Victim AV', fontsize=20, color='black', ha='center', va='center', zorder=20, path_effects=[pe.withStroke(linewidth=2, foreground='white')])
    # if 'left' in output_path and 'clean' in output_path:
    #     ax.text(adv_cur_x+1.5, adv_cur_y, 'Adversarial vehicle', fontsize=20, color='black', ha='center', va='center', zorder=20, path_effects=[pe.withStroke(linewidth=2, foreground='white')])
    #     ax.text(vic_his_traj[-1, 0]-1, vic_his_traj[-1, 1], 'Victim AV', fontsize=20, color='black', ha='center', va='center', zorder=20, path_effects=[pe.withStroke(linewidth=2, foreground='white')])
    # elif 'right' in output_path and 'clean' in output_path:
    #     ax.text(adv_cur_x-1.5, adv_cur_y, 'Adversarial vehicle', fontsize=20, color='black', ha='center', va='center', zorder=20, path_effects=[pe.withStroke(linewidth=2, foreground='white')])
    #     ax.text(vic_his_traj[-1, 0]+1, vic_his_traj[-1, 1], 'Victim AV', fontsize=20, color='black', ha='center', va='center', zorder=20, path_effects=[pe.withStroke(linewidth=2, foreground='white')])


    ### trajectories ###
    # Victim vehicle's history trajectory
    ax.plot(vic_his_traj[:, 0], vic_his_traj[:, 1], 'b-', label='Victim history', zorder=20)
    alphas = np.linspace(0.2, 0.8, vic_his_traj.shape[0])
    for i in range(vic_his_traj.shape[0]):
        ax.plot(vic_his_traj[i, 0], vic_his_traj[i, 1], 'bo', alpha=alphas[i], zorder=21)
    ax.plot(vic_his_traj[-1, 0], vic_his_traj[-1, 1], 'b*', label='Victim current', markersize=10, zorder=30)  # Blue star for victim

    # Victim vehicle's planned future trajectory
    ax.quiver(vic_plan_traj[:-1, 0], vic_plan_traj[:-1, 1],
              vic_plan_traj[1:, 0] - vic_plan_traj[:-1, 0],
              vic_plan_traj[1:, 1] - vic_plan_traj[:-1, 1],
              color='green', scale_units='xy', angles='xy', scale=1, label='Victim plan', linestyle='dashed', zorder=20)

    # Adversarial vehicle's history trajectory
    ax.plot(adv_his_traj[:, 0], adv_his_traj[:, 1], 'r-', label='Adv history', zorder=20)
    alphas = np.linspace(0.2, 0.8, adv_his_traj.shape[0])  # Gradual increase from 0.1 to 1
    for i in range(adv_his_traj.shape[0]):
        ax.plot(adv_his_traj[i, 0], adv_his_traj[i, 1], 'ro', alpha=alphas[i], zorder=21)
    ax.plot(adv_his_traj[-1, 0], adv_his_traj[-1, 1], 'r*', label='Adv current', markersize=15, zorder=30)  # Red star for adversary

    # Adversarial vehicle's predicted future trajectory
    ax.quiver(adv_pred_traj[:-1, 0], adv_pred_traj[:-1, 1],
              adv_pred_traj[1:, 0] - adv_pred_traj[:-1, 0],
              adv_pred_traj[1:, 1] - adv_pred_traj[:-1, 1],
              color='orange', scale_units='xy', angles='xy', scale=1, label='Adv predict', linestyle='dashed', zorder=20)
    ### trajectories ###

    ### Adjusting the plot limits ###
    all_x = np.concatenate([vic_his_traj[:, 0], vic_plan_traj[:, 0], 
                            adv_his_traj[:, 0], adv_pred_traj[:, 0], [adv_cur_x]])
    all_y = np.concatenate([vic_his_traj[:, 1], vic_plan_traj[:, 1], 
                            adv_his_traj[:, 1], adv_pred_traj[:, 1], [adv_cur_y]])

    x_min, x_max = np.min(all_x), np.max(all_x)
    y_min, y_max = np.min(all_y), np.max(all_y)

    # Adjust limits to center the victim vehicle and give more space on the adversarial vehicle side
    if 'left' in output_path:
        if 'right_direction' in output_path:
            fixed_x_min, fixed_x_max = -4, 2
        else:
            fixed_x_min, fixed_x_max = -4, 1
        # fixed_x_min, fixed_x_max = -4, 2  # right direction
        # fixed_x_min, fixed_x_max = -4, 0  # left direction
        fixed_y_min, fixed_y_max = 9, 23
    else:
        fixed_x_min, fixed_x_max = -1, 4
        fixed_y_min, fixed_y_max = 17, 32
    ax.set_xlim(fixed_x_min, fixed_x_max)
    ax.set_ylim(fixed_y_min, fixed_y_max)
    ### Adjusting the plot limits ###

    if 'direction' in output_path:
        # draw a line along x = 0 with union color and dashed line
        ax.plot([0, 0], [fixed_y_min, fixed_y_max], color='black', linestyle='dashed', label='orig direction', linewidth=4, zorder=0)

    # Set labels
    ax.set_xlabel('X (m)', fontsize=18)
    ax.set_ylabel('Y (m)', fontsize=18)

    # # set ticks
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(14)
    # plt.xticks(np.arange(int(fixed_x_min), fixed_x_max, 1))
    # plt.xticks([-4, -3, -2, -1, 0])

    # # # legend
    # if 'clean' in output_path:
    #     ax.legend(fontsize="18")
    # elif 'direction' in output_path:
    #     ax.legend(fontsize="16")
    # else:
    #     legend = ax.legend(fontsize="16")

    # # export legend
    # ax.grid(False)
    # def export_legend(legend, filename="legend.png"):
    #     fig  = legend.figure
    #     fig.canvas.draw()
    #     bbox  = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    #     fig.savefig(filename, dpi=1200, bbox_inches=bbox)
    # export_legend(legend)

    # Save the plot as pdf
    # ax.grid(False)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

# def calculate_heading(last_position, second_last_position):
#     # Calculate the heading angle in radians from the positions
#     delta_x = last_position[0] - second_last_position[0]
#     delta_y = last_position[1] - second_last_position[1]
#     heading = np.arctan2(delta_y, delta_x)
#     return heading


# def plot_realworld(vic_his_traj, vic_plan_traj, vic_cur_rot, adv_his_traj, adv_pred_traj, output_path):
#     # Plotting the trajectories
#     fig, ax = plt.subplots()

#     # Victim vehicle's history trajectory (blue solid line with circles)
#     ax.plot(vic_his_traj[:, 0], vic_his_traj[:, 1], 'bo-', label='Victim History')

#     # Victim vehicle's planned future trajectory (green dashed line with arrows)
#     ax.quiver(vic_plan_traj[:-1, 0], vic_plan_traj[:-1, 1],
#               vic_plan_traj[1:, 0] - vic_plan_traj[:-1, 0],
#               vic_plan_traj[1:, 1] - vic_plan_traj[:-1, 1],
#               color='green', scale_units='xy', angles='xy', scale=1, label='Victim Future', linestyle='dashed')
#     # ax.plot(vic_plan_traj[:, 0], vic_plan_traj[:, 1], 'go', markersize=3) # Additional points

#     # Adversarial vehicle's history trajectory (red solid line with circles)
#     ax.plot(adv_his_traj[:, 0], adv_his_traj[:, 1], 'ro-', label='Adversary History')

#     # Adversarial vehicle's predicted future trajectory (orange dashed line with arrows)
#     ax.quiver(adv_pred_traj[:-1, 0], adv_pred_traj[:-1, 1],
#               adv_pred_traj[1:, 0] - adv_pred_traj[:-1, 0],
#               adv_pred_traj[1:, 1] - adv_pred_traj[:-1, 1],
#               color='orange', scale_units='xy', angles='xy', scale=1, label='Adversary Future', linestyle='dashed')
#     # ax.plot(adv_pred_traj[:, 0], adv_pred_traj[:, 1], 'o', color='orange', markersize=3) # Additional points

#     # Plot victim vehicle
#     r_img = rotate(robot, vic_cur_rot * 180 / np.pi, reshape=True)
#     oi = OffsetImage(r_img, zoom=0.015, zorder=700)
#     veh_box = AnnotationBbox(oi, (vic_his_traj[-1, 0], vic_his_traj[-1, 1]), frameon=False)
#     veh_box.zorder = 700
#     ax.add_artist(veh_box)

#     # Plot adversarial vehicle
#     adv_img = rotate(adversary, vic_cur_rot * 180 / np.pi, reshape=True)
#     oi_adv = OffsetImage(adv_img, zoom=0.015, zorder=700)
#     adv_box = AnnotationBbox(oi_adv, (adv_his_traj[-1, 0], adv_his_traj[-1, 1]), frameon=False)
#     adv_box.zorder = 700
#     ax.add_artist(adv_box)

#     # Set labels and title
#     ax.set_xlabel('X Position')
#     ax.set_ylabel('Y Position')
#     ax.set_title('Real World Trajectories')
#     ax.legend()

#     # Save the plot
#     plt.savefig(output_path)
#     plt.close()
    

# Function to check if a row exists and to update or append data
def update_or_append_row(file_path, row_data):
    updated = False
    data = []

    # Read the existing data
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            reader = csv.reader(file)
            for r in reader:
                # Check if the row exists
                if r[0] == row_data[0] and r[1] == row_data[1] and r[2] == row_data[2]:
                    # Update the row
                    r[3] = row_data[3]
                    updated = True
                data.append(r)

    # Append new row if not updated
    if not updated:
        data.append(row_data)

    # Write data back to the file
    with open(file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)

