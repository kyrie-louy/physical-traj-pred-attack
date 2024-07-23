import math
import torch
import numpy as np


def calculate_board_boundaries(corner):
    """
    Calculate the minimum and maximum boundaries of the board given its corner coordinates.

    Parameters:
    corner (numpy array): A (4, 3) array containing the {x, y, z} coordinates of the board's 
                          top left, top right, bottom right, bottom left corners.

    Returns:
    tuple: Contains the min and max boundaries (min_x, max_x, min_y, max_y, min_z, max_z).
    """
    # Extract x, y, and z coordinates
    x_coords = corner[:, 0]
    y_coords = corner[:, 1]
    z_coords = corner[:, 2]

    # Calculate min and max for each coordinate
    min_x, max_x = torch.min(x_coords), torch.max(x_coords)
    min_y, max_y = torch.min(y_coords), torch.max(y_coords)
    min_z, max_z = torch.min(z_coords), torch.max(z_coords)

    return (min_x, max_x, min_y, max_y, min_z, max_z)


# def cornersraycast(corner):
#     #   simulate lidar pts based on board's corners, ray casting, 
#     #   corner:           (4,3), {x,y,z} of the board's top_left, top_right, bottom_right, bottom_left
    
#     # init board boundaries
#     board_min_x, board_max_x, board_min_y, board_max_y, board_min_z, board_max_z = calculate_board_boundaries(corner)

#     # calculate board plane parameters
#     xo1,yo1,zo1 = corner[0]
#     xo2,yo2,zo2 = corner[1]
#     xo3,yo3,zo3 = corner[2]
#     a = (yo2-yo1)*(zo3-zo1)-(zo2-zo1)*(yo3-yo1)
#     b = (xo3-xo1)*(zo2-zo1)-(xo2-xo1)*(zo3-zo1)
#     c = (xo2-xo1)*(yo3-yo1)-(xo3-xo1)*(yo2-yo1)
#     d = -(a*xo1+b*yo1+c*zo1)

#     # init ray casting parameters
#     # depending on your LiDAR angular resolution, only works for those angular resolution is evenly distributed
#     ver_res = math.radians(1)
#     hor_res = math.radians(0.4)
#     # hor_res = (np.pi / 180) * 0.24 #0.004
#     # ver_res = (np.pi / 180) * 0.4 #0.007
    
#     # ray casting
#     pts = []
#     for theta in np.arange(11*np.pi/24, 13*np.pi/24, ver_res): #elev, vertical angle range of LiDAR scanning
#         for phi in np.arange(-np.pi/5, np.pi/5, hor_res): #az, horizontal angle range of LiDAR scanning
#             m = np.sin(theta)*np.cos(phi) # x component
#             n = np.sin(theta)*np.sin(phi) # y component
#             p = np.cos(theta) # z component
#             t = (-a*m-b*n-c*p-d)/(a*m+b*n+c*p)
#             x,y,z = m*t+m,n*t+n,p*t+p

#             # check if the point is within the board boundaries
#             if board_min_x <= x <= board_max_x and board_min_y <= y <= board_max_y and board_min_z <= z <= board_max_z:
#                 pts.append([x,y,z])  # {x,y,z} is all you need, the coordiantes of each point
    
#     # add intensity
#     pts = np.array(pts)
#     rs = np.random.rand(pts.shape[0], 1) * 0.3 + 0.4
#     if pts.shape[0] == 0:
#         print('No points generated')
#         print(pts.shape)
#     pts = np.concatenate([pts, rs], axis=1)

#     return pts  # (N, 4), N is the number of points


def cornersraycast(corner):
    # Calculate board boundaries and plane parameters
    board_min_x, board_max_x, board_min_y, board_max_y, board_min_z, board_max_z = calculate_board_boundaries(corner)
    xo1, yo1, zo1 = corner[0]
    xo2, yo2, zo2 = corner[1]
    xo3, yo3, zo3 = corner[2]
    a = (yo2 - yo1) * (zo3 - zo1) - (zo2 - zo1) * (yo3 - yo1)
    b = (xo3 - xo1) * (zo2 - zo1) - (xo2 - xo1) * (zo3 - zo1)
    c = (xo2 - xo1) * (yo3 - yo1) - (xo3 - xo1) * (yo2 - yo1)
    d = -(a * xo1 + b * yo1 + c * zo1)

    # Ray casting parameters
    ver_res = torch.tensor(np.radians(1))
    hor_res = torch.tensor(np.radians(0.4))

    # Create arrays for theta and phi
    thetas = torch.arange(11 * np.pi / 24, 13 * np.pi / 24, ver_res)
    phis = torch.arange(-np.pi / 5, np.pi / 5, hor_res)

    # Meshgrid for vectorized computation
    theta_grid, phi_grid = torch.meshgrid(thetas, phis, indexing='ij')
    m = torch.sin(theta_grid) * torch.cos(phi_grid)
    n = torch.sin(theta_grid) * torch.sin(phi_grid)
    p = torch.cos(theta_grid)
    t = (-a * m - b * n - c * p - d) / (a * m + b * n + c * p)

    # Calculate points
    x = m * t + m
    y = n * t + n
    z = p * t + p

    # Filter points within the board boundaries
    valid_points = (board_min_x <= x) & (x <= board_max_x) & \
                   (board_min_y <= y) & (y <= board_max_y) & \
                   (board_min_z <= z) & (z <= board_max_z)

    pts = torch.stack([x[valid_points], y[valid_points], z[valid_points]], dim=-1)

    # Add intensity
    rs = torch.rand(pts.shape[0], 1) * 0.3 + 0.4
    pts = torch.cat([pts, rs], dim=1)

    return pts.detach().cpu().numpy()


def board_params2corners(adv_bd_list, N_pts, rot='yaw'):
    #   convert board parameters into board corners
    #   N_pts:           number of boards
    #   adv_bd_list:     (Npts, 6), each board's {x,y,z,h,w,theta}
    #   rot:             rotation axis, e.g., 'yaw' means the board is rotated along yaw axis, then theta is the inclination angle
    
    assert N_pts==len(adv_bd_list)
    
    adv_bd_list = torch.tensor(adv_bd_list)
    x,y,z,h,w,theta = torch.chunk(adv_bd_list, 6, dim=-1)  # 6 of (N_pts,1)
    cos_t = torch.cos(np.pi/2.0-theta)
    sin_t = torch.sin(np.pi/2.0-theta)
    
    if rot == 'yaw':
        top_left_x = x - w/2 * sin_t
        top_left_y = y + w/2 * cos_t
        top_left_z = z + h/2
        top_right_x = x + w/2 * sin_t
        top_right_y = y - w/2 * cos_t
        top_right_z = z + h/2
        bottom_right_x = x + w/2 * sin_t
        bottom_right_y = y - w/2 * cos_t
        bottom_right_z = z - h/2
        bottom_left_x = x - w/2 * sin_t
        bottom_left_y = y + w/2 * cos_t
        bottom_left_z = z - h/2
    elif rot == 'pitch':
        top_left_x = x + h/2 * cos_t
        top_left_y = y + w/2
        top_left_z = z + h/2 * sin_t
        top_right_x = x + h/2 * cos_t
        top_right_y = y - w/2
        top_right_z = z + h/2 * sin_t
        bottom_right_x = x - h/2 * cos_t
        bottom_right_y = y - w/2
        bottom_right_z = z - h/2 * sin_t
        bottom_left_x = x - h/2 * cos_t
        bottom_left_y = y + w/2
        bottom_left_z = z - h/2 * sin_t
    else:
        raise Exception('Unknown rotation type')
    
    adv_bd_corners = torch.cat([top_left_x,top_left_y,top_left_z,top_right_x,top_right_y,top_right_z,
                                    bottom_right_x,bottom_right_y,bottom_right_z,bottom_left_x,bottom_left_y,bottom_left_z],dim=-1)
    adv_bd_corners = adv_bd_corners.reshape(N_pts,4,3) #N_pts,4,3
    
    return adv_bd_corners

