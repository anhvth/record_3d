import numpy as np
rear_intrinsics = dict(
    fx=711.66571045,
    fy=711.66571045,
    cx=353.33404541,
    cy=480.73724365234375,
)
# def depth_map_to_point_cloud(depth_map, rows, cols, cam='rear'):
#     if cam == 'rear':
#         fx = rear_intrinsics['fx']
#         fy = rear_intrinsics['fy']
#         cx = rear_intrinsics['cx']
#         cy = rear_intrinsics['cy']
#     # Create an empty point cloud
#     point_cloud = np.empty((rows, cols, 3), dtype=np.float32)

#     # For each pixel in the depth map
#     for y in range(rows):
#         for x in range(cols):
#             # Calculate the 3D coordinates of the pixel
#             z = depth_map[y, x]
#             # z is in range (0,1) and is its actually max value is 3
#             # get the actually z in mm
#             z = z * 3 * 1000

#             u = (x - cx) * z / fx
#             v = (y - cy) * z / fy

#             # Add the 3D coordinates to the point cloud
#             point_cloud[y, x, 0] = u
#             point_cloud[y, x, 1] = v
#             point_cloud[y, x, 2] = z

#     return point_cloud

def depth_map_to_point_cloud(depth_map, rows, cols, cam='rear'):
    if cam == 'rear':
        fx = rear_intrinsics['fx']
        fy = rear_intrinsics['fy']
        cx = rear_intrinsics['cx']
        cy = rear_intrinsics['cy']
    else:
        # Assume front camera if cam is not 'rear'
        fx = front_intrinsics['fx']
        fy = front_intrinsics['fy']
        cx = front_intrinsics['cx']
        cy = front_intrinsics['cy']

    # Create arrays to hold the x, y, and z coordinates
    x = np.empty((rows, cols), dtype=np.float32)
    y = np.empty((rows, cols), dtype=np.float32)
    z = np.empty((rows, cols), dtype=np.float32)

    # Calculate the 3D coordinates for each pixel
    x = (np.arange(cols) - cx) * depth_map / fx
    y = (np.arange(rows)[:, np.newaxis] - cy) * depth_map / fy
    z = depth_map 

    # Stack the x, y, and z arrays to create the point cloud
    point_cloud = np.stack((x, y, z), axis=-1)

    return point_cloud