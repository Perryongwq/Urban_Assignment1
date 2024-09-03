import os
import numpy as np
from utils.compute_f import compute_step_positions

def create_magnetic_data(unix_time, x_pos, y_pos, m_strength):
    return {
        'unix_time': int(unix_time),
        'x_pos': float(x_pos),
        'y_pos': float(y_pos),
        'm_strength': float(m_strength)
    }

def get_waypoints(txt_path: str, xy_only=True):
    wp = []
    # parse text file
    with open(txt_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip()
            if not line or line[0] == '#':
                continue
            line_data = line.split('\t')
            if line_data[1] == 'TYPE_WAYPOINT':
                # Unix Time, X Pos, Y Pos
                wp.append([int(line_data[0]), float(line_data[2]), float(line_data[3])])

    wp = np.array(wp)

    return wp[:, 1:] if xy_only else wp

def augment_waypoints(waypoints, accelerometer_data, rotation_data):
    """
    Augments the waypoints using accelerometer and rotation vector data.
    """
    accelerometer_data = np.array(accelerometer_data)
    rotation_data = np.array(rotation_data)
    augmented_waypoints = compute_step_positions(accelerometer_data, rotation_data, waypoints)
    return augmented_waypoints

def extract_magnetic_positions(txt_path: str, augment=False):
    magnetic_data, waypoints = [], []
    accelerometer_data = []
    rotation_data = []

    with open(txt_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            line_data = line.split('\t')
            timestamp = int(line_data[0])

            if line_data[1] == 'TYPE_MAGNETIC_FIELD':
                # Unix Time, X axis, Y axis, Z axis
                magnetic_data.append([timestamp, float(line_data[2]), float(line_data[3]), float(line_data[4])])
            elif line_data[1] == 'TYPE_WAYPOINT':
                # Unix Time, X Pos, Y Pos
                waypoints.append([timestamp, float(line_data[2]), float(line_data[3])])
            elif augment:
                if line_data[1] == 'TYPE_ACCELEROMETER':
                    # Unix Time, X axis, Y axis, Z axis
                    accelerometer_data.append([timestamp, float(line_data[2]), float(line_data[3]), float(line_data[4])])
                elif line_data[1] == 'TYPE_ROTATION_VECTOR':
                    # Unix Time, X axis, Y axis, Z axis
                    rotation_data.append([timestamp, float(line_data[2]), float(line_data[3]), float(line_data[4])])

    magnetic_data, waypoints = np.array(magnetic_data), np.array(waypoints)
    original_magnetic_count = len(magnetic_data)
    
    if augment:
        waypoints = augment_waypoints(waypoints, accelerometer_data, rotation_data)

    waypoint_times = waypoints[:, 0]
    for magnetic_row in magnetic_data:
        closest_time_index = np.argmin(abs(waypoint_times - magnetic_row[0]))
        magnetic_row[0] = waypoint_times[closest_time_index]

    synchronized_waypoints = []
    for timestamp, x_pos, y_pos in waypoints:
        corresponding_magnetic_data = np.array([[m[1], m[2], m[3]] for m in magnetic_data if m[0] == timestamp])
        aggregated_data = [timestamp, x_pos, y_pos, 0.]
        
        if len(corresponding_magnetic_data) > 0:
            magnetic_strength = np.mean(np.sqrt(np.sum(corresponding_magnetic_data ** 2, axis=1)))
            aggregated_data[3] = magnetic_strength
        
        synchronized_waypoints.append(aggregated_data)
    
    waypoint_count = len(synchronized_waypoints)
    return synchronized_waypoints, original_magnetic_count, waypoint_count