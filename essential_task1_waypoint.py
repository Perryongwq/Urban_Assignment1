import os
import json
import random
import sys
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from utils.compute_f import compute_step_positions

# Existing imports from essential_task1.py
from data_processing import get_waypoints

""" CONFIG CONSTANTS """
RANDOM_SEED = 42
DATA_DIR = os.path.join(os.getcwd(), 'data')
OUTPUT_DIR = os.path.join(os.getcwd(), 'output')
PATH_DATA_DIR = 'path_data_files'
FLOOR_INFO_JSON_FILE = 'floor_info.json'
FLOOR_IMAGE_FILE = 'floor_image.png'
SAVE_IMG_DPI = 200

""" UTILITY FUNCTIONS """
def create_dir(directory: str) -> None:
    """Creates a directory if it doesn't exist."""
    os.makedirs(directory, exist_ok=True)

def get_site_floors(data_dir: str) -> list:
    """Retrieves a list of site floors from the given data directory."""
    site_floors = []
    for site in os.scandir(data_dir):
        if site.is_dir():
            site_name = site.name
            site_floors.extend((site_name, floor.name) for floor in os.scandir(site.path) if floor.is_dir())
    return site_floors

""" MAIN FUNCTIONALITY """
def get_data_from_one_txt(txtpath, augmentation=True):
    # This function handles the data extraction and augmentation
    acce = []
    magn = []
    ahrs = []
    wifi = []
    ibeacon = []
    waypoint = []

    with open(txtpath, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip()
            if not line or line[0] == '#':
                continue

            line_data = line.split('\t')

            if line_data[1] == 'TYPE_ACCELEROMETER':
                acce.append([int(line_data[0]), float(line_data[2]), float(line_data[3]), float(line_data[4])])
            elif line_data[1] == 'TYPE_MAGNETIC_FIELD':
                magn.append([int(line_data[0]), float(line_data[2]), float(line_data[3]), float(line_data[4])])
            elif line_data[1] == 'TYPE_ROTATION_VECTOR':
                ahrs.append([int(line_data[0]), float(line_data[2]), float(line_data[3]), float(line_data[4])])
            elif line_data[1] == 'TYPE_WIFI':
                sys_ts = line_data[0]
                bssid = line_data[3]
                rssi = line_data[4]
                wifi_data = [sys_ts, bssid, rssi]
                wifi.append(wifi_data)
            elif line_data[1] == 'TYPE_BEACON':
                ts = line_data[0]
                uuid = line_data[2]
                major = line_data[3]
                minor = line_data[4]
                rssi = line_data[6]
                ibeacon_data = [ts, '_'.join([uuid, major, minor]), rssi]
                ibeacon.append(ibeacon_data)
            elif line_data[1] == 'TYPE_WAYPOINT':
                waypoint.append([int(line_data[0]), float(line_data[2]), float(line_data[3])])

    acce, magn, ahrs, wifi, ibeacon, waypoint = np.array(acce), np.array(magn), np.array(ahrs), np.array(wifi), np.array(ibeacon), np.array(waypoint)

    if augmentation:
        augmented_data = compute_step_positions(acce, ahrs, waypoint)
    else:
        augmented_data = waypoint

    # Returning the augmented or original waypoint data
    return augmented_data[:, 1:3]  # Return only positions

def visualize_waypoints(site, floor, save_dir=None, save_dpi=160, wp_augment=False):
    random.seed(RANDOM_SEED)  # this ensures the color printed each time is the same
    floor_path = os.path.join(DATA_DIR, site, floor)

    floor_waypoints = []
    floor_data_path = os.path.join(floor_path, PATH_DATA_DIR)
    txt_filenames = os.listdir(floor_data_path)
    for filename in txt_filenames:
        txt_path = os.path.join(floor_data_path, filename)
        txt_waypoints = get_data_from_one_txt(txt_path, augmentation=wp_augment)
        floor_waypoints.append(txt_waypoints)

    # Read floor information to get map height and width
    json_path = os.path.join(floor_path, FLOOR_INFO_JSON_FILE)
    with open(json_path) as file:
        map_info = json.load(file)['map_info']
    map_height, map_width = map_info['height'], map_info['width']

    total_waypoints = 0
    img = mpimg.imread(os.path.join(floor_path, FLOOR_IMAGE_FILE))
    sns.reset_orig()

    # Choose colors based on whether waypoints are augmented
    if wp_augment:
        colors = sns.color_palette('bright', n_colors=10)  # Different color palette for augmented waypoints
    else:
        colors = sns.color_palette('dark', n_colors=10)

    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=colors)  # Set colors
    plt.clf()
    plt.imshow(img)
    map_scaler = (img.shape[0] / map_height + img.shape[1] / map_width) / 2

    for i, ways in enumerate(floor_waypoints):
        x, y = zip(*ways)
        total_waypoints += len(x)
        x, y = np.array(x), np.array(y)
        x, y = x * map_scaler, img.shape[0] - y * map_scaler
        plt.plot(x, y, linewidth='0.5', linestyle='-', marker='x', markersize=3)

    if not wp_augment:
        plt.title(f"{site} - {floor} - {total_waypoints} Waypoints".title())
    else:
        plt.title(f"{site} - {floor} - {total_waypoints} Augmented Waypoints".title())

    plt.xticks((np.arange(25, map_width, 25) * map_scaler).astype('uint'),
               np.arange(25, map_width, 25).astype('uint'))
    plt.yticks((img.shape[0] - np.arange(25, map_height, 25) * map_scaler).astype('uint'),
               np.arange(25, map_height, 25).astype('uint'))
    plt.tight_layout()

    if save_dir:
        if not wp_augment:
            save_path = os.path.join(save_dir, site + "--" + floor)
        else:
            save_path = os.path.join(save_dir, site + "--" + floor + "--" + "A")
        plt.savefig(save_path, dpi=save_dpi)
    else:
        plt.show()

    return total_waypoints


def main(save_dir=None, save_dpi=SAVE_IMG_DPI):
    if save_dir is None:
        save_dir = os.path.join(OUTPUT_DIR, os.path.splitext(os.path.basename(__file__))[0])
    
    print(f"Saving outputs to: {save_dir}")
    create_dir(save_dir)

    all_waypoints = {}
    for site, floor in get_site_floors(DATA_DIR):
        print(f"Processing site: {site}, floor: {floor}")
        # Process with and without augmentation
        wp_count = visualize_waypoints(site, floor, save_dir, save_dpi, wp_augment=False)
        aug_wp_count = visualize_waypoints(site, floor, save_dir, save_dpi, wp_augment=True)
        all_waypoints[(site, floor)] = {'original': wp_count, 'augmented': aug_wp_count}

    print(f"All waypoints processed: {all_waypoints}")
    print("COMPLETED")

if __name__ == '__main__':
    save_dir = sys.argv[1] if len(sys.argv) > 1 else None
    save_dpi = int(sys.argv[2]) if len(sys.argv) > 2 else SAVE_IMG_DPI
    main(save_dir, save_dpi)
