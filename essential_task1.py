import json
import os
import random
import sys

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from data_processing import get_waypoints

""" CONFIG CONSTANTS """
RANDOM_SEED = 42
DATA_DIR = os.path.join(os.getcwd(), 'data')
OUTPUT_DIR =  os.path.join(os.getcwd(), 'output')
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

def main(save_dir=None, save_dpi=SAVE_IMG_DPI):
    if save_dir is None:
        save_dir = os.path.join(OUTPUT_DIR, os.path.splitext(os.path.basename(__file__))[0])
    
    print(f"Saving outputs to: {save_dir}")
    create_dir(save_dir)

    all_waypoints = {}
    for site, floor in get_site_floors(DATA_DIR):
        print(f"Processing site: {site}, floor: {floor}")
        wp_count = visualize_waypoints(site, floor, save_dir, save_dpi)
        all_waypoints[(site, floor)] = wp_count

    print(f"All waypoints processed: {all_waypoints}")
    print("COMPLETED")

def visualize_waypoints(site, floor, save_dir=None, save_dpi=160):
    random.seed(RANDOM_SEED)  # this ensures the color printed each time is the same
    floor_path = os.path.join(DATA_DIR, site, floor)

    floor_waypoints = []
    floor_data_path = os.path.join(floor_path, PATH_DATA_DIR)
    txt_filenames = os.listdir(floor_data_path)
    for filename in txt_filenames:
        txt_path = os.path.join(floor_data_path, filename)
        txt_waypoints = get_waypoints(txt_path, xy_only=True)
        floor_waypoints.append(txt_waypoints)

    # Read floor information to get map height and width
    json_path = os.path.join(floor_path, FLOOR_INFO_JSON_FILE)
    with open(json_path) as file:
        map_info = json.load(file)['map_info']
    map_height, map_width = map_info['height'], map_info['width']

    total_waypoints = 0
    img = mpimg.imread(os.path.join(floor_path, FLOOR_IMAGE_FILE))
    sns.reset_orig()
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

    title = f"{site} - {floor} - {total_waypoints} Waypoints"
    plt.title(title.title())

    plt.xticks((np.arange(25, map_width, 25) * map_scaler).astype('uint'),
               np.arange(25, map_width, 25).astype('uint'))
    plt.yticks((img.shape[0] - np.arange(25, map_height, 25) * map_scaler).astype('uint'),
               np.arange(25, map_height, 25).astype('uint'))
    plt.tight_layout()

    if save_dir:
        save_path = os.path.join(save_dir, f"{site}--{floor}")
        plt.savefig(save_path, dpi=save_dpi)
    else:
        plt.show()

    return total_waypoints

if __name__ == '__main__':
    save_dir = sys.argv[1] if len(sys.argv) > 1 else None
    save_dpi = int(sys.argv[2]) if len(sys.argv) > 2 else SAVE_IMG_DPI

    main(save_dir, save_dpi)
