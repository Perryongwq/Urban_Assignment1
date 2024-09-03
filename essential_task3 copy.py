import argparse
import json
import os
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

from data_processing import extract_magnetic_positions

""" CONFIG CONSTANTS """
RANDOM_SEED = 42
DATA_DIR = os.path.join(os.getcwd(), 'data')
OUTPUT_DIR = os.path.join(os.getcwd(), 'output')
PATH_DATA_DIR = 'path_data_files'
FLOOR_INFO_JSON_FILE = 'floor_info.json'
FLOOR_IMAGE_FILE = 'floor_image.png'
SAVE_IMG_DPI = 200
MAGNETIC_RANGE = None  # Set this as needed

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

def main():
    save_dir = os.path.join(OUTPUT_DIR, os.path.splitext(os.path.basename(__file__))[0])

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--save_dir', help='Save directory for output images', type=str, default=save_dir)
    parser.add_argument('--dpi', help='DPI of saved images', type=int, default=SAVE_IMG_DPI)
    parser.add_argument('-a', '--augment', dest='augment', action='store_true',
                        help='Toggle to augment waypoints.', default=True)

    args = vars(parser.parse_args())
    save_dir = str(args['save_dir'])
    dpi = int(args['dpi'])
    wp_augment = bool(args['augment'])

    if save_dir:
        create_dir(save_dir)

    for site, floor in get_site_floors(DATA_DIR):
        print(site, ' ------------ ', floor)
        visualize_geomagnetic(site, floor, save_dir, save_dpi=dpi, augment_wp=wp_augment, m_range=MAGNETIC_RANGE)


def visualize_geomagnetic(site, floor, save_dir=None, save_dpi=160, augment_wp=False, m_range=None):
    random.seed(RANDOM_SEED)
    floor_path = os.path.join(DATA_DIR, site, floor)

    # Parse magnetic data
    floor_data_path = os.path.join(floor_path, PATH_DATA_DIR)
    file_list = os.listdir(floor_data_path)
    floor_magnetic_data = np.zeros((0, 3))
    total_waypoints = 0
    total_mg = 0
    for filename in file_list:
        txt_path = os.path.join(floor_data_path, filename)
        magnetic_data, ori_mg_count, wp_count = extract_magnetic_positions(txt_path, augment_wp)
        total_mg += ori_mg_count
        total_waypoints += wp_count
        magnetic_wp_str = np.array(magnetic_data)[:, 1:4].astype(float)
        floor_magnetic_data = np.append(floor_magnetic_data, magnetic_wp_str, axis=0)

    # Read floor information to get map height, width
    json_path = os.path.join(floor_path, FLOOR_INFO_JSON_FILE)
    with open(json_path) as file:
        map_info = json.load(file)['map_info']
    map_height, map_width = map_info['height'], map_info['width']

    img = mpimg.imread(os.path.join(floor_path, FLOOR_IMAGE_FILE))
    reversed_color_map = plt.cm.get_cmap('inferno').reversed()

    plt.clf()
    plt.imshow(img)
    map_scaler = (img.shape[0] / map_height + img.shape[1] / map_width) / 2
    x = floor_magnetic_data[:, 0] * map_scaler
    y = img.shape[0] - floor_magnetic_data[:, 1] * map_scaler
    m_strength = floor_magnetic_data[:, 2]

    if m_range:
        plt.scatter(x, y, c=m_strength, s=10, vmin=m_range[0], vmax=m_range[1], cmap=reversed_color_map)
    else:
        plt.scatter(x, y, c=m_strength, s=10, cmap=reversed_color_map)
    plt.colorbar(cmap=reversed_color_map)
    plt.xticks((np.arange(25, map_width, 25) * map_scaler).astype('uint'),
               np.arange(25, map_width, 25).astype('uint'))
    plt.yticks((img.shape[0] - np.arange(25, map_height, 25) * map_scaler).astype('uint'),
               np.arange(25, map_height, 25).astype('uint'))

    if not augment_wp:
        plt.title(f"{site} - {floor} -- {total_mg} Mag: Ori {total_waypoints} Waypoints".title())
    else:
        plt.title(f"{site} - {floor} -- {total_mg} Mag: Aug {total_waypoints} Waypoints".title())

    if save_dir:
        if augment_wp:
            save_path = os.path.join(save_dir, site + "--" + floor)
        else:
            save_path = os.path.join(save_dir, site + "--" + floor + "--" + "O")
        plt.savefig(save_path, dpi=save_dpi)
    else:
        plt.show()


if __name__ == '__main__':
    main()
