import os
import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import random
from collections import defaultdict
from data import read_data_file

""" CONFIG CONSTANTS """
RANDOM_SEED = 42
DATA_DIR = os.path.join(os.getcwd(), 'data')
OUTPUT_DIR =  os.path.join(os.getcwd(), 'output')
PATH_DATA_DIR = 'path_data_files'
FLOOR_INFO_JSON_FILE = 'floor_info.json'
FLOOR_IMAGE_FILE = 'floor_image.png'
SAVE_IMG_DPI = 200

def wifi_print(site, floor, savepath=None, cbarange=None, selectmethod='random 3'):
    random.seed(1)  # this ensures the color printed each time is the same
    file_path = os.path.join(DATA_DIR, site, floor)

    # Load map information
    json_path = os.path.join(file_path, 'floor_info.json')
    with open(json_path) as file:
        mapinfo = json.load(file)['map_info']
    mapheight, mapwidth = mapinfo['height'], mapinfo['width']

    file_list = os.listdir(os.path.join(file_path, "path_data_files"))

    wifi_data = defaultdict(list)
    for filename in file_list:
        txtname = os.path.join(file_path, "path_data_files", filename)
        trajectory_data = read_data_file(txtname)
        for tdata in trajectory_data:
            px, py = tdata[1], tdata[2]
            timestamp_wifis = tdata[7]
            for bssid, rssi in timestamp_wifis.items():
                wifi_data[bssid].append((px, py, rssi))

    # Print out the number of Wi-Fi APs on this floor
    print(f'This floor has {len(wifi_data)} wifi APs')

    # Handle the 'random' selection method
    method_parts = selectmethod.split()
    if len(method_parts) == 2 and method_parts[0] == 'random':
        sample_number = int(method_parts[1])
        bssids = random.sample(wifi_data.keys(), sample_number)

        savedir = savepath if savepath else './WifiHeatMap'
        os.makedirs(savedir, exist_ok=True)

        for bssid in bssids:
            target_wifi_data = np.array(wifi_data[bssid])
            save_path = os.path.join(savedir, bssid.replace(':', '-'))

            img = mpimg.imread(os.path.join(file_path, 'floor_image.png'))
            plt.clf()
            plt.imshow(img)
            plt.title(f'Wifi: {bssid} ({len(wifi_data)} APs)')
            mapscaler = (img.shape[0] / mapheight + img.shape[1] / mapwidth) / 2
            x = target_wifi_data[:, 0] * mapscaler
            y = img.shape[0] - target_wifi_data[:, 1] * mapscaler
            rssi_intensity = target_wifi_data[:, 2]
            plt.scatter(x, y, c=rssi_intensity, s=10, vmin=cbarange[0], vmax=cbarange[1]) if cbarange else plt.scatter(x, y, c=rssi_intensity, s=10)
            plt.colorbar()
            plt.xticks((np.arange(25, mapwidth, 25) * mapscaler).astype('uint'), np.arange(25, mapwidth, 25).astype('uint'))
            plt.yticks((img.shape[0] - np.arange(25, mapheight, 25) * mapscaler).astype('uint'), np.arange(25, mapheight, 25).astype('uint'))
            plt.savefig(save_path, dpi=160)
    else:
        raise ValueError('Parameter selectmethod is not in the correct form.')

if __name__ == "__main__":
    save_dir = os.path.join(OUTPUT_DIR, os.path.splitext(os.path.basename(__file__))[0])
    os.makedirs(save_dir, exist_ok=True)

    sites_and_floors = [
        ('site1', 'B1'), ('site1', 'F1'), ('site1', 'F2'), ('site1', 'F3'), ('site1', 'F4'),
        ('site2', 'B1'), ('site2', 'F1'), ('site2', 'F2'), ('site2', 'F3'), ('site2', 'F4'), 
        ('site2', 'F5'), ('site2', 'F6'), ('site2', 'F7'), ('site2', 'F8')
    ]

    for site, floor in sites_and_floors:
        print(f"Processing {site} - {floor}")
        save_path = os.path.join(save_dir, f"{site}--{floor}")
        wifi_print(site, floor, save_path)
    
    print("Done")
