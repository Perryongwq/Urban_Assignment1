import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from PIL import Image
from utils.compute_f import split_ts_seq, compute_step_positions

""" CONFIG CONSTANTS """
RANDOM_SEED = 42
DATA_DIR = os.path.join(os.getcwd(), 'data')
OUTPUT_DIR = os.path.join(os.getcwd(), 'output')
PATH_DATA_DIR = 'path_data_files'
FLOOR_INFO_JSON_FILE = 'floor_info.json'
FLOOR_IMAGE_FILE = 'floor_image.png'
SAVE_IMG_DPI = 200

@dataclass
class ReadData:
    acce: np.ndarray
    ahrs: np.ndarray
    wifi: np.ndarray
    waypoint: np.ndarray

def read_data_file(data_filename):
    acce = []
    ahrs = []
    wifi = []
    waypoint = []
    with open(data_filename, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for line_data in lines:
            line_data = line_data.strip()
            if not line_data or line_data[0] == '#':
                continue
            line_data = line_data.split('\t')
            if line_data[1] == 'TYPE_ACCELEROMETER':
                acce.append([int(line_data[0]), float(line_data[2]), float(line_data[3]), float(line_data[4])])
                continue
            if line_data[1] == 'TYPE_ROTATION_VECTOR':
                ahrs.append([int(line_data[0]), float(line_data[2]), float(line_data[3]), float(line_data[4])])
                continue
            if line_data[1] == 'TYPE_WIFI':
                sys_ts = line_data[0]
                ssid = line_data[2]
                bssid = line_data[3]
                rssi = line_data[4]
                lastseen_ts = line_data[6]
                wifi_data = [sys_ts, ssid, bssid, rssi, lastseen_ts]
                wifi.append(wifi_data)
                continue
            if line_data[1] == 'TYPE_WAYPOINT':
                waypoint.append([int(line_data[0]), float(line_data[2]), float(line_data[3])])
    acce = np.array(acce)
    ahrs = np.array(ahrs)
    wifi = np.array(wifi)
    waypoint = np.array(waypoint)
    return ReadData(acce, ahrs, wifi, waypoint)

def get_wifis_by_position(path_file_list):
    pos_wifi_datas = {}
    for path_filename in path_file_list:
        print(f'Processing {path_filename}...')
        path_datas = read_data_file(path_filename)
        acce_datas = path_datas.acce
        ahrs_datas = path_datas.ahrs
        wifi_datas = path_datas.wifi
        posi_datas = path_datas.waypoint
        step_positions = compute_step_positions(acce_datas, ahrs_datas, posi_datas)
        
        if wifi_datas.size != 0:
            sep_tss = np.unique(wifi_datas[:, 0].astype(float))
            wifi_datas_list = split_ts_seq(wifi_datas, sep_tss)
            for wifi_ds in wifi_datas_list:
                diff = np.abs(step_positions[:, 0] - float(wifi_ds[0, 0]))
                index = np.argmin(diff)
                target_xy_key = tuple(step_positions[index, 1:3])
                if target_xy_key in pos_wifi_datas:
                    pos_wifi_datas[target_xy_key] = np.append(pos_wifi_datas[target_xy_key], wifi_ds, axis=0)
                else:
                    pos_wifi_datas[target_xy_key] = wifi_ds
    return pos_wifi_datas

def extract_wifi_rssi(pos_wifi_datas):
    wifi_rssi = {}
    for position_key in pos_wifi_datas:
        wifi_data = pos_wifi_datas[position_key]
        for wifi_d in wifi_data:
            bssid = wifi_d[2]
            rssi = int(wifi_d[3])
            if bssid in wifi_rssi:
                position_rssi = wifi_rssi[bssid]
                if position_key in position_rssi:
                    old_rssi = position_rssi[position_key][0]
                    old_count = position_rssi[position_key][1]
                    position_rssi[position_key][0] = (old_rssi * old_count + rssi) / (old_count + 1)
                    position_rssi[position_key][1] = old_count + 1
                else:
                    position_rssi[position_key] = np.array([rssi, 1])
            else:
                position_rssi = {}
                position_rssi[position_key] = np.array([rssi, 1])
            wifi_rssi[bssid] = position_rssi
    return wifi_rssi

def wifi_print(site, floor, savepath=None, cbarange=None, selectmethod='random 3'):
    random.seed(RANDOM_SEED)  # Ensure consistent random selection
    file_path = os.path.join(DATA_DIR, site, floor)

    # Load map information
    json_path = os.path.join(file_path, FLOOR_INFO_JSON_FILE)
    with open(json_path) as file:
        mapinfo = json.load(file)['map_info']
    mapheight, mapwidth = mapinfo['height'], mapinfo['width']

    file_list = os.listdir(os.path.join(file_path, PATH_DATA_DIR))

    pos_wifi_datas = get_wifis_by_position([os.path.join(file_path, PATH_DATA_DIR, f) for f in file_list])
    wifi_rssi = extract_wifi_rssi(pos_wifi_datas)

    # Print out the number of Wi-Fi APs on this floor
    print(f'This floor has {len(wifi_rssi)} Wi-Fi APs')

    method_parts = selectmethod.split()
    if len(method_parts) == 2 and method_parts[0] == 'random':
        sample_number = int(method_parts[1])
        bssids = random.sample(wifi_rssi.keys(), sample_number)

        savedir = savepath if savepath else './WifiHeatMap'
        os.makedirs(savedir, exist_ok=True)

        for bssid in bssids:
            heat_positions = np.array(list(wifi_rssi[bssid].keys()))
            heat_values = np.array(list(wifi_rssi[bssid].values()))[:, 0]

            save_path = os.path.join(savedir, bssid.replace(':', '-'))

            img = mpimg.imread(os.path.join(file_path, FLOOR_IMAGE_FILE))
            plt.clf()
            plt.imshow(img)
            plt.title(f'Wi-Fi: {bssid} ({len(wifi_rssi)} APs)')
            mapscaler = (img.shape[0] / mapheight + img.shape[1] / mapwidth) / 2
            x = heat_positions[:, 0] * mapscaler
            y = img.shape[0] - heat_positions[:, 1] * mapscaler
            plt.scatter(x, y, c=heat_values, s=10, vmin=cbarange[0], vmax=cbarange[1]) if cbarange else plt.scatter(x, y, c=heat_values, s=10)
            plt.colorbar()
            plt.xticks((np.arange(25, mapwidth, 25) * mapscaler).astype('uint'), np.arange(25, mapwidth, 25).astype('uint'))
            plt.yticks((img.shape[0] - np.arange(25, mapheight, 25) * mapscaler).astype('uint'), np.arange(25, mapheight, 25).astype('uint'))
            plt.savefig(save_path, dpi=SAVE_IMG_DPI)
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
