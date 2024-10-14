import os
import numpy as np
from collections import defaultdict
import random
from data_process_utils import read_data_file  

def get_data_from_one_txt(txt_path):
    sensor_data = read_data_file(txt_path)

    acce = sensor_data['acce']
    magn = sensor_data['magn']
    ahrs = sensor_data['ahrs']
    wifi = sensor_data['wifi']
    ibeacon = sensor_data['ibeacon']
    waypoint = sensor_data['waypoint']

    index2data = [{'magn': [], 'wifi': defaultdict(list), 'ibeacon': defaultdict(list)} for _ in waypoint]
    index2time = waypoint[:, 0]

    for magn_data in magn:
        tdiff = np.abs(index2time - magn_data[0])
        idx = np.argmin(tdiff)
        index2data[idx]['magn'].append(magn_data[1:])

    for wifi_data in wifi:
        tdiff = np.abs(index2time - wifi_data[0])
        idx = np.argmin(tdiff)
        index2data[idx]['wifi'][wifi_data[2]].append(wifi_data[3])  # wifi_data[2]: BSSID, wifi_data[3]: RSSI

    for ibeacon_data in ibeacon:
        tdiff = np.abs(index2time - ibeacon_data[0])
        idx = np.argmin(tdiff)
        index2data[idx]['ibeacon'][ibeacon_data[1]].append(ibeacon_data[2])  # ibeacon_data[1]: UUID, ibeacon_data[2]: RSSI

    txt_data = []
    for idx, (t, px, py) in enumerate(waypoint):
        magn_values = index2data[idx]['magn']
        if magn_values:
            magn_mean = np.mean(magn_values, axis=0)
            magn_intensity = np.mean(np.linalg.norm(magn_values, axis=1))
        else:
            magn_mean = [0, 0, 0]
            magn_intensity = 0

        wifi_signals = defaultdict(lambda: -100)
        for bssid, rssis in index2data[idx]['wifi'].items():
            wifi_signals[bssid] = np.mean(rssis)

        ibeacon_signals = defaultdict(lambda: -100)
        for uuid, rssis in index2data[idx]['ibeacon'].items():
            ibeacon_signals[uuid] = np.mean(rssis)

        txt_data.append([t, px, py] + list(magn_mean) + [magn_intensity, wifi_signals, ibeacon_signals])

    return txt_data

def split_floor_data(site, floor, testratio=0.1):
    file_path = os.path.join('./data', site, floor)
    txt_files = os.listdir(os.path.join(file_path, "path_data_files"))
    trajectory_data = []

    for txt_file in txt_files:
        txt_path = os.path.join(file_path, "path_data_files", txt_file)
        trajectory_data.extend(get_data_from_one_txt(txt_path))

    trajectory_data = np.array(trajectory_data)
    total_samples = len(trajectory_data)
    indices = list(range(total_samples))
    random.shuffle(indices)
    split_point = int(total_samples * (1 - testratio))
    train_indices = indices[:split_point]
    test_indices = indices[split_point:]

    bssid2index, uuid2index = {}, {}
    train_data, test_data = [], []

    def process_data(indices, is_train=True):
        data_list = []
        for idx in indices:
            sample = trajectory_data[idx]
            px, py = sample[1:3]
            magn_features = sample[3:7]
            wifi_signals = sample[7]
            ibeacon_signals = sample[8]

            wifi_vector = []
            if wifi_signals:
                for bssid in wifi_signals.keys():
                    if is_train and bssid not in bssid2index:
                        bssid2index[bssid] = len(bssid2index)
                    if bssid in bssid2index:
                        wifi_vector.append((bssid2index[bssid], (100 + wifi_signals[bssid]) / 100))
            else:
                wifi_vector.append((-1, 0))

            ibeacon_vector = []
            if ibeacon_signals:
                for uuid in ibeacon_signals.keys():
                    if is_train and uuid not in uuid2index:
                        uuid2index[uuid] = len(uuid2index)
                    if uuid in uuid2index:
                        ibeacon_vector.append((uuid2index[uuid], (100 + ibeacon_signals[uuid]) / 100))
            else:
                ibeacon_vector.append((-1, 0))

            data_list.append((px, py, *magn_features, wifi_vector, ibeacon_vector))
        return data_list

    train_data = process_data(train_indices, is_train=True)
    test_data = process_data(test_indices, is_train=False)

    # Convert data lists to numpy arrays
    def convert_to_numpy(data_list, bssid2index_len, uuid2index_len):
        data_array = []
        for item in data_list:
            px, py, *features, wifi_vector, ibeacon_vector = item
            wifi_features = np.zeros(bssid2index_len)
            for idx, value in wifi_vector:
                if idx >= 0:
                    wifi_features[idx] = value
            ibeacon_features = np.zeros(uuid2index_len)
            for idx, value in ibeacon_vector:
                if idx >= 0:
                    ibeacon_features[idx] = value
            data_array.append([px, py] + features + wifi_features.tolist() + ibeacon_features.tolist())
        return np.array(data_array)

    train_set = convert_to_numpy(train_data, len(bssid2index), len(uuid2index))
    test_set = convert_to_numpy(test_data, len(bssid2index), len(uuid2index))

    return train_set, test_set, (bssid2index, uuid2index)
