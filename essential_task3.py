import os
import json
import base64
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
from typing import Tuple
from utils.compute_f import compute_step_positions

# Configuration constants
RANDOM_SEED = 42
DATA_DIR = os.path.join(os.getcwd(), 'data')
OUTPUT_DIR = os.path.join(os.getcwd(), 'output')
PATH_DATA_DIR = 'path_data_files'
FLOOR_INFO_JSON_FILE = 'floor_info.json'
FLOOR_IMAGE_FILE = 'floor_image.png'
SAVE_IMG_DPI = 200

# Function to read data file
def read_data_file(data_filename: str, verbose: bool = False) -> Tuple[np.array, np.array, np.array, np.array]:
    with open(data_filename, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    acce_data, magn_data, ahrs_data, wayp_data = [], [], [], []

    for line_data in lines:
        line_data = line_data.strip()
        if not line_data or line_data[0] == '#':
            continue
        line_data = line_data.split('\t')

        if line_data[1] == 'TYPE_ACCELEROMETER':
            acce_data.append([int(line_data[0]), float(line_data[2]), float(line_data[3]), float(line_data[4])])
        elif line_data[1] == 'TYPE_MAGNETIC_FIELD':
            magn_data.append([int(line_data[0]), float(line_data[2]), float(line_data[3]), float(line_data[4])])
        elif line_data[1] == 'TYPE_ROTATION_VECTOR':
            ahrs_data.append([int(line_data[0]), float(line_data[2]), float(line_data[3]), float(line_data[4])])
        elif line_data[1] == 'TYPE_WAYPOINT':
            wayp_data.append([int(line_data[0]), float(line_data[2]), float(line_data[3])])

    if verbose:
        print(f"# of accelerometer data: {len(acce_data)}")
        print(f"# of geomagnetic data : {len(magn_data)}")
        print(f"# of rotation vect data: {len(ahrs_data)}")
        print(f"# of waypoints data   : {len(wayp_data)}")

    return np.array(acce_data), np.array(magn_data), np.array(ahrs_data), np.array(wayp_data)

# Function to map magnetic data to positions
def get_nearest_position_to_magn_data(step_pos: np.array, magn_data: np.array) -> pd.DataFrame:
    magn_data_df = pd.DataFrame(data=magn_data, columns=['timestamp', 'x', 'y', 'z'])
    step_pos_df = pd.DataFrame(data=step_pos, columns=['timestamp', 'x', 'y'])
    magn_data_df = magn_data_df.sort_values(by=['timestamp']).drop_duplicates(keep='first')
    mag_pos_datas = pd.merge_asof(magn_data_df, step_pos_df, on="timestamp", direction='nearest', suffixes=('', '_pos'))
    return mag_pos_datas

# Function to calculate magnetic strength
def calculate_magnetic_strength(mag_pos_datas: pd.DataFrame) -> pd.DataFrame:
    mag_pos_datas['magn_strength'] = np.sqrt(mag_pos_datas['x'] ** 2 + mag_pos_datas['y'] ** 2 + mag_pos_datas['z'] ** 2)
    avg_magn_strength = mag_pos_datas.groupby(['x_pos', 'y_pos'], as_index=False)['magn_strength'].mean()
    return avg_magn_strength

# Function to plot and save geomagnetic heatmap
def plot_and_save_geomagnetic_heatmap(mag_strength_pos_data: pd.DataFrame, path_dir: str, site_id: int,
                                      floor_id: str, augment: bool = True, save_img: bool = False):
    x_pos = mag_strength_pos_data['x_pos'].values
    y_pos = mag_strength_pos_data['y_pos'].values
    strength = mag_strength_pos_data['magn_strength'].values

    data = [go.Scatter(
        x=x_pos, y=y_pos,
        mode='markers',
        marker=dict(
            color=strength,
            colorbar=dict(title="μT"),
            colorscale="Rainbow",
            cmin=0,
            cmax=100,
            size=5,
        )
    )]

    with open(os.path.join(path_dir, FLOOR_INFO_JSON_FILE)) as f:
        floor_info = json.load(f)
    height_meter = floor_info['map_info']['height']
    width_meter = floor_info['map_info']['width']

    with open(os.path.join(path_dir, FLOOR_IMAGE_FILE), 'rb') as image_file:
        floor_plan = base64.b64encode(image_file.read()).decode('utf-8')
    
    img_bg = {
        'source': f'data:image/png;base64,{floor_plan}',
        'xref': 'x', 'yref': 'y',
        'x': 0, 'y': height_meter,
        'sizex': width_meter, 'sizey': height_meter,
        'sizing': 'contain', 'opacity': 1, 'layer': "below",
    }

    layout = go.Layout(
        xaxis=dict(autorange=False, range=[0, width_meter], showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(autorange=False, range=[0, height_meter], showticklabels=False, showgrid=False, zeroline=False,
                   scaleanchor='x', scaleratio=1),
        autosize=True,
        width=900,
        height=100 + 900 * height_meter / width_meter,
        template='plotly_white',
        images=[img_bg]
    )

    fig = go.Figure(data=data, layout=layout)

    if save_img:
        output_filepath = os.path.join(OUTPUT_DIR, f"site{site_id}")
        Path(output_filepath).mkdir(parents=True, exist_ok=True)
        file_name = f'{floor_id}.png' if augment else f'{floor_id}_unaugmented.png'
        fig.write_image(os.path.join(output_filepath, file_name))

# Function to generate geomagnetic heatmap
def get_geomagnetic_heatmap(site_id: int, floor_id: str, augment: bool = True, save_img: bool = False):
    path_dir = os.path.join(DATA_DIR, f'site{site_id}', floor_id)
    data_files_path_dir = os.path.join(path_dir, PATH_DATA_DIR)
    
    if not os.path.exists(data_files_path_dir):
        print(f"Directory does not exist: {data_files_path_dir}")
        return

    data_filenames = [f for f in os.listdir(data_files_path_dir) if os.path.isfile(os.path.join(data_files_path_dir, f))]

    mag_pos_datas = []
    for data_filename in data_filenames:
        data_filename = os.path.join(data_files_path_dir, data_filename)
        acce_data, magn_data, ahrs_data, wayp_data = read_data_file(data_filename)
        if augment:
            wayp_data = compute_step_positions(acce_data, ahrs_data, wayp_data)
        mag_pos_datas.append(get_nearest_position_to_magn_data(wayp_data, magn_data))

    mag_pos_data_df = pd.concat(mag_pos_datas, ignore_index=True)
    mag_strength_pos_data = calculate_magnetic_strength(mag_pos_data_df)

    plot_and_save_geomagnetic_heatmap(mag_strength_pos_data, path_dir, site_id, floor_id, augment, save_img)

# List of sites and floors
sites_and_floors = [
    ('site1', 'B1'), ('site1', 'F1'), ('site1', 'F2'), ('site1', 'F3'), ('site1', 'F4'),
    ('site2', 'B1'), ('site2', 'F1'), ('site2', 'F2'), ('site2', 'F3'), ('site2', 'F4'), 
    ('site2', 'F5'), ('site2', 'F6'), ('site2', 'F7'), ('site2', 'F8')
]

# Function to ensure the directory exists
def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        print(f"Directory does not exist: {directory}")
        return False
    return True

# Main processing loop
if __name__ == "__main__":
    for site, floor in sites_and_floors:
        print(f"Processing {site} - {floor}")
        path_dir = os.path.join(DATA_DIR, f"site{site[-1]}", floor)
        data_files_path_dir = os.path.join(path_dir, PATH_DATA_DIR)

        # Ensure that the directory exists before attempting to process it
        if not ensure_directory_exists(data_files_path_dir):
            print(f"Skipping {site} - {floor} due to missing directory.")
            continue

        save_path = os.path.join(OUTPUT_DIR, f"{site}--{floor}")
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)

        try:
            get_geomagnetic_heatmap(site[-1], floor, augment=True, save_img=True)
        except Exception as e:
            print(f"An error occurred while processing {site} - {floor}: {e}")

    print("Done")
