import json
import os
from pathlib import Path
import plotly.graph_objs as go
from PIL import Image
import numpy as np
from utils.compute_f import split_ts_seq, compute_step_positions

def parse_file_ibeacon(path_data_files):
    ibeacons = []
    acceleration = []
    positions = []
    ahrs = []

    with open(path_data_files, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip()
            if line[0] == "#" or not line:
                continue

            line = line.split("\t")
            if line[1] == "TYPE_ACCELEROMETER":
                acceleration.append([int(line[0]), float(line[2]), float(line[3]), float(line[4])])
                continue

            if line[1] == "TYPE_BEACON":
                ts = line[0]
                uuid = line[2]
                major = line[3]
                minor = line[4]
                rssi = line[6]
                ibeacon_entry = [ts, f"{uuid}_{major}_{minor}", rssi]
                ibeacons.append(ibeacon_entry)
                continue

            if line[1] == "TYPE_WAYPOINT":
                positions.append([int(line[0]), float(line[2]), float(line[3])])
                continue

            if line[1] == "TYPE_ROTATION_VECTOR":
                ahrs.append([int(line[0]), float(line[2]), float(line[3]), float(line[4])])
                continue

    ibeacons = np.array(ibeacons)
    acceleration = np.array(acceleration)
    positions = np.array(positions)
    ahrs = np.array(ahrs)

    return {"ibeacon": ibeacons, "acce": acceleration, "waypoint": positions, "ahrs": ahrs}

def read_file_ibeacon(path_data_files):
    print(f"Reading {path_data_files}...")
    path_datas = parse_file_ibeacon(path_data_files)
    ibeacon_datas = path_datas["ibeacon"]
    acce_datas = path_datas["acce"]
    ahrs_datas = path_datas["ahrs"]
    posi_datas = path_datas["waypoint"]
    return ibeacon_datas, acce_datas, ahrs_datas, posi_datas

def get_ibeacon_to_position(ibeacon_datas, acce_datas, ahrs_datas, posi_datas, augment=False):
    ibeacon_extraction = {}
    step_positions = compute_step_positions(acce_datas, ahrs_datas, posi_datas)
    if not augment:
        step_positions = posi_datas

    if ibeacon_datas.size > 0:
        ts_list = np.unique(ibeacon_datas[:, 0].astype(float))
        ibeacon_data_list = split_ts_seq(ibeacon_datas, ts_list)
        for ibeacon_data in ibeacon_data_list:
            diff = np.abs(step_positions[:, 0] - float(ibeacon_data[0, 0]))
            index = np.argmin(diff)
            target_xy_key = tuple(step_positions[index, 1:3])
            if target_xy_key in ibeacon_extraction:
                ibeacon_extraction[target_xy_key] = np.append(ibeacon_extraction[target_xy_key], ibeacon_data, axis=0)
            else:
                ibeacon_extraction[target_xy_key] = ibeacon_data

    return ibeacon_extraction

def get_axes_values(ibeacon_extraction):
    ibeacon_rssi = {}
    for key in ibeacon_extraction.keys():
        for ibeacon_d in ibeacon_extraction[key]:
            ummid = ibeacon_d[1]
            rssi = int(ibeacon_d[2])
            if ummid in ibeacon_rssi:
                position_rssi = ibeacon_rssi[ummid]
                if key in position_rssi:
                    old_rssi = position_rssi[key][0]
                    old_count = position_rssi[key][1]
                    position_rssi[key][0] = (old_rssi * old_count + rssi) / (old_count + 1)
                    position_rssi[key][1] = old_count + 1
                else:
                    position_rssi[key] = np.array([rssi, 1])
            else:
                position_rssi = {}
                position_rssi[key] = np.array([rssi, 1])
            ibeacon_rssi[ummid] = position_rssi

    return ibeacon_rssi

def draw_heatmap(position, value, floor_plan_filename, width, height, title, filename):
    fig = go.Figure()

    # add heat map
    fig.add_trace(
        go.Scatter(x=position[:, 0],
                   y=position[:, 1],
                   mode='markers',
                   marker=dict(size=7,
                               color=value,
                               colorbar=dict(title="dBm"),
                               colorscale="Rainbow"),
                   text=value,
                   name=title))

    # add image
    floor_plan = Image.open(floor_plan_filename)
    fig.update_layout(images=[
        go.layout.Image(
            source=floor_plan,
            xref="x",
            yref="y",
            x=0,
            y=height,
            sizex=width,
            sizey=height,
            sizing="contain",
            opacity=1,
            layer="below",
        )
    ])

    # configure
    fig.update_xaxes(autorange=False, range=[0, width])
    fig.update_yaxes(autorange=False, range=[0, height], scaleanchor="x", scaleratio=1)
    fig.update_layout(
        title=go.layout.Title(
            text=title or "No title.",
            xref="paper",
            x=0,
        ),
        autosize=True,
        width=900,
        height=200 + 900 * height / width,
        template="plotly_white",
    )

    fig.write_image(filename)

if __name__ == "__main__":
    # Define the directories for the sites and floors
    floor_dirs = [
        "data/site1/B1", "data/site1/F1", "data/site1/F2", "data/site1/F3", "data/site1/F4",
        "data/site2/B1", "data/site2/F1", "data/site2/F2", "data/site2/F3", 
        "data/site2/F4", "data/site2/F5", "data/site2/F6", "data/site2/F7", "data/site2/F8"
    ]

    # Process each floor
    for floor_dir in floor_dirs:
        floor_plan_filename = os.path.join(floor_dir, "floor_image.png")
        floor_info_filename = os.path.join(floor_dir, "floor_info.json")
        path_data_dir = os.path.join(floor_dir, "path_data_files")
        output_dir = os.path.join("output", floor_dir.split("data")[1], "ibeacons")

        with open(floor_info_filename) as f:
            floor_info = json.load(f)
        width_meter = floor_info["map_info"]["width"]
        height_meter = floor_info["map_info"]["height"]

        for data_file in os.listdir(path_data_dir):
            ibeacon_datas, acce_datas, ahrs_datas, posi_datas = read_file_ibeacon(
                os.path.join(path_data_dir, data_file)
            )
            ibeacon_extractions = get_ibeacon_to_position(ibeacon_datas, acce_datas, ahrs_datas, posi_datas)
            ibeacon_rssi = get_axes_values(ibeacon_extractions)
            print(f'This file has {len(ibeacon_rssi.keys())} ibeacons')

            for target_ibeacon in ibeacon_rssi.keys():
                heat_positions = np.array(list(ibeacon_rssi[target_ibeacon].keys()))
                heat_values = np.array(list(ibeacon_rssi[target_ibeacon].values()))[:, 0]
                output_filename = os.path.join(output_dir, f"{target_ibeacon}.png")
                Path(output_dir).mkdir(parents=True, exist_ok=True)
                draw_heatmap(heat_positions, heat_values, floor_plan_filename, width_meter,
                             height_meter, f'iBeacon: {target_ibeacon} RSSI', output_filename)
