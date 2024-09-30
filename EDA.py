import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd  # New addition for table creation

# Step 2: Function to read sensor data files (no changes)
def read_data_file(data_filename):
    acce = []
    acce_uncali = []
    gyro = []
    gyro_uncali = []
    magn = []
    magn_uncali = []
    ahrs = []
    wifi = []
    ibeacon = []
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

        if line_data[1] == 'TYPE_ACCELEROMETER_UNCALIBRATED':
            acce_uncali.append([int(line_data[0]), float(line_data[2]), float(line_data[3]), float(line_data[4])])
            continue

        if line_data[1] == 'TYPE_GYROSCOPE':
            gyro.append([int(line_data[0]), float(line_data[2]), float(line_data[3]), float(line_data[4])])
            continue

        if line_data[1] == 'TYPE_GYROSCOPE_UNCALIBRATED':
            gyro_uncali.append([int(line_data[0]), float(line_data[2]), float(line_data[3]), float(line_data[4])])
            continue

        if line_data[1] == 'TYPE_MAGNETIC_FIELD':
            magn.append([int(line_data[0]), float(line_data[2]), float(line_data[3]), float(line_data[4])])
            continue

        if line_data[1] == 'TYPE_MAGNETIC_FIELD_UNCALIBRATED':
            magn_uncali.append([int(line_data[0]), float(line_data[2]), float(line_data[3]), float(line_data[4])])
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

        if line_data[1] == 'TYPE_BEACON':
            ts = line_data[0]
            uuid = line_data[2]
            major = line_data[3]
            minor = line_data[4]
            rssi = line_data[6]
            ibeacon_data = [ts, '_'.join([uuid, major, minor]), rssi]
            ibeacon.append(ibeacon_data)
            continue

        if line_data[1] == 'TYPE_WAYPOINT':
            waypoint.append([int(line_data[0]), float(line_data[2]), float(line_data[3])])

    acce = np.array(acce)
    acce_uncali = np.array(acce_uncali)
    gyro = np.array(gyro)
    gyro_uncali = np.array(gyro_uncali)
    magn = np.array(magn)
    magn_uncali = np.array(magn_uncali)
    ahrs = np.array(ahrs)
    wifi = np.array(wifi)
    ibeacon = np.array(ibeacon)
    waypoint = np.array(waypoint)

    return {
        "acce": acce,
        "acce_uncali": acce_uncali,
        "gyro": gyro,
        "gyro_uncali": gyro_uncali,
        "magn": magn,
        "magn_uncali": magn_uncali,
        "ahrs": ahrs,
        "wifi": wifi,
        "ibeacon": ibeacon,
        "waypoint": waypoint
    }

# Step 3: Function to save RSSI distributions for Wi-Fi and iBeacon signals (no changes)
def save_rssi_distribution(data, title, xlabel, filename):
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=50, color='skyblue', edgecolor='black')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

# Step 4: Process all files for all sites and floors and save the EDA results (with summary analysis)
output_dir = os.path.join(os.getcwd(), 'EDA_results')  # Create a folder for results
os.makedirs(output_dir, exist_ok=True)

summary_data = []  # To store summary data for each site/floor

src_path = os.path.dirname(__file__)
base_dir = os.path.join(src_path, 'data')

for site in os.listdir(base_dir):
    site_path = os.path.join(base_dir, site)
    
    for floor in os.listdir(site_path):
        floor_path = os.path.join(site_path, floor)
        data_files_dir = os.path.join(floor_path, 'path_data_files')
        
        if os.path.exists(data_files_dir):
            data_files = os.listdir(data_files_dir)
            
            all_wifi_rssi = []
            all_ibeacon_rssi = []
            all_magnetic_strength = []
            waypoint_count = 0  # Initialize waypoint count
            
            for data_file in data_files:
                data_file_path = os.path.join(data_files_dir, data_file)
                sensor_data = read_data_file(data_file_path)
                
                # Extract Wi-Fi data
                if len(sensor_data['wifi']) > 0:
                    wifi_rssi = np.array(sensor_data['wifi'])[:, 3].astype(float)
                    all_wifi_rssi.extend(wifi_rssi)
                
                # Extract iBeacon data if available
                if len(sensor_data['ibeacon']) > 0 and len(sensor_data['ibeacon'][0]) == 3:
                    ibeacon_rssi = np.array(sensor_data['ibeacon'])[:, 2].astype(float)
                    all_ibeacon_rssi.extend(ibeacon_rssi)
                
                # Extract magnetic field data
                if len(sensor_data['magn']) > 0:
                    magn_data = np.array(sensor_data['magn'])
                    magnetic_field_strength = np.sqrt(np.sum(magn_data[:, 1:] ** 2, axis=1))
                    all_magnetic_strength.extend(magnetic_field_strength)

                # Count waypoints
                waypoint_count += len(sensor_data['waypoint'])

            # Summarize the results for the current site and floor
            summary_data.append({
                'Site': site,
                'Floor': floor,
                'Magnetic Field': len(all_magnetic_strength),
                'Wi-Fi': len(all_wifi_rssi),
                'iBeacon': len(all_ibeacon_rssi),
                'Waypoints': waypoint_count
            })

            # Save RSSI and magnetic field distributions for this site/floor
            print(f"Processing EDA for Site: {site}, Floor: {floor}")
            
            if all_wifi_rssi:
                wifi_filename = os.path.join(output_dir, f"{site}_{floor}_wifi_rssi.png")
                save_rssi_distribution(all_wifi_rssi, f"Wi-Fi RSSI Distribution for {site}/{floor}", "RSSI (dBm)", wifi_filename)
            
            if all_ibeacon_rssi:
                ibeacon_filename = os.path.join(output_dir, f"{site}_{floor}_ibeacon_rssi.png")
                save_rssi_distribution(all_ibeacon_rssi, f"iBeacon RSSI Distribution for {site}/{floor}", "RSSI (dBm)", ibeacon_filename)
            
            if all_magnetic_strength:
                magnetic_filename = os.path.join(output_dir, f"{site}_{floor}_magnetic_strength.png")
                save_rssi_distribution(all_magnetic_strength, f"Magnetic Field Strength for {site}/{floor}", "Magnetic Strength (μT)", magnetic_filename)

# Convert summary data to a DataFrame and print or save the summary
summary_df = pd.DataFrame(summary_data)
summary_csv_path = os.path.join(output_dir, 'summary_analysis.csv')
summary_df.to_csv(summary_csv_path, index=False)

print(f"All EDA results have been saved to the folder: {output_dir}")
print("Summary of data available for fingerprinting:")
print(summary_df)
