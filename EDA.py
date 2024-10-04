import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from data_process_utils import read_data_file

def save_rssi_distribution(data, title, xlabel, filename):
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=50, color='skyblue', edgecolor='black')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

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
            waypoint_count = 0  

            for data_file in data_files:
                data_file_path = os.path.join(data_files_dir, data_file)
                sensor_data = read_data_file(data_file_path)

                # Extract Wi-Fi data
                if len(sensor_data['wifi']) > 0:
                    wifi_rssi = np.array([entry[3] for entry in sensor_data['wifi']])
                    all_wifi_rssi.extend(wifi_rssi)

                # Extract iBeacon data if available
                if len(sensor_data['ibeacon']) > 0:
                    ibeacon_rssi = np.array([entry[2] for entry in sensor_data['ibeacon']])
                    all_ibeacon_rssi.extend(ibeacon_rssi)

                # Extract magnetic field data
                if len(sensor_data['magn']) > 0:
                    magn_data = sensor_data['magn']
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


summary_df = pd.DataFrame(summary_data)
summary_csv_path = os.path.join(output_dir, 'summary_analysis.csv')
summary_df.to_csv(summary_csv_path, index=False)

print(f"All EDA results have been saved to the folder: {output_dir}")
print("Summary of data available for fingerprinting:")
print(summary_df)
