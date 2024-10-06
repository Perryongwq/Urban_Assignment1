import os
import csv

def extract_data_from_file(file_path):
    magnetic_data = []
    wifi_signals = {}
    ibeacon_signals = {}
    ground_truth_location = []
    accelerometer_data = []
    gyroscope_data = []
    rotation_vector_data = []

    with open(file_path, 'r', encoding='utf-8') as file:  # Ensure UTF-8 encoding
        for line in file:
            line_data = line.split('\t')
            if len(line_data) < 2:
                continue  # Skip invalid or incomplete lines

            if line_data[1] == 'TYPE_MAGNETIC_FIELD':
                magnetic_data = [float(line_data[2]), float(line_data[3]), float(line_data[4])]
            
            elif line_data[1] == 'TYPE_WAYPOINT':
                ground_truth_location = [float(line_data[2]), float(line_data[3])]
            
            elif line_data[1] == 'TYPE_ACCELEROMETER':
                accelerometer_data = [float(line_data[2]), float(line_data[3]), float(line_data[4])]

            elif line_data[1] == 'TYPE_GYROSCOPE':
                gyroscope_data = [float(line_data[2]), float(line_data[3]), float(line_data[4])]
            
            elif line_data[1] == 'TYPE_ROTATION_VECTOR':
                rotation_vector_data = [float(line_data[2]), float(line_data[3]), float(line_data[4])]

            elif line_data[1].startswith('TYPE_WIFI'):
                try:
                    bssid = line_data[3] 
                    rssi = float(line_data[4]) 
                    wifi_signals[bssid] = rssi
                except (ValueError, IndexError):
                    print(f"Skipping invalid WiFi line: {line.strip()}")
            
            elif line_data[1].startswith('TYPE_IBEACON'):
                try:
                    uuid = line_data[3]  
                    rssi = float(line_data[4]) 
                    ibeacon_signals[uuid] = rssi
                except (ValueError, IndexError):
                    print(f"Skipping invalid iBeacon line: {line.strip()}")

    return {
        "magnetic_data": magnetic_data,
        "wifi_signals": wifi_signals,
        "ibeacon_signals": ibeacon_signals,
        "ground_truth_location": ground_truth_location,
        "accelerometer_data": accelerometer_data,
        "gyroscope_data": gyroscope_data,
        "rotation_vector_data": rotation_vector_data
    }

def process_all_files(directory):
    all_data = []
    for file_name in os.listdir(directory):
        file_path = os.path.join(directory, file_name)
        if file_name.endswith('.txt'):
            data = extract_data_from_file(file_path)
            all_data.append(data)
    return all_data

def save_data_to_csv(data, output_csv_file):
    # Define the headers for the CSV
    headers = [
        'Magnetic Data', 'WiFi Signals', 'iBeacon Signals', 
        'Ground Truth Location', 'Accelerometer Data', 
        'Gyroscope Data', 'Rotation Vector Data'
    ]
    
    # Open the CSV file for writing
    with open(output_csv_file, 'w', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=headers)
        writer.writeheader()

        for record in data:
            writer.writerow({
                'Magnetic Data': record['magnetic_data'],
                'WiFi Signals': record['wifi_signals'],
                'iBeacon Signals': record['ibeacon_signals'],
                'Ground Truth Location': record['ground_truth_location'],
                'Accelerometer Data': record['accelerometer_data'],
                'Gyroscope Data': record['gyroscope_data'],
                'Rotation Vector Data': record['rotation_vector_data']
            })

# Update your directory path accordingly
path_data_files_dir = 'C:/Users/perry/Documents/AI6128/Urban_Assignment1/data/site1/F1/path_data_files/'
output_csv_file = 'extracted_sensor_data.csv'

# Extract and process data
all_extracted_data = process_all_files(path_data_files_dir)

# Save the processed data into a CSV file
save_data_to_csv(all_extracted_data, output_csv_file)

print(f"Data has been successfully saved to {output_csv_file}")
