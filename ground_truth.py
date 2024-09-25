import os

# Function to process each file and extract magnetic data, WiFi signals, iBeacon signals, and ground truth location
def extract_data_from_file(file_path):
    magnetic_data = []
    wifi_signals = {}
    ibeacon_signals = {}
    ground_truth_location = []
    
    with open(file_path, 'r', encoding='utf-8') as file:  # Ensure UTF-8 encoding
        for line in file:
            line_data = line.split('\t')
            if line_data[1] == 'TYPE_MAGNETIC_FIELD':
                magnetic_data = [float(line_data[2]), float(line_data[3]), float(line_data[4])]
            elif line_data[1] == 'TYPE_WAYPOINT':
                ground_truth_location = [float(line_data[2]), float(line_data[3])]
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

    return magnetic_data, wifi_signals, ibeacon_signals, ground_truth_location

# sample_txt_file_path = 'C:/Users/perry/Documents/AI6128/Urban_Assignment1/data/site1/F1/path_data_files/5dd9e7aac5b77e0006b1732b.txt'
# sample_data = extract_data_from_file(sample_txt_file_path)
# print("Magnetic Data:", sample_data[0])
# print("WiFi Signals:", sample_data[1])
# print("iBeacon Signals:", sample_data[2])
# print("Ground Truth Location:", sample_data[3])



def process_all_files(directory):
    all_data = []
    for file_name in os.listdir(directory):
        file_path = os.path.join(directory, file_name)
        if file_name.endswith('.txt'):
            data = extract_data_from_file(file_path)
            all_data.append(data)
    return all_data


path_data_files_dir = 'C:/Users/perry/Documents/AI6128/Urban_Assignment1/data/site1/F1/path_data_files/'
all_extracted_data = process_all_files(path_data_files_dir)

# Display the first 5 extracted data entries (magnetic data, WiFi, iBeacon, and ground truth location)
for data in all_extracted_data[:5]:
    print("Magnetic Data:", data[0])
    print("WiFi Signals:", data[1])
    print("iBeacon Signals:", data[2])
    print("Ground Truth Location:", data[3])
