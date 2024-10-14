import numpy as np

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
        for line_data in file:
            line_data = line_data.strip()
            if not line_data or line_data[0] == '#':
                continue

            line_data = line_data.split('\t')
            timestamp = int(line_data[0])
            data_type = line_data[1]

            if data_type == 'TYPE_ACCELEROMETER':
                acce.append([timestamp, float(line_data[2]), float(line_data[3]), float(line_data[4])])

            elif data_type == 'TYPE_ACCELEROMETER_UNCALIBRATED':
                acce_uncali.append([timestamp, float(line_data[2]), float(line_data[3]), float(line_data[4])])

            elif data_type == 'TYPE_GYROSCOPE':
                gyro.append([timestamp, float(line_data[2]), float(line_data[3]), float(line_data[4])])

            elif data_type == 'TYPE_GYROSCOPE_UNCALIBRATED':
                gyro_uncali.append([timestamp, float(line_data[2]), float(line_data[3]), float(line_data[4])])

            elif data_type == 'TYPE_MAGNETIC_FIELD':
                magn.append([timestamp, float(line_data[2]), float(line_data[3]), float(line_data[4])])

            elif data_type == 'TYPE_MAGNETIC_FIELD_UNCALIBRATED':
                magn_uncali.append([timestamp, float(line_data[2]), float(line_data[3]), float(line_data[4])])

            elif data_type == 'TYPE_ROTATION_VECTOR':
                ahrs.append([timestamp, float(line_data[2]), float(line_data[3]), float(line_data[4])])

            elif data_type == 'TYPE_WIFI':
                wifi.append([timestamp, line_data[2], line_data[3], float(line_data[4]), int(line_data[6])])

            elif data_type == 'TYPE_BEACON':
                ibeacon.append([timestamp, '_'.join(line_data[2:5]), float(line_data[6])])

            elif data_type == 'TYPE_WAYPOINT':
                waypoint.append([timestamp, float(line_data[2]), float(line_data[3])])

    # Convert lists to numpy arrays where appropriate
    acce = np.array(acce)
    acce_uncali = np.array(acce_uncali)
    gyro = np.array(gyro)
    gyro_uncali = np.array(gyro_uncali)
    magn = np.array(magn)
    magn_uncali = np.array(magn_uncali)
    ahrs = np.array(ahrs)
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
