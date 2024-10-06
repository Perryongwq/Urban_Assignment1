import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import json
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from data_process import get_data_from_one_txt

# Function to get the site-floor combinations
def get_site_floors(data_dir: str) -> list:
    site_floors = []
    for site in os.scandir(data_dir):
        if site.is_dir():
            for floor in os.scandir(site.path):
                if floor.is_dir():
                    site_floors.append((site.name, floor.name))
    return site_floors

# Dataset class for handling multi-modal data
class MultiModalFloorData(Dataset):
    def __init__(self, wifi_features, magnetic_features, ibeacon_features, labels, site_labels, floor_labels):
        self.wifi_features = wifi_features
        self.magnetic_features = magnetic_features
        self.ibeacon_features = ibeacon_features
        self.labels = labels  # Position labels
        self.site_labels = site_labels  # Site labels (integers)
        self.floor_labels = floor_labels  # Floor labels (integers)
        self.length = labels.shape[0]

    def __getitem__(self, index):
        wifi_x = self.wifi_features[index]
        magnetic_x = self.magnetic_features[index]
        ibeacon_x = self.ibeacon_features[index]
        y = self.labels[index]
        site_label = self.site_labels[index]
        floor_label = self.floor_labels[index]
        return wifi_x, magnetic_x, ibeacon_x, y, site_label, floor_label

    def __len__(self):
        return self.length

class IndoorLocModel(nn.Module):
    def __init__(self, wifi_input_dim, magnetic_input_dim, ibeacon_input_dim,
                 hidden_dim, num_sites, num_floors, num_outputs=2):
        super(IndoorLocModel, self).__init__()

        # Handle cases where input dimensions are zero (i.e., modality not used)
        self.use_wifi = wifi_input_dim > 0
        self.use_magnetic = magnetic_input_dim > 0
        self.use_ibeacon = ibeacon_input_dim > 0

        if self.use_wifi:
            self.wifi_branch = nn.Sequential(
                nn.Linear(wifi_input_dim, hidden_dim),
                nn.ReLU(),
            )
        if self.use_magnetic:
            self.magnetic_branch = nn.Sequential(
                nn.Linear(magnetic_input_dim, hidden_dim),
                nn.ReLU(),

            )
        if self.use_ibeacon:
            self.ibeacon_branch = nn.Sequential(
                nn.Linear(ibeacon_input_dim, hidden_dim),
                nn.ReLU(),

            )

        # Calculate total hidden dimension based on used modalities
        total_hidden_dim = hidden_dim * sum([self.use_wifi, self.use_magnetic, self.use_ibeacon])

        # Shared layers
        self.shared_fc = nn.Sequential(
            nn.Linear(total_hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.position_output = nn.Linear(hidden_dim, num_outputs)  # For position regression
        self.site_output = nn.Linear(hidden_dim, num_sites)  # For site classification
        self.floor_output = nn.Linear(hidden_dim, num_floors)  # For floor classification

    def forward(self, wifi_x, magnetic_x, ibeacon_x):
        features = []
        if self.use_wifi:
            wifi_feat = self.wifi_branch(wifi_x)
            features.append(wifi_feat)
        if self.use_magnetic:
            magnetic_feat = self.magnetic_branch(magnetic_x)
            features.append(magnetic_feat)
        if self.use_ibeacon:
            ibeacon_feat = self.ibeacon_branch(ibeacon_x)
            features.append(ibeacon_feat)

        combined = torch.cat(features, dim=1)
        shared_features = self.shared_fc(combined)

        # Outputs
        position_pred = self.position_output(shared_features)  # Regression output
        site_pred = self.site_output(shared_features)  # Site classification output
        floor_pred = self.floor_output(shared_features)  # Floor classification output

        return position_pred, site_pred, floor_pred

# Function to load and combine data from all site-floor combinations
def load_combined_data(site_floors, testratio=0.1, use_wifi=True, use_magnetic=True, use_ibeacon=True):
    import random
    trajectory_data = []
    bssid2index = {}
    uuid2index = {}
    idx_counter_bssid = 0
    idx_counter_uuid = 0

    site2index = {}
    floor2index = {}
    idx_counter_site = 0
    idx_counter_floor = 0

    for site, floor in site_floors:
        print(f"Loading data for site: {site}, floor: {floor}")
        file_path = os.path.join('./data', site, floor)
        txt_files_dir = os.path.join(file_path, "path_data_files")
        if not os.path.exists(txt_files_dir):
            continue
        txt_files = os.listdir(txt_files_dir)

        # Map site and floor to indices
        if site not in site2index:
            site2index[site] = idx_counter_site
            idx_counter_site += 1
        if floor not in floor2index:
            floor2index[floor] = idx_counter_floor
            idx_counter_floor += 1

        site_idx = site2index[site]
        floor_idx = floor2index[floor]

        for txt_file in txt_files:
            txt_path = os.path.join(txt_files_dir, txt_file)
            # Get data from the txt file
            txt_data = get_data_from_one_txt(txt_path)
            # Now process each data point
            for sample in txt_data:
                t, px, py = sample[0], sample[1], sample[2]
                magn_features = sample[3:7] if use_magnetic else []  # magn_x, magn_y, magn_z, magn_intensity
                wifi_signals = sample[7] if use_wifi else {}
                ibeacon_signals = sample[8] if use_ibeacon else {}

                # Update bssid2index and uuid2index
                wifi_vector = {}
                for bssid, rssi in wifi_signals.items():
                    if bssid not in bssid2index:
                        bssid2index[bssid] = idx_counter_bssid
                        idx_counter_bssid += 1
                    idx = bssid2index[bssid]
                    wifi_vector[idx] = (100 + rssi) / 100  # Normalize RSSI values

                ibeacon_vector = {}
                for uuid, rssi in ibeacon_signals.items():
                    if uuid not in uuid2index:
                        uuid2index[uuid] = idx_counter_uuid
                        idx_counter_uuid += 1
                    idx = uuid2index[uuid]
                    ibeacon_vector[idx] = (100 + rssi) / 100

                sample_data = {
                    'px': px,
                    'py': py,
                    'site_idx': site_idx,
                    'floor_idx': floor_idx,
                    'magn_features': magn_features,
                    'wifi_vector': wifi_vector,
                    'ibeacon_vector': ibeacon_vector
                }
                trajectory_data.append(sample_data)

    # Now, we know the total number of BSSIDs and UUIDs
    num_bssids = len(bssid2index)
    num_uuids = len(uuid2index)
    print(f"Total number of BSSIDs: {num_bssids}")
    print(f"Total number of UUIDs: {num_uuids}")
    print(f"Total number of sites: {len(site2index)}")
    print(f"Total number of floors: {len(floor2index)}")

    # Construct the full feature tensors
    labels = []
    site_labels = []
    floor_labels = []
    wifi_features_list = []
    magnetic_features_list = []
    ibeacon_features_list = []

    for sample in trajectory_data:
        px, py = sample['px'], sample['py']
        labels.append([px, py])
        site_labels.append(sample['site_idx'])
        floor_labels.append(sample['floor_idx'])

        # Magnetic features
        if use_magnetic:
            magn_features = sample['magn_features']
        else:
            magn_features = []
        magnetic_features_list.append(magn_features)

        # Wi-Fi features
        if use_wifi:
            wifi_features = np.zeros(num_bssids)
            for idx, value in sample['wifi_vector'].items():
                wifi_features[idx] = value
            wifi_features_list.append(wifi_features)
        else:
            wifi_features_list.append(np.zeros(0))  # Empty array

        # iBeacon features
        if use_ibeacon:
            ibeacon_features = np.zeros(num_uuids)
            for idx, value in sample['ibeacon_vector'].items():
                ibeacon_features[idx] = value
            ibeacon_features_list.append(ibeacon_features)
        else:
            ibeacon_features_list.append(np.zeros(0))  # Empty array

    labels = np.array(labels)
    site_labels = np.array(site_labels)
    floor_labels = np.array(floor_labels)
    wifi_features = np.array(wifi_features_list)
    magnetic_features = np.array(magnetic_features_list)
    ibeacon_features = np.array(ibeacon_features_list)

    # Ensure that feature arrays have consistent dimensions
    if wifi_features.ndim == 1:
        wifi_features = np.zeros((len(labels), 0))  # Empty array with zero columns
    if magnetic_features.ndim == 1:
        magnetic_features = np.zeros((len(labels), 0))  # Empty array with zero columns
    if ibeacon_features.ndim == 1:
        ibeacon_features = np.zeros((len(labels), 0))  # Empty array with zero columns

    # Split into train and test
    total_samples = len(labels)
    indices = list(range(total_samples))
    random.shuffle(indices)
    split_point = int(total_samples * (1 - testratio))
    train_indices = indices[:split_point]
    test_indices = indices[split_point:]

    train_labels = labels[train_indices]
    test_labels = labels[test_indices]
    train_site_labels = site_labels[train_indices]
    test_site_labels = site_labels[test_indices]
    train_floor_labels = floor_labels[train_indices]
    test_floor_labels = floor_labels[test_indices]
    train_wifi_features = wifi_features[train_indices]
    test_wifi_features = wifi_features[test_indices]
    train_magnetic_features = magnetic_features[train_indices]
    test_magnetic_features = magnetic_features[test_indices]
    train_ibeacon_features = ibeacon_features[train_indices]
    test_ibeacon_features = ibeacon_features[test_indices]

    return (train_wifi_features, train_magnetic_features, train_ibeacon_features, train_labels,
            train_site_labels, train_floor_labels), \
           (test_wifi_features, test_magnetic_features, test_ibeacon_features, test_labels,
            test_site_labels, test_floor_labels), \
           (bssid2index, uuid2index, site2index, floor2index)

# Trainer class for the IndoorLocModel
class IndoorLocModelTrainer:
    def __init__(self, train_data, test_data, num_sites, num_floors,
                 site2index, floor2index,
                 batchsize=64, device='cuda', error_threshold=5.0):
        self.batchsize = batchsize
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.error_threshold = error_threshold  # Threshold for accuracy calculation

        # Unpack train and test data
        (self.train_wifi_features, self.train_magnetic_features, self.train_ibeacon_features,
         self.train_labels, self.train_site_labels, self.train_floor_labels) = train_data
        (self.test_wifi_features, self.test_magnetic_features, self.test_ibeacon_features,
         self.test_labels, self.test_site_labels, self.test_floor_labels) = test_data

        self.num_sites = num_sites
        self.num_floors = num_floors
        self.site2index = site2index  # Store the mappings
        self.floor2index = floor2index

        # Initialize data loaders
        self.trainDataLoader = None
        self.testDataLoader = None

        # Initialize model
        self.model = None

        # Initialize scalers
        self.scaler_wifi = StandardScaler()
        self.scaler_magnetic = StandardScaler()
        self.scaler_ibeacon = StandardScaler()
        self.y_mean = None
        self.y_std = None

        # Initialize loss and accuracy history
        self.loss_history = []
        self.val_loss_history = []
        self.train_error_history = []
        self.test_error_history = []
        self.train_accuracy_history = []
        self.test_accuracy_history = []
        self.train_site_accuracy_history = []
        self.test_site_accuracy_history = []
        self.train_floor_accuracy_history = []
        self.test_floor_accuracy_history = []

    def initialize_model(self, hidden_dim=128):
        # Normalize Wi-Fi features
        if self.train_wifi_features.shape[1] > 0:
            self.train_wifi_features = self.scaler_wifi.fit_transform(self.train_wifi_features)
            self.test_wifi_features = self.scaler_wifi.transform(self.test_wifi_features)

        # Normalize Magnetic features (if any)
        if self.train_magnetic_features.shape[1] > 0:
            self.train_magnetic_features = self.scaler_magnetic.fit_transform(self.train_magnetic_features)
            self.test_magnetic_features = self.scaler_magnetic.transform(self.test_magnetic_features)

        # Normalize iBeacon features
        if self.train_ibeacon_features.shape[1] > 0:
            self.train_ibeacon_features = self.scaler_ibeacon.fit_transform(self.train_ibeacon_features)
            self.test_ibeacon_features = self.scaler_ibeacon.transform(self.test_ibeacon_features)

        # Convert y_mean and y_std to torch tensors on the device
        self.y_mean = torch.tensor(self.train_labels.mean(axis=0), dtype=torch.float32).to(self.device)
        self.y_std = torch.tensor(self.train_labels.std(axis=0), dtype=torch.float32).to(self.device)

        # Normalize labels using NumPy arrays
        self.train_labels = (self.train_labels - self.y_mean.cpu().numpy()) / self.y_std.cpu().numpy()
        self.test_labels = (self.test_labels - self.y_mean.cpu().numpy()) / self.y_std.cpu().numpy()

        # Convert site and floor labels to tensors
        self.train_site_labels = torch.tensor(self.train_site_labels, dtype=torch.long)
        self.test_site_labels = torch.tensor(self.test_site_labels, dtype=torch.long)
        self.train_floor_labels = torch.tensor(self.train_floor_labels, dtype=torch.long)
        self.test_floor_labels = torch.tensor(self.test_floor_labels, dtype=torch.long)

        # Create data loaders
        train_dataset = MultiModalFloorData(
            torch.tensor(self.train_wifi_features, dtype=torch.float32),
            torch.tensor(self.train_magnetic_features, dtype=torch.float32),
            torch.tensor(self.train_ibeacon_features, dtype=torch.float32),
            torch.tensor(self.train_labels, dtype=torch.float32),
            self.train_site_labels,
            self.train_floor_labels
        )
        test_dataset = MultiModalFloorData(
            torch.tensor(self.test_wifi_features, dtype=torch.float32),
            torch.tensor(self.test_magnetic_features, dtype=torch.float32),
            torch.tensor(self.test_ibeacon_features, dtype=torch.float32),
            torch.tensor(self.test_labels, dtype=torch.float32),
            self.test_site_labels,
            self.test_floor_labels
        )
        self.trainDataLoader = DataLoader(train_dataset, batch_size=self.batchsize, shuffle=True)
        self.testDataLoader = DataLoader(test_dataset, batch_size=self.batchsize, shuffle=False)

        # Initialize model
        wifi_input_dim = self.train_wifi_features.shape[1]
        magnetic_input_dim = self.train_magnetic_features.shape[1]
        ibeacon_input_dim = self.train_ibeacon_features.shape[1]
        print(f"Initializing IndoorLocModel with wifi_input_dim={wifi_input_dim}, "
              f"magnetic_input_dim={magnetic_input_dim}, ibeacon_input_dim={ibeacon_input_dim}")
        self.model = IndoorLocModel(wifi_input_dim, magnetic_input_dim, ibeacon_input_dim,
                                     hidden_dim=hidden_dim, num_sites=self.num_sites,
                                     num_floors=self.num_floors, num_outputs=2).to(self.device)

    def train(self, epochs=100, learning_rate=0.001, verbose=True):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion_regression = nn.MSELoss(reduction='mean')
        criterion_classification = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

        min_val_error = float('inf')
        stop_epoch = 0
        patience = 10  # Early stopping patience
        epochs_no_improve = 0
        for epoch in range(1, epochs + 1):
            self.model.train()
            epoch_loss = 0
            epoch_error = 0
            epoch_accuracy = 0
            epoch_site_correct = 0
            epoch_floor_correct = 0
            total_samples = 0
            batch_count = 0

            for wifi_x, magnetic_x, ibeacon_x, y, site_label, floor_label in self.trainDataLoader:
                batch_count += 1
                wifi_x = wifi_x.to(self.device)
                magnetic_x = magnetic_x.to(self.device)
                ibeacon_x = ibeacon_x.to(self.device)
                y = y.to(self.device)
                site_label = site_label.to(self.device)
                floor_label = floor_label.to(self.device)

                optimizer.zero_grad()
                outputs, site_pred, floor_pred = self.model(wifi_x, magnetic_x, ibeacon_x)
                loss_regression = criterion_regression(outputs, y)
                loss_site = criterion_classification(site_pred, site_label)
                loss_floor = criterion_classification(floor_pred, floor_label)

                # Total loss is a weighted sum (you can adjust weights)
                loss = loss_regression + loss_site + loss_floor
                loss.backward()
                optimizer.step()

                # Compute error
                errors = torch.sqrt(torch.sum(((outputs - y) * self.y_std) ** 2, dim=1))
                error = torch.mean(errors).item()
                epoch_loss += loss.item()
                epoch_error += error

                # Compute position accuracy
                correct_positions = torch.sum(errors < self.error_threshold).item()
                epoch_accuracy += correct_positions

                # Compute site accuracy
                _, site_preds = torch.max(site_pred, 1)
                site_correct = torch.sum(site_preds == site_label).item()
                epoch_site_correct += site_correct

                # Compute floor accuracy
                _, floor_preds = torch.max(floor_pred, 1)
                floor_correct = torch.sum(floor_preds == floor_label).item()
                epoch_floor_correct += floor_correct

                total_samples += y.size(0)

            mean_loss = epoch_loss / batch_count
            mean_error = epoch_error / batch_count
            mean_accuracy = epoch_accuracy / total_samples
            mean_site_accuracy = epoch_site_correct / total_samples
            mean_floor_accuracy = epoch_floor_correct / total_samples

            val_metrics = self.evaluate(criterion_regression, criterion_classification)

            # Append the losses and errors to the history for plotting
            self.loss_history.append(mean_loss)
            self.val_loss_history.append(val_metrics['val_loss'])
            self.train_error_history.append(mean_error)
            self.test_error_history.append(val_metrics['val_error'])
            self.train_accuracy_history.append(mean_accuracy)
            self.test_accuracy_history.append(val_metrics['val_accuracy'])
            self.train_site_accuracy_history.append(mean_site_accuracy)
            self.test_site_accuracy_history.append(val_metrics['val_site_accuracy'])
            self.train_floor_accuracy_history.append(mean_floor_accuracy)
            self.test_floor_accuracy_history.append(val_metrics['val_floor_accuracy'])
            scheduler.step(val_metrics['val_loss'])

            if verbose:
                print(f'Epoch {epoch}: Loss {mean_loss:.6f}, Val Loss {val_metrics["val_loss"]:.6f}, '
                      f'Train Error {mean_error:.4f}, Val Error {val_metrics["val_error"]:.4f}, '
                      f'Train Acc {mean_accuracy:.4f}, Val Acc {val_metrics["val_accuracy"]:.4f}, '
                      f'Train Site Acc {mean_site_accuracy:.4f}, Val Site Acc {val_metrics["val_site_accuracy"]:.4f}, '
                      f'Train Floor Acc {mean_floor_accuracy:.4f}, Val Floor Acc {val_metrics["val_floor_accuracy"]:.4f}')

            if val_metrics['val_error'] < min_val_error:
                min_val_error, stop_epoch = val_metrics['val_error'], epoch
                epochs_no_improve = 0
                # Save the best model
                torch.save(self.model.state_dict(), 'best_model.pt')
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print(f'Early stopping at epoch {epoch}')
                break

        # Load the best model
        self.model.load_state_dict(torch.load('best_model.pt'))
        return min_val_error, stop_epoch

    def evaluate(self, criterion_regression, criterion_classification):
        self.model.eval()
        total_loss = 0
        total_error = 0
        total_correct = 0
        total_site_correct = 0
        total_floor_correct = 0
        total_samples = 0
        batch_count = 0

        with torch.no_grad():
            for wifi_x, magnetic_x, ibeacon_x, y, site_label, floor_label in self.testDataLoader:
                batch_count += 1
                wifi_x = wifi_x.to(self.device)
                magnetic_x = magnetic_x.to(self.device)
                ibeacon_x = ibeacon_x.to(self.device)
                y = y.to(self.device)
                site_label = site_label.to(self.device)
                floor_label = floor_label.to(self.device)

                outputs, site_pred, floor_pred = self.model(wifi_x, magnetic_x, ibeacon_x)
                loss_regression = criterion_regression(outputs, y)
                loss_site = criterion_classification(site_pred, site_label)
                loss_floor = criterion_classification(floor_pred, floor_label)
                loss = loss_regression + loss_site + loss_floor
                total_loss += loss.item()

                # Compute error
                errors = torch.sqrt(torch.sum(((outputs - y) * self.y_std) ** 2, dim=1))
                error = torch.mean(errors).item()
                total_error += error

                # Compute position accuracy
                correct_positions = torch.sum(errors < self.error_threshold).item()
                total_correct += correct_positions

                # Compute site accuracy
                _, site_preds = torch.max(site_pred, 1)
                site_correct = torch.sum(site_preds == site_label).item()
                total_site_correct += site_correct

                # Compute floor accuracy
                _, floor_preds = torch.max(floor_pred, 1)
                floor_correct = torch.sum(floor_preds == floor_label).item()
                total_floor_correct += floor_correct

                total_samples += y.size(0)

        val_loss = total_loss / batch_count
        val_error = total_error / batch_count
        val_accuracy = total_correct / total_samples
        val_site_accuracy = total_site_correct / total_samples
        val_floor_accuracy = total_floor_correct / total_samples

        metrics = {
            'val_loss': val_loss,
            'val_error': val_error,
            'val_accuracy': val_accuracy,
            'val_site_accuracy': val_site_accuracy,
            'val_floor_accuracy': val_floor_accuracy
        }
        return metrics

    def plot_loss(self, save_path=None):
        epochs = list(range(1, len(self.loss_history) + 1))
        if not epochs:
            print('No epochs trained.')
            return

        plt.figure(figsize=(12, 15))

        plt.subplot(3, 1, 1)
        plt.plot(epochs, self.loss_history, label='Training Loss', color='red')
        plt.plot(epochs, self.val_loss_history, label='Validation Loss', color='orange')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss over Epochs')
        plt.legend()

        plt.subplot(3, 1, 2)
        plt.plot(epochs, self.train_accuracy_history, label='Train Position Accuracy', color='blue')
        plt.plot(epochs, self.test_accuracy_history, label='Val Position Accuracy', color='green')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Position Accuracy over Epochs')
        plt.legend()

        plt.subplot(3, 1, 3)
        plt.plot(epochs, self.train_site_accuracy_history, label='Train Site Accuracy', color='purple')
        plt.plot(epochs, self.test_site_accuracy_history, label='Val Site Accuracy', color='brown')
        plt.plot(epochs, self.train_floor_accuracy_history, label='Train Floor Accuracy', color='cyan')
        plt.plot(epochs, self.test_floor_accuracy_history, label='Val Floor Accuracy', color='magenta')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Site and Floor Accuracy over Epochs')
        plt.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"Loss and Accuracy plots saved at {save_path}")

        plt.close()

    def predict(self, wifi_x, magnetic_x, ibeacon_x, groundtruth_location=None, groundtruth_site=None, groundtruth_floor=None, save_path=None):
        self.model.eval()
        with torch.no_grad():
            wifi_x = torch.tensor(wifi_x, dtype=torch.float32).unsqueeze(0).to(self.device)
            magnetic_x = torch.tensor(magnetic_x, dtype=torch.float32).unsqueeze(0).to(self.device)
            ibeacon_x = torch.tensor(ibeacon_x, dtype=torch.float32).unsqueeze(0).to(self.device)
            output, site_pred, floor_pred = self.model(wifi_x, magnetic_x, ibeacon_x)
            pred_position = output.cpu().numpy()[0] * self.y_std.cpu().numpy() + self.y_mean.cpu().numpy()
            site_idx = torch.argmax(site_pred, dim=1).item()
            floor_idx = torch.argmax(floor_pred, dim=1).item()

            # Map site and floor indices back to names
            index2site = {idx: site for site, idx in self.site2index.items()}
            index2floor = {idx: floor for floor, idx in self.floor2index.items()}
            predicted_site = index2site.get(site_idx, 'Unknown')
            predicted_floor = index2floor.get(floor_idx, 'Unknown')

            print(f"Predicted Location: {pred_position}")
            print(f"Predicted Site: {predicted_site}")
            print(f"Predicted Floor: {predicted_floor}")

            if groundtruth_site is not None:
                print(f"Ground Truth Site: {groundtruth_site}")
            if groundtruth_floor is not None:
                print(f"Ground Truth Floor: {groundtruth_floor}")

            if groundtruth_location is not None:
                error_distance = np.linalg.norm(pred_position - np.array(groundtruth_location))
                print(f"Ground Truth Location: {groundtruth_location}")
                print(f"Position Error: {error_distance:.2f} units")
            else:
                error_distance = None

        # Visualization
        # Assuming that the floor map image and floor_info.json are available in the data directory
        site_name = predicted_site
        floor_name = predicted_floor
        data_dir = './data'
        floor_info_path = os.path.join(data_dir, site_name, floor_name, 'floor_info.json')
        floor_image_path = os.path.join(data_dir, site_name, floor_name, 'floor_image.png')

        if not os.path.exists(floor_info_path) or not os.path.exists(floor_image_path):
            print(f"Floor map or info not found for site: {site_name}, floor: {floor_name}. Skipping visualization.")
            return pred_position, predicted_site, predicted_floor

        # Load floor map info
        with open(floor_info_path, 'r') as f:
            floor_info = json.load(f)['map_info']
        map_height = floor_info['height']
        map_width = floor_info['width']

        # Load floor map image
        img = plt.imread(floor_image_path)
        img_height, img_width = img.shape[:2]

        # Calculate scaling factors
        scale_x = img_width / map_width
        scale_y = img_height / map_height

        # Transform coordinates to image pixels
        pred_x = pred_position[0] * scale_x
        pred_y = img_height - pred_position[1] * scale_y  # Invert y-axis for image coordinate system

        # Plotting
        plt.figure(figsize=(8, 8))
        plt.imshow(img)
        plt.scatter(pred_x, pred_y, color='red', marker='o', label='Prediction')

        if groundtruth_location is not None:
            gt_x = groundtruth_location[0] * scale_x
            gt_y = img_height - groundtruth_location[1] * scale_y
            plt.scatter(gt_x, gt_y, color='green', marker='x', label='Ground Truth')
            plt.legend()
        else:
            plt.legend(['Prediction'])

        plt.title(f"Prediction for Site: {predicted_site} (GT: {groundtruth_site}), Floor: {predicted_floor} (GT: {groundtruth_floor})")
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"Prediction plot saved at {save_path}")
        else:
            plt.show()

        plt.close()

        return pred_position, predicted_site, predicted_floor  # Returns [px, py], site name, floor name

# Main execution code
if __name__ == "__main__":
    data_dir = './data'
    site_floors = get_site_floors(data_dir)  # Get all sites and floors

    batch_size = 64
    epochs = 100
    learning_rate = 0.001
    error_threshold = 5.0

    # Directory to save results
    results_dir = os.path.join(os.getcwd(), 'all_experiments_results')
    os.makedirs(results_dir, exist_ok=True)

    # Define the experiments
    experiments = {
        'experiment_1': {'use_wifi': True, 'use_magnetic': True, 'use_ibeacon': False},

    }

    for exp_name, exp_params in experiments.items():
        print(f"\nStarting {exp_name}")
        # Load combined data
        train_data, test_data, (bssid2index, uuid2index, site2index, floor2index) = load_combined_data(
            site_floors,
            testratio=0.2,
            use_wifi=exp_params['use_wifi'],
            use_magnetic=exp_params['use_magnetic'],
            use_ibeacon=exp_params['use_ibeacon']
        )

        num_sites = len(site2index)
        num_floors = len(floor2index)

        model_trainer = IndoorLocModelTrainer(train_data, test_data,
                                               num_sites=num_sites, num_floors=num_floors,
                                               site2index=site2index, floor2index=floor2index,
                                               batchsize=batch_size, device='cuda', error_threshold=error_threshold)
        model_trainer.initialize_model(hidden_dim=128)

        min_val_error, stop_epoch = model_trainer.train(
            epochs=epochs,
            learning_rate=learning_rate,
            verbose=True
        )
        print(f"{exp_name} completed. Min validation error: {min_val_error:.4f}, Stop epoch: {stop_epoch}")

        loss_plot_path = os.path.join(results_dir, f"{exp_name}_training_loss_accuracy.png")
        model_trainer.plot_loss(save_path=loss_plot_path)

    print("\nAll experiments completed.")

    magnetic_data = [-41.67328, -21.322632, -47.491455]
    wifi_signals = {
        '12:74:9c:a7:b2:ba': -48.0, '0e:74:9c:a7:b2:ba': -48.0, '06:74:9c:a7:a5:ee': -52.0, '12:74:9c:2b:1a:26': -40.0,
        '06:74:9c:a7:b2:ba': -48.0, '0a:74:9c:a7:b2:ba': -48.0, '1e:74:9c:a7:b2:ba': -48.0, '0a:74:9c:a7:a5:ee': -52.0,
        '12:74:9c:a7:a5:ee': -52.0, '1e:74:9c:a7:a5:ee': -52.0, '0e:74:9c:a7:a5:ee': -52.0, '1a:74:9c:a7:a5:ed': -62.0,
        '16:74:9c:a7:a5:ed': -62.0, '06:74:9c:a7:a5:ed': -73.0, '16:74:9c:a7:b2:ba': -49.0, '0a:74:9c:a7:a5:ed': -61.0,
        '0e:74:9c:a7:a5:ed': -62.0, '12:74:9c:a7:a5:ed': -61.0, '06:74:9c:2b:1a:33': -50.0, '16:74:9c:a7:a5:ee': -53.0,
    }
    ibeacon_signals = {}
    ground_truth_location = [128.31096, 68.949165]
    ground_truth_site = 'site1'
    ground_truth_floor = 'F1'

    # Prepare the input data
    def process_wifi_signals(wifi_signals, bssid2index):
        num_bssids = len(bssid2index)
        wifi_vector = np.zeros(num_bssids)
        for bssid, rssi in wifi_signals.items():
            if bssid in bssid2index:
                idx = bssid2index[bssid]
                wifi_vector[idx] = (100 + rssi) / 100  # Normalize RSSI values as during training
            else:
                pass  
        return wifi_vector

    def process_magnetic_data(magnetic_data):
        magn_intensity = np.linalg.norm(magnetic_data)
        magnetic_vector = magnetic_data + [magn_intensity]
        return np.array(magnetic_vector)

    def process_ibeacon_signals(ibeacon_signals, uuid2index):
        num_uuids = len(uuid2index)
        ibeacon_vector = np.zeros(num_uuids)
        for uuid, rssi in ibeacon_signals.items():
            if uuid in uuid2index:
                idx = uuid2index[uuid]
                ibeacon_vector[idx] = (100 + rssi) / 100
            else:
                pass
        return ibeacon_vector

    wifi_x = process_wifi_signals(wifi_signals, bssid2index)
    magnetic_x = process_magnetic_data(magnetic_data)
    ibeacon_x = process_ibeacon_signals(ibeacon_signals, uuid2index)

    if model_trainer.train_wifi_features.shape[1] > 0:
        wifi_x = model_trainer.scaler_wifi.transform([wifi_x])[0]
    else:
        wifi_x = np.zeros(0) 

    if model_trainer.train_magnetic_features.shape[1] > 0:
        magnetic_x = model_trainer.scaler_magnetic.transform([magnetic_x])[0]
    else:
        magnetic_x = np.zeros(0)  
    if model_trainer.train_ibeacon_features.shape[1] > 0:
        ibeacon_x = model_trainer.scaler_ibeacon.transform([ibeacon_x])[0]
    else:
        ibeacon_x = np.zeros(0)  

 
    predicted_location, predicted_site, predicted_floor = model_trainer.predict(
        wifi_x, magnetic_x, ibeacon_x,
        groundtruth_location=ground_truth_location,
        groundtruth_site=ground_truth_site,
        groundtruth_floor=ground_truth_floor,
        save_path='prediction_visualization.png'  
    )

    print(f"Predicted Location: {predicted_location}")
    print(f"Predicted Site: {predicted_site}")
    print(f"Predicted Floor: {predicted_floor}")
    print(f"Ground Truth Location: {ground_truth_location}")
    print(f"Ground Truth Site: {ground_truth_site}")
    print(f"Ground Truth Floor: {ground_truth_floor}")
