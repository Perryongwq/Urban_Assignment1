import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import json
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from data_process import get_data_from_one_txt  # Ensure this function is available or implement it based on your data format
import random

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
    def __init__(self, wifi_features, magnetic_features, ibeacon_features, site_labels, floor_labels):
        self.wifi_features = wifi_features
        self.magnetic_features = magnetic_features
        self.ibeacon_features = ibeacon_features
        self.site_labels = site_labels  # Site labels (integers)
        self.floor_labels = floor_labels  # Floor labels (integers)
        self.length = site_labels.shape[0]

    def __getitem__(self, index):
        wifi_x = self.wifi_features[index]
        magnetic_x = self.magnetic_features[index]
        ibeacon_x = self.ibeacon_features[index]
        site_label = self.site_labels[index]
        floor_label = self.floor_labels[index]
        return wifi_x, magnetic_x, ibeacon_x, site_label, floor_label

    def __len__(self):
        return self.length

# Modified MultiModal model definition for site and floor prediction
class MultiModalModel(nn.Module):
    def __init__(self, wifi_input_dim, magnetic_input_dim, ibeacon_input_dim, hidden_dim, num_sites, num_floors):
        super(MultiModalModel, self).__init__()

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

        total_hidden_dim = hidden_dim * sum([self.use_wifi, self.use_magnetic, self.use_ibeacon])

        # Shared layers
        self.shared_fc = nn.Sequential(
            nn.Linear(total_hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Output layers
        self.site_output = nn.Linear(hidden_dim, num_sites)  # Site classification output
        self.floor_output = nn.Linear(hidden_dim, num_floors)  # Floor classification output

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

        # Outputs for site and floor prediction
        site_pred = self.site_output(shared_features)  # Site classification output
        floor_pred = self.floor_output(shared_features)  # Floor classification output

        return site_pred, floor_pred

# Function to load and combine data from all site-floor combinations
def load_combined_data(site_floors, testratio=0.1, use_wifi=True, use_magnetic=True, use_ibeacon=True):
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
            txt_data = get_data_from_one_txt(txt_path)
            for sample in txt_data:
                t, px, py = sample[0], sample[1], sample[2]
                magn_features = sample[3:7] if use_magnetic else []
                wifi_signals = sample[7] if use_wifi else {}
                ibeacon_signals = sample[8] if use_ibeacon else {}

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
                    'site_idx': site_idx,
                    'floor_idx': floor_idx,
                    'magn_features': magn_features,
                    'wifi_vector': wifi_vector,
                    'ibeacon_vector': ibeacon_vector
                }
                trajectory_data.append(sample_data)

    num_bssids = len(bssid2index)
    num_uuids = len(uuid2index)
    print(f"Total number of BSSIDs: {num_bssids}")
    print(f"Total number of UUIDs: {num_uuids}")
    print(f"Total number of sites: {len(site2index)}")
    print(f"Total number of floors: {len(floor2index)}")

    site_labels = []
    floor_labels = []
    wifi_features_list = []
    magnetic_features_list = []
    ibeacon_features_list = []

    for sample in trajectory_data:
        site_labels.append(sample['site_idx'])
        floor_labels.append(sample['floor_idx'])

        if use_magnetic:
            magn_features = sample['magn_features']
        else:
            magn_features = []
        magnetic_features_list.append(magn_features)

        if use_wifi:
            wifi_features = np.zeros(num_bssids)
            for idx, value in sample['wifi_vector'].items():
                wifi_features[idx] = value
            wifi_features_list.append(wifi_features)
        else:
            wifi_features_list.append(np.zeros(0))

        if use_ibeacon:
            ibeacon_features = np.zeros(num_uuids)
            for idx, value in sample['ibeacon_vector'].items():
                ibeacon_features[idx] = value
            ibeacon_features_list.append(ibeacon_features)
        else:
            ibeacon_features_list.append(np.zeros(0))

    site_labels = np.array(site_labels)
    floor_labels = np.array(floor_labels)
    wifi_features = np.array(wifi_features_list)
    magnetic_features = np.array(magnetic_features_list)
    ibeacon_features = np.array(ibeacon_features_list)

    # Handle cases where features might be empty
    if wifi_features.ndim == 1:
        wifi_features = np.zeros((len(site_labels), 0))
    if magnetic_features.ndim == 1:
        magnetic_features = np.zeros((len(site_labels), 0))
    if ibeacon_features.ndim == 1:
        ibeacon_features = np.zeros((len(site_labels), 0))

    total_samples = len(site_labels)
    indices = list(range(total_samples))
    random.shuffle(indices)
    split_point = int(total_samples * (1 - testratio))
    train_indices = indices[:split_point]
    test_indices = indices[split_point:]

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

    return (train_wifi_features, train_magnetic_features, train_ibeacon_features, train_site_labels, train_floor_labels), \
           (test_wifi_features, test_magnetic_features, test_ibeacon_features, test_site_labels, test_floor_labels), \
           (bssid2index, uuid2index, site2index, floor2index)

# Trainer class for the MultiModalModel
class MultiModalModelTrainer:
    def __init__(self, train_data, test_data, num_sites, num_floors, site2index, floor2index, batchsize=64, device='cuda'):
        self.batchsize = batchsize
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        (self.train_wifi_features, self.train_magnetic_features, self.train_ibeacon_features, self.train_site_labels, self.train_floor_labels) = train_data
        (self.test_wifi_features, self.test_magnetic_features, self.test_ibeacon_features, self.test_site_labels, self.test_floor_labels) = test_data

        self.num_sites = num_sites
        self.num_floors = num_floors
        self.site2index = site2index
        self.floor2index = floor2index

        self.trainDataLoader = None
        self.testDataLoader = None

        self.model = None

        self.scaler_wifi = StandardScaler()
        self.scaler_magnetic = StandardScaler()
        self.scaler_ibeacon = StandardScaler()

        # Initialize history lists for plotting
        self.loss_history = []
        self.val_loss_history = []
        self.train_site_accuracy_history = []
        self.test_site_accuracy_history = []
        self.train_floor_accuracy_history = []
        self.test_floor_accuracy_history = []

    def initialize_model(self, hidden_dim=128):
        if self.train_wifi_features.shape[1] > 0:
            self.train_wifi_features = self.scaler_wifi.fit_transform(self.train_wifi_features)
            self.test_wifi_features = self.scaler_wifi.transform(self.test_wifi_features)

        if self.train_magnetic_features.shape[1] > 0:
            self.train_magnetic_features = self.scaler_magnetic.fit_transform(self.train_magnetic_features)
            self.test_magnetic_features = self.scaler_magnetic.transform(self.test_magnetic_features)

        if self.train_ibeacon_features.shape[1] > 0:
            self.train_ibeacon_features = self.scaler_ibeacon.fit_transform(self.train_ibeacon_features)
            self.test_ibeacon_features = self.scaler_ibeacon.transform(self.test_ibeacon_features)

        self.train_site_labels = torch.tensor(self.train_site_labels, dtype=torch.long)
        self.test_site_labels = torch.tensor(self.test_site_labels, dtype=torch.long)
        self.train_floor_labels = torch.tensor(self.train_floor_labels, dtype=torch.long)
        self.test_floor_labels = torch.tensor(self.test_floor_labels, dtype=torch.long)

        train_dataset = MultiModalFloorData(
            torch.tensor(self.train_wifi_features, dtype=torch.float32),
            torch.tensor(self.train_magnetic_features, dtype=torch.float32),
            torch.tensor(self.train_ibeacon_features, dtype=torch.float32),
            self.train_site_labels,
            self.train_floor_labels
        )
        test_dataset = MultiModalFloorData(
            torch.tensor(self.test_wifi_features, dtype=torch.float32),
            torch.tensor(self.test_magnetic_features, dtype=torch.float32),
            torch.tensor(self.test_ibeacon_features, dtype=torch.float32),
            self.test_site_labels,
            self.test_floor_labels
        )
        self.trainDataLoader = DataLoader(train_dataset, batch_size=self.batchsize, shuffle=True)
        self.testDataLoader = DataLoader(test_dataset, batch_size=self.batchsize, shuffle=False)

        wifi_input_dim = self.train_wifi_features.shape[1]
        magnetic_input_dim = self.train_magnetic_features.shape[1]
        ibeacon_input_dim = self.train_ibeacon_features.shape[1]
        print(f"Initializing MultiModalModel with wifi_input_dim={wifi_input_dim}, magnetic_input_dim={magnetic_input_dim}, ibeacon_input_dim={ibeacon_input_dim}")
        self.model = MultiModalModel(wifi_input_dim, magnetic_input_dim, ibeacon_input_dim, hidden_dim, num_sites=self.num_sites, num_floors=self.num_floors).to(self.device)

    def train(self, epochs=100, learning_rate=0.001, verbose=True):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion_classification = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

        min_val_loss = float('inf')
        stop_epoch = 0
        patience = 10
        epochs_no_improve = 0

        for epoch in range(1, epochs + 1):
            self.model.train()
            epoch_loss = 0
            epoch_site_correct = 0
            epoch_floor_correct = 0
            total_samples = 0
            batch_count = 0

            for wifi_x, magnetic_x, ibeacon_x, site_label, floor_label in self.trainDataLoader:
                batch_count += 1
                wifi_x, magnetic_x, ibeacon_x = wifi_x.to(self.device), magnetic_x.to(self.device), ibeacon_x.to(self.device)
                site_label, floor_label = site_label.to(self.device), floor_label.to(self.device)

                optimizer.zero_grad()
                site_pred, floor_pred = self.model(wifi_x, magnetic_x, ibeacon_x)
                loss_site = criterion_classification(site_pred, site_label)
                loss_floor = criterion_classification(floor_pred, floor_label)

                loss = loss_site + loss_floor
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

                _, site_preds = torch.max(site_pred, 1)
                site_correct = torch.sum(site_preds == site_label).item()
                epoch_site_correct += site_correct

                _, floor_preds = torch.max(floor_pred, 1)
                floor_correct = torch.sum(floor_preds == floor_label).item()
                epoch_floor_correct += floor_correct

                total_samples += site_label.size(0)

            mean_loss = epoch_loss / batch_count
            mean_site_accuracy = epoch_site_correct / total_samples
            mean_floor_accuracy = epoch_floor_correct / total_samples

            # Evaluate on validation set
            val_metrics = self.evaluate(criterion_classification)

            # Record metrics
            self.loss_history.append(mean_loss)
            self.val_loss_history.append(val_metrics['val_loss'])
            self.train_site_accuracy_history.append(mean_site_accuracy)
            self.test_site_accuracy_history.append(val_metrics['val_site_accuracy'])
            self.train_floor_accuracy_history.append(mean_floor_accuracy)
            self.test_floor_accuracy_history.append(val_metrics['val_floor_accuracy'])

            if verbose:
                print(f'Epoch {epoch}: Loss {mean_loss:.6f}, '
                      f'Train Site Acc {mean_site_accuracy:.4f}, Val Site Acc {val_metrics["val_site_accuracy"]:.4f}, '
                      f'Train Floor Acc {mean_floor_accuracy:.4f}, Val Floor Acc {val_metrics["val_floor_accuracy"]:.4f}')

            # Check for improvement
            if val_metrics['val_loss'] < min_val_loss:
                min_val_loss, stop_epoch = val_metrics['val_loss'], epoch
                epochs_no_improve = 0
                torch.save(self.model.state_dict(), 'best_model_sitefloor.pt')  # Fixed filename
            else:
                epochs_no_improve += 1

            scheduler.step(val_metrics['val_loss'])

            if epochs_no_improve >= patience:
                print(f'Early stopping at epoch {epoch}')
                break

        self.model.load_state_dict(torch.load('best_model_sitefloor.pt'))
        return min_val_loss, stop_epoch

    def evaluate(self, criterion_classification):
        self.model.eval()
        total_loss = 0
        total_site_correct = 0
        total_floor_correct = 0
        total_samples = 0
        batch_count = 0

        with torch.no_grad():
            for wifi_x, magnetic_x, ibeacon_x, site_label, floor_label in self.testDataLoader:
                batch_count += 1
                wifi_x, magnetic_x, ibeacon_x = wifi_x.to(self.device), magnetic_x.to(self.device), ibeacon_x.to(self.device)
                site_label, floor_label = site_label.to(self.device), floor_label.to(self.device)

                site_pred, floor_pred = self.model(wifi_x, magnetic_x, ibeacon_x)
                loss_site = criterion_classification(site_pred, site_label)
                loss_floor = criterion_classification(floor_pred, floor_label)
                loss = loss_site + loss_floor
                total_loss += loss.item()

                _, site_preds = torch.max(site_pred, 1)
                site_correct = torch.sum(site_preds == site_label).item()
                total_site_correct += site_correct

                _, floor_preds = torch.max(floor_pred, 1)
                floor_correct = torch.sum(floor_preds == floor_label).item()
                total_floor_correct += floor_correct

                total_samples += site_label.size(0)

        val_loss = total_loss / batch_count
        val_site_accuracy = total_site_correct / total_samples
        val_floor_accuracy = total_floor_correct / total_samples

        return {
            'val_loss': val_loss,
            'val_site_accuracy': val_site_accuracy,
            'val_floor_accuracy': val_floor_accuracy
        }

    # Predict function with ground truth comparison
    def predict(self, wifi_x, magnetic_x, ibeacon_x, groundtruth_site=None, groundtruth_floor=None):
        self.model.eval()
        with torch.no_grad():
            wifi_x = torch.tensor(wifi_x, dtype=torch.float32).unsqueeze(0).to(self.device)
            magnetic_x = torch.tensor(magnetic_x, dtype=torch.float32).unsqueeze(0).to(self.device)
            ibeacon_x = torch.tensor(ibeacon_x, dtype=torch.float32).unsqueeze(0).to(self.device)

            # Make predictions
            site_pred, floor_pred = self.model(wifi_x, magnetic_x, ibeacon_x)

            # Get the predicted indices for site and floor
            site_idx = torch.argmax(site_pred, dim=1).item()
            floor_idx = torch.argmax(floor_pred, dim=1).item()

            # Map indices to site and floor names
            index2site = {idx: site for site, idx in self.site2index.items()}
            index2floor = {idx: floor for floor, idx in self.floor2index.items()}
            predicted_site = index2site.get(site_idx, 'Unknown')
            predicted_floor = index2floor.get(floor_idx, 'Unknown')

            # Print predicted site and floor
            print(f"Predicted Site: {predicted_site}")
            print(f"Predicted Floor: {predicted_floor}")

            # Compare to ground truth
            if groundtruth_site is not None and groundtruth_floor is not None:
                print(f"Ground Truth Site: {groundtruth_site}")
                print(f"Ground Truth Floor: {ground_truth_floor}")

                # Check if predictions match ground truth
                site_match = predicted_site == groundtruth_site
                floor_match = predicted_floor == groundtruth_floor

                print(f"Site Prediction Correct: {site_match}")
                print(f"Floor Prediction Correct: {floor_match}")

                return predicted_site, predicted_floor, site_match, floor_match

            return predicted_site, predicted_floor

# Main execution code
if __name__ == "__main__":
    data_dir = './data'
    site_floors = get_site_floors(data_dir)  

    batch_size = 64
    epochs = 100
    learning_rate = 0.001

    experiments = {'experiment_1': {'use_wifi': True, 'use_magnetic': True, 'use_ibeacon': False}}

    # Create a directory to save plots
    plots_dir = 'plots'
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
        print(f"Created directory '{plots_dir}' to save plots.")

    for exp_name, exp_params in experiments.items():
        print(f"\nStarting {exp_name}")
        train_data, test_data, (bssid2index, uuid2index, site2index, floor2index) = load_combined_data(
            site_floors,
            testratio=0.2,
            use_wifi=exp_params['use_wifi'],
            use_magnetic=exp_params['use_magnetic'],
            use_ibeacon=exp_params['use_ibeacon']
        )

        num_sites = len(site2index)
        num_floors = len(floor2index)

        model_trainer = MultiModalModelTrainer(train_data, test_data, num_sites=num_sites, num_floors=num_floors,
                                               site2index=site2index, floor2index=floor2index, batchsize=batch_size)
        model_trainer.initialize_model(hidden_dim=128)

        min_val_loss, stop_epoch = model_trainer.train(epochs=epochs, learning_rate=learning_rate, verbose=True)
        print(f"{exp_name} completed. Min validation loss: {min_val_loss:.4f}, Stop epoch: {stop_epoch}")

        # Plotting training and validation loss and accuracy
        plt.figure(figsize=(14, 6))

        # Plot Loss
        plt.subplot(1, 2, 1)
        plt.plot(range(1, len(model_trainer.loss_history) + 1), model_trainer.loss_history, label='Train Loss')
        plt.plot(range(1, len(model_trainer.val_loss_history) + 1), model_trainer.val_loss_history, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'{exp_name} - Loss')
        plt.legend()
        plt.grid(True)

        # Set x-axis to integer ticks
        ax1 = plt.gca()
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

        # Plot Accuracy
        plt.subplot(1, 2, 2)
        plt.plot(range(1, len(model_trainer.train_site_accuracy_history) + 1), model_trainer.train_site_accuracy_history, label='Train Site Acc')
        plt.plot(range(1, len(model_trainer.test_site_accuracy_history) + 1), model_trainer.test_site_accuracy_history, label='Validation Site Acc')
        plt.plot(range(1, len(model_trainer.train_floor_accuracy_history) + 1), model_trainer.train_floor_accuracy_history, label='Train Floor Acc', linestyle='--')
        plt.plot(range(1, len(model_trainer.test_floor_accuracy_history) + 1), model_trainer.test_floor_accuracy_history, label='Validation Floor Acc', linestyle='--')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title(f'{exp_name} - Accuracy')
        plt.legend()
        plt.grid(True)

        # Set x-axis to integer ticks
        ax2 = plt.gca()
        ax2.xaxis.set_major_locator(MaxNLocator(integer=True))

        plt.tight_layout()

        # Save the plot to the plots directory
        plot_filename = f"{exp_name}_metrics.png"
        plot_path = os.path.join(plots_dir, plot_filename)
        plt.savefig(plot_path, dpi=300)
        print(f"Saved plot to '{plot_path}'.")

        # Optionally, display the plot
        plt.show()
        plt.close()  # Close the figure to free up memory

    # Test the model by making predictions and comparing with ground truth
    magnetic_data = [-41.67328, -21.322632, -47.491455]
    wifi_signals = {
        '12:74:9c:a7:b2:ba': -48.0, '0e:74:9c:a7:b2:ba': -48.0, '06:74:9c:a7:a5:ee': -52.0, '12:74:9c:2b:1a:26': -40.0,
        '06:74:9c:a7:b2:ba': -48.0, '0a:74:9c:a7:b2:ba': -48.0, '1e:74:9c:a7:b2:ba': -48.0, '0a:74:9c:a7:a5:ee': -52.0,
        '12:74:9c:a7:a5:ee': -52.0, '1e:74:9c:a7:a5:ee': -52.0, '0e:74:9c:a7:a5:ee': -52.0, '1a:74:9c:a7:a5:ed': -62.0,
        '16:74:9c:a7:a5:ed': -62.0, '06:74:9c:a7:a5:ed': -73.0, '16:74:9c:a7:b2:ba': -49.0, '0a:74:9c:a7:a5:ed': -61.0,
        '0e:74:9c:a7:a5:ed': -62.0, '12:74:9c:a7:a5:ed': -61.0, '06:74:9c:2b:1a:33': -50.0, '16:74:9c:a7:a5:ee': -53.0,
    }
    ibeacon_signals = {}

    ground_truth_site = "site1"
    ground_truth_floor = "F1"

    def process_wifi_signals(wifi_signals, bssid2index):
        num_bssids = len(bssid2index)
        wifi_vector = np.zeros(num_bssids)
        for bssid, rssi in wifi_signals.items():
            if bssid in bssid2index:
                idx = bssid2index[bssid]
                wifi_vector[idx] = (100 + rssi) / 100  
            else:
                pass  # Unknown BSSID, can be ignored or handled accordingly
        return wifi_vector

    def process_magnetic_data(magnetic_data):
        magn_intensity = np.linalg.norm(magnetic_data)
        magnetic_vector = np.append(magnetic_data, magn_intensity)  # Concatenate intensity
        return np.array(magnetic_vector)

    def process_ibeacon_signals(ibeacon_signals, uuid2index):
        num_uuids = len(uuid2index)
        ibeacon_vector = np.zeros(num_uuids)
        for uuid, rssi in ibeacon_signals.items():
            if uuid in uuid2index:
                idx = uuid2index[uuid]
                ibeacon_vector[idx] = (100 + rssi) / 100
            else:
                pass  # Unknown UUID, can be ignored or handled accordingly
        return ibeacon_vector

    # Assuming only one experiment was run, get the last model_trainer
    if 'model_trainer' in locals():
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

        predicted_site, predicted_floor, site_match, floor_match = model_trainer.predict(
            wifi_x, magnetic_x, ibeacon_x,
            groundtruth_site=ground_truth_site,
            groundtruth_floor=ground_truth_floor
        )

        print(f"Prediction Results: \nPredicted Site: {predicted_site}, Ground Truth Site: {ground_truth_site}")
        print(f"Predicted Floor: {predicted_floor}, Ground Truth Floor: {ground_truth_floor}")
        print(f"Site Match: {site_match}, Floor Match: {floor_match}")
    else:
        print("No model_trainer found. Ensure that at least one experiment has been run.")
