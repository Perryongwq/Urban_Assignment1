import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from data_process import get_data_from_one_txt  # Ensure this function is available

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
    def __init__(self, wifi_features, magnetic_features, ibeacon_features, labels):
        self.wifi_features = wifi_features
        self.magnetic_features = magnetic_features
        self.ibeacon_features = ibeacon_features
        self.labels = labels
        self.length = labels.shape[0]

    def __getitem__(self, index):
        wifi_x = self.wifi_features[index]
        magnetic_x = self.magnetic_features[index]
        ibeacon_x = self.ibeacon_features[index]
        y = self.labels[index]
        return wifi_x, magnetic_x, ibeacon_x, y

    def __len__(self):
        return self.length

# MultiModal model definition
class MultiModalModel(nn.Module):
    def __init__(self, wifi_input_dim, magnetic_input_dim, ibeacon_input_dim, hidden_dim, num_outputs=2):
        super(MultiModalModel, self).__init__()

        # Handle cases where input dimensions are zero (i.e., modality not used)
        self.use_wifi = wifi_input_dim > 0
        self.use_magnetic = magnetic_input_dim > 0
        self.use_ibeacon = ibeacon_input_dim > 0

        if self.use_wifi:
            self.wifi_branch = nn.Sequential(
                nn.Linear(wifi_input_dim, hidden_dim),
                nn.ReLU(),
                # Add more layers if needed
            )
        if self.use_magnetic:
            self.magnetic_branch = nn.Sequential(
                nn.Linear(magnetic_input_dim, hidden_dim),
                nn.ReLU(),
                # Add more layers if needed
            )
        if self.use_ibeacon:
            self.ibeacon_branch = nn.Sequential(
                nn.Linear(ibeacon_input_dim, hidden_dim),
                nn.ReLU(),
                # Add more layers if needed
            )

        # Calculate total hidden dimension based on used modalities
        total_hidden_dim = hidden_dim * sum([self.use_wifi, self.use_magnetic, self.use_ibeacon])

        self.fc = nn.Sequential(
            nn.Linear(total_hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_outputs)
        )

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
        output = self.fc(combined)
        return output

# Function to load and combine data from all site-floor combinations
def load_combined_data(site_floors, testratio=0.1, use_wifi=True, use_magnetic=True, use_ibeacon=True):
    import random
    trajectory_data = []
    bssid2index = {}
    uuid2index = {}
    idx_counter_bssid = 0
    idx_counter_uuid = 0

    for site, floor in site_floors:
        print(f"Loading data for site: {site}, floor: {floor}")
        file_path = os.path.join('./data', site, floor)
        txt_files_dir = os.path.join(file_path, "path_data_files")
        if not os.path.exists(txt_files_dir):
            continue
        txt_files = os.listdir(txt_files_dir)

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

    # Construct the full feature tensors
    labels = []
    wifi_features_list = []
    magnetic_features_list = []
    ibeacon_features_list = []

    for sample in trajectory_data:
        px, py = sample['px'], sample['py']
        labels.append([px, py])

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
    train_wifi_features = wifi_features[train_indices]
    test_wifi_features = wifi_features[test_indices]
    train_magnetic_features = magnetic_features[train_indices]
    test_magnetic_features = magnetic_features[test_indices]
    train_ibeacon_features = ibeacon_features[train_indices]
    test_ibeacon_features = ibeacon_features[test_indices]

    return (train_wifi_features, train_magnetic_features, train_ibeacon_features, train_labels), \
           (test_wifi_features, test_magnetic_features, test_ibeacon_features, test_labels), \
           (bssid2index, uuid2index)

# Trainer class for the MultiModalModel
class MultiModalModelTrainer:
    def __init__(self, train_data, test_data, batchsize=64, device='cuda', error_threshold=5.0):
        self.batchsize = batchsize
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.error_threshold = error_threshold  # Threshold for accuracy calculation

        # Unpack train and test data
        self.train_wifi_features, self.train_magnetic_features, self.train_ibeacon_features, self.train_labels = train_data
        self.test_wifi_features, self.test_magnetic_features, self.test_ibeacon_features, self.test_labels = test_data

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

        # Create data loaders
        train_dataset = MultiModalFloorData(
            torch.tensor(self.train_wifi_features, dtype=torch.float32),
            torch.tensor(self.train_magnetic_features, dtype=torch.float32),
            torch.tensor(self.train_ibeacon_features, dtype=torch.float32),
            torch.tensor(self.train_labels, dtype=torch.float32)
        )
        test_dataset = MultiModalFloorData(
            torch.tensor(self.test_wifi_features, dtype=torch.float32),
            torch.tensor(self.test_magnetic_features, dtype=torch.float32),
            torch.tensor(self.test_ibeacon_features, dtype=torch.float32),
            torch.tensor(self.test_labels, dtype=torch.float32)
        )
        self.trainDataLoader = DataLoader(train_dataset, batch_size=self.batchsize, shuffle=True)
        self.testDataLoader = DataLoader(test_dataset, batch_size=self.batchsize, shuffle=False)

        # Initialize model
        wifi_input_dim = self.train_wifi_features.shape[1]
        magnetic_input_dim = self.train_magnetic_features.shape[1]
        ibeacon_input_dim = self.train_ibeacon_features.shape[1]
        print(f"Initializing MultiModalModel with wifi_input_dim={wifi_input_dim}, "
              f"magnetic_input_dim={magnetic_input_dim}, ibeacon_input_dim={ibeacon_input_dim}")
        self.model = MultiModalModel(wifi_input_dim, magnetic_input_dim, ibeacon_input_dim,
                                     hidden_dim=hidden_dim, num_outputs=2).to(self.device)

    def train(self, epochs=100, learning_rate=0.001, verbose=True):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss(reduction='mean')
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
            total_samples = 0
            batch_count = 0

            for wifi_x, magnetic_x, ibeacon_x, y in self.trainDataLoader:
                batch_count += 1
                wifi_x = wifi_x.to(self.device)
                magnetic_x = magnetic_x.to(self.device)
                ibeacon_x = ibeacon_x.to(self.device)
                y = y.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(wifi_x, magnetic_x, ibeacon_x)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()

                # Compute error
                errors = torch.sqrt(torch.sum(((outputs - y) * self.y_std) ** 2, dim=1))
                error = torch.mean(errors).item()
                epoch_loss += loss.item()
                epoch_error += error

                # Compute accuracy
                correct_predictions = torch.sum(errors < self.error_threshold).item()
                epoch_accuracy += correct_predictions
                total_samples += y.size(0)

            mean_loss = epoch_loss / batch_count
            mean_error = epoch_error / batch_count
            mean_accuracy = epoch_accuracy / total_samples

            val_loss, val_error, val_accuracy = self.evaluate(criterion)

            # Append the losses and errors to the history for plotting
            self.loss_history.append(mean_loss)
            self.val_loss_history.append(val_loss)
            self.train_error_history.append(mean_error)
            self.test_error_history.append(val_error)
            self.train_accuracy_history.append(mean_accuracy)
            self.test_accuracy_history.append(val_accuracy)
            scheduler.step(val_loss)

            if verbose:
                print(f'Epoch {epoch}: Loss {mean_loss:.6f}, Val Loss {val_loss:.6f}, '
                      f'Train Error {mean_error:.4f}, Val Error {val_error:.4f}, '
                      f'Train Acc {mean_accuracy:.4f}, Val Acc {val_accuracy:.4f}')

            if val_error < min_val_error:
                min_val_error, stop_epoch = val_error, epoch
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

    def evaluate(self, criterion):
        self.model.eval()
        total_loss = 0
        total_error = 0
        total_correct = 0
        total_samples = 0
        batch_count = 0

        with torch.no_grad():
            for wifi_x, magnetic_x, ibeacon_x, y in self.testDataLoader:
                batch_count += 1
                wifi_x = wifi_x.to(self.device)
                magnetic_x = magnetic_x.to(self.device)
                ibeacon_x = ibeacon_x.to(self.device)
                y = y.to(self.device)

                outputs = self.model(wifi_x, magnetic_x, ibeacon_x)
                loss = criterion(outputs, y)
                total_loss += loss.item()

                # Compute error
                errors = torch.sqrt(torch.sum(((outputs - y) * self.y_std) ** 2, dim=1))
                error = torch.mean(errors).item()
                total_error += error

                # Compute accuracy
                correct_predictions = torch.sum(errors < self.error_threshold).item()
                total_correct += correct_predictions
                total_samples += y.size(0)

        val_loss = total_loss / batch_count
        val_error = total_error / batch_count
        val_accuracy = total_correct / total_samples

        return val_loss, val_error, val_accuracy

    def plot_loss(self, save_path=None):
        epochs = list(range(1, len(self.loss_history) + 1))
        if not epochs:
            print('No epochs trained.')
            return

        plt.figure(figsize=(12, 10))

        plt.subplot(2, 1, 1)
        plt.plot(epochs, self.loss_history, label='Training Loss', color='red')
        plt.plot(epochs, self.val_loss_history, label='Validation Loss', color='orange')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss over Epochs')
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(epochs, self.train_accuracy_history, label='Training Accuracy', color='blue')
        plt.plot(epochs, self.test_accuracy_history, label='Validation Accuracy', color='green')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracy over Epochs')
        plt.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"Loss and Accuracy plots saved at {save_path}")

        plt.close()

    def predict(self, wifi_x, magnetic_x, ibeacon_x):
        self.model.eval()
        with torch.no_grad():
            wifi_x = torch.tensor(wifi_x, dtype=torch.float32).unsqueeze(0).to(self.device)
            magnetic_x = torch.tensor(magnetic_x, dtype=torch.float32).unsqueeze(0).to(self.device)
            ibeacon_x = torch.tensor(ibeacon_x, dtype=torch.float32).unsqueeze(0).to(self.device)
            output = self.model(wifi_x, magnetic_x, ibeacon_x)
            pred = output.cpu().numpy()[0] * self.y_std.cpu().numpy() + self.y_mean.cpu().numpy()
            return pred  # Returns [px, py]

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
        'experiment_1': {'use_wifi': True, 'use_magnetic': False, 'use_ibeacon': False},
        'experiment_2': {'use_wifi': True, 'use_magnetic': True, 'use_ibeacon': False},
        'experiment_3': {'use_wifi': True, 'use_magnetic': True, 'use_ibeacon': True},
    }

    for exp_name, exp_params in experiments.items():
        print(f"\nStarting {exp_name}")
        # Load combined data
        train_data, test_data, (bssid2index, uuid2index) = load_combined_data(
            site_floors,
            testratio=0.2,
            use_wifi=exp_params['use_wifi'],
            use_magnetic=exp_params['use_magnetic'],
            use_ibeacon=exp_params['use_ibeacon']
        )

        # Create the model trainer
        model_trainer = MultiModalModelTrainer(train_data, test_data,
                                               batchsize=batch_size, device='cuda', error_threshold=error_threshold)
        model_trainer.initialize_model(hidden_dim=128)

        min_val_error, stop_epoch = model_trainer.train(
            epochs=epochs,
            learning_rate=learning_rate,
            verbose=True
        )
        print(f"{exp_name} completed. Min validation error: {min_val_error:.4f}, Stop epoch: {stop_epoch}")

        # Save training loss and accuracy graphs
        loss_plot_path = os.path.join(results_dir, f"{exp_name}_training_loss_accuracy.png")
        model_trainer.plot_loss(save_path=loss_plot_path)

    print("\nAll experiments completed.")
