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

# Dataset class for handling floor data
class FloorData(Dataset):
    def __init__(self, floordata):
        self.features = floordata[:, 2:]
        self.labels = floordata[:, :2]
        self.length = floordata.shape[0]

    def __getitem__(self, index):
        return self.features[index], self.labels[index]

    def __len__(self):
        return self.length

# VAE module
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=64):
        super(VAE, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        # Latent space
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim)
        )
        # Regressor for location prediction
        self.regressor = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        recon_x = self.decoder(z)
        return recon_x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        pred = self.regressor(z)
        return recon_x, pred, mu, logvar

# Class for VAE model
class VAEModel:
    def __init__(self, train_set, test_set, bssid2index, uuid2index, batchsize=64, device='cuda', error_threshold=5.0):
        self.batchsize = batchsize
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.bssid2index = bssid2index
        self.uuid2index = uuid2index
        self.trainDataLoader, self.testDataLoader = None, None
        self.model = None
        self.scaler = StandardScaler()
        self.y_mean, self.y_std = None, None

        self.error_threshold = error_threshold  # Threshold for accuracy calculation

        # Initialize loss and accuracy history
        self.loss_history = []
        self.val_loss_history = []
        self.train_error_history = []
        self.test_error_history = []
        self.train_accuracy_history = []
        self.test_accuracy_history = []

        # Load data
        self.train_set = train_set
        self.test_set = test_set

    def initialize_model(self):
        # Normalize the features
        if self.train_set.shape[1] > 2:
            self.train_set[:, 2:] = self.scaler.fit_transform(self.train_set[:, 2:].copy())
            self.test_set[:, 2:] = self.scaler.transform(self.test_set[:, 2:].copy())
        # Compute mean and std for labels (x and y coordinates)
        self.y_mean = torch.Tensor(self.train_set[:, :2].mean(axis=0)).to(self.device)
        self.y_std = torch.Tensor(self.train_set[:, :2].std(axis=0)).to(self.device)
        self.trainDataLoader = DataLoader(FloorData(self.train_set), batch_size=self.batchsize, shuffle=True)
        self.testDataLoader = DataLoader(FloorData(self.test_set), batch_size=self.batchsize, shuffle=False)
        input_dim = self.trainDataLoader.dataset.features.shape[1]
        print(f"Initializing model with input_dim = {input_dim}")
        self.model = VAE(input_dim, latent_dim=64).to(self.device)

    def loss_function(self, recon_x, x, mu, logvar, beta=0.1):
        # Reconstruction loss using MSE Loss
        MSE = nn.functional.mse_loss(recon_x, x, reduction='sum')
        # KL divergence
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return MSE + beta * KLD

    def train(self, epochs, startlr=0.0005, beta=0.1, lambda_vae=0.1, lambda_reg=1.0, verbose=True):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=startlr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
        mse_loss = nn.MSELoss()
        self.beta = beta
        self.lambda_vae = lambda_vae
        self.lambda_reg = lambda_reg
        self.mse_loss = mse_loss  # So that we can use it in evaluate

        min_val_error = float('inf')
        stop_epoch = 0
        patience = 10  # Early stopping patience
        epochs_no_improve = 0
        for epoch in range(1, epochs + 1):
            epoch_loss, epoch_error, epoch_accuracy, batch_count, total_samples = 0, 0, 0, 0, 0
            self.model.train()

            for x, y in self.trainDataLoader:
                batch_count += 1
                x, y = x.float().to(self.device), y.float().to(self.device)
                optimizer.zero_grad()
                recon_x, pred, mu, logvar = self.model(x)
                vae_loss = self.loss_function(recon_x, x, mu, logvar, beta=beta)
                regression_loss = mse_loss(pred, (y - self.y_mean) / self.y_std)
                # Adjust the weights
                loss = lambda_vae * vae_loss + lambda_reg * regression_loss
                loss.backward()
                optimizer.step()

                errors = torch.sqrt(torch.sum((y - (pred * self.y_std + self.y_mean)) ** 2, dim=1))
                error = torch.mean(errors).item()
                epoch_loss += loss.item()
                epoch_error += error

                # Compute batch accuracy
                correct_predictions = torch.sum(errors < self.error_threshold).item()
                epoch_accuracy += correct_predictions
                total_samples += y.size(0)

            mean_loss = epoch_loss / batch_count
            mean_error = epoch_error / batch_count
            mean_accuracy = epoch_accuracy / total_samples

            val_loss, val_error, val_accuracy = self.evaluate()

            # Append the losses and errors to the history for plotting
            self.loss_history.append(mean_loss)
            self.val_loss_history.append(val_loss)
            self.train_error_history.append(mean_error)
            self.test_error_history.append(val_error)
            self.train_accuracy_history.append(mean_accuracy)
            self.test_accuracy_history.append(val_accuracy)
            scheduler.step(val_loss)

            if verbose:
                print(f'Epoch {epoch}: Loss {mean_loss:.4f}, Val Loss {val_loss:.4f}, '
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

    def evaluate(self):
        self.model.eval()
        total_error = 0
        total_loss = 0
        total_correct = 0
        total_samples = 0
        batch_count = 0

        with torch.no_grad():
            for x, y in self.testDataLoader:
                batch_count += 1
                x, y = x.float().to(self.device), y.float().to(self.device)
                recon_x, pred, mu, logvar = self.model(x)
                vae_loss = self.loss_function(recon_x, x, mu, logvar, beta=self.beta)
                regression_loss = self.mse_loss(pred, (y - self.y_mean) / self.y_std)
                loss = self.lambda_vae * vae_loss + self.lambda_reg * regression_loss
                total_loss += loss.item()

                errors = torch.sqrt(torch.sum((y - (pred * self.y_std + self.y_mean)) ** 2, dim=1))
                error = torch.mean(errors).item()
                total_error += error

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

    # Adjusted predict method
    def predict(self, data, groundtruth=None, save_path=None):
        self.model.eval()
        # Ensure consistent input dimension
        input_dim = self.trainDataLoader.dataset.features.shape[1]
        print(f"Input dimension (from training data): {input_dim}")

        # Construct the input tensor with the same dimension as the training features
        x = torch.zeros((1, input_dim))

        # Magnetic data normalization
        if len(data) > 3:
            Mx, My, Mz = data[:3]
            MI = (Mx ** 2 + My ** 2 + Mz ** 2) ** 0.5
            Mx, My, Mz, MI = self.scaler.transform([[Mx, My, Mz, MI]])[0]
            idx = 0
            x[0, idx:idx+4] = torch.tensor([Mx, My, Mz, MI])
            idx += 4
        else:
            idx = 0

        wifis = data[-2]
        ibeacons = data[-1]

        if 'wifi_det' in self.feature_names:
            wifi_det = int(bool(wifis))
            x[0, idx] = wifi_det
            idx += 1

        if 'ibeacon_det' in self.feature_names:
            ibeacon_det = int(bool(ibeacons))
            x[0, idx] = ibeacon_det
            idx += 1

        # Fill in Wi-Fi features
        for bssid, rssi in wifis.items():
            if bssid in self.bssid2index:
                feature_idx = idx + self.bssid2index[bssid]
                x[0, feature_idx] = (100 + rssi) / 100

        # Fill in iBeacon features
        for uuid, rssi in ibeacons.items():
            if uuid in self.uuid2index:
                feature_idx = idx + len(self.bssid2index) + self.uuid2index[uuid]
                x[0, feature_idx] = (100 + rssi) / 100

        x = x.float().to(self.device)
        with torch.no_grad():
            recon_x, pred, mu, logvar = self.model(x)
            pred = pred * self.y_std + self.y_mean
            pred = pred.cpu().numpy()[0]

            # Compute error distance if ground truth is provided
            if groundtruth:
                error_distance = np.linalg.norm(pred - np.array(groundtruth))
                print(f"Predicted Location: {pred}")
                print(f"Ground Truth Location: {groundtruth}")
                print(f"Error Distance: {error_distance:.2f} units")

        # Visualization
        if groundtruth:
            plt.figure(figsize=(6, 6))
            plt.scatter(groundtruth[0], groundtruth[1], color='green', marker='x', label='Ground Truth')
            plt.scatter(pred[0], pred[1], color='red', marker='o', label='Prediction')
            plt.legend()
            plt.xlabel('X Coordinate')
            plt.ylabel('Y Coordinate')
            plt.title('Prediction vs Ground Truth')
            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300)
                print(f"Prediction plot saved at {save_path}")
            plt.close()
        else:
            print(f"Predicted Location: {pred}")

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
                magn_features = sample[3:7]  # magn_x, magn_y, magn_z, magn_intensity
                wifi_signals = sample[7] if use_wifi else {}
                ibeacon_signals = sample[8] if use_ibeacon else {}

                # Update bssid2index and uuid2index
                wifi_vector = []
                for bssid, rssi in wifi_signals.items():
                    if bssid not in bssid2index:
                        bssid2index[bssid] = idx_counter_bssid
                        idx_counter_bssid += 1
                    idx = bssid2index[bssid]
                    wifi_vector.append((idx, (100 + rssi) / 100))

                ibeacon_vector = []
                for uuid, rssi in ibeacon_signals.items():
                    if uuid not in uuid2index:
                        uuid2index[uuid] = idx_counter_uuid
                        idx_counter_uuid += 1
                    idx = uuid2index[uuid]
                    ibeacon_vector.append((idx, (100 + rssi) / 100))

                sample_data = [px, py]
                if use_magnetic:
                    sample_data += list(magn_features)
                else:
                    sample_data += []
                sample_data += [wifi_vector, ibeacon_vector]
                trajectory_data.append(sample_data)

    # Now, we know the total number of bssids and uuids
    num_bssids = len(bssid2index)
    num_uuids = len(uuid2index)
    print(f"Total number of BSSIDs: {num_bssids}")
    print(f"Total number of UUIDs: {num_uuids}")

    # Now, we can construct the full feature vectors
    data_list = []
    for sample in trajectory_data:
        idx = 0
        px, py = sample[0], sample[1]
        features = []

        if use_magnetic:
            magn_features = sample[2:6]
            features += magn_features
            idx += 4
            wifi_vector = sample[6]
            ibeacon_vector = sample[7]
        else:
            wifi_vector = sample[2]
            ibeacon_vector = sample[3]

        wifi_det = int(bool(wifi_vector))
        ibeacon_det = int(bool(ibeacon_vector))
        features += [wifi_det, ibeacon_det]
        idx += 2

        wifi_features = np.zeros(num_bssids)
        for idx_bssid, value in wifi_vector:
            wifi_features[idx_bssid] = value
        features += wifi_features.tolist()
        idx += num_bssids

        ibeacon_features = np.zeros(num_uuids)
        for idx_uuid, value in ibeacon_vector:
            ibeacon_features[idx_uuid] = value
        features += ibeacon_features.tolist()

        data_list.append([px, py] + features)

    data_array = np.array(data_list)

    # Now split into train and test
    total_samples = len(data_array)
    indices = list(range(total_samples))
    random.shuffle(indices)
    split_point = int(total_samples * (1 - testratio))
    train_indices = indices[:split_point]
    test_indices = indices[split_point:]

    train_set = data_array[train_indices]
    test_set = data_array[test_indices]

    feature_names = []
    if use_magnetic:
        feature_names += ['Mx', 'My', 'Mz', 'MI']
    feature_names += ['wifi_det', 'ibeacon_det']
    feature_names += ['wifi_' + str(i) for i in range(num_bssids)]
    feature_names += ['ibeacon_' + str(i) for i in range(num_uuids)]

    return train_set, test_set, (bssid2index, uuid2index), feature_names

# Main execution code
if __name__ == "__main__":
    data_dir = './data'
    site_floors = get_site_floors(data_dir)  # Get all sites and floors

    batch_size = 64
    epochs = 300
    start_lr = 0.0001
    beta = 0.1
    lambda_vae = 0.1
    lambda_reg = 1.0
    error_threshold = 5.0  # Threshold for accuracy calculation

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
        train_set, test_set, (bssid2index, uuid2index), feature_names = load_combined_data(
            site_floors,
            testratio=0.2,
            use_wifi=exp_params['use_wifi'],
            use_magnetic=exp_params['use_magnetic'],
            use_ibeacon=exp_params['use_ibeacon']
        )

        # Create the model
        model = VAEModel(train_set, test_set, bssid2index, uuid2index,
                         batchsize=batch_size, device='cuda', error_threshold=error_threshold)
        model.feature_names = feature_names  # Store feature names for prediction
        model.initialize_model()

        min_val_error, stop_epoch = model.train(
            epochs=epochs,
            startlr=start_lr,
            beta=beta,
            lambda_vae=lambda_vae,
            lambda_reg=lambda_reg,
            verbose=True
        )
        print(f"{exp_name} completed. Min validation error: {min_val_error:.4f}, Stop epoch: {stop_epoch}")

        # Save training loss and accuracy graphs
        loss_plot_path = os.path.join(results_dir, f"{exp_name}_training_loss_accuracy.png")
        model.plot_loss(save_path=loss_plot_path)

    print("\nAll experiments completed.")
