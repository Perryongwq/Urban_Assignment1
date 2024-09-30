import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import json
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from mode1_data import split_floor_data  # Ensure this module is available

# Function to get the site floors
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
    def __init__(self, site, floor, batchsize=64, testratio=0.1, device='cuda', use_wifi=True, use_ibeacon=True):
        self.site = site
        self.floor = floor
        self.batchsize = batchsize
        self.testratio = testratio
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.use_wifi = use_wifi
        self.use_ibeacon = use_ibeacon
        self.trainDataLoader, self.testDataLoader = None, None
        self.model = None
        self.scaler = StandardScaler()
        self.y_mean, self.y_std = None, None
        self.bssid2index, self.uuid2index = None, None

        # Initialize loss history
        self.loss_history = []
        self.train_error_history = []
        self.test_error_history = []

    def initialize_model(self):
        self.load_data()
        input_dim = self.trainDataLoader.dataset.features.shape[1]
        print(f"Initializing model with input_dim = {input_dim}")
        self.model = VAE(input_dim, latent_dim=64).to(self.device)  # Increased latent_dim

    def load_data(self):
        train_set, test_set, (self.bssid2index, self.uuid2index) = split_floor_data(self.site, self.floor, self.testratio)
        # Normalize the magnetic data (columns 2 to 5)
        train_set[:, 2:6] = self.scaler.fit_transform(train_set[:, 2:6].copy())
        test_set[:, 2:6] = self.scaler.transform(test_set[:, 2:6].copy())
        # Compute mean and std for labels (x and y coordinates)
        self.y_mean = torch.Tensor(train_set[:, :2].mean(axis=0)).to(self.device)
        self.y_std = torch.Tensor(train_set[:, :2].std(axis=0)).to(self.device)
        self.trainDataLoader = DataLoader(FloorData(train_set), batch_size=self.batchsize, shuffle=True)
        self.testDataLoader = DataLoader(FloorData(test_set), batch_size=self.batchsize, shuffle=False)

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

        min_val_error = float('inf')
        stop_epoch = 0
        patience = 10  # Early stopping patience
        epochs_no_improve = 0
        for epoch in range(1, epochs + 1):
            epoch_loss, epoch_error, batch_count = 0, 0, 0
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

                error = torch.sum(torch.sqrt(torch.sum((y - (pred * self.y_std + self.y_mean)) ** 2, dim=1))) / y.size(0)
                epoch_loss += loss.item()
                epoch_error += error.item()

            mean_loss = epoch_loss / batch_count
            mean_error = epoch_error / batch_count
            test_error = self.evaluate()
            # Append the losses and errors to the history for plotting
            self.loss_history.append(mean_loss)
            self.train_error_history.append(mean_error)
            self.test_error_history.append(test_error)
            scheduler.step(test_error)

            if verbose:
                print(f'Epoch {epoch}: Loss {mean_loss:.4f}, Train Error {mean_error:.4f}, Test Error {test_error:.4f}')

            if test_error < min_val_error:
                min_val_error, stop_epoch = test_error, epoch
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
        batch_count = 0

        with torch.no_grad():
            for x, y in self.testDataLoader:
                batch_count += 1
                x, y = x.float().to(self.device), y.float().to(self.device)
                recon_x, pred, mu, logvar = self.model(x)
                error = torch.sum(torch.sqrt(torch.sum((y - (pred * self.y_std + self.y_mean)) ** 2, dim=1))) / y.size(0)
                total_error += error.item()

        return total_error / batch_count

    def plot_loss(self, save_path=None):
        epochs = list(range(1, len(self.loss_history) + 1))
        if not epochs:
            print('No epochs trained.')
            return

        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        plt.plot(epochs, self.loss_history, label='Train Loss', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss over Epochs')
        plt.legend()
        plt.subplot(2, 1, 2)
        plt.plot(epochs, self.train_error_history, label='Train Error', color='blue')
        plt.plot(epochs, self.test_error_history, label='Test Error', color='green')
        plt.xlabel('Epoch')
        plt.ylabel('Error')
        plt.title('Train and Test Error over Epochs')
        plt.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300)  # Save the plot before showing or closing it
            print(f"Loss plot saved at {save_path}")

        plt.close()  # Close the plot after saving

    # Method for making predictions
    def predict(self, data, groundtruth=None, save_path=None):
        self.model.eval()
        # Ensure consistent input dimension
        input_dim = self.trainDataLoader.dataset.features.shape[1]
        print(f"Input dimension (from training data): {input_dim}")

        # Construct the input tensor with the same dimension as the training features
        x = torch.zeros((1, input_dim))

        # Magnetic data normalization
        Mx, My, Mz = data[:3]
        MI = (Mx**2 + My**2 + Mz**2) ** 0.5
        Mx, My, Mz, MI = self.scaler.transform([[Mx, My, Mz, MI]])[0]
        wifis, ibeacons = data[3], data[4]
        wifi_det = int(bool(wifis))
        ibeacon_det = int(bool(ibeacons))
        x[0, :6] = torch.tensor([Mx, My, Mz, MI, wifi_det, ibeacon_det])

        # Fill in Wi-Fi features
        for bssid, rssi in wifis.items():
            if bssid in self.bssid2index:
                idx = 6 + self.bssid2index[bssid]
                x[0, idx] = (100 + rssi) / 100

        # Fill in iBeacon features
        for uuid, rssi in ibeacons.items():
            if uuid in self.uuid2index:
                idx = 6 + len(self.bssid2index) + self.uuid2index[uuid]
                x[0, idx] = (100 + rssi) / 100

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
        json_path = os.path.join('./data', self.site, self.floor, 'floor_info.json')
        with open(json_path) as file:
            map_info = json.load(file)['map_info']
        map_height, map_width = map_info['height'], map_info['width']
        img_path = os.path.join('./data', self.site, self.floor, 'floor_image.png')
        if not os.path.exists(img_path):
            print(f"Floor image not found at {img_path}. Skipping visualization.")
            return
        img = plt.imread(img_path)
        plt.figure(figsize=(6, 6))
        plt.imshow(img)
        scaler = (img.shape[0] / map_height + img.shape[1] / map_width) / 2
        pred_x, pred_y = pred[0] * scaler, img.shape[0] - pred[1] * scaler
        pred_point = plt.scatter(pred_x, pred_y, color='red', marker='o', label='Prediction')
        if groundtruth:
            gt_x, gt_y = groundtruth[0] * scaler, img.shape[0] - groundtruth[1] * scaler
            gt_point = plt.scatter(gt_x, gt_y, color='green', marker='x', label='Ground Truth')
            plt.legend()
        else:
            plt.legend(['Prediction'])
        
        plt.title(f"Prediction for Site: {self.site}, Floor: {self.floor}")
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"Prediction plot saved at {save_path}")

        plt.close()  # Close the plot after saving

# Main execution code for training across all sites and floors
if __name__ == "__main__":
    data_dir = './data'  
    site_floors = get_site_floors(data_dir)  # Get all sites and floors

    batch_size = 64
    epochs = 300
    start_lr = 0.0001
    beta = 0.1
    lambda_vae = 0.1
    lambda_reg = 1.0
    ground_truth_location = [128.31096, 68.949165]  

    # Magnetic data and example Wi-Fi/iBeacon signals
    magnetic_data = [-41.67328, -21.322632, -47.491455]
    wifi_signals = {
        '12:74:9c:a7:b2:ba': -48.0, '0e:74:9c:a7:b2:ba': -48.0, '06:74:9c:a7:a5:ee': -52.0, 
        '12:74:9c:2b:1a:26': -40.0, '06:74:9c:a7:b2:ba': -48.0, '0a:74:9c:a7:b2:ba': -48.0
    }
    ibeacon_signals = {
        '1a:74:9c:a7:a5:ed': -62.0, '16:74:9c:a7:b2:ba': -49.0, '0a:74:9c:a7:a5:ed': -61.0
    }

    # Directory to save results
    results_dir = os.path.join(os.getcwd(), 'all_experiments_results')
    os.makedirs(results_dir, exist_ok=True)

    # Loop over each site and floor
    for site, floor in site_floors:
        print(f"\nTraining for Site: {site}, Floor: {floor}")

        # Experiment 1: Wi-Fi RSSI and Magnetic Field Strengths only
        model_wifi_magnetic = VAEModel(
            site, floor, 
            batchsize=batch_size, 
            device='cuda', 
            testratio=0.2, 
            use_wifi=True, 
            use_ibeacon=False
        )
        model_wifi_magnetic.initialize_model()  # Initialize the model

        min_val_error, stop_epoch = model_wifi_magnetic.train(
            epochs=epochs, 
            startlr=start_lr, 
            beta=beta, 
            lambda_vae=lambda_vae, 
            lambda_reg=lambda_reg
        )
        print(f"Experiment 1 for {site}/{floor} - Min validation error: {min_val_error:.4f}, Stop epoch: {stop_epoch}")
        
        # Save training loss graphs
        exp1_loss_path = os.path.join(results_dir, f"{site}_{floor}_experiment_1_training_loss.png")
        model_wifi_magnetic.plot_loss(save_path=exp1_loss_path)  # Save with proper path
        
        # Prediction and visualization
        exp1_pred_path = os.path.join(results_dir, f"{site}_{floor}_experiment_1_prediction.png")
        model_wifi_magnetic.predict(
            data=[*magnetic_data, wifi_signals, {}], 
            groundtruth=ground_truth_location, 
            save_path=exp1_pred_path
        )

        # Experiment 2: Wi-Fi RSSI, iBeacon RSSI, and Magnetic Field Strengths
        model_wifi_magnetic_ibeacon = VAEModel(
            site, floor, 
            batchsize=batch_size, 
            device='cuda', 
            testratio=0.2, 
            use_wifi=True, 
            use_ibeacon=True
        )
        model_wifi_magnetic_ibeacon.initialize_model()  
        min_val_error, stop_epoch = model_wifi_magnetic_ibeacon.train(
            epochs=epochs, 
            startlr=start_lr, 
            beta=beta, 
            lambda_vae=lambda_vae, 
            lambda_reg=lambda_reg
        )
        print(f"Experiment 2 for {site}/{floor} - Min validation error: {min_val_error:.4f}, Stop epoch: {stop_epoch}")
        
        # Save training loss graphs for Experiment 2
        exp2_loss_path = os.path.join(results_dir, f"{site}_{floor}_experiment_2_training_loss.png")
        model_wifi_magnetic_ibeacon.plot_loss(save_path=exp2_loss_path)
        
        # Prediction and visualization for Experiment 2
        exp2_pred_path = os.path.join(results_dir, f"{site}_{floor}_experiment_2_prediction.png")
        model_wifi_magnetic_ibeacon.predict(
            data=[*magnetic_data, wifi_signals, ibeacon_signals], 
            groundtruth=ground_truth_location, 
            save_path=exp2_pred_path
        )

    print("\nTraining and evaluation completed for all sites and floors.")
