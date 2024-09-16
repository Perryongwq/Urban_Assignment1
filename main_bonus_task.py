import os
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
import json
import numpy as np
from sklearn.preprocessing import StandardScaler
from data import split_floor_data


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


# Encoder module
class Encoder(nn.Module):
    def __init__(self, wifi_dim, ibeacon_dim, output_dim=32, hidden_magn=32, hidden_wifi=64, hidden_ibeacon=64,
                 drop_rate=0.4, actfunc=nn.ReLU, use_wifi=True, use_ibeacon=True):
        super(Encoder, self).__init__()
        self.use_wifi = use_wifi
        self.use_ibeacon = use_ibeacon

        # Store the dimensions for Wi-Fi and iBeacon
        self.wifi_dim = wifi_dim if use_wifi else 0  # Ensure wifi_dim is set
        self.ibeacon_dim = ibeacon_dim if use_ibeacon else 0  # Ensure ibeacon_dim is set

        # Magnetic encoder
        self.magn_encoder = self._create_encoder(4, hidden_magn, drop_rate, actfunc)

        # Wi-Fi encoder
        if self.use_wifi:
            self.wifi_encoder = self._create_encoder(self.wifi_dim + 1, hidden_wifi, drop_rate, actfunc)

        # iBeacon encoder
        if self.use_ibeacon:
            self.ibeacon_encoder = self._create_encoder(self.ibeacon_dim + 1, hidden_ibeacon, drop_rate, actfunc)

        # Define the feature dimension for final encoding
        self.feature_dim = hidden_magn  # Magnetic features
        if use_wifi:
            self.feature_dim += hidden_wifi  # Add Wi-Fi features if used
        if use_ibeacon:
            self.feature_dim += hidden_ibeacon  # Add iBeacon features if used

        # Ensure the first linear layer matches the final feature size after concatenation
        self.encoder = nn.Sequential(
            nn.Linear(self.feature_dim, output_dim * 4),
            nn.BatchNorm1d(output_dim * 4),
            nn.Dropout(drop_rate),
            actfunc(),
            nn.Linear(output_dim * 4, output_dim * 2),
            nn.BatchNorm1d(output_dim * 2),
            actfunc(),
            nn.Linear(output_dim * 2, output_dim),
        )

    def _create_encoder(self, input_dim, hidden_dim, drop_rate, actfunc):
        layers = [nn.Linear(input_dim, hidden_dim * 2),
                  nn.BatchNorm1d(hidden_dim * 2),
                  nn.Dropout(drop_rate * 0.25),
                  actfunc(),
                  nn.Linear(hidden_dim * 2, hidden_dim)]

        return nn.Sequential(*layers)

    def forward(self, x):
        magn_out = self.magn_encoder(x[:, :4])

        # Handle Wi-Fi input
        if self.use_wifi:
            wifi_input = torch.cat([x[:, 4:5], x[:, 6:6 + self.wifi_dim]], dim=1)
            wifi_out = self.wifi_encoder(wifi_input)

        # Handle iBeacon input
        if self.use_ibeacon:
            ibeacon_input = torch.cat([x[:, 5:6], x[:, 6 + self.wifi_dim:6 + self.wifi_dim + self.ibeacon_dim]], dim=1)
            ibeacon_out = self.ibeacon_encoder(ibeacon_input)

        # Combine outputs
        if self.use_wifi and self.use_ibeacon:
            output = torch.cat([magn_out, wifi_out, ibeacon_out], dim=1)
        elif self.use_wifi:
            output = torch.cat([magn_out, wifi_out], dim=1)
        elif self.use_ibeacon:
            output = torch.cat([magn_out, ibeacon_out], dim=1)
        else:
            output = magn_out

        return self.encoder(output)


# Decoder module
class Decoder(nn.Module):
    def __init__(self, input_dim=32, hidden=64, drop_rate=0.2, actfunc=nn.Tanh):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.Dropout(drop_rate),
            actfunc(),
            nn.Linear(hidden, hidden * 2),
            nn.BatchNorm1d(hidden * 2),
            nn.Dropout(drop_rate),
            actfunc(),
            nn.Linear(hidden * 2, 2)
        )

    def forward(self, x):
        return self.decoder(x)


# Complete deep learning network model
class DLnetwork(nn.Module):
    def __init__(self, wifi_dim, ibeacon_dim, use_wifi=True, use_ibeacon=True):
        super(DLnetwork, self).__init__()
        self.encoder = Encoder(wifi_dim, ibeacon_dim, use_wifi=use_wifi, use_ibeacon=use_ibeacon)
        self.decoder = Decoder(input_dim=32)

    def forward(self, x):
        return self.decoder(self.encoder(x))


# Class for deep learning model
class DLModel:
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

    # Rename this method from initial() to initialize_model()
    def initialize_model(self):
        self.load_data()
        self.model = DLnetwork(len(self.bssid2index), len(self.uuid2index), use_wifi=self.use_wifi, use_ibeacon=self.use_ibeacon).to(self.device)

    def load_data(self):
        train_set, test_set, (self.bssid2index, self.uuid2index) = split_floor_data(self.site, self.floor, self.testratio)
        train_set[:, 2:6] = self.scaler.fit_transform(train_set[:, 2:6].copy())
        test_set[:, 2:6] = self.scaler.transform(test_set[:, 2:6].copy())
        self.y_mean, self.y_std = torch.Tensor(train_set[:, :2].mean(axis=0)).to(self.device), torch.Tensor(train_set[:, :2].std(axis=0)).to(self.device)
        self.trainDataLoader = DataLoader(FloorData(train_set), batch_size=self.batchsize, shuffle=True)
        self.testDataLoader = DataLoader(FloorData(test_set), batch_size=self.batchsize, shuffle=False)


    def train(self, epochs, startlr=0.01, verbose=True):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=startlr, weight_decay=1e-4)
        criterion = nn.MSELoss()

        min_val_error = float('inf')
        stop_epoch = 0

        for epoch in range(1, epochs + 1):
            epoch_loss, epoch_error, batch_count = 0, 0, 0
            self.model.train()

            for x, y in self.trainDataLoader:
                batch_count += 1
                x, y = x.float().to(self.device), y.float().to(self.device)
                optimizer.zero_grad()
                pred = self.model(x)
                loss = criterion(pred, (y - self.y_mean) / self.y_std)
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

            if verbose:
                print(f'Epoch {epoch}: Loss {mean_loss}, Train Error {mean_error}, Test Error {test_error}')

            if test_error < min_val_error:
                min_val_error, stop_epoch = test_error, epoch

        return min_val_error, stop_epoch

    def evaluate(self):
        self.model.eval()
        total_error = 0
        batch_count = 0

        with torch.no_grad():
            for x, y in self.testDataLoader:
                batch_count += 1
                x, y = x.float().to(self.device), y.float().to(self.device)
                pred = self.model(x)
                error = torch.sum(torch.sqrt(torch.sum((y - (pred * self.y_std + self.y_mean)) ** 2, dim=1))) / y.size(0)
                total_error += error.item()

        return total_error / batch_count

    # Continue the plot_loss method
    def plot_loss(self):
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
        plt.show()

    # Method for making predictions
    def predict(self, data, groundtruth=None):
        self.model.eval()
        x = torch.zeros((1, 6 + len(self.bssid2index) + len(self.uuid2index)))
        Mx, My, Mz = data[:3]
        MI = (Mx**2 + My**2 + Mz**2) ** 0.5
        Mx, My, Mz, MI = self.scaler.transform([[Mx, My, Mz, MI]])[0]
        wifis, ibeacons = data[3], data[4]
        wifi_det, ibeacon_det = int(bool(wifis)), int(bool(ibeacons))
        x[0, :6] = torch.tensor([Mx, My, Mz, MI, wifi_det, ibeacon_det])

        for bssid, rssi in wifis.items():
            if bssid in self.bssid2index:
                idx = 6 + self.bssid2index[bssid]
                x[0, idx] = (100 + rssi) / 100

        for uuid, rssi in ibeacons.items():
            if uuid in self.uuid2index:
                idx = 6 + len(self.bssid2index) + self.uuid2index[uuid]
                x[0, idx] = (100 + rssi) / 100

        x = x.to(self.device)
        with torch.no_grad():
            pred = self.model(x)
            pred = pred * self.y_std + self.y_mean
            pred = pred.cpu().numpy()[0]

        # Visualization
        json_path = os.path.join('./data', self.site, self.floor, 'floor_info.json')
        with open(json_path) as file:
            map_info = json.load(file)['map_info']
        map_height, map_width = map_info['height'], map_info['width']
        img = plt.imread(os.path.join('./data', self.site, self.floor, 'floor_image.png'))
        plt.figure(figsize=(6, 6))
        plt.imshow(img)
        scaler = (img.shape[0] / map_height + img.shape[1] / map_width) / 2
        pred_point = plt.scatter(pred[0] * scaler, img.shape[0] - pred[1] * scaler, color='red', marker='o')
        if groundtruth:
            gt_point = plt.scatter(groundtruth[0] * scaler, img.shape[0] - groundtruth[1] * scaler, color='green', marker='x')
            plt.legend([pred_point, gt_point], ['Prediction', 'Ground Truth'])
        else:
            plt.legend([pred_point], ['Prediction'])
        plt.show()


# Main execution code
if __name__ == "__main__":
    test_one = True
    batch_size = 32

    # For multiple models (not test_one)
    if not test_one:
        use_wifi = True
        use_ibeacon = False
        all_models = []
        site_floors = get_site_floors('./data')
        
        for site, floor in site_floors:
            model = DLModel(site, floor, batchsize=batch_size, device='cuda', testratio=0.2, use_wifi=use_wifi, use_ibeacon=use_ibeacon)
            model.initialize_model()  # Ensure the model is initialized
            all_models.append((site, floor, model))
    
    # For a single model (test_one)
    else:
        model = DLModel('site2', 'F4', batchsize=batch_size, device='cuda', testratio=0.2, use_wifi=True, use_ibeacon=False)
        model.initialize_model()  # Initialize the model

    # Training phase
    if test_one:
        min_val_error, stop_epoch = model.train(epochs=300, startlr=0.01)
        print(f"Minimum validation error: {min_val_error}, stop_epoch: {stop_epoch}")
    else:
        print(f'=> Training models for BS: {batch_size}, use_wifi: {use_wifi}, use_ibeacon: {use_ibeacon}')
        results = []
        for site, floor, model in all_models:
            print(f'=> Training model for {site} -- {floor}')
            min_val_error, stop_epoch = model.train(epochs=300, startlr=0.01, verbose=False)
            results.append([site, floor, min_val_error, stop_epoch])
        print(f"Training results: {results}")
        avg_err = np.mean([result[2] for result in results])
        print(f"Average validation error: {avg_err}")

    # Plot the training loss
    model.plot_loss()

    # Prediction phase
    model.predict([9.695435, 35.49347, -36.782837, {
        "04:40:a9:a1:9a:61": -50,
        "04:40:a9:a1:87:61": -56,
        "04:40:a9:a1:95:01": -65,
        "04:40:a9:a1:7d:e1": -72,
    }, {}])
