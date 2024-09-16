import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import numpy as np
import random
import matplotlib.pyplot as plt

random.seed(2)
torch.manual_seed(2)

# Dataset class for handling sequential data
class FloorData(Dataset):
    def __init__(self, floordata):
        self.features = floordata[:, 2:]
        self.labels = floordata[:, :2]
        self.length = floordata.shape[0]

    def __getitem__(self, index):
        return self.features[index], self.labels[index]

    def __len__(self):
        return self.length

#------------------------------------------  Encoder
class Encoder(nn.Module):
    def __init__(self, wifi_dim, ibeacon_dim, output_dim=32, hidden_magn=32, hidden_wifi=64, hidden_ibeacon=64, drop_rate=0.4, actfunc=nn.ReLU, use_wifi=True, use_ibeacon=True):
        super(Encoder, self).__init__()
        self.use_wifi = use_wifi
        self.use_ibeacon = use_ibeacon
        self.wifi_dim = wifi_dim
        self.ibeacon_dim = ibeacon_dim

        # Geomagnetic encoder
        self.magn_encoder = nn.Sequential(
            nn.Linear(4, hidden_magn * 2),
            nn.BatchNorm1d(hidden_magn * 2),
            nn.Dropout(drop_rate * 0.25),
            actfunc(),
            nn.Linear(hidden_magn * 2, hidden_magn)
        )

        # Wi-Fi encoder
        if use_wifi:
            self.wifi_encoder = nn.Sequential(
                nn.Linear(3, hidden_wifi * 4),  
                nn.BatchNorm1d(hidden_wifi * 4),
                nn.Dropout(drop_rate * 0.5),
                actfunc(),
                nn.Linear(hidden_wifi * 4, hidden_wifi * 2),
                nn.BatchNorm1d(hidden_wifi * 2),
                nn.Dropout(drop_rate),
                actfunc(),
                nn.Linear(hidden_wifi * 2, hidden_wifi)
            )
        # iBeacon encoder
            self.ibeacon_encoder = nn.Sequential(
                nn.Linear(3, hidden_ibeacon * 4),
                nn.BatchNorm1d(hidden_ibeacon * 4),
                nn.Dropout(drop_rate * 0.5),
                actfunc(),
                nn.Linear(hidden_ibeacon * 4, hidden_ibeacon * 2),
                nn.BatchNorm1d(hidden_ibeacon * 2),
                nn.Dropout(drop_rate),
                actfunc(),
                nn.Linear(hidden_ibeacon * 2, hidden_ibeacon)
            )

        # Fully connected layers for feature fusion
        self.feature_dim = 4 + hidden_magn + (hidden_wifi if use_wifi else 0) + (hidden_ibeacon if use_ibeacon else 0)
        self.encoder = nn.Sequential(
            nn.BatchNorm1d(294),
            nn.ReLU(),
            nn.Linear(294, output_dim * 4),
            nn.BatchNorm1d(output_dim * 4),
            nn.Dropout(drop_rate * 0.5),
            nn.ReLU(),
            nn.Linear(output_dim * 4, output_dim * 2),
            nn.BatchNorm1d(output_dim * 2),
            nn.ReLU(),
            nn.Linear(output_dim * 2, output_dim)
        )

    def forward(self, x):
        magn_o, wifi_det, ibeacon_det, wifi_o, ibeacon_o = x.split([4, 1, 1, 2, 2], dim=1)

        magn_out = self.magn_encoder(magn_o)

        if self.use_wifi:
            wifi = torch.cat([wifi_det, wifi_o], dim=1)
            wifi_out = self.wifi_encoder(wifi)
        else:
            wifi_out = None

        if self.use_ibeacon:
            ibeacon = torch.cat([ibeacon_det, ibeacon_o], dim=1)
            ibeacon_out = self.ibeacon_encoder(ibeacon)
        else:
            ibeacon_out = None

        features = [magn_o, magn_out]
        if self.use_wifi:
            features += [wifi_out, wifi_det]
        if self.use_ibeacon:
            features += [ibeacon_out, ibeacon_det]

        output = torch.cat(features, dim=1)
        output = self.encoder(output)

        return output


#------------------------------------------  Decoder
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
            nn.Linear(hidden * 2, 2)  # Predict (x, y)
        )

    def forward(self, x):
        return self.decoder(x)

#------------------------------------------  DLnetwork
class DLnetwork(nn.Module):
    def __init__(self, wifi_dim, ibeacon_dim, use_wifi=True, use_ibeacon=True, augmentation=True):
        super(DLnetwork, self).__init__()
        if not augmentation:
            self.encoder = Encoder(wifi_dim, ibeacon_dim, output_dim=32, hidden_magn=32, hidden_wifi=32, hidden_ibeacon=32, drop_rate=0, actfunc=nn.ReLU, use_wifi=use_wifi, use_ibeacon=use_ibeacon)
            self.decoder = Decoder(input_dim=32, hidden=64, drop_rate=0, actfunc=nn.ReLU)
        else:
            self.encoder = Encoder(wifi_dim, ibeacon_dim, output_dim=32, hidden_magn=32, hidden_wifi=128, hidden_ibeacon=128, drop_rate=0, actfunc=nn.ReLU, use_wifi=use_wifi, use_ibeacon=use_ibeacon)
            self.decoder = Decoder(input_dim=32, hidden=64, drop_rate=0, actfunc=nn.ReLU)

    def forward(self, x):
        encoded = self.encoder(x)
        return self.decoder(encoded)

#------------------------------------------ DLModel: Handles the workflow of training, evaluation, and prediction

class DLModel:
    def __init__(self, wifi_dim, ibeacon_dim, batchsize=64, device='cuda', use_wifi=True, use_ibeacon=True, use_augmentation=True):
        self.batchsize = batchsize
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.use_wifi = use_wifi
        self.use_ibeacon = use_ibeacon
        self.use_augmentation = use_augmentation
        self.model = DLnetwork(wifi_dim, ibeacon_dim, use_wifi=use_wifi, use_ibeacon=use_ibeacon, augmentation=use_augmentation).to(self.device)

    def train(self, train_loader, test_loader, epochs=100, lr=0.0001):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        # Lists to store losses and errors
        train_losses = []
        val_losses = []
        train_errors = []
        val_errors = []

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            total_error = 0

            for features, labels in train_loader:
                features, labels = features.float().to(self.device), labels.float().to(self.device)
                optimizer.zero_grad()
                output = self.model(features)

                # Compute loss and backpropagate
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()

                # Compute Euclidean distance error
                euclidean_error = torch.sqrt(torch.sum((output - labels) ** 2, dim=1)).mean().item()
                total_error += euclidean_error
                total_loss += loss.item()

            # Store training loss and error
            train_losses.append(total_loss / len(train_loader))
            train_errors.append(total_error / len(train_loader))
            
            # Evaluate the model and get validation loss and error
            val_loss, val_error = self.evaluate(test_loader, criterion)
            val_losses.append(val_loss)
            val_errors.append(val_error)

            print(f'Epoch {epoch+1}, Training Loss: {train_losses[-1]}, Training Error: {train_errors[-1]}, Validation Loss: {val_losses[-1]}, Validation Error: {val_errors[-1]}')

        # Save the trained model
        torch.save(self.model.state_dict(), 'trained_model.pth')

        # Plot losses and errors
        self.plot_training(train_losses, val_losses, train_errors, val_errors)

    def evaluate(self, test_loader, criterion):
        self.model.eval()
        total_loss = 0
        total_error = 0
        with torch.no_grad():
            for features, labels in test_loader:
                features, labels = features.float().to(self.device), labels.float().to(self.device)
                output = self.model(features)

                # Compute validation loss
                loss = criterion(output, labels)
                total_loss += loss.item()

                # Compute Euclidean distance error
                euclidean_error = torch.sqrt(torch.sum((output - labels) ** 2, dim=1)).mean().item()
                total_error += euclidean_error

        return total_loss / len(test_loader), total_error / len(test_loader)

    def predict(self, x):
        self.model.eval()
        with torch.no_grad():
            x = x.float().to(self.device)
            return self.model(x)

    def plot_training(self, train_losses, val_losses, train_errors, val_errors):
        plt.figure(figsize=(12, 6))

        # Plot losses
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training/Validation Loss')
        plt.legend()

        # Plot errors
        plt.subplot(1, 2, 2)
        plt.plot(train_errors, label='Training Error (Euclidean)')
        plt.plot(val_errors, label='Validation Error (Euclidean)')
        plt.xlabel('Epoch')
        plt.ylabel('Error')
        plt.title('Training/Validation Error (Euclidean)')
        plt.legend()

        plt.tight_layout()
        plt.savefig('training_plot.png')
        plt.show()

# Example usage
def load_data():
    num_samples = 1000
    num_features = 10
    data = np.random.rand(num_samples, num_features)
    labels = np.random.rand(num_samples, 2)  # (x, y) coordinates
    floordata = np.hstack([labels, data])
    return floordata

def visualize_predictions(predictions, ground_truth):
    plt.figure(figsize=(6, 6))
    plt.scatter(predictions[:, 0], predictions[:, 1], color='r', label='Predictions')
    plt.scatter(ground_truth[:, 0], ground_truth[:, 1], color='g', label='Ground Truth')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Predicted vs Ground Truth Locations')
    plt.legend()
    plt.show()

def main():
    # Load data
    floordata = load_data()
    train_data, test_data = floordata[:800], floordata[800:]

    train_dataset = FloorData(train_data)
    test_dataset = FloorData(test_data)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    wifi_dim = 3
    ibeacon_dim = 3

    model = DLModel(wifi_dim, ibeacon_dim, batchsize=64, device='cuda')

    # Train the model
    model.train(train_loader, test_loader, epochs=50, lr=0.001)

    # Predict and visualize
    sample_input = torch.tensor(test_data[:5, 2:], dtype=torch.float32)
    predicted_location = model.predict(sample_input).cpu().numpy()
    actual_location = test_data[:5, :2]  # Ground truth locations
    visualize_predictions(predicted_location, actual_location)

if __name__ == "__main__":
    main()
