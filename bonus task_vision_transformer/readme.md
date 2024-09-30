# Indoor Localization using Vision Transformer (ViT)

This project implements an indoor localization system using a Vision Transformer (ViT) model. It processes Wi-Fi RSSI data converted into image format to predict the location coordinates within a building.

## Dataset

**👉 To download the full dataset, please visit [Indoor Location Competition 2.0 Dataset](https://www.microsoft.com/en-us/research/publication/indoor-location-competition-2-0-dataset/).**

## Project Structure

```
indoor-localization-vit/
│
├── data/
│   └── ... (contains raw data files)
│
├── image_data/
│   └── ... (contains generated image files)
│
├── models/
│   └── vit_indoor_localization.pth
│
├── vision_transformerv_final.ipynb
├── wifi_feature final.ipynb
├── vit_indoor_localization.pth (the trained model)
│   
│
├── waypoint_mapping.json
├── site_mapping.json
├── floor_mapping.json
├── bssid_mapping.json
├── wifi_data.csv
├── requirements.txt
└── README.md
```

## Setup

1. Download the addtional large files and data required for this project :
  [Download preprocessed sensor and wifi data, trained model and processed images](https://drive.google.com/drive/folders/1gPzmf_eKa1VZv8dL9bVd8_gFxYkZNVFb?usp=share_link)


2. About the Data :
   - only 2 site data are used.
   - The following json and csv data are extracted from the dataset:
      - waypoint_mapping.json
     - site_mapping.json
     - floor_mapping.json
     - bssid_mapping.json
     - wifi_data.csv

3. Download the dataset from the link provided above and place the raw data files in the `data/` directory.

5. Prepare the data:
   - Run the data preprocessing script to generate the image data and mapping files:
     ```
     wifi_feature final.ipynb
     ```

6. Train the model:
   ```
   vision_transformerv_final.ipynb
   ```

## Usage

After training, you can use the model for inference:

```python
# Load the saved model
def load_model(model_class, filepath, **kwargs):
    model = model_class(**kwargs)
    model.load_state_dict(torch.load(filepath))
    model.eval()
    return model

# Function to make inference on a single image
def make_inference(model, image, site, level, device):
    model.eval()
    with torch.no_grad():
        image = image.unsqueeze(0).to(device)  # Add batch dimension
        site = torch.tensor([site], dtype=torch.long).to(device)
        level = torch.tensor([level], dtype=torch.long).to(device)
        output = model(image, site, level)
    return output.squeeze().cpu().numpy()

# Example usage after training
print("Evaluating on test set...")
test_avg_loss, test_predictions, test_targets = evaluate_on_test_set(model, test_loader, device, criterion)
display_test_results(test_avg_loss, test_predictions, test_targets)

# Save the trained model
model_save_path = 'vit_indoor_localization.pth'
save_model(model, model_save_path)

# Load the saved model
loaded_model = load_model(
    ViT,
    model_save_path,
    img_size=img_size,
    patch_size=patch_size,
    n_channels=n_channels,
    d_model=d_model,
    nhead=nhead,
    dim_feedforward=dim_feedforward,
    blocks=blocks,
    mlp_head_units=mlp_head_units,
    n_outputs=n_outputs,
    n_sites=n_sites,
    n_levels=n_levels
).to(device)

# Make inference on a single image
# Assume we have a single image, site, and level for this example
sample_image, sample_site, sample_level, _ = next(iter(test_loader))
sample_image = sample_image[0]  # Take the first image from the batch
sample_site = sample_site[0].item()
sample_level = sample_level[0].item()

predicted_coords = make_inference(loaded_model, sample_image, sample_site, sample_level, device)
print(f"Predicted coordinates: x={predicted_coords[0]:.2f}, y={predicted_coords[1]:.2f}")

# Visualize the sample image and prediction
plt.figure(figsize=(8, 8))
plt.imshow(sample_image.permute(1, 2, 0))  # Convert from CxHxW to HxWxC for displaying
plt.title(f"Sample Image\nPredicted coordinates: x={predicted_coords[0]:.2f}, y={predicted_coords[1]:.2f}")
plt.axis('off')
plt.show()
```

## Model Architecture

The model uses a Vision Transformer (ViT) architecture adapted for indoor localization:

- Input: Wi-Fi RSSI data converted to image format
- Embedding: Patch embedding + positional encoding
- Transformer Encoder: Self-attention layers
- MLP Head: Regression for coordinate prediction

## Performance

The model achieves a Mean Position Error of 12.6384 meters on the test set. 

