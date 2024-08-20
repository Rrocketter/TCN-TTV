import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc


class TTVDataset(Dataset):
    def __init__(self, X, y, ttvs):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()
        self.ttvs = torch.from_numpy(ttvs).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.ttvs[idx]


def load_data(file_path):
    data = np.load(file_path, mmap_mode='r')
    X = data['X']
    y = data['y']
    ttvs = data['ttvs']

    min_samples = min(len(X), len(y), len(ttvs))
    X = X[:min_samples]
    y = y[:min_samples]
    ttvs = ttvs[:min_samples]

    # Normalize the input data
    X = (X - np.mean(X, axis=(0, 2), keepdims=True)) / np.std(X, axis=(0, 2), keepdims=True)
    ttvs = (ttvs - np.mean(ttvs)) / np.std(ttvs)

    return X, y, ttvs


class SimplifiedTCNModel(nn.Module):
    def __init__(self, input_shape, nb_filters, kernel_size, nb_stacks, dilations):
        super(SimplifiedTCNModel, self).__init__()
        self.conv_blocks = nn.ModuleList()

        in_channels = input_shape[0]
        for _ in range(nb_stacks):
            for dilation in dilations:
                self.conv_blocks.append(
                    nn.Sequential(
                        nn.Conv1d(in_channels, nb_filters, kernel_size, padding='same', dilation=dilation),
                        nn.ReLU(),
                        nn.BatchNorm1d(nb_filters)
                    )
                )
                in_channels = nb_filters

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(nb_filters, 64)
        self.classification_output = nn.Linear(64, 1)
        self.regression_output = nn.Linear(64, 1)

    def forward(self, x):
        for block in self.conv_blocks:
            x = block(x)

        x = self.global_avg_pool(x).squeeze(-1)
        x = F.relu(self.fc(x))

        classification_output = self.classification_output(x).squeeze(-1)
        regression_output = self.regression_output(x).squeeze(-1)

        return classification_output, regression_output


def focal_loss(pred, target, alpha=0.25, gamma=2.0):
    bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
    pt = torch.exp(-bce_loss)
    focal_loss = alpha * (1 - pt) ** gamma * bce_loss
    return focal_loss.mean()


def custom_loss(y_pred, y_true, ttv_true):
    classification_pred, regression_pred = y_pred
    focal = focal_loss(classification_pred, y_true)
    positive_mask = y_true == 1
    mse_loss = F.mse_loss(regression_pred[positive_mask], ttv_true[positive_mask])
    return focal + 0.0001 * mse_loss


def train_model(model, train_loader, val_loader, optimizer, scheduler, scaler, num_epochs, device):
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for batch_X, batch_y, batch_ttv in train_loader:
            batch_X, batch_y, batch_ttv = batch_X.to(device), batch_y.to(device), batch_ttv.to(device)

            with autocast():
                outputs = model(batch_X)
                loss = custom_loss(outputs, batch_y, batch_ttv)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        if (epoch + 1) % 5 == 0:
            val_loss = evaluate_model(model, val_loader, device)
            print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    return model


def evaluate_model(model, data_loader, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch_X, batch_y, batch_ttv in data_loader:
            batch_X, batch_y, batch_ttv = batch_X.to(device), batch_y.to(device), batch_ttv.to(device)

            with autocast():
                outputs = model(batch_X)
                loss = custom_loss(outputs, batch_y, batch_ttv)

            total_loss += loss.item()

    return total_loss / len(data_loader)


def main():
    # Load and preprocess data
    X, y, ttvs = load_data("/content/drive/MyDrive//ml_data/ttv_detection_data.npz")

    # Split the data
    X_train, X_val, y_train, y_val, ttvs_train, ttvs_val = train_test_split(
        X, y, ttvs, test_size=0.2, random_state=42
    )

    # Create datasets and dataloaders
    train_dataset = TTVDataset(X_train, y_train, ttvs_train)
    val_dataset = TTVDataset(X_val, y_val, ttvs_val)

    batch_size = 256  # Increased batch size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4, pin_memory=True)

    # Model parameters
    input_shape = (1, X.shape[2])
    nb_filters = 32  # Reduced number of filters
    kernel_size = 3
    nb_stacks = 2  # Reduced number of stacks
    dilations = [1, 2, 4, 8]  # Reduced number of dilation levels

    # Create model
    model = SimplifiedTCNModel(input_shape, nb_filters, kernel_size, nb_stacks, dilations)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    num_epochs = 30
    scheduler = OneCycleLR(optimizer, max_lr=0.01, epochs=num_epochs, steps_per_epoch=len(train_loader))
    scaler = GradScaler()

    # Train the model
    trained_model = train_model(model, train_loader, val_loader, optimizer, scheduler, scaler, num_epochs, device)

    # Save the trained model
    torch.save(trained_model.state_dict(), 'optimized_ttv_detection_model.pth')
    print("Model training completed and saved.")


if __name__ == "__main__":
    main()