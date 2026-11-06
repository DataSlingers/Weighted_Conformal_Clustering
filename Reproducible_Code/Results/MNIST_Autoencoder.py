import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import random
import numpy as np

# ==========================================
# Reproducibility Setup
# ==========================================

def set_seed(seed=42):
    """Locks all sources of randomness for full reproducibility."""
    # 1. Set standard Python/NumPy seeds
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # 2. Set PyTorch CPU/GPU seeds
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # For multi-GPU setups
        
    # 3. Force CuDNN to be deterministic
    # This disables the auto-tuner and forces deterministic algorithms
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    """Ensures deterministic behavior if num_workers > 0 in DataLoader."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# ==========================================
# Network Architecture
# ==========================================

class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim=10):
        super(ConvAutoencoder, self).__init__()
        
        # Encoder: (1, 28, 28) -> (32, 14, 14) -> (64, 7, 7) -> (128, 3, 3)
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=0),
            nn.ReLU()
        )
        self.encoder_fc = nn.Linear(1152, latent_dim)
        
        # Decoder: Latent -> 1152 -> Reshape -> (64, 7, 7) -> (32, 14, 14) -> (1, 28, 28)
        self.decoder_fc = nn.Linear(latent_dim, 1152)
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid() 
        )

    def forward(self, x):
        encoded = self.encoder_conv(x)
        encoded = encoded.view(encoded.size(0), -1) 
        z = self.encoder_fc(encoded)
        
        decoded = self.decoder_fc(z)
        decoded = decoded.view(decoded.size(0), 128, 3, 3) 
        x_recon = self.decoder_conv(decoded)
        
        return z, x_recon

# ==========================================
# Training Routine
# ==========================================

def train_autoencoder(model, dataloader, epochs=100, lr=0.001, device='cpu'):
    print("\n--- Starting CNN Autoencoder Training ---")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for x, _ in dataloader:
            x = x.to(device)
            
            optimizer.zero_grad()
            _, x_recon = model(x)
            loss = criterion(x_recon, x)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} | MSE Loss: {total_loss / len(dataloader):.5f}")

# ==========================================
# Main Execution
# ==========================================

if __name__ == "__main__":
    # --- LOCK RANDOMNESS ---
    SEED = 0
    set_seed(SEED)
    
    EPOCHS = 100 
    LATENT_DIM = 10
    SAVE_PATH = f"MNIST_AE_{LATENT_DIM}d.pth"
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # --- MULTI-GPU BATCH SCALING ---
    BASE_BATCH_SIZE = 256
    NUM_GPUS = torch.cuda.device_count()
    
    if NUM_GPUS > 1:
        print(f"Using {NUM_GPUS} GPUs!")
        BATCH_SIZE = BASE_BATCH_SIZE * NUM_GPUS
    else:
        print(f"Using device: {DEVICE}")
        BATCH_SIZE = BASE_BATCH_SIZE

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset = datasets.MNIST(root='../Data', train=True, download=True, transform=transform)
    
    # --- DETERMINISTIC DATALOADER ---
    # Create a specific generator tied to our seed for data shuffling
    g = torch.Generator()
    g.manual_seed(SEED)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        worker_init_fn=seed_worker, # Ensures multiple workers don't duplicate randomness
        generator=g                 # Locks the shuffling order
    )

    # Initialize CNN Autoencoder
    autoencoder = ConvAutoencoder(latent_dim=LATENT_DIM)

    # --- WRAP MODEL IN DATAPARALLEL ---
    if NUM_GPUS > 1:
        autoencoder = nn.DataParallel(autoencoder)
    autoencoder = autoencoder.to(DEVICE)

    # Train the Autoencoder
    train_autoencoder(autoencoder, train_loader, epochs=EPOCHS, device=DEVICE)
    
    # Save the model
    print(f"\nTraining complete! Saving CNN model to {SAVE_PATH}...")
    
    # Best practice: Save the original state_dict, not the DataParallel wrapper state_dict
    model_to_save = autoencoder.module if hasattr(autoencoder, 'module') else autoencoder
    torch.save(model_to_save.state_dict(), SAVE_PATH)
