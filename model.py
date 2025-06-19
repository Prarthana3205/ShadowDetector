import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
from scipy import ndimage

class ShadowScoutCNN(nn.Module):
    def __init__(self, input_channels=3):
        super(ShadowScoutCNN, self).__init__()

        # Learnable scalar thresholds for rule-based approach
        self.hi_threshold = nn.Parameter(torch.tensor(0.6))
        self.i_threshold = nn.Parameter(torch.tensor(0.4))

        # Hybrid rule/CNN weights with proper initialization
        self.use_hybrid = True  # Enable hybrid mode
        self.rule_weight = nn.Parameter(torch.tensor(0.5))
        self.cnn_weight = nn.Parameter(torch.tensor(0.5))

        # CNN Architecture
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.dropout1 = nn.Dropout2d(0.25)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.dropout2 = nn.Dropout2d(0.25)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.dropout3 = nn.Dropout2d(0.25)

        # Global average pooling followed by FC layer for threshold prediction
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, 3)  # For thresholds: HI, I, S

    def freeze_cnn_layers(self):
        """Freeze CNN layers for initial training phase"""
        layers_to_freeze = [
            self.conv1, self.conv2, self.conv3,
            self.bn1, self.bn2, self.bn3,
            self.dropout1, self.dropout2, self.dropout3,
            self.fc
        ]
        
        for layer in layers_to_freeze:
            for param in layer.parameters():
                param.requires_grad = False
        
        print("CNN layers frozen - only training blending weights and thresholds")

    def unfreeze_all_layers(self):
        """Unfreeze all layers for joint training"""
        for param in self.parameters():
            param.requires_grad = True
        print("All layers unfrozen - joint training enabled")

    def forward(self, x):
        batch_size, channels, height, width = x.shape

        HI_channel = x[:, 0:1, :, :]  # HI
        I_channel  = x[:, 1:2, :, :]  # I
        S_channel  = x[:, 2:3, :, :]  # S

        # CNN-based prediction
        conv_features = F.relu(self.bn1(self.conv1(x)))
        conv_features = self.dropout1(conv_features)

        conv_features = F.relu(self.bn2(self.conv2(conv_features)))
        conv_features = self.dropout2(conv_features)

        conv_features = F.relu(self.bn3(self.conv3(conv_features)))
        conv_features = self.dropout3(conv_features)

        # Threshold prediction from CNN
        pooled = self.global_pool(conv_features)  # Shape: [B, 128, 1, 1]
        flattened = pooled.view(pooled.size(0), -1)  # Shape: [B, 128]
        cnn_thresholds = torch.sigmoid(self.fc(flattened))  # Shape: [B, 3] in range [0, 1]

        hi_thresh_cnn = cnn_thresholds[:, 0].view(-1, 1, 1, 1)
        i_thresh_cnn  = cnn_thresholds[:, 1].view(-1, 1, 1, 1)
        s_thresh_cnn  = cnn_thresholds[:, 2].view(-1, 1, 1, 1)

        # CNN-based soft masks
        cnn_hi_mask = torch.sigmoid(+10 * (HI_channel - hi_thresh_cnn))   # Higher HI = more shadow
        cnn_i_mask  = torch.sigmoid(-10 * (I_channel  - i_thresh_cnn))    # Lower I = more shadow
        cnn_s_mask  = torch.sigmoid(-10 * (S_channel  - s_thresh_cnn))    # Lower S = more shadow 


        # CNN combined mask (equal weighting for simplicity)
        cnn_mask = (cnn_hi_mask + cnn_i_mask + cnn_s_mask) / 3.0

        # Rule-based prediction using global learnable thresholds
        rule_mask = self.apply_shadowscout_rules(
            HI_channel, I_channel, S_channel, 
            self.hi_threshold, self.i_threshold
        )

        if self.use_hybrid:
            # Normalize weights to sum to 1
            weights_sum = self.rule_weight + self.cnn_weight + 1e-8  # Avoid division by zero
            normalized_rule_weight = self.rule_weight / weights_sum
            normalized_cnn_weight = self.cnn_weight / weights_sum
            
            # Blend rule-based and CNN masks
            combined_mask = (normalized_rule_weight * rule_mask + 
                           normalized_cnn_weight * cnn_mask)
        else:
            combined_mask = cnn_mask

        return torch.clamp(combined_mask, 0.0, 1.0), rule_mask, cnn_mask

    def apply_shadowscout_rules(self, HI, I, S, hi_thresh, i_thresh):
        """Apply rule-based ShadowScout algorithm"""
        high_HI = HI > hi_thresh
        low_I   = I < i_thresh
        high_I  = I > (1.0 - i_thresh)

        shadow_condition     = (high_HI.float() * low_I.float())
        vegetation_condition = (high_HI.float() * high_I.float())
        non_shadow_condition = ((1.0 - high_HI.float()) * high_I.float())

        low_S = (S < 0.5).float()
        saturation_boost = 0.2 * low_S

        rule_prob = (shadow_condition + saturation_boost 
                    - 0.6 * vegetation_condition 
                    - 0.4 * non_shadow_condition)

        return torch.clamp(rule_prob, 0.0, 1.0)

    def consistency_loss(self, rule_mask, cnn_mask):
        """Consistency loss to encourage agreement between rule and CNN masks"""
        return F.mse_loss(rule_mask, cnn_mask)

    def calinski_harabasz_loss(self, mask, image):
        """
        Calinski-Harabasz index based loss for clustering quality
        """
        B, _, H, W = mask.shape
        mask_flat = mask.view(B, -1)
        img_flat = image[:, 0:1, :, :].view(B, -1)

        total_loss = 0.0
        valid_batches = 0

        for b in range(B):
            batch_mask = mask_flat[b]
            batch_img = img_flat[b]
            
            shadow_pixels = batch_img[batch_mask > 0.5]
            non_shadow_pixels = batch_img[batch_mask <= 0.5]

            if len(shadow_pixels) < 2 or len(non_shadow_pixels) < 2:
                continue

            mean_shadow = shadow_pixels.mean()
            mean_non_shadow = non_shadow_pixels.mean()
            mean_global = batch_img.mean()

            # Between-class variance
            sb = (shadow_pixels.size(0) * (mean_shadow - mean_global) ** 2 +
                non_shadow_pixels.size(0) * (mean_non_shadow - mean_global) ** 2)

            # Within-class variance
            sw = ((shadow_pixels - mean_shadow) ** 2).sum() + ((non_shadow_pixels - mean_non_shadow) ** 2).sum()

            ch_index = sb / (sw + 1e-5)
            total_loss += -ch_index  # Negative for minimization
            valid_batches += 1

        if valid_batches == 0:
            return torch.tensor(0.0, device=mask.device, requires_grad=True)

        return total_loss / valid_batches

    def get_threshold_values(self):
        """Return current learnable threshold values"""
        return {
            'hi_threshold': self.hi_threshold.item(),
            'i_threshold': self.i_threshold.item(),
            'rule_weight': self.rule_weight.item(),
            'cnn_weight': self.cnn_weight.item()
        }


class ShadowDataset(Dataset):
    """Dataset class for loading preprocessed HI, I, S data"""
    def __init__(self, preprocessed_dir, transform=None):
        self.preprocessed_dir = preprocessed_dir
        self.files = [f for f in os.listdir(preprocessed_dir) if f.endswith('.npy')]
        self.transform = transform
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        file_path = os.path.join(self.preprocessed_dir, self.files[idx])
        
        # Load preprocessed data [H, W, 3] where channels are [HI, I, S]
        data = np.load(file_path)
        
        # Convert to PyTorch format [3, H, W]
        data = torch.from_numpy(data).permute(2, 0, 1).float()
        
        if self.transform:
            data = self.transform(data)
            
        return data, self.files[idx]


class ShadowScoutTrainer:
    """Enhanced training class with hybrid approach and phase-based training"""
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.train_losses = []
        self.val_losses = []
        self.consistency_losses = []
        self.current_epoch = 0
        
    def compute_loss(self, combined_mask, rule_mask, cnn_mask, input_data):
        """Compute combined loss with consistency term"""
        # Primary loss (Calinski-Harabasz)
        ch_loss = self.model.calinski_harabasz_loss(combined_mask, input_data)
        
        # Consistency loss between rule and CNN masks
        consistency_loss = self.model.consistency_loss(rule_mask, cnn_mask)
        
        # Spatial smoothness loss
        diff_h = torch.abs(combined_mask[:, :, 1:, :] - combined_mask[:, :, :-1, :])
        diff_w = torch.abs(combined_mask[:, :, :, 1:] - combined_mask[:, :, :, :-1])
        smoothness_loss = torch.mean(diff_h) + torch.mean(diff_w)
        
        # Combine losses with weights
        if self.current_epoch < 10:
            # Phase 1: Focus on consistency and smoothness
            total_loss = ch_loss + 2.0 * consistency_loss + 0.1 * smoothness_loss
        else:
            # Phase 2: Balanced approach
            total_loss = ch_loss + 0.5 * consistency_loss + 0.1 * smoothness_loss
        
        return total_loss, {
            'ch_loss': ch_loss.item(),
            'consistency': consistency_loss.item(),
            'smoothness': smoothness_loss.item()
        }
    
    def train_epoch(self, dataloader, optimizer):
        """Train for one epoch with phase-based approach"""
        self.model.train()
        epoch_loss = 0.0
        epoch_metrics = {'ch_loss': 0, 'consistency': 0, 'smoothness': 0}
        
        for batch_idx, (data, filenames) in enumerate(tqdm(dataloader, desc=f"Training Epoch {self.current_epoch + 1}")):
            data = data.to(self.device)
            
            optimizer.zero_grad()
            
            # Forward pass returns combined mask, rule mask, and CNN mask
            combined_mask, rule_mask, cnn_mask = self.model(data)
            
            # Compute loss
            loss, metrics = self.compute_loss(combined_mask, rule_mask, cnn_mask, data)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Accumulate metrics
            epoch_loss += loss.item()
            for key in metrics:
                epoch_metrics[key] += metrics[key]
        
        # Average metrics
        num_batches = len(dataloader)
        epoch_loss /= num_batches
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches
            
        return epoch_loss, epoch_metrics
    
    def validate(self, dataloader):
        """Validate the model"""
        self.model.eval()
        val_loss = 0.0
        val_metrics = {'ch_loss': 0, 'consistency': 0, 'smoothness': 0}
        
        with torch.no_grad():
            for data, filenames in tqdm(dataloader, desc="Validation"):
                data = data.to(self.device)
                
                combined_mask, rule_mask, cnn_mask = self.model(data)
                loss, metrics = self.compute_loss(combined_mask, rule_mask, cnn_mask, data)
                
                val_loss += loss.item()
                for key in metrics:
                    val_metrics[key] += metrics[key]
        
        num_batches = len(dataloader)
        val_loss /= num_batches
        for key in val_metrics:
            val_metrics[key] /= num_batches
            
        return val_loss, val_metrics
    
    def train(self, train_loader, val_loader, num_epochs, learning_rate=0.001):
        """Full training loop with phase-based approach"""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.7)
        
        best_val_loss = float('inf')
        
        # Phase 1: Freeze CNN layers for first 10 epochs
        self.model.freeze_cnn_layers()
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Unfreeze CNN layers after 10 epochs
            if epoch == 10:
                self.model.unfreeze_all_layers()
                print("Switched to joint training mode")
            
            # Train
            train_loss, train_metrics = self.train_epoch(train_loader, optimizer)
            self.train_losses.append(train_loss)
            self.consistency_losses.append(train_metrics['consistency'])
            
            # Validate
            val_loss, val_metrics = self.validate(val_loader)
            self.val_losses.append(val_loss)
            
            # Learning rate scheduling
            scheduler.step()
            
            # Print metrics
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            print(f"CH Loss: {train_metrics['ch_loss']:.4f}, Consistency: {train_metrics['consistency']:.4f}")
            
            thresholds = self.model.get_threshold_values()
            print(f"Thresholds - HI: {thresholds['hi_threshold']:.3f}, I: {thresholds['i_threshold']:.3f}")
            print(f"Weights - Rule: {thresholds['rule_weight']:.3f}, CNN: {thresholds['cnn_weight']:.3f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'thresholds': thresholds
                }, 'best_shadowscout_hybrid_model.pth')
    
    def plot_training_curves(self):
        """Plot comprehensive training curves"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curves
        axes[0, 0].plot(self.train_losses, label='Training Loss', color='blue')
        axes[0, 0].plot(self.val_losses, label='Validation Loss', color='red')
        axes[0, 0].axvline(x=10, color='green', linestyle='--', label='Joint Training Start')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training Curves')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Consistency loss
        axes[0, 1].plot(self.consistency_losses, label='Consistency Loss', color='purple')
        axes[0, 1].axvline(x=10, color='green', linestyle='--', label='Joint Training Start')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Consistency Loss')
        axes[0, 1].set_title('Rule-CNN Consistency')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Threshold evolution (if we track it)
        axes[1, 0].text(0.5, 0.5, 'Threshold Evolution\n(Track during training)', 
                       ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Learned Thresholds')
        
        # Weight evolution
        axes[1, 1].text(0.5, 0.5, 'Weight Evolution\n(Track during training)', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Blending Weights')
        
        plt.tight_layout()
        plt.show()


def apply_morphological_filtering(binary_mask, kernel_size=3):
    """Apply morphological filtering to clean up binary mask"""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # Opening: erosion followed by dilation (removes noise)
    opened = cv2.morphologyEx(binary_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    
    # Closing: dilation followed by erosion (fills holes)
    cleaned = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    
    return cleaned.astype(np.float32)


def inference_single_image(model, image_path, device='cuda', apply_morphology=True):
    """Run inference on a single preprocessed image with optional morphological filtering"""
    model.eval()
    
    # Load preprocessed data
    data = np.load(image_path)  # Shape: [H, W, 3]
    data_tensor = torch.from_numpy(data).permute(2, 0, 1).unsqueeze(0).float().to(device)
    
    with torch.no_grad():
        combined_mask, rule_mask, cnn_mask = model(data_tensor)
    
    # Convert to numpy
    combined_prob = combined_mask.squeeze().cpu().numpy()
    rule_prob = rule_mask.squeeze().cpu().numpy()
    cnn_prob = cnn_mask.squeeze().cpu().numpy()
    
    # Create binary mask
    binary_mask = (combined_prob > 0.5).astype(np.float32)
    
    # Apply morphological filtering if requested
    if apply_morphology:
        binary_mask = apply_morphological_filtering(binary_mask)
    
    return {
        'combined_prob': combined_prob,
        'rule_prob': rule_prob,
        'cnn_prob': cnn_prob,
        'binary_mask': binary_mask
    }


def visualize_results(original_image_path, preprocessed_path, model, device='cuda'):
    """Enhanced visualization showing all components"""
    import cv2
    
    # Load original image
    original = cv2.imread(original_image_path)
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    
    # Load preprocessed data
    preprocessed = np.load(preprocessed_path)
    HI, I, S = preprocessed[:, :, 0], preprocessed[:, :, 1], preprocessed[:, :, 2]
    
    # Get inference results
    results = inference_single_image(model, preprocessed_path, device)
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    
    # Row 1: Original and channels
    axes[0, 0].imshow(original)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(HI, cmap='viridis')
    axes[0, 1].set_title('HI Channel (Hue-Intensity)')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(I, cmap='gray')
    axes[0, 2].set_title('I Channel (Intensity)')
    axes[0, 2].axis('off')
    
    # Row 2: Saturation and individual predictions
    axes[1, 0].imshow(S, cmap='plasma')
    axes[1, 0].set_title('S Channel (Saturation)')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(results['rule_prob'], cmap='hot')
    axes[1, 1].set_title('Rule-based Prediction')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(results['cnn_prob'], cmap='hot')
    axes[1, 2].set_title('CNN Prediction')
    axes[1, 2].axis('off')
    
    # Row 3: Combined results
    axes[2, 0].imshow(results['combined_prob'], cmap='hot')
    axes[2, 0].set_title('Combined Probability Map')
    axes[2, 0].axis('off')
    
    axes[2, 1].imshow(results['binary_mask'], cmap='gray')
    axes[2, 1].set_title('Final Binary Mask (Cleaned)')
    axes[2, 1].axis('off')
    
    # Overlay on original
    overlay = original.copy()
    shadow_pixels = results['binary_mask'] > 0.5
    overlay[shadow_pixels] = [255, 0, 0]  # Red overlay for shadows
    axes[2, 2].imshow(overlay)
    axes[2, 2].set_title('Shadow Overlay on Original')
    axes[2, 2].axis('off')
    
    plt.tight_layout()
    plt.show()


def main():
    """Main function with enhanced hybrid training"""
    # Configuration
    PREPROCESSED_DIR = 'preprocessed'
    BATCH_SIZE = 8
    NUM_EPOCHS = 50
    LEARNING_RATE = 0.001
    
    # Create dataset and dataloaders
    dataset = ShadowDataset(PREPROCESSED_DIR)
    
    # Split dataset (80% train, 20% validation)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Create hybrid model
    model = ShadowScoutCNN(input_channels=3)
    
    # Create trainer
    trainer = ShadowScoutTrainer(model)
    
    print("Starting Enhanced ShadowScout Hybrid Training...")
    print(f"Dataset size: {len(dataset)} images")
    print(f"Training set: {len(train_dataset)} images") 
    print(f"Validation set: {len(val_dataset)} images")
    print("Phase 1 (Epochs 1-10): Frozen CNN layers, train blending weights")
    print("Phase 2 (Epochs 11+): Joint training of all parameters")
    
    # Train the model
    trainer.train(train_loader, val_loader, NUM_EPOCHS, LEARNING_RATE)
    
    # Plot training curves
    trainer.plot_training_curves()
    
    # Print final learned parameters
    print("\nFinal Learned Parameters:")
    thresholds = model.get_threshold_values()
    for key, value in thresholds.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()