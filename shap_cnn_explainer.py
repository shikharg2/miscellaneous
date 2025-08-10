"""
SHAP CNN Explainer for Malware Detection

This script loads a trained CNN model for malware classification and generates
SHAP (SHapley Additive exPlanations) visualizations to explain the model's 
predictions. It creates a side-by-side comparison showing:

1. One benign network traffic image and one malware image
2. Their corresponding SHAP attribution maps
3. Model predictions with confidence scores

The SHAP attributions show which pixels (network packet features) contribute
most to the malware vs benign classification decision:
- Red regions: Features that push toward "malware" classification
- Blue regions: Features that push toward "benign" classification

Usage:
    source .venv/bin/activate
    python shap_cnn_explainer.py

Output:
    Explainability/shap_cnn_comparison_plot.png
"""

import os
import random
import numpy as np
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import shap

# Import the trained model class and dataset utilities
from cnn_binary_classifier import MalwareCNN, MalwareDataset


def set_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def denormalize_and_to_numpy(t: torch.Tensor, mean=0.5, std=0.5) -> np.ndarray:
    """Convert a tensor batch (N,C,H,W) back to [0,1] HWC numpy for plotting."""
    if t.requires_grad:
        t = t.detach()
    if t.is_cuda:
        t = t.cpu()
    x = t.clone()
    x = x * std + mean
    x = torch.clamp(x, 0.0, 1.0)
    x = x.permute(0, 2, 3, 1)  # N,H,W,C
    return x.numpy()


def load_data(data_dir: str, device: torch.device, bg_size: int = 50):
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    dataset = MalwareDataset(data_dir, transform=transform)
    n = len(dataset)
    if n == 0:
        raise RuntimeError(f"No images found under '{data_dir}'.")

    # Find one benign and one malware image
    benign_idx = None
    malware_idx = None
    
    for i in range(n):
        _, label = dataset[i]
        if label == 0 and benign_idx is None:  # Benign
            benign_idx = i
        elif label == 1 and malware_idx is None:  # Malware
            malware_idx = i
        
        if benign_idx is not None and malware_idx is not None:
            break
    
    if benign_idx is None or malware_idx is None:
        raise RuntimeError("Could not find both benign and malware images in dataset")

    # Create background set for SHAP (mix of images)
    bg_indices = np.random.choice(n, size=min(bg_size, n), replace=False)
    
    def stack_by_indices(idxs):
        xs, ys = [], []
        for i in idxs:
            x, y = dataset[int(i)]
            xs.append(x)
            ys.append(y)
        x = torch.stack(xs).to(device)
        y = torch.tensor(ys)
        return x, y

    # Load background data
    background, _ = stack_by_indices(bg_indices)
    
    # Load the specific benign and malware examples
    to_explain, labels = stack_by_indices([benign_idx, malware_idx])

    return background, to_explain, labels


def build_model(weights_path: str, device: torch.device) -> MalwareCNN:
    model = MalwareCNN(input_channels=1, num_classes=2).to(device)
    state = torch.load(weights_path, map_location=device)
    # Load from a plain state_dict if possible; otherwise try full model
    try:
        model.load_state_dict(state)
    except Exception:
        try:
            model = torch.load(weights_path, map_location=device)
            model.to(device)
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {weights_path}: {e}")
    model.eval()
    return model


def compute_shap_and_plot(model: torch.nn.Module,
                           background: torch.Tensor,
                           to_explain: torch.Tensor,
                           labels: torch.Tensor,
                           out_path: str):
    """Compute SHAP values using GradientExplainer and save an image plot."""
    # Ensure gradients can flow for SHAP
    background = background.clone().requires_grad_(True)
    to_explain = to_explain.clone().requires_grad_(True)

    # For PyTorch models, pass the model and a background tensor directly
    explainer = shap.GradientExplainer(model, background)
    shap_values = explainer.shap_values(to_explain)

    if isinstance(shap_values, list):
        # For binary classification choose class 1 (Malware)
        sv = shap_values[1]
    else:
        sv = shap_values

    # Convert SHAP values to numpy and handle different shapes
    if isinstance(sv, torch.Tensor):
        sv = sv.detach().cpu().numpy()
    
    # Handle different SHAP output formats
    if sv.ndim == 4:
        if sv.shape[1] == 1:  # (N, 1, H, W)
            sv = sv[:, 0, :, :]  # -> (N, H, W)
        elif sv.shape[1] == 2:  # (N, 2, H, W) - output for each class
            sv = sv[:, 1, :, :]  # Take class 1 (malware) -> (N, H, W)
        elif sv.shape[-1] == 1:  # (N, H, W, 1)
            sv = sv[:, :, :, 0]  # -> (N, H, W)
        elif sv.shape[-1] == 2:  # (N, H, W, 2)
            sv = sv[:, :, :, 1]  # Take class 1 -> (N, H, W)
    elif sv.ndim == 5:  # (N, C, H, W, classes)
        sv = sv[:, 0, :, :, 1]  # Take first channel, class 1

    imgs_np = denormalize_and_to_numpy(to_explain)
    # Squeeze to (N,H,W) for grayscale
    if imgs_np.ndim == 4 and imgs_np.shape[-1] == 1:
        imgs_np = imgs_np[:, :, :, 0]

    # Get model predictions for display
    with torch.no_grad():
        logits = model(to_explain)
        probs = torch.softmax(logits, dim=1).cpu().numpy()

    # Create custom visualization - 2x2 grid
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    class_names = ['Benign', 'Malware']
    
    for i in range(2):  # Should be exactly 2 images (benign and malware)
        col = i
        true_label = int(labels[i])
        pred_prob = probs[i]
        pred_label = np.argmax(pred_prob)
        
        # Original image (top row)
        axes[0, col].imshow(imgs_np[i], cmap='gray')
        axes[0, col].set_title(f'{class_names[true_label]} Image\n'
                              f'Pred: {class_names[pred_label]} ({pred_prob[pred_label]:.3f})')
        axes[0, col].axis('off')
        
        # SHAP attribution (bottom row)
        shap_img = sv[i]
        if shap_img.ndim != 2:
            # Fallback for unexpected shapes
            shap_img = shap_img.flatten().reshape(28, 28) if shap_img.size == 784 else shap_img[..., 0]
        
        max_val = np.max(np.abs(shap_img))
        im = axes[1, col].imshow(shap_img, cmap='RdBu_r', vmin=-max_val, vmax=max_val)
        axes[1, col].set_title(f'SHAP Attribution\n(Red→Malware, Blue→Benign)')
        axes[1, col].axis('off')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=axes[1, col], fraction=0.046, pad=0.04)
        cbar.set_label('Attribution', rotation=270, labelpad=15)
    
    plt.suptitle('Malware Detection: SHAP Explanations', fontsize=16, y=0.95)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"Saved SHAP comparison plot to: {out_path}")
    plt.close()


if __name__ == "__main__":
    set_seeds(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    root = os.path.dirname(__file__)
    weights = os.path.join(root, 'malware_cnn_model.pth')
    data_dir = os.path.join(root, 'Explainability', 'images')
    if not os.path.exists(weights):
        raise FileNotFoundError(f"Weights not found: {weights}")

    model = build_model(weights, device)
    background, to_explain, labels = load_data(data_dir, device, bg_size=50)
    print(f"Background: {tuple(background.shape)}, To explain: {tuple(to_explain.shape)}")
    print(f"Labels: {labels.numpy()} (0=Benign, 1=Malware)")

    out_file = os.path.join(root, 'Explainability', 'shap_cnn_comparison_plot.png')
    compute_shap_and_plot(model, background, to_explain, labels, out_file)
