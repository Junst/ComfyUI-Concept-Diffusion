"""
Utility functions for ConceptAttention ComfyUI nodes
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

def resize_saliency_map(saliency_map: torch.Tensor, target_size: Tuple[int, int]) -> torch.Tensor:
    """
    Resize saliency map to target size.
    
    Args:
        saliency_map: Input saliency map tensor
        target_size: Target (height, width) tuple
    
    Returns:
        Resized saliency map
    """
    if len(saliency_map.shape) == 2:
        saliency_map = saliency_map.unsqueeze(0).unsqueeze(0)
    elif len(saliency_map.shape) == 3:
        saliency_map = saliency_map.unsqueeze(0)
    
    resized = F.interpolate(
        saliency_map,
        size=target_size,
        mode='bilinear',
        align_corners=False
    )
    
    return resized.squeeze()

def normalize_saliency_map(saliency_map: torch.Tensor, method: str = "minmax") -> torch.Tensor:
    """
    Normalize saliency map.
    
    Args:
        saliency_map: Input saliency map
        method: Normalization method ("minmax", "zscore", "softmax")
    
    Returns:
        Normalized saliency map
    """
    if method == "minmax":
        min_val = saliency_map.min()
        max_val = saliency_map.max()
        if max_val > min_val:
            return (saliency_map - min_val) / (max_val - min_val)
        else:
            return saliency_map
    
    elif method == "zscore":
        mean_val = saliency_map.mean()
        std_val = saliency_map.std()
        if std_val > 0:
            return (saliency_map - mean_val) / std_val
        else:
            return saliency_map
    
    elif method == "softmax":
        return F.softmax(saliency_map.flatten(), dim=0).reshape(saliency_map.shape)
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def create_concept_colors(concepts: List[str]) -> Dict[str, Tuple[float, float, float]]:
    """
    Create color mapping for concepts.
    
    Args:
        concepts: List of concept names
    
    Returns:
        Dictionary mapping concept names to RGB colors
    """
    # Predefined colors
    predefined_colors = {
        'person': (1.0, 0.0, 0.0),      # Red
        'car': (0.0, 1.0, 0.0),         # Green
        'tree': (0.0, 0.0, 1.0),        # Blue
        'sky': (1.0, 1.0, 0.0),         # Yellow
        'building': (1.0, 0.0, 1.0),    # Magenta
        'road': (0.5, 0.5, 0.5),        # Gray
        'grass': (0.0, 0.8, 0.0),       # Dark Green
        'water': (0.0, 0.5, 1.0),       # Light Blue
        'mountain': (0.6, 0.4, 0.2),    # Brown
        'cloud': (0.9, 0.9, 0.9),       # Light Gray
    }
    
    colors = {}
    used_colors = set()
    
    # Assign predefined colors first
    for concept in concepts:
        concept_lower = concept.lower()
        if concept_lower in predefined_colors:
            colors[concept] = predefined_colors[concept_lower]
            used_colors.add(predefined_colors[concept_lower])
    
    # Generate colors for remaining concepts
    remaining_concepts = [c for c in concepts if c not in colors]
    for i, concept in enumerate(remaining_concepts):
        # Generate a unique color
        hue = (i * 137.5) % 360  # Golden angle for good distribution
        rgb = hsv_to_rgb(hue, 0.8, 0.9)
        colors[concept] = rgb
    
    return colors

def hsv_to_rgb(h: float, s: float, v: float) -> Tuple[float, float, float]:
    """
    Convert HSV to RGB.
    
    Args:
        h: Hue (0-360)
        s: Saturation (0-1)
        v: Value (0-1)
    
    Returns:
        RGB tuple (0-1)
    """
    h = h / 360.0
    c = v * s
    x = c * (1 - abs((h * 6) % 2 - 1))
    m = v - c
    
    if h < 1/6:
        r, g, b = c, x, 0
    elif h < 2/6:
        r, g, b = x, c, 0
    elif h < 3/6:
        r, g, b = 0, c, x
    elif h < 4/6:
        r, g, b = 0, x, c
    elif h < 5/6:
        r, g, b = x, 0, c
    else:
        r, g, b = c, 0, x
    
    return (r + m, g + m, b + m)

def apply_gaussian_blur(saliency_map: torch.Tensor, kernel_size: int = 5, sigma: float = 1.0) -> torch.Tensor:
    """
    Apply Gaussian blur to saliency map.
    
    Args:
        saliency_map: Input saliency map
        kernel_size: Size of Gaussian kernel
        sigma: Standard deviation of Gaussian kernel
    
    Returns:
        Blurred saliency map
    """
    if len(saliency_map.shape) == 2:
        saliency_map = saliency_map.unsqueeze(0).unsqueeze(0)
    elif len(saliency_map.shape) == 3:
        saliency_map = saliency_map.unsqueeze(0)
    
    # Create Gaussian kernel
    kernel = create_gaussian_kernel(kernel_size, sigma, saliency_map.device)
    
    # Apply convolution
    blurred = F.conv2d(saliency_map, kernel, padding=kernel_size//2)
    
    return blurred.squeeze()

def create_gaussian_kernel(kernel_size: int, sigma: float, device: torch.device) -> torch.Tensor:
    """
    Create Gaussian kernel for blurring.
    
    Args:
        kernel_size: Size of kernel
        sigma: Standard deviation
        device: Device to create kernel on
    
    Returns:
        Gaussian kernel tensor
    """
    # Create coordinate grids
    x = torch.arange(kernel_size, dtype=torch.float32, device=device)
    y = torch.arange(kernel_size, dtype=torch.float32, device=device)
    x, y = torch.meshgrid(x, y, indexing='ij')
    
    # Center the kernel
    center = kernel_size // 2
    x = x - center
    y = y - center
    
    # Calculate Gaussian
    kernel = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    
    # Reshape for conv2d
    kernel = kernel.unsqueeze(0).unsqueeze(0)
    
    return kernel

def compute_attention_statistics(saliency_maps: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, float]]:
    """
    Compute statistics for saliency maps.
    
    Args:
        saliency_maps: Dictionary of concept -> saliency map
    
    Returns:
        Dictionary of statistics for each concept
    """
    stats = {}
    
    for concept, saliency_map in saliency_maps.items():
        saliency_flat = saliency_map.flatten()
        
        stats[concept] = {
            'mean': saliency_flat.mean().item(),
            'std': saliency_flat.std().item(),
            'min': saliency_flat.min().item(),
            'max': saliency_flat.max().item(),
            'median': saliency_flat.median().item(),
            'sum': saliency_flat.sum().item(),
        }
    
    return stats

def threshold_saliency_map(saliency_map: torch.Tensor, threshold: float, method: str = "absolute") -> torch.Tensor:
    """
    Apply threshold to saliency map.
    
    Args:
        saliency_map: Input saliency map
        threshold: Threshold value
        method: Threshold method ("absolute", "percentile", "otsu")
    
    Returns:
        Thresholded saliency map
    """
    if method == "absolute":
        return (saliency_map > threshold).float()
    
    elif method == "percentile":
        threshold_val = torch.quantile(saliency_map.flatten(), threshold)
        return (saliency_map > threshold_val).float()
    
    elif method == "otsu":
        # Simple Otsu thresholding
        saliency_flat = saliency_map.flatten()
        hist = torch.histc(saliency_flat, bins=256, min=0, max=1)
        hist = hist / hist.sum()
        
        # Find optimal threshold
        best_threshold = 0
        best_variance = 0
        
        for t in range(256):
            w0 = hist[:t+1].sum()
            w1 = hist[t+1:].sum()
            
            if w0 == 0 or w1 == 0:
                continue
            
            mu0 = (torch.arange(t+1, dtype=torch.float32) * hist[:t+1]).sum() / w0
            mu1 = (torch.arange(t+1, 256, dtype=torch.float32) * hist[t+1:]).sum() / w1
            
            variance = w0 * w1 * (mu0 - mu1) ** 2
            
            if variance > best_variance:
                best_variance = variance
                best_threshold = t / 255.0
        
        return (saliency_map > best_threshold).float()
    
    else:
        raise ValueError(f"Unknown threshold method: {method}")

def save_saliency_maps(saliency_maps: Dict[str, torch.Tensor], output_dir: str, prefix: str = "concept"):
    """
    Save saliency maps to files.
    
    Args:
        saliency_maps: Dictionary of concept -> saliency map
        output_dir: Output directory
        prefix: Filename prefix
    """
    import os
    from PIL import Image
    
    os.makedirs(output_dir, exist_ok=True)
    
    for concept, saliency_map in saliency_maps.items():
        # Convert to PIL Image
        saliency_np = saliency_map.cpu().numpy()
        saliency_np = (saliency_np * 255).astype(np.uint8)
        
        # Create PIL Image
        img = Image.fromarray(saliency_np, mode='L')
        
        # Save
        filename = f"{prefix}_{concept}.png"
        filepath = os.path.join(output_dir, filename)
        img.save(filepath)
        
        logger.info(f"Saved saliency map: {filepath}")
