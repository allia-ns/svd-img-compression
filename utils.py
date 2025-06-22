import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def compress_image_svd(image_array, compression_ratio):
    """
    Compress image using SVD decomposition
    
    Args:
        image_array: numpy array of image (H, W) or (H, W, C)
        compression_ratio: percentage of singular values to keep (1-100)
    
    Returns:
        compressed_array: reconstructed image array
        k_values: list of k values used for each channel
    """
    
    # Validate input
    if not isinstance(image_array, np.ndarray):
        raise ValueError("Input must be a numpy array")
    
    if not (1 <= compression_ratio <= 100):
        raise ValueError("Compression ratio must be between 1 and 100")
    
    # Handle grayscale vs color images
    if len(image_array.shape) == 2:
        # Grayscale image
        compressed_array, k = compress_channel_svd(image_array, compression_ratio)
        return compressed_array, [k]
    
    elif len(image_array.shape) == 3:
        # Color image - process each channel separately
        compressed_channels = []
        k_values = []
        
        for channel in range(image_array.shape[2]):
            channel_data = image_array[:, :, channel]
            compressed_channel, k = compress_channel_svd(channel_data, compression_ratio)
            compressed_channels.append(compressed_channel)
            k_values.append(k)
        
        # Stack channels back together
        compressed_array = np.stack(compressed_channels, axis=2)
        return compressed_array, k_values
    
    else:
        raise ValueError("Image must be 2D (grayscale) or 3D (color)")

def compress_channel_svd(channel_data, compression_ratio):
    """
    Compress single channel using SVD
    
    Args:
        channel_data: 2D numpy array representing one channel
        compression_ratio: percentage of singular values to keep
    
    Returns:
        reconstructed_channel: compressed channel data
        k: number of singular values used
    """
    
    try:
        # Ensure data is float for SVD calculations
        channel_data = channel_data.astype(np.float64)
        
        # Perform SVD decomposition
        U, S, Vt = np.linalg.svd(channel_data, full_matrices=False)
        
        # Calculate number of singular values to keep
        total_singular_values = len(S)
        k = max(1, int(total_singular_values * compression_ratio / 100))
        
        # Keep only top k singular values
        U_k = U[:, :k]
        S_k = S[:k]
        Vt_k = Vt[:k, :]
        
        # Reconstruct the channel
        reconstructed_channel = U_k @ np.diag(S_k) @ Vt_k
        
        # Ensure values are in valid range [0, 255]
        reconstructed_channel = np.clip(reconstructed_channel, 0, 255)
        
        return reconstructed_channel, k
        
    except Exception as e:
        raise RuntimeError(f"SVD compression failed: {str(e)}")

def calculate_compression_stats(original_size, compressed_size, runtime):
    """
    Calculate compression statistics
    
    Args:
        original_size: size in bytes of original image
        compressed_size: estimated size in bytes of compressed representation
        runtime: processing time in seconds
    
    Returns:
        dict with compression statistics
    """
    
    size_reduction = ((original_size - compressed_size) / original_size) * 100
    compression_ratio = original_size / compressed_size if compressed_size > 0 else 0
    
    return {
        'original_size': original_size,
        'compressed_size': compressed_size,
        'size_reduction': size_reduction,
        'compression_ratio': compression_ratio,
        'runtime': runtime
    }

def create_comparison_plot(original_image, compressed_image, stats):
    """
    Create side-by-side comparison plot
    
    Args:
        original_image: PIL Image object
        compressed_image: PIL Image object  
        stats: compression statistics dictionary
    
    Returns:
        matplotlib figure object
    """
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Original image
    axes[0].imshow(original_image)
    axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Compressed image
    axes[1].imshow(compressed_image)
    axes[1].set_title(
        f'Compressed Image\n'
        f'Size Reduction: {stats["size_reduction"]:.1f}%\n'
        f'Runtime: {stats["runtime"]:.3f}s',
        fontsize=14, fontweight='bold'
    )
    axes[1].axis('off')
    
    plt.tight_layout()
    return fig

def calculate_image_quality_metrics(original_array, compressed_array):
    """
    Calculate image quality metrics (optional advanced feature)
    
    Args:
        original_array: original image as numpy array
        compressed_array: compressed image as numpy array
    
    Returns:
        dict with quality metrics
    """
    
    # Mean Squared Error
    mse = np.mean((original_array - compressed_array) ** 2)
    
    # Peak Signal-to-Noise Ratio
    if mse == 0:
        psnr = float('inf')
    else:
        max_pixel = 255.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    
    return {
        'mse': mse,
        'psnr': psnr
    }

def get_optimal_k_suggestions(image_shape, target_ratios=[10, 25, 50, 75, 90]):
    """
    Suggest optimal k values for different compression ratios
    
    Args:
        image_shape: tuple of image dimensions
        target_ratios: list of target compression ratios
    
    Returns:
        dict mapping ratios to suggested k values
    """
    
    max_k = min(image_shape[:2])
    suggestions = {}
    
    for ratio in target_ratios:
        k = max(1, int(max_k * ratio / 100))
        suggestions[ratio] = k
    
    return suggestions
