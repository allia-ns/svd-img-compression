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
        compression_info: dict with detailed compression stats
    """
    
    # Validate input
    if not isinstance(image_array, np.ndarray):
        raise ValueError("Input must be a numpy array")
    
    if not (1 <= compression_ratio <= 100):
        raise ValueError("Compression ratio must be between 1 and 100")
    
    # Handle grayscale vs color images
    if len(image_array.shape) == 2:
        # Grayscale image
        compressed_array, info = compress_channel_svd(image_array, compression_ratio)
        compression_info = {
            'k_values': [info['k']],
            'original_elements': info['original_elements'],
            'compressed_elements': info['compressed_elements'],
            'channels': 1
        }
        return compressed_array, compression_info
    
    elif len(image_array.shape) == 3:
        # Color image - process each channel separately
        compressed_channels = []
        k_values = []
        total_original_elements = 0
        total_compressed_elements = 0
        
        for channel in range(image_array.shape[2]):
            channel_data = image_array[:, :, channel]
            compressed_channel, info = compress_channel_svd(channel_data, compression_ratio)
            compressed_channels.append(compressed_channel)
            k_values.append(info['k'])
            total_original_elements += info['original_elements']
            total_compressed_elements += info['compressed_elements']
        
        # Stack channels back together
        compressed_array = np.stack(compressed_channels, axis=2)
        compression_info = {
            'k_values': k_values,
            'original_elements': total_original_elements,
            'compressed_elements': total_compressed_elements,
            'channels': image_array.shape[2]
        }
        return compressed_array, compression_info
    
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
        info: dict with compression details for this channel
    """
    
    try:
        if np.any(np.isnan(channel_data)) or np.any(np.isinf(channel_data)):
            raise ValueError("Image contains invalid values (NaN or Inf)")
        # Ensure data is float for SVD calculations
        channel_data = channel_data.astype(np.float64)
        
        # Get dimensions
        m, n = channel_data.shape
        
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
        
        # Calculate actual compression statistics
        original_elements = m * n  # Original matrix size
        compressed_elements = (m * k) + k + (k * n)  # U_k + S_k + Vt_k sizes
        
        info = {
            'k': k,
            'total_singular_values': total_singular_values,
            'original_elements': original_elements,
            'compressed_elements': compressed_elements,
            'dimensions': (m, n)
        }
        
        return reconstructed_channel, info
        
    except Exception as e:
        raise RuntimeError(f"SVD compression failed: {str(e)}")

def calculate_compression_stats(original_shape, compression_info, runtime, file_size_before=None, file_size_after=None):
    """
    Calculate comprehensive compression statistics
    
    Args:
        original_shape: tuple of original image dimensions
        compression_info: compression info from compress_image_svd
        runtime: processing time in seconds
        file_size_before: optional file size before compression
        file_size_after: optional file size after compression
    
    Returns:
        dict with comprehensive compression statistics
    """
    
    # Mathematical compression (the real SVD compression)
    original_elements = compression_info['original_elements']
    compressed_elements = compression_info['compressed_elements']
    
    mathematical_compression_ratio = (original_elements - compressed_elements) / original_elements * 100
    mathematical_space_savings = original_elements / compressed_elements if compressed_elements > 0 else 0
    
    # Calculate average k across channels
    avg_k = np.mean(compression_info['k_values'])
    max_possible_k = min(original_shape[:2])
    k_percentage = (avg_k / max_possible_k) * 100
    
    stats = {
        # Mathematical compression (the real compression)
        'mathematical_compression_ratio': mathematical_compression_ratio,
        'mathematical_space_savings': mathematical_space_savings,
        'original_elements': original_elements,
        'compressed_elements': compressed_elements,
        
        # SVD-specific stats
        'k_values': compression_info['k_values'],
        'avg_k': avg_k,
        'max_possible_k': max_possible_k,
        'k_percentage': k_percentage,
        'channels': compression_info['channels'],
        
        # Performance
        'runtime': runtime,
        
        # Image info
        'original_shape': original_shape,
    }
    
    # File size comparison (if provided)
    if file_size_before and file_size_after:
        file_compression_ratio = (file_size_before - file_size_after) / file_size_before * 100
        stats.update({
            'file_size_before': file_size_before,
            'file_size_after': file_size_after,
            'file_compression_ratio': file_compression_ratio,
        })
    
    return stats

def create_compression_summary(stats):
    """
    Create a human-readable summary of compression results
    
    Args:
        stats: compression statistics dictionary
    
    Returns:
        formatted string summary
    """
    
    summary = f"""
📊 SVD Compression Results:

🔢 Mathematical Compression:
   • Space reduction: {stats['mathematical_compression_ratio']:.1f}%
   • Compression ratio: {stats['mathematical_space_savings']:.2f}:1
   • Original data elements: {stats['original_elements']:,}
   • Compressed data elements: {stats['compressed_elements']:,}

🎯 SVD Parameters:
   • Singular values used (k): {stats['avg_k']:.0f} out of {stats['max_possible_k']}
   • K percentage: {stats['k_percentage']:.1f}%
   • Channels processed: {stats['channels']}
   • Per-channel k values: {stats['k_values']}

⚡ Performance:
   • Processing time: {stats['runtime']:.2f} seconds
   • Image dimensions: {stats['original_shape']}
"""
    
    if 'file_compression_ratio' in stats:
        summary += f"""
💾 File Size Comparison:
   • Before: {stats['file_size_before']:,} bytes
   • After: {stats['file_size_after']:,} bytes  
   • File size change: {stats['file_compression_ratio']:.1f}%
   
📝 Note: File size may increase due to PNG encoding of reconstructed image.
The real compression is in the mathematical reduction shown above!
"""
    
    return summary

def create_comparison_plot(original_image, compressed_image, stats):
    """
    Create comprehensive comparison plot with statistics
    
    Args:
        original_image: PIL Image object
        compressed_image: PIL Image object  
        stats: compression statistics dictionary
    
    Returns:
        matplotlib figure object
    """
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Original image
    axes[0, 0].imshow(original_image)
    axes[0, 0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Compressed image
    axes[0, 1].imshow(compressed_image)
    axes[0, 1].set_title(
        f'Compressed Image (k={stats["avg_k"]:.0f})\n'
        f'Mathematical Compression: {stats["mathematical_compression_ratio"]:.1f}%',
        fontsize=14, fontweight='bold'
    )
    axes[0, 1].axis('off')
    
    # Compression statistics bar chart
    categories = ['Original\nElements', 'Compressed\nElements']
    values = [stats['original_elements'], stats['compressed_elements']]
    colors = ['lightcoral', 'lightblue']
    
    bars = axes[1, 0].bar(categories, values, color=colors)
    axes[1, 0].set_title('Data Elements Comparison', fontweight='bold')
    axes[1, 0].set_ylabel('Number of Elements')
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:,}', ha='center', va='bottom')
    
    # Performance and compression metrics
    axes[1, 1].axis('off')
    metrics_text = f"""
Compression Metrics:

Mathematical Reduction: {stats['mathematical_compression_ratio']:.1f}%
Space Savings: {stats['mathematical_space_savings']:.2f}:1

SVD Parameters:
K values used: {stats['k_values']}
Max possible K: {stats['max_possible_k']}
K percentage: {stats['k_percentage']:.1f}%

Performance:
Runtime: {stats['runtime']:.2f}s
Channels: {stats['channels']}
Image size: {stats['original_shape']}
"""
    
    axes[1, 1].text(0.1, 0.9, metrics_text, transform=axes[1, 1].transAxes,
                    fontsize=11, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    return fig

def calculate_image_quality_metrics(original_array, compressed_array):
    """
    Calculate image quality metrics
    
    Args:
        original_array: original image as numpy array
        compressed_array: compressed image as numpy array
    
    Returns:
        dict with quality metrics
    """
    
    # Ensure same data type for comparison
    original_array = original_array.astype(np.float64)
    compressed_array = compressed_array.astype(np.float64)
    
    # Mean Squared Error
    mse = np.mean((original_array - compressed_array) ** 2)
    
    # Peak Signal-to-Noise Ratio
    if mse == 0:
        psnr = float('inf')
    else:
        max_pixel = 255.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    
    # Structural Similarity (simplified version)
    mean_orig = np.mean(original_array)
    mean_comp = np.mean(compressed_array)
    var_orig = np.var(original_array)
    var_comp = np.var(compressed_array)
    
    return {
        'mse': mse,
        'psnr': psnr,
        'mean_difference': abs(mean_orig - mean_comp),
        'variance_ratio': var_comp / var_orig if var_orig > 0 else 0
    }

def get_optimal_k_suggestions(image_shape, target_ratios=[10, 25, 50, 75, 90]):
    """
    Suggest optimal k values for different compression ratios based on typical use cases
    
    Args:
        image_shape: tuple of image dimensions
        target_ratios: list of target compression ratios
    
    Returns:
        dict mapping ratios to suggested k values and descriptions
    """
    
    max_k = min(image_shape[:2])
    suggestions = {}
    
    descriptions = {
        10: "Heavy compression - basic structure only",
        25: "High compression - good for thumbnails", 
        50: "Medium compression - balanced quality/size",
        75: "Light compression - high quality",
        90: "Minimal compression - near original quality"
    }
    
    for ratio in target_ratios:
        k = max(1, int(max_k * ratio / 100))
        suggestions[ratio] = {
            'k': k,
            'description': descriptions.get(ratio, f"{ratio}% compression"),
            'estimated_quality': 'High' if ratio >= 75 else 'Medium' if ratio >= 50 else 'Low'
        }
    
    return suggestions

def analyze_singular_values(S, k):
    """Analyze singular value distribution"""
    return {
        'total_energy': np.sum(S**2),
        'retained_energy': np.sum(S[:k]**2),
        'energy_ratio': np.sum(S[:k]**2) / np.sum(S**2) * 100,
        'largest_sv': S[0],
        'smallest_retained_sv': S[k-1] if k > 0 else 0
    }
