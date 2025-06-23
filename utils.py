import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def compress_image_svd(image_array, compression_ratio):
    """
    Kompresi gambar menggunakan dekomposisi SVD
    """
    
    # Validasi input
    if not isinstance(image_array, np.ndarray):
        raise ValueError("Input harus berupa numpy array")
    
    if not (1 <= compression_ratio <= 100):
        raise ValueError("Rasio kompresi harus antara 1 dan 100")
    
    # Handle grayscale vs color
    if len(image_array.shape) == 2:
        # Gambar grayscale
        compressed_array, info = compress_channel_svd(image_array, compression_ratio)
        compression_info = {
            'k_values': [info['k']],
            'original_elements': info['original_elements'],
            'compressed_elements': info['compressed_elements'],
            'channels': 1
        }
        return compressed_array, compression_info
    
    elif len(image_array.shape) == 3:
        # Gambar berwarna - proses setiap channel terpisah
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
        
        # Stack channels kembali
        compressed_array = np.stack(compressed_channels, axis=2)
        compression_info = {
            'k_values': k_values,
            'original_elements': total_original_elements,
            'compressed_elements': total_compressed_elements,
            'channels': image_array.shape[2]
        }
        return compressed_array, compression_info
    
    else:
        raise ValueError("Gambar harus 2D (grayscale) atau 3D (berwarna)")

def compress_channel_svd(channel_data, compression_ratio):
    """
    Kompresi single channel menggunakan SVD
    """
    
    try:
        if np.any(np.isnan(channel_data)) or np.any(np.isinf(channel_data)):
            raise ValueError("Gambar mengandung nilai tidak valid (NaN atau Inf)")
        
        # Konversi ke float untuk kalkulasi SVD
        channel_data = channel_data.astype(np.float64)
        
        # Dapatkan dimensi
        m, n = channel_data.shape
        
        # Lakukan dekomposisi SVD
        U, S, Vt = np.linalg.svd(channel_data, full_matrices=False)
        
        # Hitung jumlah singular values yang dipertahankan
        total_singular_values = len(S)
        k = max(1, int(total_singular_values * compression_ratio / 100))
        
        # Ambil hanya k singular values teratas
        U_k = U[:, :k]
        S_k = S[:k]
        Vt_k = Vt[:k, :]
        
        # Rekonstruksi channel
        reconstructed_channel = U_k @ np.diag(S_k) @ Vt_k
        
        # Pastikan nilai dalam range [0, 255]
        reconstructed_channel = np.clip(reconstructed_channel, 0, 255)
        
        # Hitung statistik kompresi
        original_elements = m * n
        compressed_elements = (m * k) + k + (k * n)  # U_k + S_k + Vt_k
        
        info = {
            'k': k,
            'total_singular_values': total_singular_values,
            'original_elements': original_elements,
            'compressed_elements': compressed_elements,
            'dimensions': (m, n)
        }
        
        return reconstructed_channel, info
        
    except Exception as e:
        raise RuntimeError(f"Kompresi SVD gagal: {str(e)}")

def calculate_compression_stats(original_shape, compression_info, runtime, file_size_before=None, file_size_after=None):
    """
    Hitung statistik kompresi komprehensif
    """
    
    # Kompresi matematis (kompresi SVD sebenarnya)
    original_elements = compression_info['original_elements']
    compressed_elements = compression_info['compressed_elements']
    
    mathematical_compression_ratio = (original_elements - compressed_elements) / original_elements * 100
    mathematical_space_savings = original_elements / compressed_elements if compressed_elements > 0 else 0
    
    # Hitung rata-rata k across channels
    avg_k = np.mean(compression_info['k_values'])
    max_possible_k = min(original_shape[:2])
    k_percentage = (avg_k / max_possible_k) * 100
    
    stats = {
        # Kompresi matematis (kompresi sebenarnya)
        'mathematical_compression_ratio': mathematical_compression_ratio,
        'mathematical_space_savings': mathematical_space_savings,
        'original_elements': original_elements,
        'compressed_elements': compressed_elements,
        
        # Stats khusus SVD
        'k_values': compression_info['k_values'],
        'avg_k': avg_k,
        'max_possible_k': max_possible_k,
        'k_percentage': k_percentage,
        'channels': compression_info['channels'],
        
        # Performa
        'runtime': runtime,
        
        # Info gambar
        'original_shape': original_shape,
    }
    
    # Perbandingan ukuran file (jika disediakan)
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
    Buat ringkasan hasil kompresi yang mudah dibaca
    """
    
    summary = f"""
📊 Hasil Kompresi SVD:

🔢 Kompresi Matematis:
   • Reduksi ruang: {stats['mathematical_compression_ratio']:.1f}%
   • Rasio kompresi: {stats['mathematical_space_savings']:.2f}:1
   • Elemen data asli: {stats['original_elements']:,}
   • Elemen data terkompresi: {stats['compressed_elements']:,}

🎯 Parameter SVD:
   • Singular values digunakan (k): {stats['avg_k']:.0f} dari {stats['max_possible_k']}
   • Persentase K: {stats['k_percentage']:.1f}%
   • Channel diproses: {stats['channels']}
   • Nilai k per channel: {stats['k_values']}

⚡ Performa:
   • Waktu pemrosesan: {stats['runtime']:.2f} detik
   • Dimensi gambar: {stats['original_shape']}
"""
    
    if 'file_compression_ratio' in stats:
        summary += f"""
💾 Perbandingan Ukuran File:
   • Sebelum: {stats['file_size_before']:,} bytes
   • Sesudah: {stats['file_size_after']:,} bytes  
   • Perubahan ukuran file: {stats['file_compression_ratio']:.1f}%
   
📝 Catatan: Ukuran file mungkin bertambah karena encoding PNG dari gambar yang direkonstruksi.
Kompresi sebenarnya terlihat pada reduksi matematis di atas!
"""
    
    return summary

def create_comparison_plot(original_image, compressed_image, stats):
    """
    Buat plot perbandingan komprehensif dengan statistik
    """
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Gambar asli
    axes[0, 0].imshow(original_image)
    axes[0, 0].set_title('Gambar Asli', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Gambar terkompresi
    axes[0, 1].imshow(compressed_image)
    axes[0, 1].set_title(
        f'Gambar Terkompresi (k={stats["avg_k"]:.0f})\n'
        f'Kompresi Matematis: {stats["mathematical_compression_ratio"]:.1f}%',
        fontsize=14, fontweight='bold'
    )
    axes[0, 1].axis('off')
    
    # Bar chart statistik kompresi
    categories = ['Elemen\nAsli', 'Elemen\nTerkompresi']
    values = [stats['original_elements'], stats['compressed_elements']]
    colors = ['lightcoral', 'lightblue']
    
    bars = axes[1, 0].bar(categories, values, color=colors)
    axes[1, 0].set_title('Perbandingan Elemen Data', fontweight='bold')
    axes[1, 0].set_ylabel('Jumlah Elemen')
    
    # Tambah label nilai pada bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:,}', ha='center', va='bottom')
    
    # Metrik performa dan kompresi
    axes[1, 1].axis('off')
    metrics_text = f"""
Metrik Kompresi:

Reduksi Matematis: {stats['mathematical_compression_ratio']:.1f}%
Penghematan Ruang: {stats['mathematical_space_savings']:.2f}:1

Parameter SVD:
Nilai K digunakan: {stats['k_values']}
K maksimum: {stats['max_possible_k']}
Persentase K: {stats['k_percentage']:.1f}%

Performa:
Runtime: {stats['runtime']:.2f}s
Channel: {stats['channels']}
Ukuran gambar: {stats['original_shape']}
"""
    
    axes[1, 1].text(0.1, 0.9, metrics_text, transform=axes[1, 1].transAxes,
                    fontsize=11, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    return fig

def calculate_image_quality_metrics(original_array, compressed_array):
    """
    Hitung metrik kualitas gambar
    """
    
    # Pastikan tipe data sama untuk perbandingan
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
    
    # Structural Similarity (versi sederhana)
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
    Saran nilai k optimal untuk rasio kompresi berbeda berdasarkan kasus penggunaan
    """
    
    max_k = min(image_shape[:2])
    suggestions = {}
    
    descriptions = {
        10: "Kompresi berat - struktur dasar saja",
        25: "Kompresi tinggi - bagus untuk thumbnail", 
        50: "Kompresi sedang - kualitas/ukuran seimbang",
        75: "Kompresi ringan - kualitas tinggi",
        90: "Kompresi minimal - mendekati kualitas asli"
    }
    
    for ratio in target_ratios:
        k = max(1, int(max_k * ratio / 100))
        suggestions[ratio] = {
            'k': k,
            'description': descriptions.get(ratio, f"Kompresi {ratio}%"),
            'estimated_quality': 'Tinggi' if ratio >= 75 else 'Sedang' if ratio >= 50 else 'Rendah'
        }
    
    return suggestions

def analyze_singular_values(S, k):
    """Analisis distribusi singular values"""
    return {
        'total_energy': np.sum(S**2),
        'retained_energy': np.sum(S[:k]**2),
        'energy_ratio': np.sum(S[:k]**2) / np.sum(S**2) * 100,
        'largest_sv': S[0],
        'smallest_retained_sv': S[k-1] if k > 0 else 0
    }
