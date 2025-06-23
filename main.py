import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time
import io
from utils import (
    compress_image_svd, 
    calculate_compression_stats, 
    create_comparison_plot,
    create_compression_summary,
    calculate_image_quality_metrics,
    get_optimal_k_suggestions
)

def resize_image_if_needed(image, max_dimension=800):
    """
    Mengubah ukuran gambar jika terlalu besar untuk mencegah masalah memori
    """
    width, height = image.size
    
    if max(width, height) > max_dimension:
        # Hitung dimensi baru sambil mempertahankan rasio aspek
        if width > height:
            new_width = max_dimension
            new_height = int(height * max_dimension / width)
        else:
            new_height = max_dimension
            new_width = int(width * max_dimension / height)
        
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        return image, True
    
    return image, False

def main():
    st.set_page_config(
        page_title="SVD Image Compression",
        page_icon="🖼️",
        layout="wide"
    )
    
    st.title("🖼️ Kompresi Gambar dengan SVD")
    st.write("Kompres gambar menggunakan Singular Value Decomposition")
    
    # Inisialisasi session state untuk menyimpan hasil
    if 'compression_results' not in st.session_state:
        st.session_state.compression_results = None
    if 'original_image' not in st.session_state:
        st.session_state.original_image = None
    if 'compressed_image' not in st.session_state:
        st.session_state.compressed_image = None
    
    # Sidebar untuk kontrol
    with st.sidebar:
        st.header("⚙️ Pengaturan")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Pilih file gambar",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff']
        )
        
        # Info ukuran file
        if uploaded_file is not None:
            file_size_mb = uploaded_file.size / (1024 * 1024)
            st.info(f"📁 Ukuran file: {file_size_mb:.2f} MB")
            
            # Peringatan untuk file besar
            if file_size_mb > 2:
                st.warning("⚠️ File besar terdeteksi! Gambar akan otomatis diresize untuk pemrosesan.")
        
        # Slider rasio kompresi
        compression_ratio = st.slider(
            "Rasio Kompresi (%)",
            min_value=1,
            max_value=100,
            value=50,
            step=1,
            help="Persentase lebih tinggi = kompresi lebih sedikit, kualitas lebih baik"
        )
        
        # Tampilkan saran k optimal
        if uploaded_file is not None:
            try:
                temp_image = Image.open(uploaded_file)
                temp_processed, _ = resize_image_if_needed(temp_image)
                temp_array = np.array(temp_processed)
                
                suggestions = get_optimal_k_suggestions(temp_array.shape)
                
                with st.expander("💡 Panduan Kompresi", expanded=False):
                    for ratio, info in suggestions.items():
                        quality_emoji = "🟢" if info['estimated_quality'] == 'High' else "🟡" if info['estimated_quality'] == 'Medium' else "🔴"
                        st.write(f"{quality_emoji} **{ratio}%**: {info['description']} (k≈{info['k']})")
            except:
                pass  # Skip jika gambar tidak bisa dimuat sementara
        
        # Tombol proses
        process_button = st.button("🚀 Kompres Gambar", type="primary")
        
        # Tombol hapus hasil
        if st.session_state.compression_results is not None:
            if st.button("🗑️ Hapus Hasil"):
                st.session_state.compression_results = None
                st.session_state.original_image = None
                st.session_state.compressed_image = None
                st.rerun()
    
    # Area konten utama
    if uploaded_file is not None:
        # Muat gambar
        try:
            original_image = Image.open(uploaded_file)
            
            # Auto-resize jika diperlukan
            processed_image, was_resized = resize_image_if_needed(original_image)
            
            if was_resized:
                st.warning(f"🔄 Gambar otomatis diresize dari {original_image.size} ke {processed_image.size} untuk pemrosesan optimal")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📸 Gambar Asli")
                st.image(processed_image, use_container_width=True)
                st.write(f"**Dimensi:** {processed_image.size[0]} × {processed_image.size[1]}")
                
            with col2:
                st.subheader("🗜️ Gambar Terkompresi")
                
                if process_button:
                    # Tampilkan progress
                    progress_container = st.container()
                    with progress_container:
                        progress_bar = st.progress(0, text="Memulai kompresi...")
                        
                        # Konversi PIL ke numpy array
                        img_array = np.array(processed_image)
                        progress_bar.progress(10, text="Gambar dimuat...")
                        
                        # Kompres gambar dengan error handling
                        try:
                            start_time = time.time()
                            progress_bar.progress(20, text="Melakukan dekomposisi SVD...")
                            
                            # Gunakan return values yang benar dari utils
                            compressed_array, compression_info = compress_image_svd(img_array, compression_ratio)
                            progress_bar.progress(70, text="Merekonstruksi gambar...")
                            
                            end_time = time.time()
                            runtime = end_time - start_time
                            
                            # Konversi kembali ke PIL
                            compressed_image = Image.fromarray(compressed_array.astype(np.uint8))
                            progress_bar.progress(80, text="Menghitung statistik...")
                            
                            # Gunakan parameter yang benar untuk perhitungan stats
                            stats = calculate_compression_stats(
                                original_shape=img_array.shape,
                                compression_info=compression_info,
                                runtime=runtime,
                                file_size_before=uploaded_file.size,
                                file_size_after=len(io.BytesIO().getvalue()) if compressed_image else None
                            )
                            
                            progress_bar.progress(90, text="Menghitung metrik kualitas...")
                            
                            # Hitung metrik kualitas gambar
                            quality_metrics = calculate_image_quality_metrics(
                                img_array.astype(np.float64), 
                                compressed_array.astype(np.float64)
                            )
                            
                            # Simpan hasil dalam session state
                            st.session_state.compression_results = {
                                'stats': stats,
                                'compression_info': compression_info,
                                'quality_metrics': quality_metrics,
                                'img_array': img_array,
                                'runtime': runtime
                            }
                            st.session_state.original_image = processed_image
                            st.session_state.compressed_image = compressed_image
                            
                            progress_bar.progress(100, text="✅ Kompresi selesai!")
                            time.sleep(0.5)  # Jeda singkat untuk menampilkan penyelesaian
                            
                        except Exception as e:
                            st.error(f"❌ Kompresi gagal: {str(e)}")
                            st.info("💡 Coba dengan gambar yang lebih kecil atau rasio kompresi yang berbeda")
                            import traceback
                            st.error(f"Info debug: {traceback.format_exc()}")
                        finally:
                            # Hapus progress setelah selesai
                            progress_container.empty()
                
                # Tampilkan hasil jika tersedia
                if st.session_state.compression_results is not None and st.session_state.compressed_image is not None:
                    st.image(st.session_state.compressed_image, use_container_width=True)
                    
                    # Tampilkan statistik yang tepat
                    stats = st.session_state.compression_results['stats']
                    quality_metrics = st.session_state.compression_results['quality_metrics']
                    
                    # Metrik utama dalam kolom
                    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                    
                    with col_stat1:
                        st.metric(
                            "Waktu Proses", 
                            f"{stats['runtime']:.3f}s"
                        )
                    
                    with col_stat2:
                        st.metric(
                            "Kompresi Matematis", 
                            f"{stats['mathematical_compression_ratio']:.1f}%"
                        )
                    
                    with col_stat3:
                        st.metric(
                            "Penghematan Ruang", 
                            f"{stats['mathematical_space_savings']:.2f}:1"
                        )
                    
                    with col_stat4:
                        st.metric(
                            "Kualitas PSNR",
                            f"{quality_metrics['psnr']:.1f} dB" if quality_metrics['psnr'] != float('inf') else "Sempurna"
                        )
                    
                    # Ringkasan detail menggunakan fungsi utils
                    with st.expander("📊 Ringkasan Kompresi Detail", expanded=True):
                        summary = create_compression_summary(stats)
                        st.code(summary, language=None)
                    
                    # Bagian metrik kualitas
                    with st.expander("🎯 Analisis Kualitas Gambar", expanded=False):
                        col_q1, col_q2 = st.columns(2)
                        
                        with col_q1:
                            st.metric("Mean Squared Error", f"{quality_metrics['mse']:.2f}")
                            st.metric("Rata-rata Perbedaan", f"{quality_metrics['mean_difference']:.2f}")
                        
                        with col_q2:
                            st.metric("PSNR (dB)", f"{quality_metrics['psnr']:.2f}" if quality_metrics['psnr'] != float('inf') else "∞")
                            st.metric("Rasio Varians", f"{quality_metrics['variance_ratio']:.3f}")
                        
                        # Interpretasi kualitas
                        psnr = quality_metrics['psnr']
                        if psnr == float('inf'):
                            quality_desc = "🟢 Sempurna (tidak ada kehilangan terdeteksi)"
                        elif psnr > 40:
                            quality_desc = "🟢 Kualitas sangat baik"
                        elif psnr > 30:
                            quality_desc = "🟡 Kualitas baik"
                        elif psnr > 20:
                            quality_desc = "🟠 Kualitas cukup"
                        else:
                            quality_desc = "🔴 Kualitas kurang"
                        
                        st.info(f"**Penilaian Kualitas:** {quality_desc}")
                    
                    # Tampilkan detail SVD
                    st.subheader("🔢 Detail Teknis SVD")
                    compression_info = st.session_state.compression_results['compression_info']
                    img_array = st.session_state.compression_results['img_array']
                    
                    col_svd1, col_svd2 = st.columns(2)
                    
                    with col_svd1:
                        st.write(f"**Dimensi asli:** {img_array.shape}")
                        st.write(f"**Channel yang diproses:** {compression_info['channels']}")
                        st.write(f"**Nilai K yang digunakan:** {compression_info['k_values']}")
                    
                    with col_svd2:
                        st.write(f"**Elemen asli:** {compression_info['original_elements']:,}")
                        st.write(f"**Elemen terkompresi:** {compression_info['compressed_elements']:,}")
                        avg_k = np.mean(compression_info['k_values'])
                        st.write(f"**Rata-rata K:** {avg_k:.1f}")
                    
                    # Buat plot perbandingan menggunakan utils
                    with st.expander("📈 Perbandingan Visual & Analisis", expanded=True):
                        try:
                            fig = create_comparison_plot(
                                st.session_state.original_image, 
                                st.session_state.compressed_image, 
                                stats
                            )
                            st.pyplot(fig)
                            plt.close(fig)  # Mencegah memory leak
                        except Exception as e:
                            st.error(f"Pembuatan plot gagal: {str(e)}")
                    
                    # Bagian unduh
                    st.subheader("💾 Unduh Gambar Terkompresi")
                    
                    # Konversi ke bytes untuk unduh
                    buf = io.BytesIO()
                    st.session_state.compressed_image.save(buf, format='PNG')
                    byte_im = buf.getvalue()
                    
                    col_download1, col_download2 = st.columns([2, 1])
                    
                    with col_download1:
                        st.download_button(
                            label="📥 Unduh Gambar Terkompresi",
                            data=byte_im,
                            file_name=f"compressed_{uploaded_file.name.split('.')[0]}_ratio{compression_ratio}.png",
                            mime="image/png",
                            help="Klik untuk mengunduh gambar terkompresi"
                        )
                    
                    with col_download2:
                        download_kb = len(byte_im) / 1024
                        st.metric("Ukuran Unduhan", f"{download_kb:.1f} KB")
                        
                        # Tampilkan perbandingan ukuran file jika tersedia
                        if 'file_compression_ratio' in stats:
                            if stats['file_compression_ratio'] > 0:
                                st.success(f"📉 File dikurangi sebesar {stats['file_compression_ratio']:.1f}%")
                            else:
                                st.info("📈 Ukuran file bertambah (encoding PNG)")
                        
        except Exception as e:
            st.error(f"❌ Error memuat gambar: {str(e)}")
            st.info("💡 Silakan coba dengan file gambar yang berbeda")
            import traceback
            st.error(f"Debug: {traceback.format_exc()}")
    
    else:
        st.info("👆 Silakan upload file gambar untuk memulai!")
        
        # Tampilkan info gambar contoh
        with st.expander("📝 Cara Kerja Kompresi SVD & Tips", expanded=True):
            col_tips1, col_tips2 = st.columns(2)
            
            with col_tips1:
                st.write("""
                **🧮 Cara Kerja SVD:**
                - Mendekomposisi gambar menjadi 3 matriks: U, Σ, V^T
                - Hanya menyimpan k singular value teratas
                - Merekonstruksi menggunakan komponen yang lebih sedikit
                - k% lebih tinggi = kualitas lebih baik, kompresi lebih sedikit
                """)
                
                st.write("""
                **📏 Ukuran Gambar yang Direkomendasikan:**
                - **Tes kecil:** 200×200 hingga 500×500 piksel
                - **Sedang:** 500×500 hingga 800×800 piksel  
                - **Besar:** 800×800 hingga 1200×1200 piksel
                - **Ukuran file:** Di bawah 2MB untuk performa terbaik
                """)
            
            with col_tips2:
                st.write("""
                **✨ Tips untuk Hasil Terbaik:**
                - Gambar dengan detail menunjukkan efek kompresi lebih baik
                - Coba rasio berbeda: 10% (berat), 50% (seimbang), 90% (ringan)
                - Foto lanskap dan potret bekerja dengan baik
                - Perhatikan metrik kualitas PSNR (>30 dB bagus)
                """)
                
                st.write("""
                **🎯 Panduan Kualitas:**
                - 🟢 >40 dB PSNR: Kualitas sangat baik
                - 🟡 30-40 dB: Kualitas baik  
                - 🟠 20-30 dB: Kualitas cukup
                - 🔴 <20 dB: Kualitas kurang
                """)
            
            st.info("🚀 **Performa:** Gambar otomatis diresize ke maksimal 800×800 untuk kecepatan pemrosesan optimal!")

if __name__ == "__main__":
    main()
