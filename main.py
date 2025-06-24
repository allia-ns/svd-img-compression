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
    calculate_image_quality_metrics
    # ❌ REMOVED: get_optimal_k_suggestions - this function doesn't exist in utils.py!
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
                st.warning("⚠️ File besar akan otomatis diresize")
        
        # Slider kompresi
        compression_ratio = st.slider(
            "Level Kompresi (%)",
            min_value=1,
            max_value=100,
            value=50,
            step=1,
            help="Semakin tinggi = kompresi semakin ringan"
        )
        
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
                st.info(f"🔄 Gambar diresize dari {original_image.size} ke {processed_image.size}")
            
            # Layout gambar
            col1, col2 = st.columns(2, gap="large")
            
            with col1:
                st.subheader("📸 Gambar Asli")
                st.image(processed_image, width=450, caption=f"Dimensi: {processed_image.size[0]} × {processed_image.size[1]}")
                
            with col2:
                st.subheader("🗜️ Hasil Kompresi")
                
                if process_button:
                    # Progress
                    progress_container = st.container()
                    with progress_container:
                        progress_bar = st.progress(0, text="Memulai kompresi...")
                        
                        # Konversi PIL ke numpy array
                        img_array = np.array(processed_image)
                        progress_bar.progress(10, text="Gambar dimuat...")
                        
                        # Kompres gambar
                        try:
                            start_time = time.time()
                            progress_bar.progress(20, text="Melakukan SVD...")
                            
                            compressed_array, compression_info = compress_image_svd(img_array, compression_ratio)
                            progress_bar.progress(70, text="Merekonstruksi gambar...")
                            
                            end_time = time.time()
                            runtime = end_time - start_time
                            
                            # Konversi kembali ke PIL
                            compressed_image = Image.fromarray(compressed_array.astype(np.uint8))
                            progress_bar.progress(80, text="Menghitung statistik...")
                            
                            # ✅ Fixed: Calculate file size properly
                            buf = io.BytesIO()
                            compressed_image.save(buf, format='PNG')
                            compressed_file_size = len(buf.getvalue())
                            
                            # Hitung stats dengan logic yang benar
                            stats = calculate_compression_stats(
                                original_shape=img_array.shape,
                                compression_info=compression_info,
                                runtime=runtime,
                                file_size_before=uploaded_file.size,
                                file_size_after=compressed_file_size
                            )
                            
                            # Fix rasio kompresi untuk display
                            actual_compression_ratio = compression_ratio
                            stats['display_compression_ratio'] = actual_compression_ratio
                            
                            progress_bar.progress(90, text="Menghitung kualitas...")
                            
                            # Hitung metrik kualitas gambar
                            quality_metrics = calculate_image_quality_metrics(
                                img_array.astype(np.float64), 
                                compressed_array.astype(np.float64)
                            )
                            
                            # Simpan hasil
                            st.session_state.compression_results = {
                                'stats': stats,
                                'compression_info': compression_info,
                                'quality_metrics': quality_metrics,
                                'img_array': img_array,
                                'runtime': runtime
                            }
                            st.session_state.original_image = processed_image
                            st.session_state.compressed_image = compressed_image
                            
                            progress_bar.progress(100, text="✅ Selesai!")
                            time.sleep(0.5)
                            
                        except Exception as e:
                            st.error(f"❌ Kompresi gagal: {str(e)}")
                            st.info("💡 Coba dengan gambar yang lebih kecil atau level kompresi berbeda")
                        finally:
                            progress_container.empty()
                
                # Tampilkan hasil
                if st.session_state.compression_results is not None and st.session_state.compressed_image is not None:
                    st.image(st.session_state.compressed_image, width=450, caption="Hasil kompresi SVD")
            
            # Quick metrics
            if st.session_state.compression_results is not None:
                stats = st.session_state.compression_results['stats']
                quality_metrics = st.session_state.compression_results['quality_metrics']
                
                st.markdown("---")
                st.subheader("📊 Hasil Kompresi")
                
                # 4 metrics utama
                metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                
                with metric_col1:
                    st.metric("⏱️ Waktu", f"{stats['runtime']:.2f}s")
                
                with metric_col2:
                    # Tampilkan level kompresi yang diinput user
                    st.metric("🗜️ Level Kompresi", f"{compression_ratio}%")
                
                with metric_col3:
                    psnr_display = f"{quality_metrics['psnr']:.1f} dB" if quality_metrics['psnr'] != float('inf') else "Perfect"
                    st.metric("🎯 PSNR", psnr_display)
                
                with metric_col4:
                    # Status kualitas
                    psnr = quality_metrics['psnr']
                    if psnr == float('inf'):
                        quality_status = "🟢 Sempurna"
                    elif psnr > 40:
                        quality_status = "🟢 Sangat Baik"
                    elif psnr > 30:
                        quality_status = "🟡 Baik"
                    elif psnr > 20:
                        quality_status = "🟠 Cukup"
                    else:
                        quality_status = "🔴 Kurang"
                    
                    st.metric("✨ Kualitas", quality_status)
            
            # Detail analysis - simplified
            if st.session_state.compression_results is not None:
                st.markdown("---")
                st.header("📊 Detail Analisis")
                
                tab1, tab2, tab3 = st.tabs(["📈 Ringkasan", "🎯 Kualitas", "📥 Download"])
                
                with tab1:
                    col_summary1, col_summary2 = st.columns([2, 1])
                    
                    with col_summary1:
                        st.subheader("📋 Ringkasan")
                        summary = create_compression_summary(stats)
                        st.code(summary, language=None)
                    
                    with col_summary2:
                        st.subheader("🎨 Perbandingan")
                        try:
                            fig = create_comparison_plot(
                                st.session_state.original_image, 
                                st.session_state.compressed_image, 
                                stats
                            )
                            st.pyplot(fig, use_container_width=True)
                            plt.close(fig)
                        except Exception as e:
                            st.error(f"Plot gagal: {str(e)}")
                
                with tab2:
                    st.subheader("🎯 Metrik Kualitas")
                    
                    col_q1, col_q2 = st.columns(2)
                    
                    with col_q1:
                        st.metric("Mean Squared Error", f"{quality_metrics['mse']:.2f}")
                        st.info("MSE mengukur rata-rata kesalahan piksel. Semakin kecil semakin baik.")
                    
                    with col_q2:
                        psnr_val = f"{quality_metrics['psnr']:.2f}" if quality_metrics['psnr'] != float('inf') else "∞"
                        st.metric("PSNR (dB)", psnr_val)
                        
                        # Interpretasi PSNR
                        psnr = quality_metrics['psnr']
                        if psnr == float('inf'):
                            quality_desc = "🟢 Tidak ada perbedaan terdeteksi"
                        elif psnr > 40:
                            quality_desc = "🟢 Sangat Baik - hampir tidak terlihat perbedaan"
                        elif psnr > 30:
                            quality_desc = "🟡 Baik - perbedaan minimal"
                        elif psnr > 20:
                            quality_desc = "🟠 Cukup - ada perbedaan visible"
                        else:
                            quality_desc = "🔴 Kurang - perbedaan sangat terlihat"
                        
                        st.info(quality_desc)
                
                with tab3:
                    st.subheader("💾 Download Hasil")
                    
                    # Konversi ke bytes
                    buf = io.BytesIO()
                    st.session_state.compressed_image.save(buf, format='PNG')
                    byte_im = buf.getvalue()
                    
                    col_dl1, col_dl2 = st.columns([3, 1])
                    
                    with col_dl1:
                        st.download_button(
                            label="📥 Download Gambar Terkompresi",
                            data=byte_im,
                            file_name=f"compressed_{uploaded_file.name.split('.')[0]}_level{compression_ratio}.png",
                            mime="image/png"
                        )
                    
                    with col_dl2:
                        download_kb = len(byte_im) / 1024
                        st.metric("Ukuran File", f"{download_kb:.1f} KB")
                        
        except Exception as e:
            st.error(f"❌ Error memuat gambar: {str(e)}")
            st.info("💡 Silakan coba dengan file gambar yang berbeda")
    
    else:
        st.info("👆 Upload file gambar untuk memulai!")
        
        # Simple info
        with st.expander("📝 Cara Kerja SVD", expanded=False):
            st.write("""
            **Singular Value Decomposition (SVD)** memecah gambar menjadi komponen-komponen utama, 
            kemudian merekonstruksi dengan hanya menggunakan komponen yang paling penting.
            
            **Level Kompresi:**
            - **81-10%**: Kompresi ringan, kualitas tinggi
            - **51-80%**: Kompresi sedang, seimbang  
            - **21-50%**: Kompresi berat, ukuran kecil
            - **1-20%**: Kompresi sangat berat, kualitas rendah
            """)

if __name__ == "__main__":
    main()
