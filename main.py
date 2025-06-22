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
    Resize image if it's too large to prevent memory issues
    """
    width, height = image.size
    
    if max(width, height) > max_dimension:
        # Calculate new dimensions while maintaining aspect ratio
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
    
    st.title("🖼️ SVD Image Compression")
    st.write("Compress images using Singular Value Decomposition")
    
    # Initialize session state to preserve results
    if 'compression_results' not in st.session_state:
        st.session_state.compression_results = None
    if 'original_image' not in st.session_state:
        st.session_state.original_image = None
    if 'compressed_image' not in st.session_state:
        st.session_state.compressed_image = None
    
    # Sidebar for controls
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff']
        )
        
        # File size info
        if uploaded_file is not None:
            file_size_mb = uploaded_file.size / (1024 * 1024)
            st.info(f"📁 File size: {file_size_mb:.2f} MB")
            
            # Warning for large files
            if file_size_mb > 2:
                st.warning("⚠️ Large file detected! Image will be auto-resized for processing.")
        
        # Compression ratio slider
        compression_ratio = st.slider(
            "Compression Ratio (%)",
            min_value=1,
            max_value=100,
            value=50,
            step=1,
            help="Higher percentage = less compression, better quality"
        )
        
        # Show optimal k suggestions
        if uploaded_file is not None:
            try:
                temp_image = Image.open(uploaded_file)
                temp_processed, _ = resize_image_if_needed(temp_image)
                temp_array = np.array(temp_processed)
                
                suggestions = get_optimal_k_suggestions(temp_array.shape)
                
                with st.expander("💡 Compression Guidelines", expanded=False):
                    for ratio, info in suggestions.items():
                        quality_emoji = "🟢" if info['estimated_quality'] == 'High' else "🟡" if info['estimated_quality'] == 'Medium' else "🔴"
                        st.write(f"{quality_emoji} **{ratio}%**: {info['description']} (k≈{info['k']})")
            except:
                pass  # Skip if image can't be loaded temporarily
        
        # Process button
        process_button = st.button("🚀 Compress Image", type="primary")
        
        # Clear results button
        if st.session_state.compression_results is not None:
            if st.button("🗑️ Clear Results"):
                st.session_state.compression_results = None
                st.session_state.original_image = None
                st.session_state.compressed_image = None
                st.rerun()
    
    # Main content area
    if uploaded_file is not None:
        # Load image
        try:
            original_image = Image.open(uploaded_file)
            
            # Auto-resize if needed
            processed_image, was_resized = resize_image_if_needed(original_image)
            
            if was_resized:
                st.warning(f"🔄 Image auto-resized from {original_image.size} to {processed_image.size} for optimal processing")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📸 Original Image")
                st.image(processed_image, use_container_width=True)
                st.write(f"**Dimensions:** {processed_image.size[0]} × {processed_image.size[1]}")
                
            with col2:
                st.subheader("🗜️ Compressed Image")
                
                if process_button:
                    # Show progress
                    progress_container = st.container()
                    with progress_container:
                        progress_bar = st.progress(0, text="Starting compression...")
                        
                        # Convert PIL to numpy array
                        img_array = np.array(processed_image)
                        progress_bar.progress(10, text="Image loaded...")
                        
                        # Compress image with error handling
                        try:
                            start_time = time.time()
                            progress_bar.progress(20, text="Performing SVD decomposition...")
                            
                            # 🔥 FIXED: Use correct return values from utils
                            compressed_array, compression_info = compress_image_svd(img_array, compression_ratio)
                            progress_bar.progress(70, text="Reconstructing image...")
                            
                            end_time = time.time()
                            runtime = end_time - start_time
                            
                            # Convert back to PIL
                            compressed_image = Image.fromarray(compressed_array.astype(np.uint8))
                            progress_bar.progress(80, text="Calculating statistics...")
                            
                            # 🔥 FIXED: Use correct parameters for stats calculation
                            stats = calculate_compression_stats(
                                original_shape=img_array.shape,
                                compression_info=compression_info,
                                runtime=runtime,
                                file_size_before=uploaded_file.size,
                                file_size_after=len(io.BytesIO().getvalue()) if compressed_image else None
                            )
                            
                            progress_bar.progress(90, text="Calculating quality metrics...")
                            
                            # 🆕 NEW: Calculate image quality metrics
                            quality_metrics = calculate_image_quality_metrics(
                                img_array.astype(np.float64), 
                                compressed_array.astype(np.float64)
                            )
                            
                            # Store results in session state
                            st.session_state.compression_results = {
                                'stats': stats,
                                'compression_info': compression_info,
                                'quality_metrics': quality_metrics,
                                'img_array': img_array,
                                'runtime': runtime
                            }
                            st.session_state.original_image = processed_image
                            st.session_state.compressed_image = compressed_image
                            
                            progress_bar.progress(100, text="✅ Compression complete!")
                            time.sleep(0.5)  # Brief pause to show completion
                            
                        except Exception as e:
                            st.error(f"❌ Compression failed: {str(e)}")
                            st.info("💡 Try with a smaller image or different compression ratio")
                            import traceback
                            st.error(f"Debug info: {traceback.format_exc()}")
                        finally:
                            # Clear progress after completion
                            progress_container.empty()
                
                # Display results if available
                if st.session_state.compression_results is not None and st.session_state.compressed_image is not None:
                    st.image(st.session_state.compressed_image, use_container_width=True)
                    
                    # 🔥 FIXED: Display proper statistics
                    stats = st.session_state.compression_results['stats']
                    quality_metrics = st.session_state.compression_results['quality_metrics']
                    
                    # Key metrics in columns
                    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                    
                    with col_stat1:
                        st.metric(
                            "Runtime", 
                            f"{stats['runtime']:.3f}s"
                        )
                    
                    with col_stat2:
                        st.metric(
                            "Math Compression", 
                            f"{stats['mathematical_compression_ratio']:.1f}%"
                        )
                    
                    with col_stat3:
                        st.metric(
                            "Space Savings", 
                            f"{stats['mathematical_space_savings']:.2f}:1"
                        )
                    
                    with col_stat4:
                        st.metric(
                            "PSNR Quality",
                            f"{quality_metrics['psnr']:.1f} dB" if quality_metrics['psnr'] != float('inf') else "Perfect"
                        )
                    
                    # 🆕 NEW: Beautiful summary using your utils function
                    with st.expander("📊 Detailed Compression Summary", expanded=True):
                        summary = create_compression_summary(stats)
                        st.code(summary, language=None)
                    
                    # 🆕 NEW: Quality metrics section
                    with st.expander("🎯 Image Quality Analysis", expanded=False):
                        col_q1, col_q2 = st.columns(2)
                        
                        with col_q1:
                            st.metric("Mean Squared Error", f"{quality_metrics['mse']:.2f}")
                            st.metric("Mean Difference", f"{quality_metrics['mean_difference']:.2f}")
                        
                        with col_q2:
                            st.metric("PSNR (dB)", f"{quality_metrics['psnr']:.2f}" if quality_metrics['psnr'] != float('inf') else "∞")
                            st.metric("Variance Ratio", f"{quality_metrics['variance_ratio']:.3f}")
                        
                        # Quality interpretation
                        psnr = quality_metrics['psnr']
                        if psnr == float('inf'):
                            quality_desc = "🟢 Perfect (no loss detected)"
                        elif psnr > 40:
                            quality_desc = "🟢 Excellent quality"
                        elif psnr > 30:
                            quality_desc = "🟡 Good quality"
                        elif psnr > 20:
                            quality_desc = "🟠 Fair quality"
                        else:
                            quality_desc = "🔴 Poor quality"
                        
                        st.info(f"**Quality Assessment:** {quality_desc}")
                    
                    # Display SVD details
                    st.subheader("🔢 SVD Technical Details")
                    compression_info = st.session_state.compression_results['compression_info']
                    img_array = st.session_state.compression_results['img_array']
                    
                    col_svd1, col_svd2 = st.columns(2)
                    
                    with col_svd1:
                        st.write(f"**Original dimensions:** {img_array.shape}")
                        st.write(f"**Channels processed:** {compression_info['channels']}")
                        st.write(f"**K values used:** {compression_info['k_values']}")
                    
                    with col_svd2:
                        st.write(f"**Original elements:** {compression_info['original_elements']:,}")
                        st.write(f"**Compressed elements:** {compression_info['compressed_elements']:,}")
                        avg_k = np.mean(compression_info['k_values'])
                        st.write(f"**Average K:** {avg_k:.1f}")
                    
                    # 🔥 FIXED: Create proper comparison plot using your utils
                    with st.expander("📈 Visual Comparison & Analysis", expanded=True):
                        try:
                            fig = create_comparison_plot(
                                st.session_state.original_image, 
                                st.session_state.compressed_image, 
                                stats
                            )
                            st.pyplot(fig)
                            plt.close(fig)  # Prevent memory leaks
                        except Exception as e:
                            st.error(f"Plot generation failed: {str(e)}")
                    
                    # Download section
                    st.subheader("💾 Download Compressed Image")
                    
                    # Convert to bytes for download
                    buf = io.BytesIO()
                    st.session_state.compressed_image.save(buf, format='PNG')
                    byte_im = buf.getvalue()
                    
                    col_download1, col_download2 = st.columns([2, 1])
                    
                    with col_download1:
                        st.download_button(
                            label="📥 Download Compressed Image",
                            data=byte_im,
                            file_name=f"compressed_{uploaded_file.name.split('.')[0]}_ratio{compression_ratio}.png",
                            mime="image/png",
                            help="Click to download the compressed image"
                        )
                    
                    with col_download2:
                        download_kb = len(byte_im) / 1024
                        st.metric("Download Size", f"{download_kb:.1f} KB")
                        
                        # Show file size comparison if available
                        if 'file_compression_ratio' in stats:
                            if stats['file_compression_ratio'] > 0:
                                st.success(f"📉 File reduced by {stats['file_compression_ratio']:.1f}%")
                            else:
                                st.info("📈 File size increased (PNG encoding)")
                        
        except Exception as e:
            st.error(f"❌ Error loading image: {str(e)}")
            st.info("💡 Please try with a different image file")
            import traceback
            st.error(f"Debug: {traceback.format_exc()}")
    
    else:
        st.info("👆 Please upload an image file to get started!")
        
        # Show sample images info
        with st.expander("📝 How SVD Compression Works & Tips", expanded=True):
            col_tips1, col_tips2 = st.columns(2)
            
            with col_tips1:
                st.write("""
                **🧮 How SVD Works:**
                - Decomposes image into 3 matrices: U, Σ, V^T
                - Keeps only top k singular values
                - Reconstructs using fewer components
                - Higher k% = better quality, less compression
                """)
                
                st.write("""
                **📏 Recommended Image Sizes:**
                - **Small test:** 200×200 to 500×500 pixels
                - **Medium:** 500×500 to 800×800 pixels  
                - **Large:** 800×800 to 1200×1200 pixels
                - **File size:** Under 2MB for best performance
                """)
            
            with col_tips2:
                st.write("""
                **✨ Tips for Best Results:**
                - Images with details show compression effects better
                - Try different ratios: 10% (heavy), 50% (balanced), 90% (light)
                - Landscapes and portraits work great
                - Watch the PSNR quality metric (>30 dB is good)
                """)
                
                st.write("""
                **🎯 Quality Guidelines:**
                - 🟢 >40 dB PSNR: Excellent quality
                - 🟡 30-40 dB: Good quality  
                - 🟠 20-30 dB: Fair quality
                - 🔴 <20 dB: Poor quality
                """)
            
            st.info("🚀 **Performance:** Images are auto-resized to max 800×800 for optimal processing speed!")

if __name__ == "__main__":
    main()
