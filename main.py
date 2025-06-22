import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time
import io
from utils import compress_image_svd, calculate_compression_stats, create_comparison_plot

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
            help="Lower percentage = higher compression"
        )
        
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
                            
                            compressed_array, k_values = compress_image_svd(img_array, compression_ratio)
                            progress_bar.progress(70, text="Reconstructing image...")
                            
                            end_time = time.time()
                            
                            # Convert back to PIL
                            compressed_image = Image.fromarray(compressed_array.astype(np.uint8))
                            progress_bar.progress(90, text="Finalizing...")
                            
                            # Calculate stats
                            original_size = img_array.size * img_array.itemsize
                            compressed_size = sum(k_values) * (img_array.shape[0] + img_array.shape[1] + 1) * img_array.itemsize
                            runtime = end_time - start_time
                            
                            compression_stats = calculate_compression_stats(
                                original_size, compressed_size, runtime
                            )
                            
                            # Store results in session state
                            st.session_state.compression_results = {
                                'stats': compression_stats,
                                'k_values': k_values,
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
                        finally:
                            # Clear progress after completion
                            progress_container.empty()
                
                # Display results if available
                if st.session_state.compression_results is not None and st.session_state.compressed_image is not None:
                    st.image(st.session_state.compressed_image, use_container_width=True)
                    
                    # Display statistics
                    st.subheader("📊 Compression Statistics")
                    
                    stats = st.session_state.compression_results['stats']
                    
                    col_stat1, col_stat2, col_stat3 = st.columns(3)
                    
                    with col_stat1:
                        st.metric(
                            "Runtime", 
                            f"{stats['runtime']:.3f}s"
                        )
                    
                    with col_stat2:
                        st.metric(
                            "Size Reduction", 
                            f"{stats['size_reduction']:.1f}%"
                        )
                    
                    with col_stat3:
                        st.metric(
                            "Compression Ratio", 
                            f"{stats['compression_ratio']:.2f}:1"
                        )
                    
                    # Display singular values info
                    st.subheader("🔢 SVD Details")
                    k_values = st.session_state.compression_results['k_values']
                    img_array = st.session_state.compression_results['img_array']
                    
                    if len(img_array.shape) == 3:  # Color image
                        st.write(f"**Original dimensions:** {img_array.shape[1]} × {img_array.shape[0]} × {img_array.shape[2]}")
                        st.write(f"**Singular values used per channel:** R={k_values[0]}, G={k_values[1]}, B={k_values[2]}")
                    else:  # Grayscale
                        total_singular_values = min(img_array.shape[:2])
                        st.write(f"**Original dimensions:** {img_array.shape[1]} × {img_array.shape[0]}")
                        st.write(f"**Singular values used:** {k_values[0]} out of {total_singular_values}")
                    
                    # Create comparison plot
                    with st.expander("📈 Detailed Comparison", expanded=True):
                        fig = create_comparison_plot(
                            st.session_state.original_image, 
                            st.session_state.compressed_image, 
                            stats
                        )
                        st.pyplot(fig)
                    
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
                        original_kb = len(byte_im) / 1024
                        st.metric("Download Size", f"{original_kb:.1f} KB")
                        
        except Exception as e:
            st.error(f"❌ Error loading image: {str(e)}")
            st.info("💡 Please try with a different image file")
    
    else:
        st.info("👆 Please upload an image file to get started!")
        
        # Show sample images info
        with st.expander("📝 Recommended Test Images & Tips", expanded=True):
            col_tips1, col_tips2 = st.columns(2)
            
            with col_tips1:
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
                - Images with details show better compression effects
                - Try different compression ratios (10%, 50%, 90%)
                - Landscapes and portraits work great
                - Formats supported: PNG, JPG, JPEG, BMP, TIFF
                """)
            
            st.info("🚀 **Performance:** Images are auto-resized to max 800×800 for optimal processing speed!")

if __name__ == "__main__":
    main()
