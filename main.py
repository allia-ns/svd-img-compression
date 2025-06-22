import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time
import io
from utils import compress_image_svd, calculate_compression_stats, create_comparison_plot

def main():
    st.set_page_config(
        page_title="SVD Image Compression",
        page_icon="🖼️",
        layout="wide"
    )
    
    st.title("🖼️ SVD Image Compression")
    st.write("Compress images using Singular Value Decomposition")
    
    # Sidebar for controls
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff']
        )
        
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
    
    # Main content area
    if uploaded_file is not None:
        # Load and display original image
        original_image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📸 Original Image")
            st.image(original_image, use_container_width=True)
            
        with col2:
            st.subheader("🗜️ Compressed Image")
            
            if process_button:
                # Show progress
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Convert PIL to numpy array
                img_array = np.array(original_image)
                
                # Compress image
                status_text.text("Compressing image...")
                progress_bar.progress(30)
                
                start_time = time.time()
                compressed_array, k_values = compress_image_svd(img_array, compression_ratio)
                end_time = time.time()
                
                progress_bar.progress(70)
                
                # Convert back to PIL
                compressed_image = Image.fromarray(compressed_array.astype(np.uint8))
                
                # Display compressed image
                st.image(compressed_image, use_container_width=True)
                
                progress_bar.progress(100)
                status_text.text("✅ Compression complete!")
                
                # Calculate stats
                original_size = img_array.size * img_array.itemsize
                compressed_size = sum(k_values) * (img_array.shape[0] + img_array.shape[1] + 1) * img_array.itemsize
                runtime = end_time - start_time
                
                compression_stats = calculate_compression_stats(
                    original_size, compressed_size, runtime
                )
                
                # Display statistics
                st.subheader("📊 Compression Statistics")
                
                col_stat1, col_stat2, col_stat3 = st.columns(3)
                
                with col_stat1:
                    st.metric(
                        "Runtime", 
                        f"{compression_stats['runtime']:.3f}s"
                    )
                
                with col_stat2:
                    st.metric(
                        "Size Reduction", 
                        f"{compression_stats['size_reduction']:.1f}%"
                    )
                
                with col_stat3:
                    st.metric(
                        "Compression Ratio", 
                        f"{compression_stats['compression_ratio']:.2f}:1"
                    )
                
                # Display singular values info
                st.subheader("🔢 SVD Details")
                total_singular_values = min(img_array.shape[:2])
                
                if len(img_array.shape) == 3:  # Color image
                    st.write(f"**Original dimensions:** {img_array.shape[1]} × {img_array.shape[0]} × {img_array.shape[2]}")
                    st.write(f"**Singular values used per channel:** R={k_values[0]}, G={k_values[1]}, B={k_values[2]}")
                else:  # Grayscale
                    st.write(f"**Original dimensions:** {img_array.shape[1]} × {img_array.shape[0]}")
                    st.write(f"**Singular values used:** {k_values[0]} out of {total_singular_values}")
                
                # Create comparison plot
                fig = create_comparison_plot(original_image, compressed_image, compression_stats)
                st.pyplot(fig)
                
                # Download button
                st.subheader("💾 Download Compressed Image")
                
                # Convert to bytes for download
                buf = io.BytesIO()
                compressed_image.save(buf, format='PNG')
                byte_im = buf.getvalue()
                
                st.download_button(
                    label="📥 Download Compressed Image",
                    data=byte_im,
                    file_name=f"compressed_{uploaded_file.name.split('.')[0]}_ratio{compression_ratio}.png",
                    mime="image/png"
                )
                
                # Clear progress
                progress_bar.empty()
                status_text.empty()
    
    else:
        st.info("👆 Please upload an image file to get started!")
        
        # Show sample images info
        st.subheader("📝 Recommended Test Images")
        st.write("""
        **Good test images:**
        - **Portraits/faces:** 500×500 to 1000×1000 pixels
        - **Landscapes:** 800×600 to 1920×1080 pixels  
        - **Simple graphics:** 300×300 to 800×800 pixels
        - **Detailed photos:** 1000×1000+ pixels
        
        **Formats supported:** PNG, JPG, JPEG, BMP, TIFF
        
        **Tips:** 
        - Images with more details show better compression effects
        - Try different compression ratios to see quality vs size trade-offs
        """)

if __name__ == "__main__":
    main()
