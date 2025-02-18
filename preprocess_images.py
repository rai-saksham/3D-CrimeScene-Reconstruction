import os
import sys
from PIL import Image, ImageEnhance
import cv2
import numpy as np
from skimage import exposure, img_as_float, filters

def enhance_image(image_path):
    image = cv2.imread(image_path)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Analyze histogram for contrast and brightness
    hist, hist_centers = exposure.histogram(image_gray)
    p2, p98 = np.percentile(image_gray, (2, 98))
    image_rescale = exposure.rescale_intensity(image, in_range=(p2, p98))
    
    # Estimate noise
    noise_estimation = filters.rank.entropy(image_gray, np.ones((5, 5)))
    noise_mean = np.mean(noise_estimation)
    
    # Set denoising strength based on noise estimation
    h = 10 if noise_mean > 5 else 5
    image_denoised = cv2.fastNlMeansDenoisingColored(image_rescale, None, h, h, 7, 21)
    
    # Edge detection for sharpening
    edges = filters.sobel(image_gray)
    edge_mean = np.mean(edges)
    kernel = np.array([[0, -1, 0], [-1, 5 + edge_mean * 2, -1], [0, -1, 0]])
    image_sharpened = cv2.filter2D(image_denoised, -1, kernel)
    
    # Convert to PIL Image for further processing
    image_pil = Image.fromarray(cv2.cvtColor(image_sharpened, cv2.COLOR_BGR2RGB))
    
    # Apply gamma correction
    gamma_corrector = ImageEnhance.Brightness(image_pil)
    image_pil = gamma_corrector.enhance(1.1)
    
    # Apply color balance
    color_corrector = ImageEnhance.Color(image_pil)
    image_pil = color_corrector.enhance(1.2)
    
    # Apply contrast adjustment
    contrast_corrector = ImageEnhance.Contrast(image_pil)
    image_pil = contrast_corrector.enhance(1.3)
    
    # Apply brightness adjustment
    brightness_corrector = ImageEnhance.Brightness(image_pil)
    image_pil = brightness_corrector.enhance(1.1)
    
    return cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

def preprocess_images(base_dir):
    input_dir = os.path.join(base_dir, 'og_images')
    output_dir_conv = os.path.join(base_dir, 'conv_images')
    output_dir_resized = os.path.join(base_dir, 'resized_images')
    output_dir_final = os.path.join(base_dir, 'input')
    
    if not os.path.exists(output_dir_conv):
        os.makedirs(output_dir_conv)
    
    if not os.path.exists(output_dir_resized):
        os.makedirs(output_dir_resized)
    
    if not os.path.exists(output_dir_final):
        os.makedirs(output_dir_final)

    # Convert images to JPEG and save to conv_images
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.tif', '.tiff', '.jpg', '.png', '.jpeg')):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir_conv, os.path.splitext(filename)[0] + '.jpg')
            
            with Image.open(input_path) as img:
                img = img.convert('RGB')
                img.save(output_path, 'JPEG')
                print(f"Converted {input_path} to {output_path}")
    
    # Resize images to be below 1600 pixels in any dimension and save to resized_images
    for filename in os.listdir(output_dir_conv):
        if filename.lower().endswith('.jpg'):
            img_path = os.path.join(output_dir_conv, filename)
            with Image.open(img_path) as img:
                max_dimension = max(img.size)
                if max_dimension > 1600:
                    scale = 1600 / max_dimension
                    new_size = (int(img.width * scale), int(img.height * scale))
                    img = img.resize(new_size, Image.Resampling.LANCZOS)
                img.save(os.path.join(output_dir_resized, filename))
                print(f"Resized and saved {filename} to {output_dir_resized}")

    # Apply enhancements and save to input
    for filename in os.listdir(output_dir_resized):
        if filename.lower().endswith('.jpg'):
            img_path = os.path.join(output_dir_resized, filename)
            enhanced_image = enhance_image(img_path)
            final_path = os.path.join(output_dir_final, filename)
            cv2.imwrite(final_path, enhanced_image)
            print(f"Enhanced and saved {filename} to {output_dir_final}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python preprocess_images.py <base_directory>")
        sys.exit(1)
    
    base_directory = sys.argv[1]
    preprocess_images(base_directory)
