import os
import logging
from argparse import ArgumentParser
import shutil
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

def preprocess_images(source_path):
    input_dir = os.path.join(source_path, 'og_images')
    output_dir_conv = os.path.join(source_path, 'conv_images')
    output_dir_resized = os.path.join(source_path, 'resized_images')
    output_dir_final = os.path.join(source_path, 'input')
    
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

# This Python script is based on the shell converter script provided in the MipNerF 360 repository.
parser = ArgumentParser("Colmap converter")
parser.add_argument("--no_gpu", action='store_true')
parser.add_argument("--skip_matching", action='store_true')
parser.add_argument("--source_path", "-s", required=True, type=str)
parser.add_argument("--camera", default="OPENCV", type=str)
parser.add_argument("--colmap_executable", default="", type=str)
parser.add_argument("--resize", action="store_true")
parser.add_argument("--magick_executable", default="", type=str)
args = parser.parse_args()
colmap_command = '"{}"'.format(args.colmap_executable) if len(args.colmap_executable) > 0 else "colmap"
magick_command = '"{}"'.format(args.magick_executable) if len(args.magick_executable) > 0 else "magick"
use_gpu = 1 if not args.no_gpu else 0

# Preprocess the images
preprocess_images(args.source_path)

if not args.skip_matching:
    os.makedirs(args.source_path + "/distorted/sparse", exist_ok=True)

    ## Feature extraction
    feat_extracton_cmd = colmap_command + " feature_extractor "\
        "--database_path " + args.source_path + "/distorted/database.db \
        --image_path " + args.source_path + "/input \
        --ImageReader.single_camera 1 \
        --ImageReader.camera_model " + args.camera + " \
        --SiftExtraction.use_gpu " + str(use_gpu) + " \
        --SiftExtraction.max_num_features 8192"
    exit_code = os.system(feat_extracton_cmd)
    if exit_code != 0:
        logging.error(f"Feature extraction failed with code {exit_code}. Exiting.")
        exit(exit_code)

    ## Feature matching
    feat_matching_cmd = colmap_command + " exhaustive_matcher \
        --database_path " + args.source_path + "/distorted/database.db \
        --SiftMatching.use_gpu " + str(use_gpu) + " \
        --SiftMatching.max_num_matches 32768"
    exit_code = os.system(feat_matching_cmd)
    if exit_code != 0:
        logging.error(f"Feature matching failed with code {exit_code}. Exiting.")
        exit(exit_code)

    ### Bundle adjustment
    # The default Mapper tolerance is unnecessarily large,
    # decreasing it speeds up bundle adjustment steps.
    mapper_cmd = (colmap_command + " mapper \
        --database_path " + args.source_path + "/distorted/database.db \
        --image_path "  + args.source_path + "/input \
        --output_path "  + args.source_path + "/distorted/sparse \
        --Mapper.ba_global_function_tolerance=0.000001 \
        --Mapper.num_threads=16")
    exit_code = os.system(mapper_cmd)
    if exit_code != 0:
        logging.error(f"Mapper failed with code {exit_code}. Exiting.")
        exit(exit_code)

### Image undistortion
## We need to undistort our images into ideal pinhole intrinsics.
img_undist_cmd = (colmap_command + " image_undistorter \
    --image_path " + args.source_path + "/input \
    --input_path " + args.source_path + "/distorted/sparse/0 \
    --output_path " + args.source_path + "\
    --output_type COLMAP")
exit_code = os.system(img_undist_cmd)
if exit_code != 0:
    logging.error(f"Image undistorter failed with code {exit_code}. Exiting.")
    exit(exit_code)

files = os.listdir(args.source_path + "/sparse")
os.makedirs(args.source_path + "/sparse/0", exist_ok=True)
# Copy each file from the source directory to the destination directory
for file in files:
    if file == '0':
        continue
    source_file = os.path.join(args.source_path, "sparse", file)
    destination_file = os.path.join(args.source_path, "sparse", "0", file)
    shutil.move(source_file, destination_file)

if(args.resize):
    print("Copying and resizing...")

    # Resize images.
    os.makedirs(args.source_path + "/images_2", exist_ok=True)
    os.makedirs(args.source_path + "/images_4", exist_ok=True)
    os.makedirs(args.source_path + "/images_8", exist_ok=True)
    # Get the list of files in the source directory
    files = os.listdir(args.source_path + "/images")
    # Copy each file from the source directory to the destination directory
    for file in files:
        source_file = os.path.join(args.source_path, "images", file)

        destination_file = os.path.join(args.source_path, "images_2", file)
        shutil.copy2(source_file, destination_file)
        exit_code = os.system(magick_command + " mogrify -resize 50% " + destination_file)
        if exit_code != 0:
            logging.error(f"50% resize failed with code {exit_code}. Exiting.")
            exit(exit_code)

        destination_file = os.path.join(args.source_path, "images_4", file)
        shutil.copy2(source_file, destination_file)
        exit_code = os.system(magick_command + " mogrify -resize 25% " + destination_file)
        if exit_code != 0:
            logging.error(f"25% resize failed with code {exit_code}. Exiting.")
            exit(exit_code)

        destination_file = os.path.join(args.source_path, "images_8", file)
        shutil.copy2(source_file, destination_file)
        exit_code = os.system(magick_command + " mogrify -resize 12.5% " + destination_file)
        if exit_code != 0:
            logging.error(f"12.5% resize failed with code {exit_code}. Exiting.")
            exit(exit_code)

print("Done.")
