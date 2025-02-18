import os
import logging
from argparse import ArgumentParser
import shutil
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import cv2

def adjust_gamma(image, gamma=1.0):
    inv_gamma = 1.0 / gamma
    table = (np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)])).astype("uint8")
    return cv2.LUT(image, table)

def apply_clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

def enhance_sharpness(image):
    enhancer = ImageEnhance.Sharpness(image)
    return enhancer.enhance(2.0)

def adjust_contrast(image, factor):
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(factor)

def adjust_brightness(image, factor):
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)

def adjust_color(image, factor):
    enhancer = ImageEnhance.Color(image)
    return enhancer.enhance(factor)

def auto_adjust_image(image_path):
    image = Image.open(image_path)
    image_np = np.array(image)

    # Auto gamma correction
    gamma_value = 0.9 if np.mean(image_np) < 128 else 1.1
    gamma_corrected = adjust_gamma(image_np, gamma=gamma_value)

    # Auto contrast enhancement using CLAHE
    contrast_enhanced = apply_clahe(gamma_corrected)

    # Convert back to PIL Image
    enhanced_image = Image.fromarray(contrast_enhanced)

    # Enhance sharpness
    enhanced_image = enhance_sharpness(enhanced_image)

    # Auto adjust contrast
    contrast_factor = 1.2 if np.std(np.array(enhanced_image)) < 64 else 0.9
    enhanced_image = adjust_contrast(enhanced_image, contrast_factor)

    # Auto adjust brightness
    brightness_factor = 1.2 if np.mean(np.array(enhanced_image)) < 128 else 0.9
    enhanced_image = adjust_brightness(enhanced_image, brightness_factor)

    # Auto adjust color
    color_factor = 1.2 if np.mean(np.array(enhanced_image)) < 128 else 1.0
    enhanced_image = adjust_color(enhanced_image, color_factor)

    return enhanced_image

def resize_image(image, max_dim=1600):
    width, height = image.size
    if max(width, height) > max_dim:
        scaling_factor = max_dim / float(max(width, height))
        new_size = (int(width * scaling_factor), int(height * scaling_factor))
        return image.resize(new_size, Image.Resampling.LANCZOS)
    return image

def process_images(input_dir, output_dir, resize_max_dim=1600):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff')):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename.rsplit('.', 1)[0] + '.jpg')

            # Enhance and resize image
            enhanced_image = auto_adjust_image(input_path)
            resized_image = resize_image(enhanced_image, max_dim=resize_max_dim)

            # Save the processed image
            resized_image.save(output_path, 'JPEG')
            print(f"Processed and saved {output_path}")

def run_colmap(args):
    colmap_command = '"{}"'.format(args.colmap_executable) if len(args.colmap_executable) > 0 else "colmap"
    use_gpu = 1 if not args.no_gpu else 0

    if not args.skip_matching:
        os.makedirs(args.source_path + "/distorted/sparse", exist_ok=True)

        ## Feature extraction
        feat_extracton_cmd = colmap_command + " feature_extractor "\
            "--database_path " + args.source_path + "/distorted/database.db \
            --image_path " + args.source_path + "/input \
            --ImageReader.single_camera 1 \
            --ImageReader.camera_model " + args.camera + " \
            --SiftExtraction.use_gpu " + str(use_gpu)
        exit_code = os.system(feat_extracton_cmd)
        if exit_code != 0:
            logging.error(f"Feature extraction failed with code {exit_code}. Exiting.")
            exit(exit_code)

        ## Feature matching
        feat_matching_cmd = colmap_command + " exhaustive_matcher \
            --database_path " + args.source_path + "/distorted/database.db \
            --SiftMatching.use_gpu " + str(use_gpu)
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
            --Mapper.ba_global_function_tolerance=0.000001")
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
        logging.error(f"Image undistortion failed with code {exit_code}. Exiting.")
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

def main():
    parser = ArgumentParser("Colmap converter")
    parser.add_argument("--no_gpu", action='store_true')
    parser.add_argument("--skip_matching", action='store_true')
    parser.add_argument("--source_path", "-s", required=True, type=str)
    parser.add_argument("--camera", default="OPENCV", type=str)
    parser.add_argument("--colmap_executable", default="", type=str)
    parser.add_argument("--resize", action="store_true")
    parser.add_argument("--magick_executable", default="", type=str)
    args = parser.parse_args()

    input_dir = os.path.join(args.source_path, 'og_images')
    enhanced_dir = os.path.join(args.source_path, 'input')
    final_dir = os.path.join(args.source_path, 'images')

    # Enhance and resize images before running COLMAP
    process_images(input_dir, enhanced_dir)

    # Run COLMAP
    run_colmap(args)

    print("Image preprocessing and COLMAP execution completed successfully.")

if __name__ == "__main__":
    main()
