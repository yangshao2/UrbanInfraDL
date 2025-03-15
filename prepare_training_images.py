import rasterio
import os
import re

# Function to extract image ID from filename
def get_image_id(filepath):
    filename = os.path.basename(filepath)
    image_id, _ = os.path.splitext(filename)
    return image_id

# Function to clip images into patches
def clip_image(image_path, output_dir, patch_size=(256, 256), stride=(128, 128)):
    images_output_dir = os.path.join(output_dir, 'images')

    # Create the images folder if it doesn't exist
    os.makedirs(images_output_dir, exist_ok=True)

    # Open image file
    with rasterio.open(image_path) as src_image:
        image = src_image.read()
        image_meta = src_image.meta.copy()
        image_transform = src_image.transform

    # Extract dimensions
    image_height, image_width = image.shape[1], image.shape[2]
    patch_height, patch_width = patch_size
    stride_height, stride_width = stride

    # Update metadata for patches
    image_meta.update({
        'height': patch_height,
        'width': patch_width,
        'count': image.shape[0]  # Number of channels
    })

    image_id = get_image_id(image_path)
    patch_id = 0

    # Generate patches
    for i in range(0, image_height, stride_height):
        for j in range(0, image_width, stride_width):
            # Ensure the patch doesn't exceed image dimensions
            end_i = min(i + patch_height, image_height)
            end_j = min(j + patch_width, image_width)
            start_i = end_i - patch_height if end_i - i < patch_height else i
            start_j = end_j - patch_width if end_j - j < patch_width else j

            image_patch = image[:, start_i:end_i, start_j:end_j]

            # Ensure patch size is correct
            if image_patch.shape[1] == patch_height and image_patch.shape[2] == patch_width:
                new_transform = rasterio.Affine(
                    image_transform.a, image_transform.b, image_transform.c + start_j * image_transform.a,
                    image_transform.d, image_transform.e, image_transform.f + start_i * image_transform.e
                )

                image_meta.update({'transform': new_transform})

                image_patch_filename = os.path.join(images_output_dir, f'image_patch_{patch_id}_{image_id}.tif')

                # Write patches to files
                with rasterio.open(image_patch_filename, 'w', **image_meta) as dst_image:
                    dst_image.write(image_patch)

                print(f"Saved {image_patch_filename}")
                patch_id += 1

# Process all tif files in the folder
def process_folder(image_dir, output_dir, patch_size=(256, 256), stride=(128, 128)):
    # Get all tif files in the directory
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.tif')])

    # Loop through each image
    for img_file in image_files:
        image_path = os.path.join(image_dir, img_file)

        print(f"Processing {img_file}")
        clip_image(image_path, output_dir, patch_size, stride)

# Folder paths
image_dir = '/media/newhd/yshao/bog/rgb_raw/'
output_dir = '/media/newhd/yshao/bog/training_fourclass'

# Patch size and stride
patch_size = (256, 256)
stride = (128, 128)

# Process all files in the folder
process_folder(image_dir, output_dir, patch_size, stride)