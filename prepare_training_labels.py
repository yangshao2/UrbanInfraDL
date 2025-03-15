import rasterio
import os
import re

# Function to extract the numeric identifier from the filename
def extract_identifier(filename):
    print(f"Debug: Extracting identifier from {filename}")
    # Match the numeric part preceded by "_road_" and followed by ".tif"
    match = re.search(r'_road_(\d+)\.tif$', filename)
    if match:
        return match.group(1)  # Extract the number
    else:
        raise ValueError(f"Unable to extract numeric identifier from {filename}")

# Function to extract image ID from the filepath
def get_image_id(filepath):
    filename = os.path.basename(filepath)
    image_id, _ = os.path.splitext(filename)
    return image_id

# Function to clip images into patches
def clip_image(image_path, output_dir, patch_size=(256, 256), stride=(128, 128)):
    labels_output_dir = os.path.join(output_dir, 'labels')

    # Create the labels folder if it doesn't exist
    os.makedirs(labels_output_dir, exist_ok=True)

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

    # Extract unique identifier
    filename = os.path.basename(image_path)  # Get the full filename with extension
    identifier = extract_identifier(filename)  # Pass the full filename to extract_identifier
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

                # Format the output filename
                label_patch_filename = os.path.join(
                    labels_output_dir,
                    f'label_patch_{patch_id}_t{identifier}.tif'
                )

                # Write patches to files
                with rasterio.open(label_patch_filename, 'w', **image_meta) as dst_image:
                    dst_image.write(image_patch)

                print(f"Saved {label_patch_filename}")
                patch_id += 1

# Function to process all TIFF files in the folder
def process_folder(image_dir, output_dir, patch_size=(256, 256), stride=(128, 128)):
    # Get all TIFF files in the directory
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.tif')])

    if not image_files:
        raise ValueError(f"No .tif files found in the directory: {image_dir}")

    # Debugging: Print the files found
    print("Files found:", image_files)

    # Loop through each image
    for img_file in image_files:
        print(f"Processing {img_file}")

        # Debugging: Check identifier extraction
        identifier = extract_identifier(img_file)
        print(f"Extracted Identifier for {img_file}: {identifier}")

        image_path = os.path.join(image_dir, img_file)
        clip_image(image_path, output_dir, patch_size, stride)

# Folder paths
image_dir = '/media/newhd/yshao/bog/reference_labels/'  # Update this to your input directory
output_dir = '/media/newhd/yshao/bog/training_fourclass'  # Update this to your output directory

# Patch size and stride
patch_size = (256, 256)
stride = (128, 128)

# Run the processing function
process_folder(image_dir, output_dir, patch_size, stride)
