import rasterio
import os
import re
import argparse

def extract_identifier(filename):
    """Extract the numeric identifier from the filename for label images."""
    print(f"Debug: Extracting identifier from {filename}")
    # Match the numeric part preceded by '_road_' and followed by '.tif'
    match = re.search(r'_road_(\d+)\.tif$', filename)
    if match:
        return match.group(1)
    else:
        raise ValueError(f"Unable to extract numeric identifier from {filename}")

def get_image_id(filepath):
    """Extract the image ID from the filepath for raw images."""
    filename = os.path.basename(filepath)
    image_id, _ = os.path.splitext(filename)
    return image_id

def clip_image(image_path, output_dir, patch_size=(256, 256), stride=(128, 128), mode="raw"):
    """
    Clip the given image into patches.

    For raw images, the output filename format is:
        image_patch_<patch_id>_<image_id>.tif
    For reference label images, it is:
        label_patch_<patch_id>_t<identifier>.tif
    """
    if mode == "raw":
        output_subdir = os.path.join(output_dir, "images")
        os.makedirs(output_subdir, exist_ok=True)
        image_id = get_image_id(image_path)
        file_prefix = "image_patch"
        out_filename_format = f"{file_prefix}_{{patch_id}}_{image_id}.tif"
    elif mode == "label":
        output_subdir = os.path.join(output_dir, "labels")
        os.makedirs(output_subdir, exist_ok=True)
        filename = os.path.basename(image_path)
        identifier = extract_identifier(filename)
        file_prefix = "label_patch"
        out_filename_format = f"{file_prefix}_{{patch_id}}_t{identifier}.tif"
    else:
        raise ValueError("Invalid mode. Use 'raw' for raw images or 'label' for reference labels.")

    with rasterio.open(image_path) as src_image:
        image = src_image.read()
        image_meta = src_image.meta.copy()
        image_transform = src_image.transform

    image_height, image_width = image.shape[1], image.shape[2]
    patch_height, patch_width = patch_size
    stride_height, stride_width = stride

    image_meta.update({
        'height': patch_height,
        'width': patch_width,
        'count': image.shape[0]
    })

    patch_id = 0
    for i in range(0, image_height, stride_height):
        for j in range(0, image_width, stride_width):
            # Calculate patch boundaries ensuring the patch is the correct size
            end_i = min(i + patch_height, image_height)
            end_j = min(j + patch_width, image_width)
            start_i = end_i - patch_height if end_i - i < patch_height else i
            start_j = end_j - patch_width if end_j - j < patch_width else j

            image_patch = image[:, start_i:end_i, start_j:end_j]

            if image_patch.shape[1] == patch_height and image_patch.shape[2] == patch_width:
                new_transform = rasterio.Affine(
                    image_transform.a, image_transform.b, image_transform.c + start_j * image_transform.a,
                    image_transform.d, image_transform.e, image_transform.f + start_i * image_transform.e
                )
                image_meta.update({'transform': new_transform})

                out_filename = os.path.join(output_subdir, out_filename_format.format(patch_id=patch_id))
                with rasterio.open(out_filename, 'w', **image_meta) as dst_image:
                    dst_image.write(image_patch)

                print(f"Saved {out_filename}")
                patch_id += 1

def process_folder(input_dir, output_dir, patch_size=(256, 256), stride=(128, 128), mode="raw"):
    """Process all .tif files in the specified folder."""
    image_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.tif')])
    if not image_files:
        raise ValueError(f"No .tif files found in the directory: {input_dir}")
    
    print("Files found:", image_files)
    for img_file in image_files:
        print(f"Processing {img_file}")
        image_path = os.path.join(input_dir, img_file)
        clip_image(image_path, output_dir, patch_size, stride, mode=mode)

def main():
    parser = argparse.ArgumentParser(description="Extract patches from images for training.")
    parser.add_argument("--mode", choices=["raw", "label"], required=True,
                        help="Specify 'raw' for raw images or 'label' for reference labels")
    parser.add_argument("--input_dir", required=True,
                        help="Directory containing input .tif images")
    parser.add_argument("--output_dir", required=True,
                        help="Directory to save output patches")
    parser.add_argument("--patch_size", type=int, nargs=2, default=(256, 256),
                        help="Patch size as two integers: height width")
    parser.add_argument("--stride", type=int, nargs=2, default=(128, 128),
                        help="Stride as two integers: vertical horizontal")
    
    args = parser.parse_args()
    
    process_folder(args.input_dir, args.output_dir,
                   patch_size=tuple(args.patch_size),
                   stride=tuple(args.stride),
                   mode=args.mode)

if __name__ == "__main__":
    main()
