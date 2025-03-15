patch_extractor.py
patch_extractor.py is a command-line utility designed to generate image patches from large TIFF files. It supports two modes:

raw: Processes raw images, saving patches in an images folder.
label: Processes reference label images (e.g., road, sidewalk, and bicycle lane annotations), saving patches in a labels folder.
Users can specify patch size and stride via command-line arguments, making it easy to tailor the output for deep learning training data. 

Example:
python patch_extractor.py --mode raw --input_dir /path/to/raw_images --output_dir /path/to/output --patch_size 256 256 --stride 128 128

