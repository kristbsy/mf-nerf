from transformers import pipeline, utils
from PIL import Image
import requests
import numpy as np
from pathlib import Path
import argparse

utils.logging.set_verbosity_error()

# Define the class names and color palette from Cityscapes
CLASSES = [
    "road", "sidewalk", "building", "wall", "fence", "pole", "traffic light", "traffic sign", 
    "vegetation", "terrain", "sky", "person", "rider", "car", "truck", "bus", "train", 
    "motorcycle", "bicycle"
]
PALETTE = np.array([[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156], [190, 153, 153], 
                    [153, 153, 153], [250, 170, 30], [220, 220, 0], [107, 142, 35], [152, 251, 152], 
                    [70, 130, 180], [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100], 
                    [0, 80, 100], [0, 0, 230], [119, 11, 32]])

FILTER = ['car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle', 'person', 'rider']


def load_image(image_path: str):
    """Load an image from a file or URL."""
    if image_path.startswith('http://') or image_path.startswith('https://'):
        return Image.open(requests.get(image_path, stream=True).raw).convert('RGB')
    else:
        return Image.open(image_path).convert('RGB')
    

def segment_image(image_path: str):
    """Segment the image at the given path."""
    image = load_image(image_path)
    original_size = image.size

    semantic_segmentation = pipeline("image-segmentation", "nvidia/segformer-b1-finetuned-cityscapes-1024-1024", feature_extractor="nvidia/segformer-b1-finetuned-cityscapes-1024-1024")
    results = semantic_segmentation(image)
    mask = np.zeros((original_size[1], original_size[0]), dtype=bool)
    for r in results:
        if r['label'] in FILTER:
            mask = mask | r['mask'] > 0

    # Apply transparency based on segmentation
    alpha_channel = np.where(mask, 0, 255).astype(np.uint8)  # Set alpha to 0 for masked areas
    image_rgba = image.convert("RGBA")
    image_array = np.array(image_rgba)
    image_array[:, :, 3] = alpha_channel
    
    # Save the modified image inplace
    result_image = Image.fromarray(image_array)
    result_image.save(image_path)


def main():
    parser = argparse.ArgumentParser(description='Segment images')
    parser.add_argument('data', type=str, help='Folder containing images to segment')
    data_folder = Path(parser.parse_args().data)
    
    for image_path in data_folder.glob('*.png'):
        segment_image(image_path.as_posix())


if __name__ == '__main__':
    main()