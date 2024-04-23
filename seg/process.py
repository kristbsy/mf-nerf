import json
from tqdm import tqdm
from transformers import pipeline, utils
from PIL import Image
import requests
import numpy as np
from simple_chalk import chalk
from pathlib import Path
import argparse

utils.logging.set_verbosity_error()
semantic_segmentation = pipeline(task="image-segmentation", model="nvidia/segformer-b1-finetuned-cityscapes-1024-1024", feature_extractor="nvidia/segformer-b1-finetuned-cityscapes-1024-1024", device=0)

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
        return Image.open(requests.get(image_path, stream=True).raw)
    else:
        return Image.open(image_path)


def segment_images(folder: Path):
    image_dir = folder / 'images'
    pathlist = list(image_dir.glob('*.png'))
    mask_dir = folder / 'masks'
    Path.mkdir(mask_dir, exist_ok=True)
    
    print(f"Segmenting {len(pathlist)} images in {folder}")
    
    with open(folder / 'transforms.json', 'r') as f:
        transformsFile = json.load(f)
        
        ## use tqdm to show progress bar
        for index, image_path in enumerate(tqdm(pathlist)):
            image_path = image_path.as_posix()
            mask_img_path = image_path.split('/')[-1].replace('.png', '_mask.jpeg')
            mask_path_complete = mask_dir / mask_img_path
            transformsFile["frames"][index]["mask_path"] = "masks/" + mask_img_path
            
            image = load_image(image_path)
            original_size = image.size

            results = semantic_segmentation(image)
            mask = np.zeros((original_size[1], original_size[0]), dtype=bool)
            for r in results:
                if r['label'] in FILTER:
                    mask = mask | r['mask'] > 0

            # Apply transparency based on segmentation
            alpha_channel = np.where(mask, 0, 255).astype(np.uint8)  # Set alpha to 0 for masked areas
            
            # Save the modified image inplace
            result_image = Image.fromarray(alpha_channel, mode='L')
            result_image.save(mask_path_complete, 'jpeg')
        
        print(chalk.green("Segmentation complete, saving transforms file..."))
        json.dump(transformsFile, open(folder / 'transforms.json', 'w'))


def main():
    parser = argparse.ArgumentParser(description='Segment images')
    parser.add_argument('data', type=str, help='Folder containing images to segment')
    data_folder = Path(parser.parse_args().data)
    segment_images(data_folder)
    


if __name__ == '__main__':
    main()