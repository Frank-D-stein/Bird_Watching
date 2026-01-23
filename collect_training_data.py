#!/usr/bin/env python3
"""
Collect bird and human images for training the classifier.

This script provides multiple methods:
1. Download from public datasets (Flickr, iNaturalist via URLs)
2. Capture from your own camera feed
3. Use existing sightings from the app

Storage requirements:
- ~50-100 images per species is a good starting point
- Each image ~50-200KB (resized to 224x224)
- 51 species × 100 images × 100KB = ~500MB total
- With data augmentation, you can get good results with fewer images

Usage:
    python collect_training_data.py --method flickr --species "Northern Cardinal"
    python collect_training_data.py --method capture --output-dir ./data/training
    python collect_training_data.py --method existing --source ./data/images
"""
import argparse
import hashlib
import json
import logging
import os
import random
import shutil
import time
from pathlib import Path
from urllib.parse import quote

import requests

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Southeastern US bird species
SE_US_BIRDS = [
    "Northern Cardinal", "Blue Jay", "American Robin", "Carolina Chickadee",
    "Carolina Wren", "Tufted Titmouse", "Eastern Bluebird", "Mourning Dove",
    "Ruby-throated Hummingbird", "Red-bellied Woodpecker", "Downy Woodpecker",
    "Pileated Woodpecker", "Red-headed Woodpecker", "Pine Warbler",
    "Yellow-rumped Warbler", "Prothonotary Warbler", "Common Yellowthroat",
    "Summer Tanager", "Scarlet Tanager", "Indigo Bunting", "Painted Bunting",
    "House Finch", "American Goldfinch", "Chipping Sparrow", "Song Sparrow",
    "White-throated Sparrow", "Dark-eyed Junco", "Eastern Towhee",
    "Brown Thrasher", "Gray Catbird", "Northern Mockingbird", "European Starling",
    "American Crow", "Fish Crow", "Common Grackle", "Boat-tailed Grackle",
    "Red-winged Blackbird", "Brown-headed Cowbird", "Loggerhead Shrike",
    "Blue Grosbeak", "Rose-breasted Grosbeak", "Cedar Waxwing", "Barn Swallow",
    "Purple Martin", "Chimney Swift", "Great Crested Flycatcher", "Eastern Phoebe",
    "Eastern Kingbird", "Wild Turkey", "Northern Bobwhite", "Killdeer"
]


def download_from_flickr(species, output_dir, num_images=50, api_key=None):
    """
    Download bird images from Flickr.
    
    Note: Requires a Flickr API key (free at https://www.flickr.com/services/api/)
    Without an API key, this uses a fallback method with limited results.
    """
    species_dir = Path(output_dir) / species.replace(' ', '_')
    species_dir.mkdir(parents=True, exist_ok=True)
    
    if api_key:
        # Use Flickr API
        search_url = "https://api.flickr.com/services/rest/"
        params = {
            'method': 'flickr.photos.search',
            'api_key': api_key,
            'text': f'{species} bird',
            'sort': 'relevance',
            'per_page': num_images,
            'format': 'json',
            'nojsoncallback': 1,
            'license': '1,2,3,4,5,6',  # Creative Commons licenses
            'content_type': 1,  # Photos only
            'media': 'photos'
        }
        
        try:
            response = requests.get(search_url, params=params, timeout=30)
            data = response.json()
            
            if data.get('stat') != 'ok':
                logger.error(f"Flickr API error: {data}")
                return 0
            
            photos = data.get('photos', {}).get('photo', [])
            downloaded = 0
            
            for photo in photos:
                img_url = f"https://live.staticflickr.com/{photo['server']}/{photo['id']}_{photo['secret']}_w.jpg"
                img_path = species_dir / f"{photo['id']}.jpg"
                
                if img_path.exists():
                    continue
                
                try:
                    img_response = requests.get(img_url, timeout=10)
                    if img_response.status_code == 200:
                        with open(img_path, 'wb') as f:
                            f.write(img_response.content)
                        downloaded += 1
                        logger.info(f"Downloaded {img_path.name}")
                except Exception as e:
                    logger.warning(f"Failed to download {img_url}: {e}")
                
                time.sleep(0.5)  # Rate limiting
            
            return downloaded
            
        except Exception as e:
            logger.error(f"Flickr API error: {e}")
            return 0
    else:
        logger.warning("No Flickr API key provided. Using alternative sources.")
        return download_from_inaturalist(species, output_dir, num_images)


def download_from_inaturalist(species, output_dir, num_images=50):
    """
    Download images from iNaturalist (free, no API key required).
    
    iNaturalist has high-quality, research-grade bird photos.
    """
    species_dir = Path(output_dir) / species.replace(' ', '_')
    species_dir.mkdir(parents=True, exist_ok=True)
    
    # Search for the species
    search_url = "https://api.inaturalist.org/v1/taxa"
    params = {
        'q': species,
        'rank': 'species',
        'per_page': 1
    }
    
    try:
        response = requests.get(search_url, params=params, timeout=30)
        data = response.json()
        
        if not data.get('results'):
            logger.warning(f"Species not found on iNaturalist: {species}")
            return 0
        
        taxon_id = data['results'][0]['id']
        taxon_name = data['results'][0]['name']
        logger.info(f"Found taxon: {taxon_name} (ID: {taxon_id})")
        
        # Get observations with photos
        obs_url = "https://api.inaturalist.org/v1/observations"
        obs_params = {
            'taxon_id': taxon_id,
            'quality_grade': 'research',  # Only verified IDs
            'photos': 'true',
            'per_page': num_images,
            'order': 'desc',
            'order_by': 'votes'  # Most popular first
        }
        
        obs_response = requests.get(obs_url, params=obs_params, timeout=30)
        obs_data = obs_response.json()
        
        downloaded = 0
        for obs in obs_data.get('results', []):
            for photo in obs.get('photos', [])[:1]:  # First photo per observation
                img_url = photo.get('url', '').replace('square', 'medium')
                if not img_url:
                    continue
                
                # Create unique filename from URL hash
                url_hash = hashlib.md5(img_url.encode()).hexdigest()[:8]
                img_path = species_dir / f"inat_{url_hash}.jpg"
                
                if img_path.exists():
                    continue
                
                try:
                    img_response = requests.get(img_url, timeout=10)
                    if img_response.status_code == 200:
                        with open(img_path, 'wb') as f:
                            f.write(img_response.content)
                        downloaded += 1
                        logger.info(f"Downloaded {img_path.name}")
                except Exception as e:
                    logger.warning(f"Failed to download: {e}")
                
                time.sleep(0.3)  # Rate limiting
                
                if downloaded >= num_images:
                    break
            
            if downloaded >= num_images:
                break
        
        return downloaded
        
    except Exception as e:
        logger.error(f"iNaturalist error: {e}")
        return 0


def download_humans(output_dir, num_images=100):
    """
    Download human/person images for the object detector.
    
    Uses the COCO dataset URLs or other public sources.
    """
    humans_dir = Path(output_dir) / "person"
    humans_dir.mkdir(parents=True, exist_ok=True)
    
    # Use a simple public API for sample images
    # In production, you'd use COCO dataset or similar
    logger.info("Downloading sample human images...")
    
    # Use Unsplash API (free, requires signup for production use)
    # For demo purposes, we'll create placeholder guidance
    
    readme_path = humans_dir / "README.txt"
    with open(readme_path, 'w') as f:
        f.write("""Human Training Images

To train the object detector to recognize humans, you need person images.

Recommended sources:
1. COCO Dataset (https://cocodataset.org/)
   - Download the 'person' category images
   - High quality, well-labeled

2. Open Images Dataset (https://storage.googleapis.com/openimages/web/index.html)
   - Large scale, diverse images

3. Your own camera captures
   - Run: python collect_training_data.py --method capture --category person

4. Unsplash API (https://unsplash.com/developers)
   - Free for non-commercial use
   - Search for 'person', 'people', 'human'

Minimum recommended: 100-500 images
""")
    
    logger.info(f"Created guidance at {readme_path}")
    logger.info("For human images, consider using COCO dataset or your own captures.")
    return 0


def capture_from_camera(output_dir, category="bird", num_images=50, camera_index=0):
    """
    Capture training images directly from camera.
    
    Press 's' to save a frame, 'q' to quit.
    """
    try:
        import cv2
    except ImportError:
        logger.error("OpenCV not installed. Run: pip install opencv-python")
        return 0
    
    category_dir = Path(output_dir) / category.replace(' ', '_')
    category_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        logger.error(f"Cannot open camera {camera_index}")
        return 0
    
    logger.info(f"Camera opened. Press 's' to save, 'q' to quit.")
    logger.info(f"Saving to: {category_dir}")
    
    saved = 0
    while saved < num_images:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Show frame with instructions
        display = frame.copy()
        cv2.putText(display, f"Category: {category}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display, f"Saved: {saved}/{num_images}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display, "Press 's' to save, 'q' to quit", (10, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        
        cv2.imshow('Capture Training Data', display)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            timestamp = int(time.time() * 1000)
            img_path = category_dir / f"capture_{timestamp}.jpg"
            cv2.imwrite(str(img_path), frame)
            saved += 1
            logger.info(f"Saved: {img_path.name} ({saved}/{num_images})")
        elif key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    return saved


def use_existing_sightings(source_dir, output_dir, min_confidence=0.5):
    """
    Use existing bird sighting images from the app.
    
    Organizes images by detected species.
    """
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    if not source_path.exists():
        logger.error(f"Source directory not found: {source_path}")
        return 0
    
    # Look for JSON records with species info
    json_files = list(source_path.parent.glob('logs/*.json'))
    
    copied = 0
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                record = json.load(f)
            
            species_info = record.get('species', {})
            species = species_info.get('species', 'Unknown')
            confidence = species_info.get('confidence', 0)
            image_path = record.get('image_path')
            
            if species == 'Unknown' or confidence < min_confidence:
                continue
            
            if image_path and Path(image_path).exists():
                species_dir = output_path / species.replace(' ', '_')
                species_dir.mkdir(parents=True, exist_ok=True)
                
                dest_path = species_dir / Path(image_path).name
                if not dest_path.exists():
                    shutil.copy2(image_path, dest_path)
                    copied += 1
                    logger.info(f"Copied: {dest_path.name} -> {species}")
        except Exception as e:
            logger.warning(f"Error processing {json_file}: {e}")
    
    return copied


def download_all_species(output_dir, num_per_species=50, api_key=None):
    """Download images for all SE US bird species."""
    total = 0
    for species in SE_US_BIRDS:
        logger.info(f"\n{'='*50}")
        logger.info(f"Downloading: {species}")
        logger.info(f"{'='*50}")
        
        count = download_from_inaturalist(species, output_dir, num_per_species)
        total += count
        logger.info(f"Downloaded {count} images for {species}")
        
        time.sleep(1)  # Rate limiting between species
    
    return total


def create_data_augmentation_script(output_dir):
    """Create a script to augment training data."""
    script_path = Path(output_dir) / "augment_data.py"
    
    script_content = '''#!/usr/bin/env python3
"""
Augment training images to increase dataset size.

This can multiply your dataset by 5-10x using transformations like:
- Rotation, flipping, cropping
- Color adjustments
- Noise addition

Usage:
    python augment_data.py --input ./data/training --output ./data/training_augmented
"""
import argparse
from pathlib import Path
import random

try:
    from PIL import Image, ImageEnhance, ImageFilter
    import numpy as np
except ImportError:
    print("Install requirements: pip install pillow numpy")
    exit(1)


def augment_image(img):
    """Apply random augmentations to an image."""
    augmented = []
    
    # Original
    augmented.append(img)
    
    # Horizontal flip
    augmented.append(img.transpose(Image.FLIP_LEFT_RIGHT))
    
    # Rotation variations
    for angle in [-15, -10, 10, 15]:
        rotated = img.rotate(angle, fillcolor=(128, 128, 128))
        augmented.append(rotated)
    
    # Brightness variations
    enhancer = ImageEnhance.Brightness(img)
    augmented.append(enhancer.enhance(0.8))
    augmented.append(enhancer.enhance(1.2))
    
    # Contrast variations
    enhancer = ImageEnhance.Contrast(img)
    augmented.append(enhancer.enhance(0.8))
    augmented.append(enhancer.enhance(1.2))
    
    # Random crop and resize
    w, h = img.size
    for _ in range(2):
        left = random.randint(0, w // 10)
        top = random.randint(0, h // 10)
        right = w - random.randint(0, w // 10)
        bottom = h - random.randint(0, h // 10)
        cropped = img.crop((left, top, right, bottom))
        cropped = cropped.resize((w, h), Image.LANCZOS)
        augmented.append(cropped)
    
    return augmented


def process_directory(input_dir, output_dir):
    """Process all images in directory."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    total = 0
    for species_dir in input_path.iterdir():
        if not species_dir.is_dir():
            continue
        
        out_species_dir = output_path / species_dir.name
        out_species_dir.mkdir(parents=True, exist_ok=True)
        
        for img_file in species_dir.glob("*.jpg"):
            try:
                img = Image.open(img_file).convert('RGB')
                img = img.resize((224, 224), Image.LANCZOS)
                
                augmented = augment_image(img)
                
                for i, aug_img in enumerate(augmented):
                    out_path = out_species_dir / f"{img_file.stem}_aug{i}.jpg"
                    aug_img.save(out_path, quality=90)
                    total += 1
                
            except Exception as e:
                print(f"Error processing {img_file}: {e}")
    
    return total


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input directory")
    parser.add_argument("--output", required=True, help="Output directory")
    args = parser.parse_args()
    
    count = process_directory(args.input, args.output)
    print(f"Created {count} augmented images")
'''
    
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    logger.info(f"Created augmentation script: {script_path}")
    return script_path


def main():
    parser = argparse.ArgumentParser(description='Collect training images for bird classifier')
    parser.add_argument('--method', choices=['flickr', 'inaturalist', 'capture', 'existing', 'all'],
                       default='inaturalist', help='Collection method')
    parser.add_argument('--species', default=None, help='Specific species to download')
    parser.add_argument('--output-dir', default='./data/training', help='Output directory')
    parser.add_argument('--num-images', type=int, default=50, help='Number of images per species')
    parser.add_argument('--flickr-api-key', default=None, help='Flickr API key')
    parser.add_argument('--camera-index', type=int, default=0, help='Camera index for capture')
    parser.add_argument('--source', default='./data/images', help='Source for existing method')
    parser.add_argument('--include-humans', action='store_true', help='Also download human images')
    parser.add_argument('--create-augmentation', action='store_true', help='Create augmentation script')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    total = 0
    
    if args.method == 'flickr':
        if args.species:
            total = download_from_flickr(args.species, output_dir, args.num_images, args.flickr_api_key)
        else:
            logger.info("Downloading all species from Flickr...")
            for species in SE_US_BIRDS:
                count = download_from_flickr(species, output_dir, args.num_images, args.flickr_api_key)
                total += count
                time.sleep(1)
    
    elif args.method == 'inaturalist':
        if args.species:
            total = download_from_inaturalist(args.species, output_dir, args.num_images)
        else:
            total = download_all_species(output_dir, args.num_images)
    
    elif args.method == 'capture':
        category = args.species or 'bird'
        total = capture_from_camera(output_dir, category, args.num_images, args.camera_index)
    
    elif args.method == 'existing':
        total = use_existing_sightings(args.source, output_dir)
    
    elif args.method == 'all':
        total = download_all_species(output_dir, args.num_images, args.flickr_api_key)
    
    if args.include_humans:
        download_humans(output_dir)
    
    if args.create_augmentation:
        create_data_augmentation_script(output_dir)
    
    # Print summary
    print(f"\n{'='*50}")
    print("Collection Summary")
    print(f"{'='*50}")
    print(f"Total images downloaded: {total}")
    print(f"Output directory: {output_dir}")
    
    # Count by species
    species_counts = {}
    for species_dir in output_dir.iterdir():
        if species_dir.is_dir():
            count = len(list(species_dir.glob("*.jpg")))
            if count > 0:
                species_counts[species_dir.name] = count
    
    if species_counts:
        print(f"\nImages per species:")
        for species, count in sorted(species_counts.items()):
            print(f"  {species}: {count}")
    
    # Storage estimate
    total_size = sum(f.stat().st_size for f in output_dir.rglob("*.jpg"))
    print(f"\nTotal storage used: {total_size / 1024 / 1024:.1f} MB")
    
    print(f"\nNext steps:")
    print(f"  1. Review downloaded images for quality")
    print(f"  2. Run augmentation: python {output_dir}/augment_data.py --input {output_dir} --output {output_dir}_augmented")
    print(f"  3. Train the model: python download_model.py --model mobilenet --create-training-script")
    print(f"     Then: python data/models/train_bird_model.py --data-dir {output_dir}_augmented")


if __name__ == '__main__':
    main()
