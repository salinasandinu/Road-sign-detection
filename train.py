"""
Training script for the CeyRo Traffic Sign and Traffic Light Detection Dataset
This script trains a YOLOv8 model to detect 3 classes:
- DWS-01: Class 0
- DWS-02: Class 1  
- other: Class 2 (all other traffic signs/lights)

Dataset structure:
train_n/
|___ train/
     |___ 1.jpg, 1.xml
     |___ 2.jpg, 2.xml
     |___ ...
test/
|___ test/
     |___ 10.xml
     |___ 10.jpg
     |___ ...
"""

import os
import sys
import yaml
import torch
from pathlib import Path
from ultralytics import YOLO
from xml.etree import ElementTree as ET
from sklearn.model_selection import train_test_split
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Original class dictionary for reading existing label files (75 classes)
ORIGINAL_CLASS_DICT = {
    "DWS-01": 0, "DWS-02": 1, "DWS-03": 2, "DWS-04": 3, "DWS-09": 4, "DWS-10": 5, "DWS-11": 6,
    "DWS-12": 7, "DWS-13": 8, "DWS-14": 9, "DWS-15": 10, "DWS-16": 11, "DWS-17": 12, "DWS-18": 13,
    "DWS-19": 14, "DWS-20": 15, "DWS-21": 16, "DWS-25": 17, "DWS-26": 18, "DWS-27": 19, "DWS-28": 20,
    "DWS-29": 21, "DWS-32": 22, "DWS-33": 23, "DWS-35": 24, "DWS-36": 25, "DWS-40": 26, "DWS-41": 27,
    "DWS-42": 28, "DWS-44": 29, "DWS-46": 30, "MNS-01": 31, "MNS-02": 32, "MNS-03": 33, "MNS-04": 34,
    "MNS-05": 35, "MNS-06": 36, "MNS-07": 37, "MNS-09": 38, "OSD-01": 39, "OSD-02": 40, "OSD-03": 41,
    "OSD-04": 42, "OSD-06": 43, "OSD-07": 44, "OSD-16": 45, "OSD-17": 46, "OSD-26": 47, "PHS-01": 48,
    "PHS-02": 49, "PHS-03": 50, "PHS-04": 51, "PHS-09": 52, "PHS-23": 53, "PHS-24": 54, "PRS-01": 55,
    "PRS-02": 56, "RSS-02": 57, "SLS-100": 58, "SLS-15": 59, "SLS-40": 60, "SLS-50": 61, "SLS-60": 62,
    "SLS-70": 63, "SLS-80": 64, "APR-09": 65, "APR-10": 66, "APR-11": 67, "APR-12": 68, "APR-14": 69,
    "TLS-C": 70, "TLS-E": 71, "TLS-G": 72, "TLS-R": 73, "TLS-Y": 74
}

# Target class mapping for training: DWS-01 -> 0, DWS-02 -> 1, all others -> 2
TARGET_CLASS_DICT = {
    "DWS-01": 0,
    "DWS-02": 1,
    "other": 2
}

# Classes of interest - all other labels will be mapped to "other" (class 2)
TARGET_CLASS_IDS = {0, 1}  # Original class IDs for DWS-01 and DWS-02

NUM_CLASSES = 3  # DWS-01, DWS-02, other

def remap_class_id(original_id):
    """Remap original class ID to target class ID.
    
    DWS-01 (0) -> 0
    DWS-02 (1) -> 1
    All others (2-74) -> 2 (other)
    """
    if original_id in TARGET_CLASS_IDS:
        return original_id  # Keep DWS-01 and DWS-02 as-is
    return 2  # Map everything else to "other"

def remap_existing_labels(labels_dir):
    """Remap class IDs in existing YOLO label files.
    
    This function modifies existing .txt label files to use the 3-class mapping:
    - Class 0 (DWS-01) stays as 0
    - Class 1 (DWS-02) stays as 1  
    - All other classes (2-74) become 2 (other)
    """
    labels_path = Path(labels_dir)
    if not labels_path.exists():
        logger.warning(f"Labels directory not found: {labels_dir}")
        return 0
    
    remapped_count = 0
    for label_file in labels_path.glob("*.txt"):
        with open(label_file, 'r') as f:
            lines = f.readlines()
        
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                original_class = int(parts[0])
                new_class = remap_class_id(original_class)
                parts[0] = str(new_class)
                new_lines.append(' '.join(parts))
        
        with open(label_file, 'w') as f:
            f.write('\n'.join(new_lines))
        remapped_count += 1
    
    logger.info(f"Remapped {remapped_count} label files in {labels_dir}")
    return remapped_count

def get_image_dimensions(image_path):
    """Get image dimensions from XML file."""
    xml_path = str(image_path).replace('.jpg', '.xml')
    if not os.path.exists(xml_path):
        return None, None
    
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    size_elem = root.find('size')
    if size_elem is not None:
        width = int(size_elem.find('width').text)
        height = int(size_elem.find('height').text)
        return width, height
    return None, None

def convert_xml_to_yolo(xml_path, image_width, image_height):
    """Convert XML annotations to YOLO format.
    
    Maps DWS-01 -> 0, DWS-02 -> 1, all other classes -> 2 (other)
    """
    yolo_annotations = []
    
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    for obj in root.findall('object'):
        label = obj.find('name').text
        bndbox = obj.find('bndbox')
        
        xmin = float(bndbox.find('xmin').text)
        ymin = float(bndbox.find('ymin').text)
        xmax = float(bndbox.find('xmax').text)
        ymax = float(bndbox.find('ymax').text)
        
        # Convert to YOLO format (center_x, center_y, width, height) normalized to [0, 1]
        center_x = (xmin + xmax) / 2.0 / image_width
        center_y = (ymin + ymax) / 2.0 / image_height
        width = (xmax - xmin) / image_width
        height = (ymax - ymin) / image_height
        
        # Get original class ID and remap to target classes
        if label in ORIGINAL_CLASS_DICT:
            original_id = ORIGINAL_CLASS_DICT[label]
            class_id = remap_class_id(original_id)
            yolo_annotations.append(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}")
    
    return yolo_annotations

def prepare_dataset(train_dir, output_dir):
    """Prepare dataset in YOLO format."""
    logger.info(f"Preparing dataset from {train_dir}")
    
    train_images_dir = Path(output_dir) / "images" / "train"
    train_labels_dir = Path(output_dir) / "labels" / "train"
    val_images_dir = Path(output_dir) / "images" / "val"
    val_labels_dir = Path(output_dir) / "labels" / "val"
    
    for directory in [train_images_dir, train_labels_dir, val_images_dir, val_labels_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    image_files = sorted([f for f in os.listdir(train_dir) if f.endswith('.jpg')])
    logger.info(f"Found {len(image_files)} images")
    
    # Split into train and validation sets
    train_files, val_files = train_test_split(image_files, test_size=0.2, random_state=42)
    
    logger.info(f"Train: {len(train_files)}, Val: {len(val_files)}")
    
    # Process training set
    for img_file in train_files:
        img_path = os.path.join(train_dir, img_file)
        xml_path = img_path.replace('.jpg', '.xml')
        
        if not os.path.exists(xml_path):
            logger.warning(f"No XML file for {img_file}")
            continue
        
        # Get image dimensions
        width, height = get_image_dimensions(img_path)
        if width is None or height is None:
            logger.warning(f"Could not get dimensions for {img_file}")
            continue
        
        # Convert annotations
        yolo_anns = convert_xml_to_yolo(xml_path, width, height)
        
        # Copy image
        import shutil
        shutil.copy(img_path, train_images_dir / img_file)
        
        # Save YOLO annotations
        label_file = train_labels_dir / img_file.replace('.jpg', '.txt')
        with open(label_file, 'w') as f:
            f.write('\n'.join(yolo_anns))
    
    # Process validation set
    for img_file in val_files:
        img_path = os.path.join(train_dir, img_file)
        xml_path = img_path.replace('.jpg', '.xml')
        
        if not os.path.exists(xml_path):
            logger.warning(f"No XML file for {img_file}")
            continue
        
        # Get image dimensions
        width, height = get_image_dimensions(img_path)
        if width is None or height is None:
            logger.warning(f"Could not get dimensions for {img_file}")
            continue
        
        # Convert annotations
        yolo_anns = convert_xml_to_yolo(xml_path, width, height)
        
        # Copy image
        import shutil
        shutil.copy(img_path, val_images_dir / img_file)
        
        # Save YOLO annotations
        label_file = val_labels_dir / img_file.replace('.jpg', '.txt')
        with open(label_file, 'w') as f:
            f.write('\n'.join(yolo_anns))
    
    # Create dataset YAML
    dataset_yaml = {
        'path': str(Path(output_dir).absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'nc': NUM_CLASSES,
        'names': {v: k for k, v in TARGET_CLASS_DICT.items()}
    }
    
    yaml_path = Path(output_dir) / "data.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(dataset_yaml, f, default_flow_style=False)
    
    logger.info(f"Dataset prepared. YAML file created at {yaml_path}")
    return str(yaml_path)

def train_model(data_yaml, model_name='yolov8m', epochs=100, imgsz=640, batch_size=16, device=0):
    """Train YOLOv8 model."""
    logger.info(f"Starting training with {model_name}")
    logger.info(f"Epochs: {epochs}, Image size: {imgsz}, Batch size: {batch_size}")
    
    # Load model
    model = YOLO(f'{model_name}.pt')
    
    # Check GPU availability (CUDA for NVIDIA, DirectML for AMD)
    if torch.cuda.is_available():
        device_str = str(device)
        logger.info(f"Using NVIDIA GPU (CUDA): {torch.cuda.get_device_name(device)}")
    else:
        # Try DirectML for AMD GPU support
        try:
            import torch_directml  # type: ignore
            if torch_directml.is_available():
                # Note: YOLO may have limited DirectML support
                device_str = 'cpu'  # YOLO doesn't support DirectML device string directly
                logger.warning("AMD GPU detected via DirectML, but YOLO requires CUDA.")
                logger.warning("Training will run on CPU. For GPU training, use NVIDIA GPU or Google Colab.")
            else:
                device_str = 'cpu'
        except ImportError:
            device_str = 'cpu'
            logger.info("torch_directml not installed. To enable AMD GPU support, run: pip install torch-directml")
        logger.info(f"Using device: {device_str}")
    
    # Train
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        device=device_str,
        patience=20,  # Early stopping patience
        save=True,
        project='runs/detect',
        name='traffic_signs',
        exist_ok=True,
        verbose=True,
        augment=True,
        mosaic=1.0,
        flipud=0.5,
        fliplr=0.5,
        degrees=10,
        translate=0.1,
        scale=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        perspective=0.0,
        cfg=None,
        optimizer='SGD',
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
    )
    
    logger.info("Training completed!")
    return results

def main():
    
    if 'ipykernel' in sys.modules:
        # Running in Jupyter / Colab
        class Args:
            train_dir = '../input/road-signs/train_n/train'
            output_dir = 'dataset'
            model = 'yolov8m'
            epochs = 100
            batch_size = 16
            imgsz = 640
            device = 0
            skip_prepare = False
            remap_labels = False

        args = Args()
    else:
        parser = argparse.ArgumentParser(description='Train YOLOv8 model on traffic signs dataset')
        parser.add_argument('--train_dir', type=str, default='train_n/train', 
                            help='Path to training images directory')
        parser.add_argument('--output_dir', type=str, default='dataset', 
                            help='Output directory for prepared dataset')
        parser.add_argument('--model', type=str, default='yolov8m', 
                            choices=['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x'],
                            help='YOLOv8 model size')
        parser.add_argument('--epochs', type=int, default=100, 
                            help='Number of training epochs')
        parser.add_argument('--batch_size', type=int, default=16, 
                            help='Batch size for training')
        parser.add_argument('--imgsz', type=int, default=640, 
                            help='Image size for training')
        parser.add_argument('--device', type=int, default=0, 
                            help='GPU device ID (0 for first GPU, -1 for CPU)')
        parser.add_argument('--skip_prepare', action='store_true', 
                            help='Skip dataset preparation (use existing dataset)')
        parser.add_argument('--remap_labels', action='store_true',
                            help='Remap existing label files to 3-class format (DWS-01, DWS-02, other)')
    
        args = parser.parse_args()
    
    # Prepare dataset
    if not args.skip_prepare:
        data_yaml = prepare_dataset(args.train_dir, args.output_dir)
    else:
        data_yaml = os.path.join(args.output_dir, 'data.yaml')
        if not os.path.exists(data_yaml):
            logger.error(f"Dataset YAML not found at {data_yaml}")
            return
        
        # Remap existing labels if requested
        if args.remap_labels:
            logger.info("Remapping existing labels to 3-class format...")
            train_labels = os.path.join(args.output_dir, 'labels', 'train')
            val_labels = os.path.join(args.output_dir, 'labels', 'val')
            remap_existing_labels(train_labels)
            remap_existing_labels(val_labels)
            
            # Update data.yaml with new class configuration
            yaml_path = Path(args.output_dir) / "data.yaml"
            dataset_yaml = {
                'path': str(Path(args.output_dir).absolute()),
                'train': 'images/train',
                'val': 'images/val',
                'nc': NUM_CLASSES,
                'names': {v: k for k, v in TARGET_CLASS_DICT.items()}
            }
            with open(yaml_path, 'w') as f:
                yaml.dump(dataset_yaml, f, default_flow_style=False)
            logger.info(f"Updated {yaml_path} with 3-class configuration")
    
    # Train model
    train_model(
        data_yaml=data_yaml,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        imgsz=args.imgsz,
        device=args.device
    )

if __name__ == "__main__":
    main()