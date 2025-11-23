from __future__ import annotations

import re
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision.transforms import InterpolationMode, functional as F

import cv2 # pip install opencv-contrib-python
import xml.etree.ElementTree as ET
import json
import numpy as np

# Default dataset roots provided by the user.
DEFAULT_DATA_ROOTS = {
	"pothole": Path("/dtu/datasets1/02516/potholes"),
}

IMAGE_SIZE = (256, 256)

def resize_image(image, resize_to, output_type='numpy', interpolation=transforms.InterpolationMode.BILINEAR, antialias=True):
    """
    return: Resized image in the specified format.

    resize_to: Tuple (width, height) to resize the image to.
    output_type: Specify 'PIL', 'numpy', or 'tensor' for the output format.
    interpolation: Interpolation mode to use for resizing.
    antialias: Whether to apply antialiasing
    """
    # 1. Define & apply the transformation
    resize_transform = transforms.Resize(size=resize_to, interpolation=interpolation, antialias=antialias)
    resized_image = resize_transform(image)

    # 2. Convert to the desired output type
    if output_type == 'numpy':
        return np.array(resized_image)  
    elif output_type == 'tensor':
        return transforms.ToTensor()(resized_image)  
    else:
        return resized_image  

def ensure_rgb(image):
    # If image has 4 channels, convert to 3 channels (RGB)
    if image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    return image

def extract_index(filename):
    match = re.search(r'(\d+)', filename)
    return int(match.group(1)) if match else -1

def read_data(directory, image_size): 
    """
    Image: returns as np arrays 

    Annotation: for each image, theres a corresponding dict in the format of:
                    'filename': image_name,
                    'image dimension': {'width': width, 'height': height},
                    'bboxes': a list of bounding box coordinates
    """

    # 1. Read image data and annotations from the given directory
    # 1.1 Images
    image_path = os.path.join(directory, 'images')
    image_files = sorted(
        [f for f in os.listdir(image_path) if f.lower().endswith('.png')],
        key=extract_index
    )
    images = []
    
    for filename in image_files:
        file_path = os.path.join(image_path, filename)
        
        # Check if the file is an image based on the file extension
        try:
            image = Image.open(file_path)
            if image_size is not None:
                image_array = resize_image(image, image_size)
            else:
                image_array = np.array(image)

            image_array = ensure_rgb(np.array(image_array))
            # image_array = image_array / 255.0
            images.append(image_array)
        except Exception as e:
            print(f"Error loading image {file_path}: {e}")

    # 1.2 Annotations
    annotation_path = os.path.join(directory, 'annotations')
    annotation_files = sorted(
        [f for f in os.listdir(annotation_path) if f.lower().endswith('.xml')],
        key=extract_index
    )
    annotations_dict_list = []

    for filename in annotation_files:
        file_path = os.path.join(annotation_path, filename)
        tree = ET.parse(file_path)
        root = tree.getroot()

        # 1.2.1 Extract filename
        image_name = root.find('filename').text

        # 1.2.2 Extract image size
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)
        # depth = int(size.find('depth').text)

        # 1.2.3 Extract bounding boxes
        bboxes = []
        for obj in root.findall('object'):
            bndbox = obj.find('bndbox')
            bbox = {
                'xmin': int(bndbox.find('xmin').text),
                'ymin': int(bndbox.find('ymin').text),
                'xmax': int(bndbox.find('xmax').text),
                'ymax': int(bndbox.find('ymax').text)
            }
            bboxes.append(bbox)

        # 1.2.4 Add to annotations_dict_list
        annotations_dict_list.append({
            'filename': image_name,
            'image dimension': {'width': width, 'height': height},
            'bboxes': bboxes
        })

    return images, annotations_dict_list

def get_proposals_for_images(images, image_name_prefix='potholes', image_extension ='png'):
    """
    Assumption: the images are ordered ascendingly by file name in the list
    """
    selective_search_proposals = []
    edge_box_proposals = []
    
    for i in range(len(images)):
        print(f'Generating proposal for image {i} ...')
        selective_search_proposal = get_selective_search_proposals(images[i])
        selective_search_proposals.append((f'{image_name_prefix}{i}.{image_extension}', selective_search_proposal))

        edge_box_proposal = get_edge_box_proposal(images[i])
        edge_box_proposals.append((f'{image_name_prefix}{i}.{image_extension}', edge_box_proposal))

    return selective_search_proposals, edge_box_proposals

def get_selective_search_proposals(image,max_boxes=1000, min_size=None):
    """
    Assumption: the images are numpy arrays

    min_size: min size of bounding boxes e.g: (200,200)
    max_boxes: e.g: 1000

    Returns a list of bounding box proposals in the format of (x_min, y_min, x_max, y_max)
    """
    segments = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    segments.setBaseImage(image)
    segments.switchToSelectiveSearchFast()

    rects = segments.process()

    proposals = []
    for (x, y, w, h) in rects:
        # Filter based on minimum size
        if min_size is None or (min_size is not None and w >= min_size[0] and h >= min_size[1]):
            proposals.append({
                "x_min": x,
                "y_min": y,
                "x_max": x + w,
                "y_max": y + h
            })

        # Break if we've reached the maximum number of boxes
        if len(proposals) >= max_boxes:
            break

    return proposals

def get_edge_box_proposal(image, model_path = 'model.yml.gz',max_boxes=1000, min_size=None):  
    """
    Assumption: the images are numpy arrays
    Edge detector: Canny

    min_size: min size of bounding boxes e.g: (200,200)

    Returns a list of bounding box proposals in the format of (x_min, y_min, x_max, y_max)
    """
    # 1. Load struture edge detection model
    edge_detection = cv2.ximgproc.createStructuredEdgeDetection(model_path)

    # 2. Detect edges and orientation map
    edges = edge_detection.detectEdges(np.float32(image) / 255.0) 
    orientation_map = edge_detection.computeOrientation(edges)
    edges = edge_detection.edgesNms(edges, orientation_map) # Enhance edges

    # 3. Generate edge boxes
    edge_boxes = cv2.ximgproc.createEdgeBoxes()
    edge_boxes.setMaxBoxes(max_boxes)

    # 4. Get bounding boxes using the edges
    boxes, _ = edge_boxes.getBoundingBoxes(edges, orientation_map)

    proposals = []
    for box in boxes:
        x_min, y_min, width, height = box

        # Filter based on minimum size
        if min_size is None or (min_size is not None and width >= min_size[0] and height >= min_size[1]):
            proposals.append({
                "x_min": x_min,
                "y_min": y_min,
                "x_max": x_min + width,
                "y_max": y_min + height
            })

    return proposals

def convert_numpy_types(obj):
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    else:
        return obj

def export_proposals_to_json(proposals, output_json_file):
    # 1. Convert the list of tuples to a dictionary
    proposals_dict = {identifier: convert_numpy_types(proposal) for identifier, proposal in proposals}

    # 2. Write the dictionary to a JSON file
    with open(output_json_file, 'w') as json_file:
        json.dump(proposals_dict, json_file, indent=4)

def load_proposals_from_json(json_file):
    """Load proposals from JSON file."""
    with open(json_file, 'r') as f:
        proposals_dict = json.load(f)
    return proposals_dict

def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        box1: dict with keys 'xmin', 'ymin', 'xmax', 'ymax' or 'x_min', 'y_min', 'x_max', 'y_max'
        box2: dict with keys 'xmin', 'ymin', 'xmax', 'ymax' or 'x_min', 'y_min', 'x_max', 'y_max'
    
    Returns:
        IoU score (float between 0 and 1)
    """
    # Handle different key formats
    x1_min = box1.get('xmin', box1.get('x_min'))
    y1_min = box1.get('ymin', box1.get('y_min'))
    x1_max = box1.get('xmax', box1.get('x_max'))
    y1_max = box1.get('ymax', box1.get('y_max'))
    
    x2_min = box2.get('xmin', box2.get('x_min'))
    y2_min = box2.get('ymin', box2.get('y_min'))
    x2_max = box2.get('xmax', box2.get('x_max'))
    y2_max = box2.get('ymax', box2.get('y_max'))
    
    # Calculate intersection
    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)
    
    inter_width = max(0, inter_xmax - inter_xmin)
    inter_height = max(0, inter_ymax - inter_ymin)
    intersection = inter_width * inter_height
    
    # Calculate union
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union = box1_area + box2_area - intersection
    
    # Avoid division by zero
    if union == 0:
        return 0.0
    
    return intersection / union

class PotholeProposalDataset(Dataset):
    """
    PyTorch Dataset for pothole detection using region proposals.
    
    Args:
        data_root: Path to the dataset root directory
        proposal_type: Either 'selective_search' or 'edge_box'
        proposal_json: Path to the JSON file containing proposals
        image_size: Tuple (height, width) to resize cropped regions to
        iou_threshold: IoU threshold to classify a proposal as pothole (default: 0.7)
        positive_ratio: Target ratio of positive (pothole) samples (default: 0.25)
        transform: Optional torchvision transforms to apply
        seed: Random seed for reproducibility
    """
    
    def __init__(
        self,
        data_root: Path,
        proposal_type: str = 'selective_search',
        proposal_json: Optional[str] = None,
        image_size: Tuple[int, int] = (256, 256),
        iou_threshold: float = 0.7,
        positive_ratio: float = 0.25,
        transform: Optional[Callable] = None,
        seed: int = 42
    ):
        self.data_root = Path(data_root)
        self.proposal_type = proposal_type
        self.image_size = image_size
        self.iou_threshold = iou_threshold
        self.positive_ratio = positive_ratio
        self.transform = transform
        
        random.seed(seed)
        np.random.seed(seed)
        
        # Load proposals
        if proposal_json is None:
            if proposal_type == 'selective_search':
                proposal_json = 'selective_search_proposals.json'
            elif proposal_type == 'edge_box':
                proposal_json = 'edge_box_proposals.json'
            else:
                raise ValueError(f"Unknown proposal_type: {proposal_type}")
        
        self.proposals_dict = load_proposals_from_json(proposal_json)
        
        # Load images and annotations
        self.images, self.annotations = read_data(self.data_root, image_size=None)
        
        # Build dataset samples
        self._build_dataset()
    
    def _build_dataset(self):
        """Build the dataset by pairing proposals with labels."""
        positive_samples = []
        negative_samples = []
        
        for img_idx, (image, annotation) in enumerate(zip(self.images, self.annotations)):
            filename = annotation['filename']
            gt_bboxes = annotation['bboxes']
            
            # Get proposals for this image
            proposals = self.proposals_dict.get(filename, [])
            
            # Scale proposals to match original image dimensions
            orig_width = annotation['image dimension']['width']
            orig_height = annotation['image dimension']['height']
            
            for proposal in proposals:
                # Calculate max IoU with all ground truth boxes
                max_iou = 0.0
                for gt_bbox in gt_bboxes:
                    # Scale proposal to original image dimensions
                    scaled_proposal = {
                        'x_min': int(proposal['x_min'] * orig_width / self.images[img_idx].shape[1]),
                        'y_min': int(proposal['y_min'] * orig_height / self.images[img_idx].shape[0]),
                        'x_max': int(proposal['x_max'] * orig_width / self.images[img_idx].shape[1]),
                        'y_max': int(proposal['y_max'] * orig_height / self.images[img_idx].shape[0])
                    }
                    iou = calculate_iou(scaled_proposal, gt_bbox)
                    max_iou = max(max_iou, iou)
                
                # Classify as positive or negative
                label = 1 if max_iou >= self.iou_threshold else 0
                
                sample = {
                    'image_idx': img_idx,
                    'proposal': proposal,
                    'label': label,
                    'iou': max_iou
                }
                
                if label == 1:
                    positive_samples.append(sample)
                else:
                    negative_samples.append(sample)
        
        # Balance the dataset according to positive_ratio
        num_positive = len(positive_samples)
        num_negative_needed = int(num_positive * (1 - self.positive_ratio) / self.positive_ratio)
        
        # Sample negatives
        if num_negative_needed < len(negative_samples):
            negative_samples = random.sample(negative_samples, num_negative_needed)
        
        # Combine and shuffle
        self.samples = positive_samples + negative_samples
        random.shuffle(self.samples)
        
        print(f"Dataset built: {len(positive_samples)} positive, {len(negative_samples)} negative samples")
        print(f"Actual positive ratio: {len(positive_samples) / len(self.samples):.2%}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_idx = sample['image_idx']
        proposal = sample['proposal']
        label = sample['label']
        
        # Get the full image
        image = self.images[image_idx]
        
        # Crop the proposal region
        x_min = int(proposal['x_min'])
        y_min = int(proposal['y_min'])
        x_max = int(proposal['x_max'])
        y_max = int(proposal['y_max'])
        
        # Ensure valid crop coordinates
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(image.shape[1], x_max)
        y_max = min(image.shape[0], y_max)
        
        cropped_region = image[y_min:y_max, x_min:x_max]
        
        # Convert to PIL Image for resizing
        cropped_pil = Image.fromarray(cropped_region)
        
        # Resize to target size with bilinear interpolation
        resized_region = resize_image(
            cropped_pil, 
            self.image_size, 
            output_type='tensor',
            interpolation=InterpolationMode.BILINEAR,
            antialias=True
        )
        
        # Apply additional transforms if provided
        if self.transform:
            resized_region = self.transform(resized_region)
        
        return resized_region, torch.tensor(label, dtype=torch.long)

# Generating image proposal (commented out to prevent auto-execution)
# Uncomment these lines to generate proposals
# images, annotations_dict_list = read_data(DEFAULT_DATA_ROOTS["pothole"], IMAGE_SIZE)
# selective_search_proposals, edge_box_proposals = get_proposals_for_images(images)
# export_proposals_to_json(selective_search_proposals, 'selective_search_proposals.json')
# export_proposals_to_json(edge_box_proposals, 'edge_box_proposals.json')



