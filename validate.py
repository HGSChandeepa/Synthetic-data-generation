import os
import random
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import glob

# Configuration
DATASET_DIR = "dataset"
IMAGES_DIR = os.path.join(DATASET_DIR, "images")
LABELS_DIR = os.path.join(DATASET_DIR, "labels")
OUTPUT_DIR = "visualization"
NUM_SAMPLES = 100

CLASSES = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]

# Colors for different classes (BGR format for OpenCV)
COLORS = [
    (255, 0, 0),    # Red for class 0
    (0, 255, 0),    # Green for class 1
    (0, 0, 255),    # Blue for class 2
    (255, 255, 0),  # Cyan for additional classes
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Yellow
]

def load_yolo_annotations(label_file):
    """Load YOLO format annotations from text file"""
    annotations = []
    if not os.path.exists(label_file):
        return annotations
    
    with open(label_file, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        if line:
            parts = line.split()
            if len(parts) == 5:
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                annotations.append({
                    'class_id': class_id,
                    'x_center': x_center,
                    'y_center': y_center,
                    'width': width,
                    'height': height
                })
    return annotations

def yolo_to_bbox(annotation, img_width, img_height):
    """Convert YOLO format to bounding box coordinates"""
    x_center = annotation['x_center'] * img_width
    y_center = annotation['y_center'] * img_height
    width = annotation['width'] * img_width
    height = annotation['height'] * img_height
    
    x1 = int(x_center - width / 2)
    y1 = int(y_center - height / 2)
    x2 = int(x_center + width / 2)
    y2 = int(y_center + height / 2)
    
    return x1, y1, x2, y2

def visualize_annotations(image_path, label_path, output_path):
    """Visualize YOLO annotations on image"""
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return False
    
    img_height, img_width = image.shape[:2]
    
    # Load annotations
    annotations = load_yolo_annotations(label_path)
    
    if not annotations:
        print(f"Warning: No annotations found for {os.path.basename(image_path)}")
        # Save image without annotations
        cv2.imwrite(output_path, image)
        return True
    
    print(f"Found {len(annotations)} annotations for {os.path.basename(image_path)}")
    
    # Draw bounding boxes and labels
    for i, ann in enumerate(annotations):
        class_id = ann['class_id']
        x1, y1, x2, y2 = yolo_to_bbox(ann, img_width, img_height)
        
        # Get color for this class
        color = COLORS[class_id % len(COLORS)]
        
        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # Prepare label text
        class_name = CLASSES[class_id] if class_id < len(CLASSES) else f"Class_{class_id}"
        label_text = f"{class_name}"
        
        # Draw label background
        label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(image, 
                     (x1, y1 - label_size[1] - 10), 
                     (x1 + label_size[0], y1), 
                     color, -1)
        
        # Draw label text
        cv2.putText(image, label_text, 
                   (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                   (255, 255, 255), 2)
        
        # Print annotation details
        print(f"  Box {i+1}: Class={class_name}, "
              f"Center=({ann['x_center']:.3f}, {ann['y_center']:.3f}), "
              f"Size=({ann['width']:.3f}, {ann['height']:.3f}), "
              f"Pixel coords=({x1}, {y1}, {x2}, {y2})")
    
    # Save annotated image
    cv2.imwrite(output_path, image)
    return True

def create_summary_image(visualized_images):
    """Create a grid summary of all visualized images"""
    if not visualized_images:
        return
    
    # Load first image to get dimensions
    sample_img = cv2.imread(visualized_images[0])
    if sample_img is None:
        return
    
    img_height, img_width = sample_img.shape[:2]
    
    # Calculate grid dimensions
    grid_cols = min(5, len(visualized_images))
    grid_rows = (len(visualized_images) + grid_cols - 1) // grid_cols
    
    # Resize images for grid (make them smaller)
    target_width = 300
    target_height = int(target_width * img_height / img_width)
    
    # Create summary image
    summary_width = grid_cols * target_width
    summary_height = grid_rows * target_height
    summary_img = np.zeros((summary_height, summary_width, 3), dtype=np.uint8)
    
    for i, img_path in enumerate(visualized_images):
        if i >= grid_rows * grid_cols:
            break
            
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        # Resize image
        img_resized = cv2.resize(img, (target_width, target_height))
        
        # Calculate position in grid
        row = i // grid_cols
        col = i % grid_cols
        
        y1 = row * target_height
        y2 = y1 + target_height
        x1 = col * target_width
        x2 = x1 + target_width
        
        summary_img[y1:y2, x1:x2] = img_resized
    
    # Save summary
    summary_path = os.path.join(OUTPUT_DIR, "summary_grid.jpg")
    cv2.imwrite(summary_path, summary_img)
    print(f"\nSummary grid saved to: {summary_path}")

def main():
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Check if dataset directories exist
    if not os.path.exists(IMAGES_DIR):
        print(f"Error: Images directory '{IMAGES_DIR}' does not exist!")
        return
    
    if not os.path.exists(LABELS_DIR):
        print(f"Error: Labels directory '{LABELS_DIR}' does not exist!")
        return
    
    # Get all image files
    image_files = glob.glob(os.path.join(IMAGES_DIR, "*.jpg"))
    image_files.extend(glob.glob(os.path.join(IMAGES_DIR, "*.png")))
    
    if not image_files:
        print(f"Error: No image files found in '{IMAGES_DIR}'!")
        return
    
    print(f"Found {len(image_files)} images in dataset")
    
    # Select random images
    num_samples = min(NUM_SAMPLES, len(image_files))
    selected_images = random.sample(image_files, num_samples)
    
    print(f"\nVisualizing {num_samples} random images...")
    print("=" * 60)
    
    visualized_images = []
    
    for i, image_path in enumerate(selected_images):
        image_name = os.path.basename(image_path)
        label_name = os.path.splitext(image_name)[0] + '.txt'
        label_path = os.path.join(LABELS_DIR, label_name)
        
        output_name = f"visualized_{i+1:02d}_{image_name}"
        output_path = os.path.join(OUTPUT_DIR, output_name)
        
        print(f"\n[{i+1}/{num_samples}] Processing: {image_name}")
        
        success = visualize_annotations(image_path, label_path, output_path)
        if success:
            visualized_images.append(output_path)
            print(f"Saved visualization: {output_name}")
        else:
            print(f"Failed to process: {image_name}")
    
    print("\n" + "=" * 60)
    print(f"Visualization complete! Check the '{OUTPUT_DIR}' folder")
    print(f"Successfully processed {len(visualized_images)} images")
    
    # Create summary grid
    if visualized_images:
        create_summary_image(visualized_images)
    
    # Print color legend
    print(f"\nColor Legend:")
    for i, class_name in enumerate(CLASSES):
        color = COLORS[i % len(COLORS)]
        print(f"  Class '{class_name}': RGB{color}")

if __name__ == "__main__":
    main()