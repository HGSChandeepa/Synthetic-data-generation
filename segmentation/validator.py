import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import glob
import yaml
import matplotlib.pyplot as plt
import json
from collections import defaultdict, Counter
import random

# Configuration
DATASET_DIR = "dataset"
IMAGES_DIR = os.path.join(DATASET_DIR, "images")
LABELS_DIR = os.path.join(DATASET_DIR, "labels")
OUTPUT_DIR = "validation_output"
NUM_SAMPLES = 50  # Number of random samples to visualize

# Colors for different classes (RGB format for PIL)
COLORS = [
    (255, 0, 0),    # Red
    (0, 255, 0),    # Green
    (0, 0, 255),    # Blue
    (255, 255, 0),  # Yellow
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Cyan
    (128, 0, 128),  # Purple
    (255, 165, 0),  # Orange
    (128, 128, 128), # Gray
    (0, 128, 0),    # Dark Green
]

class SegmentationValidator:
    def __init__(self, dataset_dir=DATASET_DIR, output_dir=OUTPUT_DIR):
        self.dataset_dir = dataset_dir
        self.images_dir = os.path.join(dataset_dir, "images")
        self.labels_dir = os.path.join(dataset_dir, "labels")
        self.output_dir = output_dir
        self.classes = []
        self.validation_results = {
            'total_images': 0,
            'total_annotations': 0,
            'valid_images': 0,
            'invalid_images': 0,
            'errors': [],
            'warnings': [],
            'class_distribution': defaultdict(int),
            'polygon_stats': {
                'min_points': float('inf'),
                'max_points': 0,
                'avg_points': 0,
                'total_polygons': 0
            },
            'annotation_stats': {
                'min_objects_per_image': float('inf'),
                'max_objects_per_image': 0,
                'avg_objects_per_image': 0
            }
        }
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load class names from data.yaml if available
        self._load_class_names()
    
    def _load_class_names(self):
        """Load class names from data.yaml"""
        yaml_path = os.path.join(self.dataset_dir, "data.yaml")
        if os.path.exists(yaml_path):
            try:
                with open(yaml_path, 'r') as f:
                    data = yaml.safe_load(f)
                    self.classes = data.get('names', [])
                    print(f"[INFO] Loaded {len(self.classes)} classes from data.yaml: {self.classes}")
            except Exception as e:
                print(f"[WARNING] Could not load data.yaml: {e}")
                self.classes = [str(i) for i in range(10)]  # Default fallback
        else:
            print("[WARNING] data.yaml not found, using default class names")
            self.classes = [str(i) for i in range(10)]  # Default fallback
    
    def load_segmentation_annotations(self, label_file):
        """Load YOLO segmentation format annotations from text file"""
        annotations = []
        if not os.path.exists(label_file):
            return annotations
        
        try:
            with open(label_file, 'r') as f:
                lines = f.readlines()
            
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) >= 7:  # At least class_id + 3 points (6 coordinates)
                        try:
                            class_id = int(parts[0])
                            polygon_coords = [float(x) for x in parts[1:]]
                            
                            # Validate that we have even number of coordinates (x,y pairs)
                            if len(polygon_coords) % 2 != 0:
                                self.validation_results['errors'].append(
                                    f"{os.path.basename(label_file)} line {line_num}: Odd number of coordinates"
                                )
                                continue
                            
                            # Validate that we have at least 3 points (6 coordinates)
                            if len(polygon_coords) < 6:
                                self.validation_results['errors'].append(
                                    f"{os.path.basename(label_file)} line {line_num}: Less than 3 points in polygon"
                                )
                                continue
                            
                            # Validate coordinate ranges (should be 0-1)
                            invalid_coords = [coord for coord in polygon_coords if coord < 0 or coord > 1]
                            if invalid_coords:
                                self.validation_results['errors'].append(
                                    f"{os.path.basename(label_file)} line {line_num}: Coordinates out of range [0,1]"
                                )
                                continue
                            
                            annotations.append({
                                'class_id': class_id,
                                'polygon': polygon_coords,
                                'num_points': len(polygon_coords) // 2
                            })
                            
                        except ValueError as e:
                            self.validation_results['errors'].append(
                                f"{os.path.basename(label_file)} line {line_num}: Invalid number format - {e}"
                            )
                    else:
                        self.validation_results['errors'].append(
                            f"{os.path.basename(label_file)} line {line_num}: Insufficient data (need at least 7 values)"
                        )
                        
        except Exception as e:
            self.validation_results['errors'].append(
                f"Error reading {os.path.basename(label_file)}: {e}"
            )
        
        return annotations
    
    def polygon_to_mask(self, polygon_coords, img_width, img_height):
        """Convert normalized polygon coordinates to binary mask"""
        if len(polygon_coords) < 6:  # Need at least 3 points
            return None
        
        # Convert normalized coordinates to pixel coordinates
        pixel_coords = []
        for i in range(0, len(polygon_coords), 2):
            x = int(polygon_coords[i] * img_width)
            y = int(polygon_coords[i + 1] * img_height)
            pixel_coords.append((x, y))
        
        # Create mask using PIL
        mask = Image.new('L', (img_width, img_height), 0)
        ImageDraw.Draw(mask).polygon(pixel_coords, fill=255)
        
        return np.array(mask) > 0
    
    def visualize_segmentation(self, image_path, annotations, output_path):
        """Visualize segmentation annotations on image"""
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            img_width, img_height = image.size
            
            # Create overlay for transparency
            overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)
            
            # Draw each annotation
            for i, ann in enumerate(annotations):
                class_id = ann['class_id']
                polygon_coords = ann['polygon']
                
                # Convert to pixel coordinates
                pixel_coords = []
                for j in range(0, len(polygon_coords), 2):
                    x = int(polygon_coords[j] * img_width)
                    y = int(polygon_coords[j + 1] * img_height)
                    pixel_coords.append((x, y))
                
                # Get color for this class
                color = COLORS[class_id % len(COLORS)]
                
                # Draw filled polygon with transparency
                draw.polygon(pixel_coords, fill=(*color, 100))  # Semi-transparent fill
                
                # Draw polygon outline
                draw.polygon(pixel_coords, outline=(*color, 255), width=3)
                
                # Add class label
                if pixel_coords:
                    # Find centroid for label placement
                    centroid_x = sum(p[0] for p in pixel_coords) // len(pixel_coords)
                    centroid_y = sum(p[1] for p in pixel_coords) // len(pixel_coords)
                    
                    class_name = self.classes[class_id] if class_id < len(self.classes) else f"Class_{class_id}"
                    
                    # Draw text background
                    bbox = draw.textbbox((centroid_x, centroid_y), class_name)
                    draw.rectangle(bbox, fill=(*color, 200))
                    
                    # Draw text
                    draw.text((centroid_x, centroid_y), class_name, fill=(255, 255, 255, 255))
            
            # Combine original image with overlay
            result = Image.alpha_composite(image.convert('RGBA'), overlay)
            result = result.convert('RGB')
            
            # Save result
            result.save(output_path, 'JPEG', quality=95)
            return True
            
        except Exception as e:
            self.validation_results['errors'].append(f"Visualization error for {os.path.basename(image_path)}: {e}")
            return False
    
    def calculate_polygon_area(self, polygon_coords, img_width, img_height):
        """Calculate polygon area using shoelace formula"""
        if len(polygon_coords) < 6:
            return 0
        
        # Convert to pixel coordinates
        points = []
        for i in range(0, len(polygon_coords), 2):
            x = polygon_coords[i] * img_width
            y = polygon_coords[i + 1] * img_height
            points.append((x, y))
        
        # Shoelace formula
        area = 0
        n = len(points)
        for i in range(n):
            j = (i + 1) % n
            area += points[i][0] * points[j][1]
            area -= points[j][0] * points[i][1]
        
        return abs(area) / 2
    
    def validate_dataset(self):
        """Main validation function"""
        print("=" * 60)
        print("SEGMENTATION DATASET VALIDATION")
        print("=" * 60)
        
        # Check directory structure
        if not os.path.exists(self.images_dir):
            print(f"❌ ERROR: Images directory '{self.images_dir}' does not exist!")
            return
        
        if not os.path.exists(self.labels_dir):
            print(f"❌ ERROR: Labels directory '{self.labels_dir}' does not exist!")
            return
        
        # Get all image files
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_files.extend(glob.glob(os.path.join(self.images_dir, ext)))
            image_files.extend(glob.glob(os.path.join(self.images_dir, ext.upper())))
        
        if not image_files:
            print(f"❌ ERROR: No image files found in '{self.images_dir}'!")
            return
        
        print(f"📁 Found {len(image_files)} images in dataset")
        self.validation_results['total_images'] = len(image_files)
        
        # Validate each image and its annotations
        valid_pairs = []
        objects_per_image = []
        
        for image_path in image_files:
            image_name = os.path.basename(image_path)
            label_name = os.path.splitext(image_name)[0] + '.txt'
            label_path = os.path.join(self.labels_dir, label_name)
            
            # Load and validate annotations
            annotations = self.load_segmentation_annotations(label_path)
            
            if not annotations:
                self.validation_results['warnings'].append(f"No annotations found for {image_name}")
                continue
            
            # Validate image dimensions
            try:
                with Image.open(image_path) as img:
                    img_width, img_height = img.size
                    
                    # Check each annotation
                    valid_annotations = []
                    for ann in annotations:
                        class_id = ann['class_id']
                        
                        # Update class distribution
                        self.validation_results['class_distribution'][class_id] += 1
                        
                        # Update polygon statistics
                        num_points = ann['num_points']
                        poly_stats = self.validation_results['polygon_stats']
                        poly_stats['min_points'] = min(poly_stats['min_points'], num_points)
                        poly_stats['max_points'] = max(poly_stats['max_points'], num_points)
                        poly_stats['total_polygons'] += 1
                        
                        # Calculate area
                        area = self.calculate_polygon_area(ann['polygon'], img_width, img_height)
                        if area < 10:  # Very small objects
                            self.validation_results['warnings'].append(
                                f"{image_name}: Very small object (area={area:.1f}px²)"
                            )
                        
                        valid_annotations.append(ann)
                    
                    if valid_annotations:
                        valid_pairs.append((image_path, label_path, valid_annotations))
                        objects_per_image.append(len(valid_annotations))
                        self.validation_results['valid_images'] += 1
                        self.validation_results['total_annotations'] += len(valid_annotations)
                    else:
                        self.validation_results['invalid_images'] += 1
                        
            except Exception as e:
                self.validation_results['errors'].append(f"Error processing {image_name}: {e}")
                self.validation_results['invalid_images'] += 1
        
        # Calculate statistics
        if objects_per_image:
            ann_stats = self.validation_results['annotation_stats']
            ann_stats['min_objects_per_image'] = min(objects_per_image)
            ann_stats['max_objects_per_image'] = max(objects_per_image)
            ann_stats['avg_objects_per_image'] = sum(objects_per_image) / len(objects_per_image)
        
        if self.validation_results['polygon_stats']['total_polygons'] > 0:
            poly_stats = self.validation_results['polygon_stats']
            # Calculate average points per polygon (this would need to be calculated during the loop)
            poly_stats['avg_points'] = poly_stats['max_points']  # Placeholder
        
        # Print validation summary
        self._print_validation_summary()
        
        # Generate visualizations
        if valid_pairs:
            self._generate_visualizations(valid_pairs)
        
        # Save validation report
        self._save_validation_report()
        
        return self.validation_results
    
    def _print_validation_summary(self):
        """Print detailed validation summary"""
        results = self.validation_results
        
        print("\n" + "=" * 60)
        print("VALIDATION SUMMARY")
        print("=" * 60)
        
        print(f"📊 Dataset Statistics:")
        print(f"   Total Images: {results['total_images']}")
        print(f"   Valid Images: {results['valid_images']}")
        print(f"   Invalid Images: {results['invalid_images']}")
        print(f"   Total Annotations: {results['total_annotations']}")
        
        if results['total_annotations'] > 0:
            print(f"\n📈 Annotation Statistics:")
            ann_stats = results['annotation_stats']
            print(f"   Objects per image: {ann_stats['min_objects_per_image']:.1f} - {ann_stats['max_objects_per_image']:.1f} "
                  f"(avg: {ann_stats['avg_objects_per_image']:.1f})")
            
            poly_stats = results['polygon_stats']
            print(f"   Points per polygon: {poly_stats['min_points']} - {poly_stats['max_points']}")
            print(f"   Total polygons: {poly_stats['total_polygons']}")
        
        print(f"\n🎯 Class Distribution:")
        total_objects = sum(results['class_distribution'].values())
        for class_id, count in sorted(results['class_distribution'].items()):
            class_name = self.classes[class_id] if class_id < len(self.classes) else f"Class_{class_id}"
            percentage = (count / total_objects * 100) if total_objects > 0 else 0
            print(f"   {class_name}: {count} objects ({percentage:.1f}%)")
        
        if results['errors']:
            print(f"\n❌ Errors ({len(results['errors'])}):")
            for error in results['errors'][:10]:  # Show first 10 errors
                print(f"   • {error}")
            if len(results['errors']) > 10:
                print(f"   ... and {len(results['errors']) - 10} more errors")
        
        if results['warnings']:
            print(f"\n⚠️  Warnings ({len(results['warnings'])}):")
            for warning in results['warnings'][:10]:  # Show first 10 warnings
                print(f"   • {warning}")
            if len(results['warnings']) > 10:
                print(f"   ... and {len(results['warnings']) - 10} more warnings")
        
        # Overall assessment
        print(f"\n🎯 Overall Assessment:")
        if results['valid_images'] == 0:
            print("   ❌ CRITICAL: No valid images found!")
        elif results['valid_images'] < results['total_images'] * 0.9:
            print("   ⚠️  WARNING: Less than 90% of images are valid")
        elif len(results['errors']) > 0:
            print("   ⚠️  WARNING: Some errors found, but most images are valid")
        else:
            print("   ✅ EXCELLENT: Dataset appears to be well-formed!")
    
    def _generate_visualizations(self, valid_pairs):
        """Generate sample visualizations"""
        print(f"\n🖼️  Generating visualizations...")
        
        # Create visualization subdirectory
        viz_dir = os.path.join(self.output_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        
        # Select random samples
        num_samples = min(NUM_SAMPLES, len(valid_pairs))
        sample_pairs = random.sample(valid_pairs, num_samples)
        
        successful_viz = 0
        for i, (image_path, label_path, annotations) in enumerate(sample_pairs):
            image_name = os.path.basename(image_path)
            output_name = f"viz_{i+1:02d}_{os.path.splitext(image_name)[0]}.jpg"
            output_path = os.path.join(viz_dir, output_name)
            
            if self.visualize_segmentation(image_path, annotations, output_path):
                successful_viz += 1
                if i < 5:  # Show first 5
                    print(f"   ✅ {output_name} ({len(annotations)} objects)")
        
        print(f"   Generated {successful_viz}/{num_samples} visualizations")
        
        # Create summary grid if we have successful visualizations
        if successful_viz > 0:
            self._create_summary_grid(viz_dir)
    
    def _create_summary_grid(self, viz_dir):
        """Create a grid summary of visualizations"""
        viz_files = glob.glob(os.path.join(viz_dir, "viz_*.jpg"))
        if len(viz_files) < 4:
            return
        
        # Select first 12 images for grid
        viz_files = sorted(viz_files)[:12]
        
        # Load images and create grid
        images = []
        for viz_file in viz_files:
            img = cv2.imread(viz_file)
            if img is not None:
                # Resize for grid
                img = cv2.resize(img, (300, 300))
                images.append(img)
        
        if len(images) >= 4:
            # Create grid
            grid_cols = 4
            grid_rows = (len(images) + grid_cols - 1) // grid_cols
            
            grid_img = np.zeros((grid_rows * 300, grid_cols * 300, 3), dtype=np.uint8)
            
            for i, img in enumerate(images):
                row = i // grid_cols
                col = i % grid_cols
                
                y1 = row * 300
                y2 = y1 + 300
                x1 = col * 300
                x2 = x1 + 300
                
                grid_img[y1:y2, x1:x2] = img
            
            # Save grid
            grid_path = os.path.join(self.output_dir, "validation_grid.jpg")
            cv2.imwrite(grid_path, grid_img)
            print(f"   📋 Summary grid saved: validation_grid.jpg")
    
    def _save_validation_report(self):
        """Save detailed validation report to JSON"""
        report_path = os.path.join(self.output_dir, "validation_report.json")
        
        # Convert defaultdict to regular dict for JSON serialization
        report = dict(self.validation_results)
        report['class_distribution'] = dict(report['class_distribution'])
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"   📄 Detailed report saved: validation_report.json")

def main():
    """Main execution function"""
    print("🔍 YOLO Segmentation Dataset Validator")
    print("=" * 60)
    
    validator = SegmentationValidator()
    results = validator.validate_dataset()
    
    print(f"\n✅ Validation complete! Check '{OUTPUT_DIR}' for detailed results.")

if __name__ == "__main__":
    main()