import os
import random
from PIL import Image, ImageEnhance, ImageDraw
import glob
import yaml 
import numpy as np 
import cv2

# ==== CONFIGURATION ====
PRODUCTS_DIR = "../products"  # Directory containing product images organized by class
BACKGROUNDS_DIR = "../backgrounds"  
OUTPUT_DIR = "seg_out"
NUM_IMAGES = 20                
MIN_PRODUCTS_PER_IMAGE = 1      
MAX_PRODUCTS_PER_IMAGE = 5       

CLASSES = ["1", "2", "3", "4", "5"]

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "images"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "labels"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "masks"), exist_ok=True)  # For visualization

def load_images_from_folder(folder):
    """Load all PNG images from a folder and convert to RGBA"""
    image_files = glob.glob(os.path.join(folder, "*.png"))
    if not image_files:
        print(f"Warning: No PNG images found in {folder}")
        return []
    return [Image.open(f).convert("RGBA") for f in image_files]

def apply_random_lighting(img):
    """Apply random brightness and contrast adjustments"""
    # Brightness
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(random.uniform(0.7, 1.3))
    # Contrast
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(random.uniform(0.8, 1.2))
    return img

def get_segmentation_mask(img, threshold=10):
    """
    Extract segmentation mask from RGBA image
    Returns binary mask where True = object pixel, False = transparent
    """
    img_array = np.array(img)
    
    # Get alpha channel (transparency)
    if img_array.shape[2] == 4:  # RGBA
        alpha = img_array[:, :, 3]
    else:
        # If no alpha channel, assume all pixels are visible
        alpha = np.ones((img_array.shape[0], img_array.shape[1]), dtype=np.uint8) * 255
    
    # Create binary mask
    mask = alpha > threshold
    return mask

def mask_to_polygon(mask, simplify_tolerance=2.0):
    """
    Convert binary mask to polygon coordinates
    Returns list of (x, y) coordinates for YOLO segmentation format
    """
    mask_uint8 = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    largest_contour = max(contours, key=cv2.contourArea)
    
    epsilon = simplify_tolerance
    simplified_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    # Convert to list of coordinates
    if len(simplified_contour) < 3:  # Need at least 3 points for a polygon
        return None
    
    polygon_points = []
    for point in simplified_contour:
        x, y = point[0]
        polygon_points.extend([x, y])
    
    return polygon_points

def normalize_polygon(polygon_points, img_width, img_height):
    """
    Normalize polygon coordinates to 0-1 range for YOLO format
    """
    normalized_points = []
    for i in range(0, len(polygon_points), 2):
        x = polygon_points[i] / img_width
        y = polygon_points[i + 1] / img_height
        normalized_points.extend([x, y])
    
    return normalized_points

def place_product_on_bg(bg, product_img, max_attempts=10):
    """
    Place a product image on background with proper size constraints
    Returns placement info or None if placement fails
    """
    bg_w, bg_h = bg.size
    
    # Try different scale factors if the first one doesn't fit
    for attempt in range(max_attempts):
        # Use scale factors to make products appropriate size
        scale_factor = random.uniform(0.35, 0.55)
        
        # Calculate new dimensions
        new_w = int(bg_w * scale_factor)
        aspect_ratio = product_img.width / product_img.height
        new_h = int(new_w / aspect_ratio)
        
        # Check if the product fits in the background
        if new_w <= bg_w and new_h <= bg_h:
            # Resize the product image
            product_img_resized = product_img.resize((new_w, new_h), Image.LANCZOS)
            
            # Get segmentation mask of the resized product
            mask = get_segmentation_mask(product_img_resized)
            if not np.any(mask):
                continue  # Skip if no visible pixels
            
            # Calculate valid placement range
            max_x = bg_w - new_w
            max_y = bg_h - new_h
            
            # Ensure we have valid ranges
            if max_x >= 0 and max_y >= 0:
                x = random.randint(0, max_x)
                y = random.randint(0, max_y)
                
                return (product_img_resized, mask, x, y, new_w, new_h)
    
    # If we couldn't place the product after all attempts
    return None

def check_mask_overlap(new_mask, new_x, new_y, existing_masks, overlap_threshold=0.15):
    """
    Check if a new mask overlaps significantly with existing masks
    """
    for existing_mask, existing_x, existing_y in existing_masks:
        # Calculate overlap region
        left = max(new_x, existing_x)
        right = min(new_x + new_mask.shape[1], existing_x + existing_mask.shape[1])
        top = max(new_y, existing_y)
        bottom = min(new_y + new_mask.shape[0], existing_y + existing_mask.shape[0])
        
        if left < right and top < bottom:
            # Extract overlapping regions
            new_region = new_mask[top - new_y:bottom - new_y, left - new_x:right - new_x]
            existing_region = existing_mask[top - existing_y:bottom - existing_y, left - existing_x:right - existing_x]
            
            # Calculate overlap
            overlap = np.logical_and(new_region, existing_region)
            overlap_area = np.sum(overlap)
            
            new_area = np.sum(new_mask)
            existing_area = np.sum(existing_mask)
            
            # Check if overlap exceeds threshold
            if overlap_area > overlap_threshold * min(new_area, existing_area):
                return True
    
    return False

def create_data_yaml():
    """Create YOLO data.yaml configuration file for segmentation"""
    data = {
        "path": os.path.abspath(OUTPUT_DIR),
        "train": "images",
        "val": "images",  # Using same images for val (can split later)
        "nc": len(CLASSES),
        "names": CLASSES
    }
    
    yaml_path = os.path.join(OUTPUT_DIR, "data.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)
    print(f"[INFO] data.yaml created at {yaml_path}")

def create_visualization_mask(placed_objects, img_width, img_height):
    """
    Create a colored mask image for visualization
    """
    # Create RGB mask image
    mask_img = np.zeros((img_height, img_width, 3), dtype=np.uint8)
    
    # Colors for different classes
    colors = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
    ]
    
    for mask, x, y, class_id in placed_objects:
        color = colors[class_id % len(colors)]
        
        # Apply mask with class color
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if mask[i, j]:
                    mask_y = y + i
                    mask_x = x + j
                    if 0 <= mask_y < img_height and 0 <= mask_x < img_width:
                        mask_img[mask_y, mask_x] = color
    
    return Image.fromarray(mask_img)

def main():
    # Load products and validate
    product_images = {}
    for cls in CLASSES:
        folder_path = os.path.join(PRODUCTS_DIR, cls)
        if not os.path.exists(folder_path):
            print(f"Error: Product folder '{folder_path}' does not exist!")
            return
        
        images = load_images_from_folder(folder_path)
        if not images:
            print(f"Error: No images found in '{folder_path}'!")
            return
        
        product_images[cls] = images
        print(f"[INFO] Loaded {len(images)} images for class '{cls}'")
    
    # Load backgrounds and validate
    if not os.path.exists(BACKGROUNDS_DIR):
        print(f"Error: Background folder '{BACKGROUNDS_DIR}' does not exist!")
        return
    
    background_files = glob.glob(os.path.join(BACKGROUNDS_DIR, "*"))
    background_files = [f for f in background_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not background_files:
        print(f"Error: No background images found in '{BACKGROUNDS_DIR}'!")
        return
    
    backgrounds = []
    for bg_file in background_files:
        try:
            bg_img = Image.open(bg_file).convert("RGB")
            backgrounds.append(bg_img)
        except Exception as e:
            print(f"Warning: Could not load background {bg_file}: {e}")
    
    if not backgrounds:
        print("Error: No valid background images loaded!")
        return
    
    print(f"[INFO] Loaded {len(backgrounds)} background images")
    
    # Generate dataset
    successful_images = 0
    max_image_attempts = NUM_IMAGES * 3
    
    for attempt in range(max_image_attempts):
        if successful_images >= NUM_IMAGES:
            break
            
        # Select random background
        bg = random.choice(backgrounds).copy()
        img_w, img_h = bg.size
        
        # Determine number of products for this image
        max_products = min(MAX_PRODUCTS_PER_IMAGE, len(CLASSES))
        num_products = random.randint(MIN_PRODUCTS_PER_IMAGE, max_products)
        
        # Select unique classes (no duplicates)
        selected_classes = random.sample(CLASSES, num_products)
        
        print(f"\n[Attempt {attempt+1}] Trying to place {num_products} products: {selected_classes}")
        
        # Try to place ALL selected products
        placed_products = []  # Store (resized_img, mask, paste_x, paste_y, class_id)
        placed_masks = []     # Store (mask, x, y) for overlap checking
        placement_successful = True
        
        for cls in selected_classes:
            # Select random product image from class
            original_product_img = random.choice(product_images[cls]).copy()
            
            # Apply random lighting effects
            product_img = apply_random_lighting(original_product_img)
            
            # Try to place this specific product
            product_placed = False
            max_placement_attempts = 100
            
            for placement_attempt in range(max_placement_attempts):
                placement_result = place_product_on_bg(bg.copy(), product_img)
                
                if placement_result is not None:
                    resized_img, mask, paste_x, paste_y, new_w, new_h = placement_result
                    
                    # Check for overlap with existing products
                    if not check_mask_overlap(mask, paste_x, paste_y, placed_masks):
                        # Success! Store this placement
                        cls_id = CLASSES.index(cls)
                        placed_products.append((resized_img, mask, paste_x, paste_y, cls_id))
                        placed_masks.append((mask, paste_x, paste_y))
                        product_placed = True
                        print(f"  ✓ Placed class '{cls}' at ({paste_x}, {paste_y})")
                        break
            
            if not product_placed:
                print(f"  ✗ Failed to place class '{cls}' after {max_placement_attempts} attempts")
                placement_successful = False
                break
        
        # Only proceed if ALL products were placed successfully
        if placement_successful and len(placed_products) == num_products:
            # Now actually paste all products and create annotations
            final_bg = bg.copy()
            annotation_lines = []
            placed_objects_for_viz = []  # For visualization mask
            
            for resized_img, mask, paste_x, paste_y, cls_id in placed_products:
                # Paste the product
                final_bg.paste(resized_img, (paste_x, paste_y), resized_img)
                
                # Convert mask to polygon coordinates
                # Create a full-image mask for this object
                full_mask = np.zeros((img_h, img_w), dtype=bool)
                mask_h, mask_w = mask.shape
                
                # Place the object mask in the full image
                for i in range(mask_h):
                    for j in range(mask_w):
                        if mask[i, j]:
                            full_y = paste_y + i
                            full_x = paste_x + j
                            if 0 <= full_y < img_h and 0 <= full_x < img_w:
                                full_mask[full_y, full_x] = True
                
                # Convert mask to polygon
                polygon_points = mask_to_polygon(full_mask)
                
                if polygon_points is not None:
                    # Normalize coordinates
                    normalized_polygon = normalize_polygon(polygon_points, img_w, img_h)
                    
                    # Create YOLO segmentation annotation
                    polygon_str = ' '.join([f"{coord:.6f}" for coord in normalized_polygon])
                    annotation_lines.append(f"{cls_id} {polygon_str}")
                    
                    # Store for visualization
                    placed_objects_for_viz.append((mask, paste_x, paste_y, cls_id))
            
            # Only save if we have valid annotations
            if annotation_lines:
                # Save the successful image
                img_filename = f"image_{successful_images:04d}.jpg"
                txt_filename = f"image_{successful_images:04d}.txt"
                mask_filename = f"image_{successful_images:04d}_mask.png"
                
                img_path = os.path.join(OUTPUT_DIR, "images", img_filename)
                txt_path = os.path.join(OUTPUT_DIR, "labels", txt_filename)
                mask_path = os.path.join(OUTPUT_DIR, "masks", mask_filename)
                
                # Save image
                final_bg.save(img_path, "JPEG", quality=95)
                
                # Save segmentation annotations
                with open(txt_path, "w") as f:
                    f.write("\n".join(annotation_lines))
                
                # Save visualization mask
                viz_mask = create_visualization_mask(placed_objects_for_viz, img_w, img_h)
                viz_mask.save(mask_path)
                
                successful_images += 1
                print(f"✓ SUCCESS! Generated {img_filename} with {len(placed_products)} segmented products")
            else:
                print(f"✗ FAILED - No valid segmentation annotations created")
        else:
            print(f"✗ FAILED - Could not place all {num_products} products")
    
    print(f"\n[INFO] Successfully generated {successful_images} segmentation images out of {NUM_IMAGES} requested")
    
    if successful_images > 0:
        # Create data.yaml after all images are generated
        create_data_yaml()
        print(f"[INFO] Segmentation dataset created successfully in '{OUTPUT_DIR}'")
        print(f"[INFO] Files created:")
        print(f"  - images/: Original images")
        print(f"  - labels/: YOLO segmentation annotations (polygons)")
        print(f"  - masks/: Colored visualization masks")
        print(f"[INFO] Each image contains EXACTLY 1-{min(MAX_PRODUCTS_PER_IMAGE, len(CLASSES))} unique products")
    else:
        print("[ERROR] No segmentation images were generated successfully!")

if __name__ == "__main__":
    main()