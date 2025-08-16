import os
import random
from PIL import Image, ImageEnhance
import glob
import yaml 
import numpy as np 

# ==== CONFIGURATION ====
PRODUCTS_DIR = "products"        
BACKGROUNDS_DIR = "backgrounds"  
OUTPUT_DIR = "dataset"
NUM_IMAGES = 5                 
MIN_PRODUCTS_PER_IMAGE = 1      
MAX_PRODUCTS_PER_IMAGE = 5       


CLASSES = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "images"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "labels"), exist_ok=True)

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

def get_tight_bbox_from_alpha(img, threshold=10):
    """
    Get tight bounding box around non-transparent pixels
    Returns (left, top, right, bottom) or None if no visible pixels
    """
    # Convert to numpy array
    img_array = np.array(img)
    
    # Get alpha channel (transparency)
    if img_array.shape[2] == 4:  # RGBA
        alpha = img_array[:, :, 3]
    else:
        # If no alpha channel, assume all pixels are visible
        alpha = np.ones((img_array.shape[0], img_array.shape[1]), dtype=np.uint8) * 255
    
    # Find pixels that are not transparent
    visible_pixels = alpha > threshold
    
    # Find bounding box coordinates
    rows = np.any(visible_pixels, axis=1)
    cols = np.any(visible_pixels, axis=0)
    
    if not np.any(rows) or not np.any(cols):
        return None
    
    top, bottom = np.where(rows)[0][[0, -1]]
    left, right = np.where(cols)[0][[0, -1]]
    
    return (left, top, right + 1, bottom + 1)

def place_product_on_bg(bg, product_img, max_attempts=10):
    """
    Place a product image on background with proper size constraints
    Returns (tight_bbox_x, tight_bbox_y, tight_bbox_width, tight_bbox_height) or None if placement fails
    """
    bg_w, bg_h = bg.size
    
    # Try different scale factors if the first one doesn't fit
    for attempt in range(max_attempts):
        # Use larger scale factors to make products more prominent
        scale_factor = random.uniform(0.2, 0.4)
        
        # Calculate new dimensions
        new_w = int(bg_w * scale_factor)
        aspect_ratio = product_img.width / product_img.height
        new_h = int(new_w / aspect_ratio)
        
        # Check if the product fits in the background
        if new_w <= bg_w and new_h <= bg_h:
            # Resize the product image
            product_img_resized = product_img.resize((new_w, new_h), Image.LANCZOS)
            
            # Get tight bounding box of the resized product before placing
            tight_bbox = get_tight_bbox_from_alpha(product_img_resized)
            if tight_bbox is None:
                continue  # Skip if no visible pixels
            
            rel_left, rel_top, rel_right, rel_bottom = tight_bbox
            tight_w = rel_right - rel_left
            tight_h = rel_bottom - rel_top
            
            # Calculate valid placement range
            max_x = bg_w - new_w
            max_y = bg_h - new_h
            
            # Ensure we have valid ranges
            if max_x >= 0 and max_y >= 0:
                x = random.randint(0, max_x)
                y = random.randint(0, max_y)
                
                # Calculate absolute coordinates of the tight bounding box
                abs_left = x + rel_left
                abs_top = y + rel_top
                
                return (product_img_resized, x, y, abs_left, abs_top, tight_w, tight_h)
    
    # If we couldn't place the product after all attempts
    return None

def convert_to_yolo_format(x, y, w, h, img_w, img_h):
    """Convert bounding box to YOLO format (normalized center x, center y, width, height)"""
    x_center = (x + w / 2) / img_w
    y_center = (y + h / 2) / img_h
    norm_w = w / img_w
    norm_h = h / img_h
    return f"{x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}"

def check_overlap(new_box, existing_boxes, overlap_threshold=0.15):
    """Check if a new bounding box overlaps significantly with existing ones"""
    x1, y1, w1, h1 = new_box
    
    for x2, y2, w2, h2 in existing_boxes:
        # Calculate overlap
        left = max(x1, x2)
        right = min(x1 + w1, x2 + w2)
        top = max(y1, y2)
        bottom = min(y1 + h1, y2 + h2)
        
        if left < right and top < bottom:
            overlap_area = (right - left) * (bottom - top)
            box1_area = w1 * h1
            box2_area = w2 * h2
            
            # If overlap is more than threshold of either box, consider it overlapping
            if overlap_area > overlap_threshold * min(box1_area, box2_area):
                return True
    
    return False

def create_data_yaml():
    """Create YOLO data.yaml configuration file"""
    data = {
        "train": os.path.abspath(os.path.join(OUTPUT_DIR, "images")),
        "val": os.path.abspath(os.path.join(OUTPUT_DIR, "images")),  # Using same images for val (can split later)
        "nc": len(CLASSES),
        "names": CLASSES
    }
    
    yaml_path = os.path.join(OUTPUT_DIR, "data.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)
    print(f"[INFO] data.yaml created at {yaml_path}")

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
    max_image_attempts = NUM_IMAGES * 3  # Allow more attempts to get exactly NUM_IMAGES
    
    for attempt in range(max_image_attempts):
        if successful_images >= NUM_IMAGES:
            break
            
        # Select random background
        bg = random.choice(backgrounds).copy()
        img_w, img_h = bg.size
        
        # Determine number of products for this image (1 to number of classes, max 3)
        max_products = min(MAX_PRODUCTS_PER_IMAGE, len(CLASSES))
        num_products = random.randint(MIN_PRODUCTS_PER_IMAGE, max_products)
        
        # Select unique classes (no duplicates) - THIS ENFORCES THE LIMIT
        selected_classes = random.sample(CLASSES, num_products)
        
        print(f"\n[Attempt {attempt+1}] Trying to place {num_products} products: {selected_classes}")
        
        # Try to place ALL selected products
        placed_products = []  # Store (resized_img, paste_x, paste_y, bbox_x, bbox_y, bbox_w, bbox_h, class_id)
        placed_boxes = []     # Store bounding boxes for overlap checking
        placement_successful = True
        
        for cls_idx, cls in enumerate(selected_classes):
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
                    resized_img, paste_x, paste_y, bbox_x, bbox_y, bbox_w, bbox_h = placement_result
                    
                    # Check for overlap with existing products
                    if not check_overlap((bbox_x, bbox_y, bbox_w, bbox_h), placed_boxes):
                        # Success! Store this placement
                        cls_id = CLASSES.index(cls)
                        placed_products.append((resized_img, paste_x, paste_y, bbox_x, bbox_y, bbox_w, bbox_h, cls_id))
                        placed_boxes.append((bbox_x, bbox_y, bbox_w, bbox_h))
                        product_placed = True
                        print(f"  ✓ Placed class '{cls}' at ({bbox_x}, {bbox_y}) size ({bbox_w}x{bbox_h})")
                        break
            
            if not product_placed:
                print(f"  ✗ Failed to place class '{cls}' after {max_placement_attempts} attempts")
                placement_successful = False
                break  # If we can't place one product, abandon this image
        
        # Only proceed if ALL products were placed successfully
        if placement_successful and len(placed_products) == num_products:
            # Now actually paste all products onto the background
            final_bg = bg.copy()
            annotation_lines = []
            
            for resized_img, paste_x, paste_y, bbox_x, bbox_y, bbox_w, bbox_h, cls_id in placed_products:
                # Paste the product
                final_bg.paste(resized_img, (paste_x, paste_y), resized_img)
                
                # Create YOLO annotation
                yolo_annotation = convert_to_yolo_format(bbox_x, bbox_y, bbox_w, bbox_h, img_w, img_h)
                annotation_lines.append(f"{cls_id} {yolo_annotation}")
            
            # Save the successful image
            img_filename = f"image_{successful_images:04d}.jpg"
            txt_filename = f"image_{successful_images:04d}.txt"
            
            img_path = os.path.join(OUTPUT_DIR, "images", img_filename)
            txt_path = os.path.join(OUTPUT_DIR, "labels", txt_filename)
            
            # Save image
            final_bg.save(img_path, "JPEG", quality=95)
            
            # Save annotations
            with open(txt_path, "w") as f:
                f.write("\n".join(annotation_lines))
            
            successful_images += 1
            print(f"✓ SUCCESS! Generated {img_filename} with exactly {len(placed_products)} products")
            
        else:
            print(f"✗ FAILED - Could not place all {num_products} products")
    
    print(f"\n[INFO] Successfully generated {successful_images} images out of {NUM_IMAGES} requested")
    
    if successful_images > 0:
        # Create data.yaml after all images are generated
        create_data_yaml()
        print(f"[INFO] Dataset created successfully in '{OUTPUT_DIR}'")
        print(f"[INFO] Each image contains EXACTLY 1-{min(MAX_PRODUCTS_PER_IMAGE, len(CLASSES))} unique products")
    else:
        print("[ERROR] No images were generated successfully!")

if __name__ == "__main__":
    main()