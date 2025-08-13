import os
import random
from PIL import Image, ImageEnhance
import glob
import yaml
import numpy as np
from multiprocessing import Pool
from itertools import repeat

# ==== CONFIGURATION ====
PRODUCTS_DIR = "/content/drive/MyDrive/Retail AI/Products-png"
BACKGROUNDS_DIR = "/content/drive/MyDrive/Retail AI/ScaledBackgrounds"
OUTPUT_DIR = "/content/dataset"
NUM_IMAGES = 2000
MIN_PRODUCTS_PER_IMAGE = 1
MAX_PRODUCTS_PER_IMAGE = 5

CLASSES = ["1", "2", "3", "4", "5"]

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
    img_array = np.array(img)
    if img_array.shape[2] == 4:  # RGBA
        alpha = img_array[:, :, 3]
    else:
        alpha = np.ones((img_array.shape[0], img_array.shape[1]), dtype=np.uint8) * 255
    
    visible_pixels = alpha > threshold
    
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
    Returns (resized_img, paste_x, paste_y, bbox_x, bbox_y, bbox_w, bbox_h) or None
    """
    bg_w, bg_h = bg.size
    
    for _ in range(max_attempts):
        scale_factor = random.uniform(0.35, 0.55)
        
        new_w = int(bg_w * scale_factor)
        aspect_ratio = product_img.width / product_img.height
        new_h = int(new_w / aspect_ratio)
        
        if new_w <= bg_w and new_h <= bg_h:
            product_img_resized = product_img.resize((new_w, new_h), Image.LANCZOS)
            
            tight_bbox = get_tight_bbox_from_alpha(product_img_resized)
            if tight_bbox is None:
                continue
            
            rel_left, rel_top, rel_right, rel_bottom = tight_bbox
            tight_w = rel_right - rel_left
            tight_h = rel_bottom - rel_top
            
            max_x = bg_w - new_w
            max_y = bg_h - new_h
            
            if max_x >= 0 and max_y >= 0:
                x = random.randint(0, max_x)
                y = random.randint(0, max_y)
                
                abs_left = x + rel_left
                abs_top = y + rel_top
                
                return (product_img_resized, x, y, abs_left, abs_top, tight_w, tight_h)
    
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
        left = max(x1, x2)
        right = min(x1 + w1, x2 + w2)
        top = max(y1, y2)
        bottom = min(y1 + h1, y2 + h2)
        
        if left < right and top < bottom:
            overlap_area = (right - left) * (bottom - top)
            box1_area = w1 * h1
            box2_area = h2 * w2
            
            if overlap_area > overlap_threshold * min(box1_area, box2_area):
                return True
    
    return False

def create_data_yaml():
    """Create YOLO data.yaml configuration file"""
    data = {
        "train": os.path.abspath(os.path.join(OUTPUT_DIR, "images")),
        "val": os.path.abspath(os.path.join(OUTPUT_DIR, "images")),
        "nc": len(CLASSES),
        "names": CLASSES
    }
    
    yaml_path = os.path.join(OUTPUT_DIR, "data.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)
    print(f"[INFO] data.yaml created at {yaml_path}")


def generate_single_image(args):
    """
    Worker function to generate a single image and its annotations.
    Returns (final_bg, annotation_lines, index) or None on failure.
    """
    index, all_product_images, all_backgrounds = args

    # Select random background
    bg = random.choice(all_backgrounds).copy()
    img_w, img_h = bg.size
    
    # Determine number of products and select unique classes
    max_products = min(MAX_PRODUCTS_PER_IMAGE, len(CLASSES))
    num_products = random.randint(MIN_PRODUCTS_PER_IMAGE, max_products)
    selected_classes = random.sample(CLASSES, num_products)
    
    placed_products_data = []
    placed_boxes = []
    
    for cls in selected_classes:
        original_product_img = random.choice(all_product_images[cls]).copy()
        product_img = apply_random_lighting(original_product_img)
        
        placement_successful = False
        max_placement_attempts = 100
        
        for _ in range(max_placement_attempts):
            placement_result = place_product_on_bg(bg.copy(), product_img)
            
            if placement_result:
                resized_img, paste_x, paste_y, bbox_x, bbox_y, bbox_w, bbox_h = placement_result
                
                if not check_overlap((bbox_x, bbox_y, bbox_w, bbox_h), placed_boxes):
                    cls_id = CLASSES.index(cls)
                    placed_products_data.append((resized_img, paste_x, paste_y, bbox_x, bbox_y, bbox_w, bbox_h, cls_id))
                    placed_boxes.append((bbox_x, bbox_y, bbox_w, bbox_h))
                    placement_successful = True
                    break
        
        if not placement_successful:
            return None # Fail this image generation if any product can't be placed
    
    # If all products were placed successfully, create the final image and annotations
    final_bg = bg.copy()
    annotation_lines = []
    for resized_img, paste_x, paste_y, bbox_x, bbox_y, bbox_w, bbox_h, cls_id in placed_products_data:
        final_bg.paste(resized_img, (paste_x, paste_y), resized_img)
        yolo_annotation = convert_to_yolo_format(bbox_x, bbox_y, bbox_w, bbox_h, img_w, img_h)
        annotation_lines.append(f"{cls_id} {yolo_annotation}")

    return final_bg, annotation_lines, index


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "images"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "labels"), exist_ok=True)

    # --- Load all data once before starting the parallel processes ---
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

    # --- Use a multiprocessing pool to generate images in parallel ---
    print(f"[INFO] Starting to generate {NUM_IMAGES} images using multiprocessing...")
    with Pool() as pool:
        # Create a list of arguments for the worker function
        args_list = zip(range(NUM_IMAGES), repeat(product_images), repeat(backgrounds))
        
        # Use imap_unordered for better progress reporting
        results = pool.imap_unordered(generate_single_image, args_list)
        
        successful_images = 0
        for result in results:
            if result:
                final_bg, annotation_lines, index = result
                
                img_filename = f"image_{index:04d}.jpg"
                txt_filename = f"image_{index:04d}.txt"
                
                img_path = os.path.join(OUTPUT_DIR, "images", img_filename)
                txt_path = os.path.join(OUTPUT_DIR, "labels", txt_filename)
                
                final_bg.save(img_path, "JPEG", quality=95)
                
                with open(txt_path, "w") as f:
                    f.write("\n".join(annotation_lines))
                
                successful_images += 1
                # print(f"✓ Generated {img_filename} with {len(annotation_lines)} products. ({successful_images}/{NUM_IMAGES})")
    
    print(f"\n[INFO] Successfully generated {successful_images} images out of {NUM_IMAGES} requested")
    
    if successful_images > 0:
        create_data_yaml()
        print(f"[INFO] Dataset created successfully in '{OUTPUT_DIR}'")
    else:
        print("[ERROR] No images were generated successfully!")

if __name__ == "__main__":
    main()
