import cv2
import numpy as np
from PIL import Image, ImageFilter

def create_blurred_background(image, target_size, blur_radius=10):
    """
    Create a blurred background image using COVER mode (fills entire target size, may crop)
    """
    # Convert to PIL if it's a numpy array (OpenCV image)
    if isinstance(image, np.ndarray):
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else:
        pil_image = image

    target_width, target_height = target_size
    
    # Get original dimensions
    orig_width, orig_height = pil_image.size
    
    # COVER MODE: Calculate scaling to fill the entire target size (may crop)
    scale_x = target_width / orig_width
    scale_y = target_height / orig_height
    scale = max(scale_x, scale_y)  # Use MAX for cover mode - ensures full coverage
    
    # Resize image
    new_width = int(orig_width * scale)
    new_height = int(orig_height * scale)
    resized_image = pil_image.resize((new_width, new_height), Image.LANCZOS)
    
    # Center crop to target size (this is what creates the "cover" effect)
    left = (new_width - target_width) // 2
    top = (new_height - target_height) // 2
    right = left + target_width
    bottom = top + target_height
    
    cropped_image = resized_image.crop((left, top, right, bottom))
    
    # Apply blur
    blurred_image = cropped_image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    
    return blurred_image


def fit_image_to_size(image, target_size, maintain_aspect=True):
    """
    Fit image using CONTAIN mode (fits within target size, maintains aspect ratio, may have empty space)
    """
    if isinstance(image, np.ndarray):
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else:
        pil_image = image

    target_width, target_height = target_size
    orig_width, orig_height = pil_image.size
    
    if not maintain_aspect:
        return pil_image.resize(target_size, Image.LANCZOS)
    
    # CONTAIN MODE: Calculate scaling to fit within target size (may leave empty space)
    scale_x = target_width / orig_width
    scale_y = target_height / orig_height
    scale = min(scale_x, scale_y)  # Use MIN for contain mode - ensures it fits within bounds
    
    # Calculate new dimensions
    new_width = int(orig_width * scale)
    new_height = int(orig_height * scale)
    
    # Resize the image
    fitted_image = pil_image.resize((new_width, new_height), Image.LANCZOS)
    
    return fitted_image


def compose_blurred_image(image, target_size, blur_radius=10):
    """
    Create a composite image with:
    - Background: COVER mode (blurred, fills entire screen)
    - Foreground: CONTAIN mode (sharp, maintains aspect ratio, centered)
    """
    target_width, target_height = target_size
    
    # Convert OpenCV image to PIL if needed
    if isinstance(image, np.ndarray):
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else:
        pil_image = image
    
    print(f"Original: {pil_image.size} -> Target: {target_size}")
    
    # Background: COVER mode (fills entire screen, blurred)
    blurred_bg = create_blurred_background(pil_image, target_size, blur_radius)
    print(f"Background (cover + blur): {blurred_bg.size}")
    
    # Foreground: CONTAIN mode (fits within screen, maintains aspect ratio)
    fitted_fg = fit_image_to_size(pil_image, target_size, maintain_aspect=True)
    print(f"Foreground (contain): {fitted_fg.size}")
    
    # Start with the blurred background (covers entire screen)
    composite = blurred_bg.copy()
    
    # Center the fitted foreground on top
    fg_width, fg_height = fitted_fg.size
    fg_x = (target_width - fg_width) // 2
    fg_y = (target_height - fg_height) // 2
    
    # Paste the sharp foreground over the blurred background
    composite.paste(fitted_fg, (fg_x, fg_y))
    print(f"Composite: foreground centered at ({fg_x}, {fg_y})")
    
    return composite

