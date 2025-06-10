import cv2 as cv
import numpy as np
from PIL import Image, ImageEnhance
import base64
from io import BytesIO
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def adjust_luminosity(base64_image: str, percentage: float) -> str:
    """
    Adjust the luminosity/brightness of a base64 encoded image.

    Args:
        base64_image (str): Base64 encoded image data (with or without data URL prefix)
        percentage (float): Percentage to adjust luminosity (-100 to 100)
                          Positive values brighten, negative values darken

    Returns:
        str: Base64 encoded modified image
    """
    if not -100 <= percentage <= 100:
        raise ValueError("Percentage must be between -100 and 100")

    # Remove data URL prefix if present
    if "base64," in base64_image:
        base64_image = base64_image.split("base64,")[1]

    # Decode base64 to image
    image_data = base64.b64decode(base64_image)
    pil_image = Image.open(BytesIO(image_data))

    # Convert to OpenCV format
    cv_image = cv.cvtColor(np.array(pil_image), cv.COLOR_RGB2BGR)

    # Method 1: Using PIL ImageEnhance
    enhancer = ImageEnhance.Brightness(pil_image)
    factor = 1.0 + (percentage / 100.0)
    factor = max(0.0, factor)
    pil_image = enhancer.enhance(factor)

    # Method 2: Using OpenCV HSV
    hsv = cv.cvtColor(cv_image, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsv)
    v = v.astype(np.float32)

    if percentage > 0:
        v = cv.add(v, np.ones(v.shape, dtype=np.float32) * (percentage * 2.55))
    else:
        v = cv.subtract(
            v, np.ones(v.shape, dtype=np.float32) * (abs(percentage) * 2.55)
        )

    v = np.clip(v, 0, 255).astype(np.uint8)
    hsv_adjusted = cv.merge([h, s, v])
    cv_image = cv.cvtColor(hsv_adjusted, cv.COLOR_HSV2BGR)

    # Convert back to base64
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


def zoom_region(
    base64_image: str, x: int, y: int, width: int, height: int, zoom_factor: float = 2.0
) -> str:
    """
    Zoom into a specific region of a base64 encoded image.

    Args:
        base64_image (str): Base64 encoded image data
        x (int): X coordinate of the top-left corner of the region
        y (int): Y coordinate of the top-left corner of the region
        width (int): Width of the region to zoom
        height (int): Height of the region to zoom
        zoom_factor (float): Factor by which to zoom (default: 2.0)

    Returns:
        str: Base64 encoded modified image
    """
    if zoom_factor <= 0:
        raise ValueError("Zoom factor must be positive")

    # Remove data URL prefix if present
    if "base64," in base64_image:
        base64_image = base64_image.split("base64,")[1]

    # Decode base64 to image
    image_data = base64.b64decode(base64_image)
    pil_image = Image.open(BytesIO(image_data))
    cv_image = cv.cvtColor(np.array(pil_image), cv.COLOR_RGB2BGR)

    # Get image dimensions
    img_height, img_width = cv_image.shape[:2]

    # Validate coordinates
    if x < 0 or y < 0 or x + width > img_width or y + height > img_height:
        raise ValueError("Zoom region exceeds image boundaries")

    # Extract and zoom region
    roi = cv_image[y : y + height, x : x + width]
    new_width = int(width * zoom_factor)
    new_height = int(height * zoom_factor)

    # Resize with appropriate interpolation
    if zoom_factor > 1.0:
        zoomed_roi = cv.resize(
            roi, (new_width, new_height), interpolation=cv.INTER_CUBIC
        )
    else:
        zoomed_roi = cv.resize(
            roi, (new_width, new_height), interpolation=cv.INTER_AREA
        )

    # Create new image with zoomed region
    if zoom_factor > 1.0:
        crop_x = max(0, (new_width - img_width) // 2)
        crop_y = max(0, (new_height - img_height) // 2)
        end_x = min(new_width, crop_x + img_width)
        end_y = min(new_height, crop_y + img_height)
        cv_image = zoomed_roi[crop_y:end_y, crop_x:end_x]
    else:
        result = np.zeros_like(cv_image)
        start_x = (img_width - new_width) // 2
        start_y = (img_height - new_height) // 2
        result[start_y : start_y + new_height, start_x : start_x + new_width] = (
            zoomed_roi
        )
        cv_image = result

    # Convert back to base64
    pil_image = Image.fromarray(cv.cvtColor(cv_image, cv.COLOR_BGR2RGB))
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


def change_color_scheme(base64_image: str, scheme: str) -> str:
    """
    Apply color scheme transformation to a base64 encoded image.

    Args:
        base64_image (str): Base64 encoded image data
        scheme (str): Color scheme to apply. Options:
                     'grayscale', 'sepia', 'vintage', 'cool', 'warm',
                     'high_contrast', 'negative', 'enhance_red',
                     'enhance_green', 'enhance_blue'

    Returns:
        str: Base64 encoded modified image
    """
    # Remove data URL prefix if present
    if "base64," in base64_image:
        base64_image = base64_image.split("base64,")[1]

    # Decode base64 to image
    image_data = base64.b64decode(base64_image)
    pil_image = Image.open(BytesIO(image_data))
    cv_image = cv.cvtColor(np.array(pil_image), cv.COLOR_RGB2BGR)

    scheme = scheme.lower()

    if scheme == "grayscale":
        cv_image = cv.cvtColor(cv_image, cv.COLOR_BGR2GRAY)
        cv_image = cv.cvtColor(cv_image, cv.COLOR_GRAY2BGR)
        pil_image = pil_image.convert("L").convert("RGB")

    elif scheme == "sepia":
        sepia_filter = np.array(
            [[0.272, 0.534, 0.131], [0.349, 0.686, 0.168], [0.393, 0.769, 0.189]]
        )
        cv_image = cv.transform(cv_image, sepia_filter)
        cv_image = np.clip(cv_image, 0, 255).astype(np.uint8)

    elif scheme == "vintage":
        hsv = cv.cvtColor(cv_image, cv.COLOR_BGR2HSV)
        hsv[:, :, 1] = hsv[:, :, 1] * 0.6
        cv_image = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
        sepia_filter = np.array([[0.8, 0.6, 0.4], [0.9, 0.7, 0.5], [0.4, 0.3, 0.2]])
        cv_image = cv.transform(cv_image, sepia_filter)
        cv_image = np.clip(cv_image, 0, 255).astype(np.uint8)

    elif scheme == "cool":
        b, g, r = cv.split(cv_image)
        b = cv.add(b, 30)
        r = cv.subtract(r, 15)
        cv_image = cv.merge([b, g, r])

    elif scheme == "warm":
        b, g, r = cv.split(cv_image)
        r = cv.add(r, 30)
        g = cv.add(g, 15)
        b = cv.subtract(b, 20)
        cv_image = cv.merge([b, g, r])

    elif scheme == "high_contrast":
        enhancer = ImageEnhance.Contrast(pil_image)
        pil_image = enhancer.enhance(2.0)
        cv_image = cv.cvtColor(np.array(pil_image), cv.COLOR_RGB2BGR)

    elif scheme == "negative":
        cv_image = 255 - cv_image

    elif scheme == "enhance_red":
        b, g, r = cv.split(cv_image)
        r = cv.multiply(r, 1.3)
        cv_image = cv.merge([b, g, r])

    elif scheme == "enhance_green":
        b, g, r = cv.split(cv_image)
        g = cv.multiply(g, 1.3)
        cv_image = cv.merge([b, g, r])

    elif scheme == "enhance_blue":
        b, g, r = cv.split(cv_image)
        b = cv.multiply(b, 1.3)
        cv_image = cv.merge([b, g, r])

    else:
        raise ValueError(f"Unknown color scheme: {scheme}")

    # Convert back to base64
    pil_image = Image.fromarray(cv.cvtColor(cv_image, cv.COLOR_BGR2RGB))
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


def blur_region(base64_image: str, region: str, blur_radius: int = 30) -> str:
    """
    Blur a specific region of a base64 encoded image.

    Args:
        base64_image (str): Base64 encoded image data
        region (str): Region to blur. Options: 'top_left', 'top_right', 'bottom_left', 'bottom_right'
        blur_radius (int): Radius of the Gaussian blur (default: 30)

    Returns:
        str: Base64 encoded modified image
    """
    if region not in ["top_left", "top_right", "bottom_left", "bottom_right"]:
        raise ValueError(
            "Region must be one of: 'top_left', 'top_right', 'bottom_left', 'bottom_right'"
        )

    # Remove data URL prefix if present
    if "base64," in base64_image:
        base64_image = base64_image.split("base64,")[1]

    # Decode base64 to image
    image_data = base64.b64decode(base64_image)
    pil_image = Image.open(BytesIO(image_data))
    cv_image = cv.cvtColor(np.array(pil_image), cv.COLOR_RGB2BGR)

    # Get image dimensions
    height, width = cv_image.shape[:2]
    half_height = height // 2
    half_width = width // 2

    # Define region coordinates
    regions = {
        "top_left": (0, 0, half_width, half_height),
        "top_right": (half_width, 0, width, half_height),
        "bottom_left": (0, half_height, half_width, height),
        "bottom_right": (half_width, half_height, width, height),
    }

    # Get coordinates for the specified region
    x1, y1, x2, y2 = regions[region]

    # Extract the region
    region_to_blur = cv_image[y1:y2, x1:x2]

    # Apply Gaussian blur
    blurred_region = cv.GaussianBlur(
        region_to_blur, (blur_radius * 2 + 1, blur_radius * 2 + 1), 0
    )

    # Replace the region in the original image
    cv_image[y1:y2, x1:x2] = blurred_region

    # Convert back to base64
    pil_image = Image.fromarray(cv.cvtColor(cv_image, cv.COLOR_BGR2RGB))
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


def adjust_contrast(base64_image: str, percentage: float) -> str:
    """
    Adjust the contrast of a base64 encoded image by a percentage.

    Args:
        base64_image (str): Base64 encoded image data (with or without data URL prefix)
        percentage (float): Percentage to adjust contrast (-100 to 100)
                          Positive values increase contrast, negative values decrease contrast

    Returns:
        str: Base64 encoded modified image
    """
    if not -100 <= percentage <= 100:
        raise ValueError("Percentage must be between -100 and 100")

    # Remove data URL prefix if present
    if "base64," in base64_image:
        base64_image = base64_image.split("base64,")[1]

    # Decode base64 to image
    image_data = base64.b64decode(base64_image)
    pil_image = Image.open(BytesIO(image_data))

    # Convert to OpenCV format
    cv_image = cv.cvtColor(np.array(pil_image), cv.COLOR_RGB2BGR)

    # Method 1: Using PIL ImageEnhance
    enhancer = ImageEnhance.Contrast(pil_image)
    factor = 1.0 + (percentage / 100.0)
    factor = max(0.0, factor)  # Ensure factor is not negative
    pil_image = enhancer.enhance(factor)

    # Method 2: Using OpenCV
    # Convert to LAB color space
    lab = cv.cvtColor(cv_image, cv.COLOR_BGR2LAB)
    l, a, b = cv.split(lab)

    # Apply contrast adjustment to L channel
    clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    # Merge channels and convert back to BGR
    lab = cv.merge([l, a, b])
    cv_image = cv.cvtColor(lab, cv.COLOR_LAB2BGR)

    # Convert back to base64
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


def adjust_saturation(base64_image: str, percentage: float) -> str:
    """
    Adjust the saturation of a base64 encoded image by a percentage.

    Args:
        base64_image (str): Base64 encoded image data (with or without data URL prefix)
        percentage (float): Percentage to adjust saturation (-100 to 100)
                          Positive values increase saturation, negative values decrease saturation

    Returns:
        str: Base64 encoded modified image
    """
    if not -100 <= percentage <= 100:
        raise ValueError("Percentage must be between -100 and 100")

    # Remove data URL prefix if present
    if "base64," in base64_image:
        base64_image = base64_image.split("base64,")[1]

    # Decode base64 to image
    image_data = base64.b64decode(base64_image)
    pil_image = Image.open(BytesIO(image_data))

    # Convert to OpenCV format
    cv_image = cv.cvtColor(np.array(pil_image), cv.COLOR_RGB2BGR)

    # Method 1: Using PIL ImageEnhance
    enhancer = ImageEnhance.Color(pil_image)
    factor = 1.0 + (percentage / 100.0)
    factor = max(0.0, factor)  # Ensure factor is not negative
    pil_image = enhancer.enhance(factor)

    # Method 2: Using OpenCV HSV
    hsv = cv.cvtColor(cv_image, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsv)

    # Adjust saturation
    if percentage > 0:
        s = cv.add(s, np.ones(s.shape, dtype=np.uint8) * (percentage * 2.55))
    else:
        s = cv.subtract(s, np.ones(s.shape, dtype=np.uint8) * (abs(percentage) * 2.55))

    s = np.clip(s, 0, 255).astype(np.uint8)
    hsv_adjusted = cv.merge([h, s, v])
    cv_image = cv.cvtColor(hsv_adjusted, cv.COLOR_HSV2BGR)

    # Convert back to base64
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


def display_image(base64_image: str):
    """
    Display an image from base64 encoded data.
    """
    image_data = base64.b64decode(base64_image)
    pil_image = Image.open(BytesIO(image_data))
    pil_image.show()


def add_text(
    base64_image: str,
    text: str,
    position: str = "center",
    font_scale: float = 1.0,
    color: tuple = (0, 0, 0),
    thickness: int = 2,
) -> str:
    """
    Add text to a base64 encoded image at specified position.

    Args:
        base64_image (str): Base64 encoded image data
        text (str): Text to add to the image
        position (str): Position of the text. Options: 'center', 'top_left', 'top_right', 'bottom_left'
        font_scale (float): Scale of the font (default: 1.0)
        color (tuple): BGR color tuple (default: black (0, 0, 0))
                      Common colors:
                      - Black: (0, 0, 0)
                      - White: (255, 255, 255)
                      - Red: (0, 0, 255)
                      - Green: (0, 255, 0)
                      - Blue: (255, 0, 0)
                      - Yellow: (0, 255, 255)
                      - Purple: (255, 0, 255)
                      - Cyan: (255, 255, 0)
        thickness (int): Thickness of the text (default: 2)

    Returns:
        str: Base64 encoded modified image
    """
    if position not in ["center", "top_left", "top_right", "bottom_left"]:
        raise ValueError(
            "Position must be one of: 'center', 'top_left', 'top_right', 'bottom_left'"
        )

    # Remove data URL prefix if present
    if "base64," in base64_image:
        base64_image = base64_image.split("base64,")[1]

    # Decode base64 to image
    image_data = base64.b64decode(base64_image)
    pil_image = Image.open(BytesIO(image_data))
    cv_image = cv.cvtColor(np.array(pil_image), cv.COLOR_RGB2BGR)

    # Get image dimensions
    height, width = cv_image.shape[:2]

    # Get text size
    font = cv.FONT_HERSHEY_SIMPLEX
    (text_width, text_height), baseline = cv.getTextSize(
        text, font, font_scale, thickness
    )

    # Calculate position
    if position == "center":
        x = (width - text_width) // 2
        y = (height + text_height) // 2
    elif position == "top_left":
        x = 10
        y = text_height + 10
    elif position == "top_right":
        x = width - text_width - 10
        y = text_height + 10
    else:  # bottom_left
        x = 10
        y = height - 10

    # Add text to image
    cv.putText(cv_image, text, (x, y), font, font_scale, color, thickness)

    # Convert back to base64
    pil_image = Image.fromarray(cv.cvtColor(cv_image, cv.COLOR_BGR2RGB))
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


# Example usage:
if __name__ == "__main__":
    image_path = r"image_modifications_test\kermit.jpg"

    # get image and convert to base64
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")

    # Test adding text in different positions with different colors
    print("\nTesting text addition to Kermit image...")
    positions = ["center", "top_left", "top_right", "bottom_left"]
    colors = {
        "Black": (0, 0, 0),
        "White": (255, 255, 255),
        "Red": (0, 0, 255),
        "Green": (0, 255, 0),
        "Blue": (255, 0, 0),
        "Yellow": (0, 255, 255),
        "Purple": (255, 0, 255),
        "Cyan": (255, 255, 0),
    }

    for pos in positions:
        print(f"\nAdding text to {pos}...")
        # Test each color for the current position
        for color_name, color_value in colors.items():
            print(f"Color: {color_name}")
            text_image = add_text(
                base64_image,
                f"Kermit ({pos})",
                position=pos,
                font_scale=1.5,
                color=color_value,
            )
            display_image(text_image)
