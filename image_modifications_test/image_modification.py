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


def display_image(base64_image: str):
    """
    Display an image from base64 encoded data.
    """
    image_data = base64.b64decode(base64_image)
    pil_image = Image.open(BytesIO(image_data))
    pil_image.show()


# Example usage:
if __name__ == "__main__":
    image_path = r"image_modifications_test\kermit.jpg"

    # get image and convert to base64
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")

    # Example with base64 image
    # base64_image = "data:image/jpeg;base64,..."  # Your base64 image here

    # Adjust luminosity
    brightened = adjust_luminosity(base64_image, 20)

    # Apply color scheme
    sepia = change_color_scheme(base64_image, "sepia")

    # Zoom region
    zoomed = zoom_region(base64_image, 100, 100, 200, 200, 2.0)

    # Chain operations (if needed)
    result = change_color_scheme(
        adjust_luminosity(zoom_region(base64_image, 100, 100, 200, 200, 1.5), 15),
        "warm",
    )

    # display the image
    display_image(result)
