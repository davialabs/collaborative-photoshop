from PIL import Image, ImageEnhance
import base64
from io import BytesIO
from agents import function_tool
from agents import RunContextWrapper
import cv2 as cv
import numpy as np
import requests
import json

from collaborative_photoshop.model import AgentContext
from collaborative_photoshop.utils import decode_image


@function_tool
def adjust_luminosity_base64(
    ctx: RunContextWrapper[AgentContext], percentage: float
) -> str:
    """
    Adjusts the luminosity of the image by a percentage (-100 to 100).
    DO NOT ASK FOR THE IMAGE, only the percentage is needed. The image is provided as base64 in the context.image_b64 field.
    THIS FUNCTION DOES NOT NEED THE IMAGE, ONLY THE PERCENTAGE IS A PARAMETER.
    The new image is stored in the context.modified_images_b64 field.

    Args:
        percentage (float): The percentage to adjust the luminosity by.

    Returns:
        str: A success message.
    """
    print(f"Adjusting luminosity by {percentage}%")
    current_image_b64 = ctx.context.modified_images_b64[ctx.context.current_image_index]
    # Decode base64 to bytes
    image_data = decode_image(current_image_b64)
    # Open image from bytes
    image = Image.open(BytesIO(image_data)).convert("RGB")
    # Calculate enhancement factor
    factor = 1.0 + (percentage / 100.0)
    factor = max(0.0, factor)  # Prevent negative factor
    # Adjust brightness
    enhancer = ImageEnhance.Brightness(image)
    enhanced_image = enhancer.enhance(factor)
    # Save to buffer
    buffer = BytesIO()
    enhanced_image.save(buffer, format="JPEG")
    buffer.seek(0)
    # Encode back to base64
    modified_b64 = base64.b64encode(buffer.read()).decode("utf-8")
    ctx.context.modified_images_b64.append(modified_b64)
    ctx.context.current_image_index = len(ctx.context.modified_images_b64) - 1
    return f"Luminosity adjusted by {percentage}%."


@function_tool
def change_color_scheme(ctx: RunContextWrapper[AgentContext], scheme: str) -> str:
    """
    Apply color scheme transformation.
    The image is already available in the context and will be automatically processed.
    You don't need to ask for the image, it is already in the context.
    Just specify the color scheme to apply.
    THIS FUNCTION DOES NOT NEED THE IMAGE, ONLY THE SCHEME IS A PARAMETER.


    Args:
        scheme (str): Color scheme to apply. Options:
                     'grayscale', 'sepia', 'vintage', 'cool', 'warm',
                     'high_contrast', 'negative', 'enhance_red',
                     'enhance_green', 'enhance_blue'

    Returns:
        str: Confirmation message with the applied color scheme
    """
    print(f"Starting color scheme application with scheme: {scheme}")
    current_image_b64 = ctx.context.modified_images_b64[ctx.context.current_image_index]

    # Remove data URL prefix if present
    if "base64," in current_image_b64:
        print("Found base64 prefix, removing it")
        base64_image = current_image_b64.split("base64,")[1]
    else:
        print("No base64 prefix found, using image as is")
        base64_image = current_image_b64

    try:
        # Decode base64 to image
        print("Decoding base64 image")
        image_data = base64.b64decode(base64_image)
        print(f"Decoded image size: {len(image_data)} bytes")

        pil_image = Image.open(BytesIO(image_data))
        print(f"PIL Image size: {pil_image.size}, mode: {pil_image.mode}")

        cv_image = cv.cvtColor(np.array(pil_image), cv.COLOR_RGB2BGR)
        print(f"OpenCV image shape: {cv_image.shape}")

        scheme = scheme.lower()
        print(f"Processing scheme: {scheme}")

        if scheme == "grayscale":
            print("Applying grayscale transformation")
            cv_image = cv.cvtColor(cv_image, cv.COLOR_BGR2GRAY)
            cv_image = cv.cvtColor(cv_image, cv.COLOR_GRAY2BGR)
            pil_image = pil_image.convert("L").convert("RGB")

        elif scheme == "sepia":
            print("Applying sepia transformation")
            sepia_filter = np.array(
                [[0.272, 0.534, 0.131], [0.349, 0.686, 0.168], [0.393, 0.769, 0.189]]
            )
            cv_image = cv.transform(cv_image, sepia_filter)
            cv_image = np.clip(cv_image, 0, 255).astype(np.uint8)

        elif scheme == "vintage":
            print("Applying vintage transformation")
            hsv = cv.cvtColor(cv_image, cv.COLOR_BGR2HSV)
            hsv[:, :, 1] = hsv[:, :, 1] * 0.6
            cv_image = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
            sepia_filter = np.array([[0.8, 0.6, 0.4], [0.9, 0.7, 0.5], [0.4, 0.3, 0.2]])
            cv_image = cv.transform(cv_image, sepia_filter)
            cv_image = np.clip(cv_image, 0, 255).astype(np.uint8)

        elif scheme == "cool":
            print("Applying cool transformation")
            b, g, r = cv.split(cv_image)
            b = cv.add(b, 30)
            r = cv.subtract(r, 15)
            cv_image = cv.merge([b, g, r])

        elif scheme == "warm":
            print("Applying warm transformation")
            b, g, r = cv.split(cv_image)
            r = cv.add(r, 30)
            g = cv.add(g, 15)
            b = cv.subtract(b, 20)
            cv_image = cv.merge([b, g, r])

        elif scheme == "high_contrast":
            print("Applying high contrast transformation")
            enhancer = ImageEnhance.Contrast(pil_image)
            pil_image = enhancer.enhance(2.0)
            cv_image = cv.cvtColor(np.array(pil_image), cv.COLOR_RGB2BGR)

        elif scheme == "negative":
            print("Applying negative transformation")
            cv_image = 255 - cv_image

        elif scheme == "enhance_red":
            print("Applying red enhancement")
            b, g, r = cv.split(cv_image)
            r = cv.multiply(r, 1.3)
            cv_image = cv.merge([b, g, r])

        elif scheme == "enhance_green":
            print("Applying green enhancement")
            b, g, r = cv.split(cv_image)
            g = cv.multiply(g, 1.3)
            cv_image = cv.merge([b, g, r])

        elif scheme == "enhance_blue":
            print("Applying blue enhancement")
            b, g, r = cv.split(cv_image)
            b = cv.multiply(b, 1.3)
            cv_image = cv.merge([b, g, r])

        else:
            raise ValueError(f"Unknown color scheme: {scheme}")

        print("Converting back to base64")
        # Convert back to base64
        pil_image = Image.fromarray(cv.cvtColor(cv_image, cv.COLOR_BGR2RGB))
        buffered = BytesIO()
        pil_image.save(buffered, format="PNG")

        modified_b64 = base64.b64encode(buffered.getvalue()).decode()
        ctx.context.modified_images_b64.append(modified_b64)
        ctx.context.current_image_index = len(ctx.context.modified_images_b64) - 1
        print(f"Final base64 length: {len(modified_b64)}")
        print(f"Color scheme {scheme} applied to image.")
        return f"Color scheme {scheme} applied to image."

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise


@function_tool
def next_image(ctx: RunContextWrapper[AgentContext]) -> str:
    """
    Navigate to the next image in the modified_images_b64 list.

    Returns:
        str: Confirmation message with the new current image index, or a message if already at the last image.
    """
    if ctx.context.current_image_index < len(ctx.context.modified_images_b64) - 1:
        ctx.context.current_image_index += 1
        return f"Moved to next image at index {ctx.context.current_image_index}."
    else:
        return "Already at the last image."


@function_tool
def previous_image(ctx: RunContextWrapper[AgentContext]) -> str:
    """
    Navigate to the previous image in the modified_images_b64 list.

    Returns:
        str: Confirmation message with the new current image index, or a message if already at the first image.
    """
    if ctx.context.current_image_index > 0:
        ctx.context.current_image_index -= 1
        return f"Moved to previous image at index {ctx.context.current_image_index}."
    else:
        return "Already at the first image."


@function_tool
def go_to_image_index(ctx: RunContextWrapper[AgentContext], index: int) -> str:
    """
    Navigate to a specific image in the modified_images_b64 list by index.
    """
    print(f"Navigating to image at index {index}")
    if 0 <= index < len(ctx.context.modified_images_b64):
        ctx.context.current_image_index = index
        return f"Navigated to image at index {index}."
    else:
        return f"Invalid index: {index}. There are {len(ctx.context.modified_images_b64)} images."


@function_tool
def adjust_contrast(ctx: RunContextWrapper[AgentContext], percentage: float) -> str:
    """
    Adjust the contrast of a base64 encoded image by a percentage.
    THIS FUNCTION DOES NOT NEED THE IMAGE, ONLY THE PERCENTAGE IS A PARAMETER.

    Args:
        base64_image (str): Base64 encoded image data (with or without data URL prefix)
        percentage (float): Percentage to adjust contrast (-100 to 100)
                          Positive values increase contrast, negative values decrease contrast

    Returns:
        str: Base64 encoded modified image
    """
    current_image_b64 = ctx.context.modified_images_b64[ctx.context.current_image_index]

    if not -100 <= percentage <= 100:
        raise ValueError("Percentage must be between -100 and 100")

    # Decode base64 to image
    image_data = decode_image(current_image_b64)
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
    modified_b64 = base64.b64encode(buffered.getvalue()).decode()
    ctx.context.modified_images_b64.append(modified_b64)
    ctx.context.current_image_index = len(ctx.context.modified_images_b64) - 1
    return f"Contrast adjusted by {percentage}%."


@function_tool
def remove_background(ctx: RunContextWrapper[AgentContext], iterations: int = 5) -> str:
    """
    Remove the background from an image using OpenCV's GrabCut algorithm.

    Args:
        base64_image (str): Base64 encoded image data
        iterations (int): Number of iterations for grabcut (default: 5)

    Returns:
        str: Base64 encoded image with transparent background
    """
    current_image_b64 = ctx.context.modified_images_b64[ctx.context.current_image_index]

    # Decode base64 to image
    image_data = decode_image(current_image_b64)
    pil_image = Image.open(BytesIO(image_data))
    cv_image = cv.cvtColor(np.array(pil_image), cv.COLOR_RGB2BGR)

    # Create a mask
    mask = np.zeros(cv_image.shape[:2], np.uint8)

    # Create temporary arrays for grabcut
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    # Define rectangle for grabcut (assuming object is in center)
    height, width = cv_image.shape[:2]
    rect = (width // 8, height // 8, width * 3 // 4, height * 3 // 4)

    # Apply grabcut
    cv.grabCut(
        cv_image, mask, rect, bgd_model, fgd_model, iterations, cv.GC_INIT_WITH_RECT
    )

    # Create mask where sure and likely fg pixels are marked
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")

    # Create alpha channel
    alpha = mask2 * 255

    # Convert to RGBA
    rgba = cv.cvtColor(cv_image, cv.COLOR_BGR2BGRA)
    rgba[:, :, 3] = alpha

    # Convert back to base64
    pil_image = Image.fromarray(cv.cvtColor(rgba, cv.COLOR_BGRA2RGBA))
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")
    modified_b64 = base64.b64encode(buffered.getvalue()).decode()
    ctx.context.modified_images_b64.append(modified_b64)
    ctx.context.current_image_index = len(ctx.context.modified_images_b64) - 1
    return f"Background removed from image."


@function_tool
def apply_artistic_style(ctx: RunContextWrapper[AgentContext], style: str) -> str:
    """
    Apply an artistic style to the current image using AI-powered style transfer.
    Available styles: 'van_gogh', 'picasso', 'monet', 'kandinsky', 'dali', 'warhol',
    'hokusai', 'munch', 'pollock', 'matisse', 'cezanne', 'gauguin', 'seurat', 'turner'

    Args:
        style (str): The artistic style to apply to the image.

    Returns:
        str: Confirmation message with the applied style.
    """
    print(f"Applying artistic style: {style}")
    current_image_b64 = ctx.context.modified_images_b64[ctx.context.current_image_index]

    # Decode base64 to image
    image_data = decode_image(current_image_b64)
    pil_image = Image.open(BytesIO(image_data))

    # Convert to OpenCV format
    cv_image = cv.cvtColor(np.array(pil_image), cv.COLOR_RGB2BGR)

    # Apply different artistic styles using OpenCV and PIL
    style = style.lower().replace(" ", "_")

    if style == "van_gogh":
        # Van Gogh style: bold brushstrokes, vibrant colors, swirling patterns
        cv_image = _apply_van_gogh_style(cv_image)
    elif style == "picasso":
        # Picasso style: cubist geometric shapes, bold lines
        cv_image = _apply_picasso_style(cv_image)
    elif style == "monet":
        # Monet style: impressionist, soft brushstrokes, light effects
        cv_image = _apply_monet_style(cv_image)
    elif style == "kandinsky":
        # Kandinsky style: abstract geometric shapes, bold colors
        cv_image = _apply_kandinsky_style(cv_image)
    elif style == "dali":
        # Dali style: surreal, melting effects, dreamlike
        cv_image = _apply_dali_style(cv_image)
    elif style == "warhol":
        # Warhol style: pop art, high contrast, bold colors
        cv_image = _apply_warhol_style(cv_image)
    elif style == "hokusai":
        # Hokusai style: Japanese woodblock, bold outlines, flat colors
        cv_image = _apply_hokusai_style(cv_image)
    elif style == "munch":
        # Munch style: expressionist, emotional, bold brushstrokes
        cv_image = _apply_munch_style(cv_image)
    elif style == "pollock":
        # Pollock style: abstract expressionist, splattered paint
        cv_image = _apply_pollock_style(cv_image)
    elif style == "matisse":
        # Matisse style: fauvist, bold colors, simplified forms
        cv_image = _apply_matisse_style(cv_image)
    elif style == "cezanne":
        # Cezanne style: post-impressionist, geometric brushstrokes
        cv_image = _apply_cezanne_style(cv_image)
    elif style == "gauguin":
        # Gauguin style: post-impressionist, tropical colors, flat areas
        cv_image = _apply_gauguin_style(cv_image)
    elif style == "seurat":
        # Seurat style: pointillist, small dots of color
        cv_image = _apply_seurat_style(cv_image)
    elif style == "turner":
        # Turner style: romantic, atmospheric, light effects
        cv_image = _apply_turner_style(cv_image)
    else:
        raise ValueError(
            f"Unknown artistic style: {style}. Available styles: van_gogh, picasso, monet, kandinsky, dali, warhol, hokusai, munch, pollock, matisse, cezanne, gauguin, seurat, turner"
        )

    # Convert back to base64
    pil_image = Image.fromarray(cv.cvtColor(cv_image, cv.COLOR_BGR2RGB))
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")
    modified_b64 = base64.b64encode(buffered.getvalue()).decode()

    ctx.context.modified_images_b64.append(modified_b64)
    ctx.context.current_image_index = len(ctx.context.modified_images_b64) - 1
    return f"Applied {style} artistic style to the image."


def _apply_van_gogh_style(cv_image):
    """Apply Van Gogh style: bold brushstrokes, vibrant colors, swirling patterns"""
    # Enhance colors and add swirling brushstroke effect
    hsv = cv.cvtColor(cv_image, cv.COLOR_BGR2HSV)
    hsv[:, :, 1] = hsv[:, :, 1] * 1.3  # Increase saturation
    hsv[:, :, 2] = hsv[:, :, 2] * 1.1  # Increase brightness
    cv_image = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

    # Add swirling effect
    height, width = cv_image.shape[:2]
    y, x = np.ogrid[:height, :width]
    center_x, center_y = width // 2, height // 2
    distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    angle = np.arctan2(y - center_y, x - center_x)

    # Create swirling distortion
    swirl_strength = 0.1
    new_angle = angle + swirl_strength * distance
    new_x = center_x + distance * np.cos(new_angle)
    new_y = center_y + distance * np.sin(new_angle)

    # Apply the transformation
    map_x = new_x.astype(np.float32)
    map_y = new_y.astype(np.float32)
    cv_image = cv.remap(cv_image, map_x, map_y, cv.INTER_LINEAR)

    # Add brushstroke texture
    kernel = np.ones((3, 3), np.float32) / 9
    cv_image = cv.filter2D(cv_image, -1, kernel)

    return cv_image


def _apply_picasso_style(cv_image):
    """Apply Picasso style: cubist geometric shapes, bold lines"""
    # Convert to grayscale for edge detection
    gray = cv.cvtColor(cv_image, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, 50, 150)

    # Create geometric shapes
    height, width = cv_image.shape[:2]
    mask = np.zeros_like(edges)

    # Add geometric divisions
    cv.line(mask, (width // 3, 0), (width // 3, height), 255, 3)
    cv.line(mask, (2 * width // 3, 0), (2 * width // 3, height), 255, 3)
    cv.line(mask, (0, height // 3), (width, height // 3), 255, 3)
    cv.line(mask, (0, 2 * height // 3), (width, 2 * height // 3), 255, 3)

    # Combine edges with geometric lines
    combined_edges = cv.add(edges, mask)

    # Apply posterization for flat colors
    cv_image = cv.posterize(cv_image, 4)

    # Add bold outlines
    cv_image = cv.addWeighted(
        cv_image, 0.8, cv.cvtColor(combined_edges, cv.COLOR_GRAY2BGR), 0.2, 0
    )

    return cv_image


def _apply_monet_style(cv_image):
    """Apply Monet style: impressionist, soft brushstrokes, light effects"""
    # Soften the image
    kernel = np.ones((5, 5), np.float32) / 25
    cv_image = cv.filter2D(cv_image, -1, kernel)

    # Enhance light and create soft, dreamy effect
    hsv = cv.cvtColor(cv_image, cv.COLOR_BGR2HSV)
    hsv[:, :, 2] = hsv[:, :, 2] * 1.2  # Increase brightness
    cv_image = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

    # Add soft brushstroke texture
    noise = np.random.normal(0, 10, cv_image.shape).astype(np.uint8)
    cv_image = cv.add(cv_image, noise)

    # Apply Gaussian blur for softness
    cv_image = cv.GaussianBlur(cv_image, (3, 3), 0)

    return cv_image


def _apply_kandinsky_style(cv_image):
    """Apply Kandinsky style: abstract geometric shapes, bold colors"""
    # Create geometric shapes
    height, width = cv_image.shape[:2]
    mask = np.zeros_like(cv_image)

    # Add circles, triangles, and rectangles
    cv.circle(mask, (width // 4, height // 4), 50, (255, 255, 255), -1)
    cv.circle(mask, (3 * width // 4, 3 * height // 4), 30, (255, 255, 255), -1)

    # Add triangles
    triangle = np.array(
        [
            [width // 2, height // 4],
            [width // 2 - 30, height // 2],
            [width // 2 + 30, height // 2],
        ],
        np.int32,
    )
    cv.fillPoly(mask, [triangle], (255, 255, 255))

    # Combine with original image
    cv_image = cv.addWeighted(cv_image, 0.7, mask, 0.3, 0)

    # Enhance colors
    hsv = cv.cvtColor(cv_image, cv.COLOR_BGR2HSV)
    hsv[:, :, 1] = hsv[:, :, 1] * 1.5  # Increase saturation
    cv_image = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

    return cv_image


def _apply_dali_style(cv_image):
    """Apply Dali style: surreal, melting effects, dreamlike"""
    # Create melting effect
    height, width = cv_image.shape[:2]
    y, x = np.ogrid[:height, :width]

    # Create vertical melting distortion
    melting_factor = 0.05
    new_x = x + np.sin(y * melting_factor) * 20
    new_y = y

    map_x = new_x.astype(np.float32)
    map_y = new_y.astype(np.float32)
    cv_image = cv.remap(cv_image, map_x, map_y, cv.INTER_LINEAR)

    # Add dreamlike colors
    hsv = cv.cvtColor(cv_image, cv.COLOR_BGR2HSV)
    hsv[:, :, 0] = (hsv[:, :, 0] + 30) % 180  # Shift hue
    hsv[:, :, 1] = hsv[:, :, 1] * 1.3  # Increase saturation
    cv_image = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

    return cv_image


def _apply_warhol_style(cv_image):
    """Apply Warhol style: pop art, high contrast, bold colors"""
    # Convert to grayscale
    gray = cv.cvtColor(cv_image, cv.COLOR_BGR2GRAY)

    # Create high contrast
    gray = cv.equalizeHist(gray)

    # Posterize to reduce colors
    gray = cv.posterize(gray, 3)

    # Apply different colors to different regions
    height, width = gray.shape[:2]
    result = np.zeros_like(cv_image)

    # Divide into quadrants with different colors
    result[: height // 2, : width // 2] = [0, 0, 255]  # Red
    result[: height // 2, width // 2 :] = [0, 255, 0]  # Green
    result[height // 2 :, : width // 2] = [255, 0, 0]  # Blue
    result[height // 2 :, width // 2 :] = [255, 255, 0]  # Yellow

    # Use grayscale as mask
    mask = gray / 255.0
    mask = np.stack([mask, mask, mask], axis=2)

    cv_image = cv_image * (1 - mask) + result * mask
    cv_image = cv_image.astype(np.uint8)

    return cv_image


def _apply_hokusai_style(cv_image):
    """Apply Hokusai style: Japanese woodblock, bold outlines, flat colors"""
    # Convert to grayscale for edge detection
    gray = cv.cvtColor(cv_image, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, 50, 150)

    # Dilate edges for bold outlines
    kernel = np.ones((3, 3), np.uint8)
    edges = cv.dilate(edges, kernel, iterations=2)

    # Reduce colors (posterize)
    cv_image = cv.posterize(cv_image, 4)

    # Add bold outlines
    cv_image = cv.addWeighted(
        cv_image, 0.8, cv.cvtColor(edges, cv.COLOR_GRAY2BGR), 0.2, 0
    )

    return cv_image


def _apply_munch_style(cv_image):
    """Apply Munch style: expressionist, emotional, bold brushstrokes"""
    # Enhance colors dramatically
    hsv = cv.cvtColor(cv_image, cv.COLOR_BGR2HSV)
    hsv[:, :, 1] = hsv[:, :, 1] * 1.5  # Increase saturation
    hsv[:, :, 2] = hsv[:, :, 2] * 1.2  # Increase brightness
    cv_image = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

    # Add bold brushstroke effect
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    cv_image = cv.filter2D(cv_image, -1, kernel)

    # Add emotional distortion
    height, width = cv_image.shape[:2]
    y, x = np.ogrid[:height, :width]
    center_x, center_y = width // 2, height // 2
    distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

    # Create wave distortion
    wave_factor = 0.1
    new_x = x + np.sin(distance * wave_factor) * 10
    new_y = y + np.cos(distance * wave_factor) * 10

    map_x = new_x.astype(np.float32)
    map_y = new_y.astype(np.float32)
    cv_image = cv.remap(cv_image, map_x, map_y, cv.INTER_LINEAR)

    return cv_image


def _apply_pollock_style(cv_image):
    """Apply Pollock style: abstract expressionist, splattered paint"""
    # Create splatter effect
    height, width = cv_image.shape[:2]
    splatter = np.zeros_like(cv_image)

    # Add random splatters
    for _ in range(100):
        x = np.random.randint(0, width)
        y = np.random.randint(0, height)
        radius = np.random.randint(5, 20)
        color = (
            np.random.randint(0, 255),
            np.random.randint(0, 255),
            np.random.randint(0, 255),
        )
        cv.circle(splatter, (x, y), radius, color, -1)

    # Combine with original image
    cv_image = cv.addWeighted(cv_image, 0.6, splatter, 0.4, 0)

    # Add motion blur effect
    kernel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], np.float32) / 9
    cv_image = cv.filter2D(cv_image, -1, kernel)

    return cv_image


def _apply_matisse_style(cv_image):
    """Apply Matisse style: fauvist, bold colors, simplified forms"""
    # Enhance colors dramatically
    hsv = cv.cvtColor(cv_image, cv.COLOR_BGR2HSV)
    hsv[:, :, 1] = hsv[:, :, 1] * 1.8  # Increase saturation
    cv_image = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

    # Simplify forms (reduce detail)
    cv_image = cv.bilateralFilter(cv_image, 9, 75, 75)

    # Add flat color areas
    cv_image = cv.posterize(cv_image, 5)

    return cv_image


def _apply_cezanne_style(cv_image):
    """Apply Cezanne style: post-impressionist, geometric brushstrokes"""
    # Create geometric brushstroke effect
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    cv_image = cv.filter2D(cv_image, -1, kernel)

    # Add structured brushstrokes
    height, width = cv_image.shape[:2]
    for i in range(0, height, 20):
        for j in range(0, width, 20):
            cv.rectangle(cv_image, (j, i), (j + 15, i + 15), (0, 0, 0), 1)

    return cv_image


def _apply_gauguin_style(cv_image):
    """Apply Gauguin style: post-impressionist, tropical colors, flat areas"""
    # Enhance tropical colors
    hsv = cv.cvtColor(cv_image, cv.COLOR_BGR2HSV)
    hsv[:, :, 0] = (hsv[:, :, 0] + 20) % 180  # Shift to warmer colors
    hsv[:, :, 1] = hsv[:, :, 1] * 1.4  # Increase saturation
    cv_image = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

    # Create flat color areas
    cv_image = cv.posterize(cv_image, 4)

    # Add bold outlines
    gray = cv.cvtColor(cv_image, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, 50, 150)
    cv_image = cv.addWeighted(
        cv_image, 0.9, cv.cvtColor(edges, cv.COLOR_GRAY2BGR), 0.1, 0
    )

    return cv_image


def _apply_seurat_style(cv_image):
    """Apply Seurat style: pointillist, small dots of color"""
    # Create pointillist effect
    height, width = cv_image.shape[:2]
    result = np.zeros_like(cv_image)

    # Sample colors and create dots
    for i in range(0, height, 3):
        for j in range(0, width, 3):
            color = cv_image[i, j]
            cv.circle(result, (j, i), 1, color.tolist(), -1)

    cv_image = result

    return cv_image


def _apply_turner_style(cv_image):
    """Apply Turner style: romantic, atmospheric, light effects"""
    # Create atmospheric effect
    cv_image = cv.GaussianBlur(cv_image, (5, 5), 0)

    # Enhance light effects
    hsv = cv.cvtColor(cv_image, cv.COLOR_BGR2HSV)
    hsv[:, :, 2] = hsv[:, :, 2] * 1.3  # Increase brightness
    cv_image = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

    # Add soft, dreamy effect
    kernel = np.ones((3, 3), np.float32) / 9
    cv_image = cv.filter2D(cv_image, -1, kernel)

    return cv_image
