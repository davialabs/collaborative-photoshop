from PIL import Image, ImageEnhance
import base64
from io import BytesIO
from agents import function_tool
from agents import RunContextWrapper

from openai_vivalehack.model import AgentContext
from openai_vivalehack.utils import decode_image


@function_tool
def adjust_luminosity_base64(
    ctx: RunContextWrapper[AgentContext], percentage: float
) -> str:
    """
    Adjusts the luminosity of the image by a percentage (-100 to 100).
    DO NOT ASK FOR THE IMAGE, only the percentage is needed. The image is provided as base64 in the context.image_b64 field.
    The new image is stored in the context.modified_images_b64 field.

    Args:
        percentage (float): The percentage to adjust the luminosity by.

    Returns:
        str: A success message.
    """
    print(f"Adjusting luminosity by {percentage}%")
    # Decode base64 to bytes
    image_data = decode_image(ctx.context.image_b64)
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
    return f"Luminosity adjusted by {percentage}%."
