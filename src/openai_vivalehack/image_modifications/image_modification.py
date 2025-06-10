from PIL import Image, ImageEnhance
import base64
from io import BytesIO
from agents import function_tool
from agents import RunContextWrapper

from openai_vivalehack.model import AgentContext


@function_tool
def adjust_luminosity_base64(
    ctx: RunContextWrapper[AgentContext], percentage: float
) -> str:
    """
    Given an image as base64, adjust the luminosity of the image by a percentage (-100 to 100).
    The new image is stored in the context.new_image_b64 field.
    The only thing you need is the percentage, don't ask for the image.
    """
    print("adjusting luminosity")
    # Decode base64 to bytes
    image_data = base64.b64decode(ctx.context.image_b64)
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
    ctx.context.new_image_b64 = modified_b64
    print("luminosity adjusted")
    return f"Luminosity adjusted by {percentage}% for provided image."


@function_tool
def name_image(ctx: RunContextWrapper[AgentContext]) -> str:
    """
    Name the image.
    You don't need the image, the image is provided as base64, in the context.new_image_b64 field.
    """
    print("naming image")
    return f"Image named: {ctx.context.image_b64}"
