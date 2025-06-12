from agents import function_tool
from agents import RunContextWrapper

from collaborative_photoshop.model import AgentContext

import base64
from openai import OpenAI
import io


@function_tool
def generate_image(ctx: RunContextWrapper[AgentContext], prompt: str) -> str:
    print(f"Generating image with prompt: {prompt}")
    """
    Generate an image based on a prompt.
    """
    client = OpenAI()
    model = "gpt-4.1-mini"
    response = client.responses.create(
        model=model,
        input=prompt,
        tools=[{"type": "image_generation"}],
    )
    # the generated image is in the response.output[0].result , should be saved in the context.modified_images_b64 list
    ctx.context.modified_images_b64.append(response.output[0].result)
    ctx.context.current_image_index = len(ctx.context.modified_images_b64) - 1
    return f"Image generated for prompt: {prompt}"


@function_tool
def edit_image(ctx: RunContextWrapper[AgentContext], prompt: str) -> str:
    """
    Edit the current image based on a prompt.
    """
    print(f"Editing image with prompt: {prompt}")
    client = OpenAI()

    # Ensure there is a valid image to edit
    try:
        current_index = ctx.context.current_image_index
        image_b64 = ctx.context.modified_images_b64[current_index]
    except (IndexError, AttributeError) as e:
        print(
            f"Error: No image found at index {getattr(ctx.context, 'current_image_index', None)}. {e}"
        )
        return "No image available to edit."

    # Convert base64 string to bytes
    try:
        image_bytes = base64.b64decode(image_b64)
        image_file = io.BytesIO(image_bytes)
        image_file.name = "image.png"  # Set a filename with a supported extension
    except Exception as e:
        print(f"Error decoding base64 image: {e}")
        return "Failed to decode the image."

    # Call the OpenAI image edit API
    try:
        result = client.images.edit(
            model="gpt-image-1",
            image=image_file,
            prompt=prompt,
        )
        image_base64 = result.data[0].b64_json
    except Exception as e:
        print(f"Error editing image: {e}")
        return "Failed to edit the image."

    # Update context with the new image
    ctx.context.modified_images_b64.append(image_base64)
    ctx.context.current_image_index = len(ctx.context.modified_images_b64) - 1
    return f"Image edited with prompt: {prompt}"
