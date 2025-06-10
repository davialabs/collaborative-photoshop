import asyncio
import base64

from agents import Agent, Runner
from openai_vivalehack.image_modifications import (
    adjust_luminosity_base64,
    change_color_scheme,
    go_to_image_index,
    next_image,
    previous_image,
)
from openai_vivalehack.image_generator import generate_image, edit_image
from openai_vivalehack.model import AgentContext


agent = Agent(
    name="Image Modifier Agent",
    tools=[
        adjust_luminosity_base64,
        change_color_scheme,
        go_to_image_index,
        next_image,
        previous_image,
        generate_image,
        edit_image,
    ],
)


async def main():
    agent_context = AgentContext(modified_images_b64=[], current_image_index=0)

    result = await Runner.run(
        starting_agent=agent,
        context=agent_context,
        input="Create a new image of a cat",
    )

    print(result.final_output)

    # Save the modified image as PNG
    for i, modified_image_b64 in enumerate(
        result.context_wrapper.context.modified_images_b64
    ):
        modified_image_data = base64.b64decode(modified_image_b64)
        with open(f"modified_image_{i}.png", "wb") as f:
            f.write(modified_image_data)

    final_image_b64 = result.context_wrapper.context.modified_images_b64[
        result.context_wrapper.context.current_image_index
    ]
    final_image_data = base64.b64decode(final_image_b64)
    with open("final_image.png", "wb") as f:
        f.write(final_image_data)


if __name__ == "__main__":
    asyncio.run(main())
