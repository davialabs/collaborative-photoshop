from typing_extensions import TypedDict, Any
import asyncio
import base64
import tempfile
import os

from agents import Agent, RunContextWrapper, function_tool, Runner
from openai_vivalehack.image_modifications.image_modification import (
    adjust_luminosity_base64,
    name_image,
)
from openai_vivalehack.model import AgentContext


agent = Agent(
    name="Image Modifier Agent",
    tools=[adjust_luminosity_base64, name_image],
)


async def main():
    with open(
        "/Users/rubenillouz/project/openai-vivalehack/image_modifications_test/kermit.jpg",
        "rb",
    ) as img_file:
        image_b64 = base64.b64encode(img_file.read()).decode("utf-8")

    agent_context = AgentContext(image_b64=image_b64, new_image_b64="")

    result = await Runner.run(
        starting_agent=agent,
        context=agent_context,
        input="Adjust the luminosity of the image by 60%",
    )
    print(result.final_output)

    # Save the modified image as PNG
    modified_image_data = base64.b64decode(result.context_wrapper.context.new_image_b64)  # type: ignore
    with open("modified_image.png", "wb") as f:
        f.write(modified_image_data)


if __name__ == "__main__":
    asyncio.run(main())
