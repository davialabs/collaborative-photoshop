from typing_extensions import TypedDict, Any
import asyncio

from agents import Agent, RunContextWrapper, function_tool, Runner
from openai_vivalehack.image_modifications.image_modification import ImageModifier


class Location(TypedDict):
    lat: float
    long: float


@function_tool
async def fetch_weather(location: Location) -> str:
    """Fetch the weather for a given location.

    Args:
        location: The location to fetch the weather for.
    """
    # In real life, we'd fetch the weather from a weather API
    return "sunny"


@function_tool(name_override="fetch_data")
def read_file(
    ctx: RunContextWrapper[Any], path: str, directory: str | None = None
) -> str:
    """Read the contents of a file.

    Args:
        path: The path to the file to read.
        directory: The directory to read the file from.
    """
    # In real life, we'd read the file from the file system
    return "<file contents>"


@function_tool
def modify_image(image_path: str) -> str:
    """Modify the image at the given path.

    Args:
        image_path: The path to the image to modify.
    """
    return f"image modified: {image_path}"


agent = Agent(
    name="Assistant",
    tools=[fetch_weather, read_file, modify_image],
)


async def main():
    result = await Runner.run(
        agent,
        "modify the image at the given path: /Users/jason/Desktop/IMG_0001.jpg",
    )
    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
