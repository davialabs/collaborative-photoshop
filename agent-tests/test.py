from typing_extensions import TypedDict, Any
import asyncio

from agents import Agent, RunContextWrapper, function_tool, Runner


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


agent = Agent(
    name="Assistant",
    tools=[fetch_weather, read_file],
)


async def main():
    result = await Runner.run(
        agent,
        "What is the weather in SF?",
    )
    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
