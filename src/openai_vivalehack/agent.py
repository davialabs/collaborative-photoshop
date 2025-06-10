from agents import Agent

from openai_vivalehack.image_modifications import (
    adjust_luminosity_base64,
)
from openai_vivalehack.model import AgentContext

agent = Agent[AgentContext](
    name="Image Modifier Agent",
    tools=[
        adjust_luminosity_base64,
    ],
)
