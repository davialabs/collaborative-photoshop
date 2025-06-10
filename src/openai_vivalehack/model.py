from pydantic import BaseModel
from pydantic import Field


class AgentContext(BaseModel):
    image_b64: str
    modified_images_b64: list[str] = Field(default_factory=list)
