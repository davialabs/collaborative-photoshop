from pydantic import BaseModel
from pydantic import Field


class AgentContext(BaseModel):
    image_b64: str
    current_image_index: int = Field(default=0)
    modified_images_b64: list[str] = Field(default_factory=list)
