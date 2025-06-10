from pydantic import BaseModel


class AgentContext(BaseModel):
    image_b64: str
    new_image_b64: str
