from davia import Davia
from fastapi import UploadFile
import tempfile
import os
from openai import AsyncOpenAI
from pathlib import Path
from agents import Runner
from pydantic import BaseModel

from openai_vivalehack.agent import agent
from openai_vivalehack.model import AgentContext
from openai_vivalehack.utils import encode_image

app = Davia()
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class ProcessTextRequest(BaseModel):
    text: str
    input_image: UploadFile = None
    current_image_index: int = 0
    images: list[str] = []


class ProcessAudioRequest(BaseModel):
    audio: UploadFile
    input_image: UploadFile = None
    current_image_index: int = 0
    images: list[str] = []


@app.post("/process-text")
async def process_text(
    request: ProcessTextRequest,
):
    """
    Given a text, input image, and current image index, and a list of images,
    process the text and return the modified images.
    If input_image is provided and images is empty, it will be used as the first image.
    Images are provided as base64 encoded strings.

    Args:
        text: The text to process.
        input_image: The input image to use.
        current_image_index: The current image index.
        images: The list of images representing all modified images by AI.

    Returns:
        A dictionary containing the transcription, current image index, and the new list of images.
    """
    # Prepare context
    if request.input_image and len(request.images) == 0:
        image_bytes = await request.input_image.read()
        base64_image = encode_image(image_bytes)
        agent_context = AgentContext(
            modified_images_b64=[base64_image],
            current_image_index=request.current_image_index,
        )
    elif len(request.images) > 0:
        base64_images = request.images
        agent_context = AgentContext(
            modified_images_b64=base64_images,
            current_image_index=request.current_image_index,
        )
    else:
        raise ValueError("No input image or images provided")
    # Run agent
    result = await Runner.run(
        starting_agent=agent,
        context=agent_context,
        input=request.text,
    )
    result_context: AgentContext = result.context_wrapper.context

    return {
        "transcription": request.text,
        "current_image_index": result_context.current_image_index,
        "images": result_context.modified_images_b64,
    }


@app.post("/process-audio")
async def process_audio(
    request: ProcessAudioRequest,
):
    """
    Given an audio file, input image, and current image index, and a list of images,
    process the audio and return the modified images.
    If input_image is provided and images is empty, it will be used as the first image.
    Images are provided as base64 encoded strings.

    Args:
        audio: The audio file to process.
        input_image: The input image to use.
        current_image_index: The current image index.
        images: The list of images representing all modified images by AI.

    Returns:
        A dictionary containing the transcription, current image index, and the new list of images.
    """
    # Prepare context
    if request.input_image and len(request.images) == 0:
        image_bytes = await request.input_image.read()
        base64_image = encode_image(image_bytes)
        agent_context = AgentContext(
            modified_images_b64=[base64_image],
            current_image_index=request.current_image_index,
        )
    elif len(request.images) > 0:
        base64_images = request.images
        agent_context = AgentContext(
            modified_images_b64=base64_images,
            current_image_index=request.current_image_index,
        )
    else:
        raise ValueError("No input image or images provided")
    # Prepare input
    audio_bytes = await request.audio.read()
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=True) as temp_file:
        temp_file.write(audio_bytes)
        temp_file.flush()  # Ensure all data is written
        transcription = await client.audio.transcriptions.create(
            model="gpt-4o-transcribe",
            file=Path(temp_file.name),
            language="en",
        )
    # Run agent
    result = await Runner.run(
        starting_agent=agent,
        context=agent_context,
        input=transcription.text,
    )
    result_context: AgentContext = result.context_wrapper.context

    return {
        "transcription": transcription.text,
        "current_image_index": result_context.current_image_index,
        "images": result_context.modified_images_b64,
    }


if __name__ == "__main__":
    app.run(browser=False)
