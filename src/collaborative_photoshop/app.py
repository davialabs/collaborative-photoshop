from davia import Davia
from fastapi import UploadFile
import tempfile
import os
from openai import AsyncOpenAI
from pathlib import Path
from agents import Runner
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from collaborative_photoshop.agent import agent
from collaborative_photoshop.model import AgentContext
from collaborative_photoshop.utils import encode_image


class LimitUploadSizeMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, max_upload_size: int):
        super().__init__(app)
        self.max_upload_size = max_upload_size

    async def dispatch(self, request: Request, call_next):
        # Only check for POST, PUT, PATCH (upload) requests
        if request.method in ("POST", "PUT", "PATCH"):
            if request.headers.get("content-length"):
                if int(request.headers["content-length"]) > self.max_upload_size:
                    return Response("File too large", status_code=413)
        return await call_next(request)


app = Davia()
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app.add_middleware(LimitUploadSizeMiddleware, max_upload_size=10 * 1024 * 1024)  # 10MB


@app.post("/process-audio")
async def process_audio(
    audio: UploadFile,
    input_image: UploadFile = None,
    current_image_index: int = 0,
    images: list[str] = [],
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
    if input_image and len(images) == 0:
        image_bytes = await input_image.read()
        base64_image = encode_image(image_bytes)
        agent_context = AgentContext(
            modified_images_b64=[base64_image],
            current_image_index=current_image_index,
        )
    elif len(images) > 0:
        base64_images = images
        agent_context = AgentContext(
            modified_images_b64=base64_images,
            current_image_index=current_image_index,
        )
    else:
        raise ValueError("No input image or images provided")
    # Prepare input
    audio_bytes = await audio.read()
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
    app.run()
