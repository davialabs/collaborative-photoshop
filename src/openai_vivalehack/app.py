from davia import Davia
from fastapi import UploadFile
import tempfile
import os
from openai import AsyncOpenAI
from pathlib import Path
from agents import Runner

from openai_vivalehack.agent import agent
from openai_vivalehack.model import AgentContext
from openai_vivalehack.utils import encode_image

app = Davia()
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


@app.get("/")
def read_root():
    return {"message": "Hello World"}


@app.post("/process-request")
async def process_request(image: UploadFile, audio: UploadFile):
    # Prepare context
    image_bytes = await image.read()
    base64_image = encode_image(image_bytes)
    agent_context = AgentContext(image_b64=base64_image)
    # Read audio
    audio_bytes = await audio.read()
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=True) as temp_file:
        temp_file.write(audio_bytes)
        temp_file.flush()  # Ensure all data is written
        transcription = await client.audio.transcriptions.create(
            model="gpt-4o-transcribe",
            file=Path(temp_file.name),
            language="en",
        )
        print(transcription)
    # Run agent
    result = await Runner.run(
        starting_agent=agent,
        context=agent_context,
        input=transcription.text,
    )
    print(result.final_output)
    result_context: AgentContext = result.context_wrapper.context
    print(result_context.modified_images_b64)

    return {
        "transcription": transcription.text,
        "images": result_context.modified_images_b64,
    }


if __name__ == "__main__":
    app.run(browser=False)
