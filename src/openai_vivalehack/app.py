from davia import Davia
from fastapi import UploadFile
import tempfile
import os
from openai import AsyncOpenAI
from pathlib import Path
from openai_vivalehack.utils import encode_image

app = Davia()
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


@app.get("/")
def read_root():
    return {"message": "Hello World"}


@app.post("/process-request")
async def process_request(image: UploadFile, audio: UploadFile):
    image_bytes = await image.read()
    base64_image = encode_image(image_bytes)
    completion = await client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what's in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    },
                ],
            }
        ],
    )

    print(completion.choices[0].message.content)

    # Read audio
    audio_bytes = await audio.read()
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=True) as temp_file:
        temp_file.write(audio_bytes)
        temp_file.flush()  # Ensure all data is written
        transcription = await client.audio.transcriptions.create(
            model="gpt-4o-transcribe",
            file=Path(temp_file.name),
        )
        print(transcription)

    return {"transcription": transcription}


if __name__ == "__main__":
    app.run(browser=False)
