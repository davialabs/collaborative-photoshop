import base64
from openai import OpenAI

client = OpenAI()

img = client.images.generate(
    model="dall-e-2",
    prompt="A cute baby sea otter",
)

image_bytes = base64.b64decode(img.data[0].b64_json)
with open("output.png", "wb") as f:
    f.write(image_bytes)
