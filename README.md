# OpenAI VivaLeHack

A real-time collaborative image editing platform powered by AI, enabling multiple users to modify and enhance images simultaneously through natural language commands or voice instructions.

## 🎨 Features

- **Natural Language Image Editing**: Transform your images using simple text commands
- **Voice-Controlled Editing**: Speak your editing instructions naturally
- **Real-time Collaboration**: Multiple users can work on the same image simultaneously
- **Smart Image Processing**: Powered by OpenAI's advanced AI models

### Available Image Modifications

- Remove image backgrounds
- Adjust contrast and luminosity
- Change color schemes
- Generate new images from descriptions
- Edit existing images with natural language instructions
- Make images more realistic
- Navigate through image history

## 🚀 How It Works

The platform is built around an intelligent Image Modifier Agent that:

1. Receives instructions (text or transcribed audio)
2. Processes them through AI to understand the intended modifications
3. Applies the changes in real-time
4. Maintains a history of modifications for easy navigation

### API Endpoints

- `/process-text`: Accept text commands for image modification
- `/process-audio`: Accept voice commands for image modification

Both endpoints support:

- Uploading new images
- Modifying existing images
- Maintaining modification history
- Real-time collaboration through shared image states

## 🛠️ Technical Requirements

- Python 3.12+
- OpenAI API key
- FastAPI
- Base64 image processing capabilities

## 🔑 Environment Setup

1. Clone the repository
2. Set up your environment variables:
   ```bash
   export OPENAI_API_KEY="your-api-key"
   ```

## 🎯 Example Usage

```python
# Text-based editing
"Remove the background of the image. Then adjust the contrast by 50%. Then make it more realistic."

# Voice-based editing
"Make the colors more vibrant and remove the background"
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

[Add your license here]
