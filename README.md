# Real-Time Collaborative Photoshop

A real-time collaborative image editing platform powered by AI, enabling multiple users to modify and enhance images simultaneously through natural language commands or voice instructions.

## 🎨 Features

- **Natural Language Image Editing**: Transform your images using simple text commands
- **Voice-Controlled Editing**: Speak your editing instructions naturally
- **Real-time Collaboration**: Multiple users can work on the same image simultaneously
- **Smart Image Processing**: Powered by advanced AI models

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

## 🚦 Getting Started

1. **Run the app:**

   ```bash
   python src/openai_vivalehack/app.py
   ```

   This will open Davia in your browser.

2. **Prompt the interface you like:**
   - Once Davia is open, simply describe or prompt the interface or workflow you want.
   - For example:
     - "Build a web app with a layout and design inspired by an AI-powered Photoshop.

Functionality (Audio-only):
• On the left panel:
• Let users upload or record an audio file
• While recording, show a live waveform or animation to indicate that recording is active
• On the right panel:
• Display the image corresponding to current_image_index from the images list (shown large)
• Show the transcription or result from the audio processing

Requirements:
• The app must clearly call and reflect the behavior of the process_audio endpoint.
• The design should resemble an AI-enhanced Photoshop-style editor, with intuitive panels and interactive feedback."

- Davia will help you generate the code or design for your desired interface.

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
