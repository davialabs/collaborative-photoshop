# Real-Time Collaborative Photoshop

**🌐 The app is available online:** [https://sonic-canvas-davia.vercel.app/](https://sonic-canvas-davia.vercel.app/)

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
- **NEW: Apply artistic styles** - Transform images with famous artistic styles like Van Gogh, Picasso, Monet, Warhol, and more!

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
   python src/collaborative-photoshop/app.py
   ```

   This will open Davia in your browser.

2. **Create Your Interface:**

   - Once Davia is running, you can describe your desired interface using natural language
   - Example prompt:

     ```
     Create an AI-powered image editor interface with:

     Left Panel:
     - Audio input section with upload/record functionality
     - Live waveform visualization during recording

     Right Panel:
     - Large image preview showing the current image
     - Real-time transcription display

     Design Requirements:
     - Photoshop-inspired layout with intuitive controls
     - Seamless integration with process_audio endpoint
     - Interactive feedback for user actions
     ```

- Davia will help you generate the code or design for your desired interface.

## 🎨 Artistic Style Transfer

Transform your images with famous artistic styles! The new AI-powered style transfer feature supports:

- **Van Gogh**: Bold brushstrokes, vibrant colors, swirling patterns
- **Picasso**: Cubist geometric shapes, bold lines
- **Monet**: Impressionist, soft brushstrokes, light effects
- **Warhol**: Pop art, high contrast, bold colors
- **Dali**: Surreal, melting effects, dreamlike
- **Kandinsky**: Abstract geometric shapes, bold colors
- **Hokusai**: Japanese woodblock, bold outlines, flat colors
- **Munch**: Expressionist, emotional, bold brushstrokes
- **Pollock**: Abstract expressionist, splattered paint
- **Matisse**: Fauvist, bold colors, simplified forms
- **Cezanne**: Post-impressionist, geometric brushstrokes
- **Gauguin**: Tropical colors, flat areas
- **Seurat**: Pointillist, small dots of color
- **Turner**: Romantic, atmospheric, light effects

## 🎯 Example Usage

```python
# Text-based editing
"Remove the background of the image. Then adjust the contrast by 50%. Then make it more realistic."

# Voice-based editing
"Make the colors more vibrant and remove the background"

# Artistic style transfer
"Apply Van Gogh style to this image" or "Make it look like a Picasso painting"
```

## 🤝 Contributing

Contributions are welcome!
