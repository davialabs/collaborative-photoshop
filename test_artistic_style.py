#!/usr/bin/env python3
"""
Test script for the new artistic style transfer feature.
This script creates a simple test image and applies various artistic styles.
"""

import asyncio
import base64
import numpy as np
from PIL import Image
from io import BytesIO

from src.collaborative_photoshop.agent import agent
from src.collaborative_photoshop.model import AgentContext
from agents import Runner


def create_test_image():
    """Create a simple test image for style transfer testing."""
    # Create a colorful test image
    width, height = 400, 300
    image = np.zeros((height, width, 3), dtype=np.uint8)

    # Add some colorful shapes
    # Red circle
    center_x, center_y = width // 3, height // 2
    y, x = np.ogrid[:height, :width]
    mask = (x - center_x) ** 2 + (y - center_y) ** 2 < 50**2
    image[mask] = [255, 0, 0]  # Red

    # Blue rectangle
    image[height // 4 : 3 * height // 4, 2 * width // 3 : width] = [0, 0, 255]  # Blue

    # Green triangle (simplified)
    for i in range(height // 4, 3 * height // 4):
        for j in range(width // 2, width // 2 + (i - height // 4)):
            if j < width:
                image[i, j] = [0, 255, 0]  # Green

    # Convert to PIL Image
    pil_image = Image.fromarray(image)

    # Convert to base64
    buffer = BytesIO()
    pil_image.save(buffer, format="PNG")
    buffer.seek(0)
    image_b64 = base64.b64encode(buffer.read()).decode("utf-8")

    return image_b64


async def test_artistic_styles():
    """Test the artistic style transfer feature with various styles."""
    print("🎨 Testing Artistic Style Transfer Feature")
    print("=" * 50)

    # Create test image
    test_image_b64 = create_test_image()
    print("✅ Created test image")

    # Test styles
    test_styles = ["van_gogh", "picasso", "monet", "warhol", "dali"]

    for style in test_styles:
        print(f"\n🎭 Testing {style} style...")

        try:
            # Create agent context
            agent_context = AgentContext(
                modified_images_b64=[test_image_b64], current_image_index=0
            )

            # Apply artistic style
            result = await Runner.run(
                starting_agent=agent,
                context=agent_context,
                input=f"Apply {style} artistic style to this image",
            )

            result_context: AgentContext = result.context_wrapper.context

            if len(result_context.modified_images_b64) > 1:
                print(f"✅ {style} style applied successfully!")
                print(f"   - Original images: 1")
                print(
                    f"   - Modified images: {len(result_context.modified_images_b64)}"
                )
                print(f"   - Current index: {result_context.current_image_index}")
            else:
                print(f"❌ {style} style failed - no new image created")

        except Exception as e:
            print(f"❌ Error applying {style} style: {str(e)}")

    print(f"\n🎉 Artistic style transfer testing completed!")
    print(
        "The new feature allows users to transform images with famous artistic styles!"
    )
    print(
        "Available styles: van_gogh, picasso, monet, kandinsky, dali, warhol, hokusai, munch, pollock, matisse, cezanne, gauguin, seurat, turner"
    )


if __name__ == "__main__":
    asyncio.run(test_artistic_styles())
