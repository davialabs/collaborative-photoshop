"""
Example usage of the ImageModifier class.

This script demonstrates how to use the ImageModifier class to:
1. Adjust image luminosity
2. Zoom into specific regions
3. Apply various color schemes
4. Chain multiple modifications together

Based on the Neptune.ai blog post about image processing libraries.
"""

from image_modification import ImageModifier
import os


def demonstrate_image_modifications():
    """
    Demonstrate various image modification capabilities.
    """

    # Use forward slashes for path to avoid escape sequence issues
    image_path = "image_modifications/kermit.jpg"

    # Check if image exists
    if not os.path.exists(image_path):
        print(f"⚠️  Demo image '{image_path}' not found!")
        print("To run this demo:")
        print(
            "1. Place an image file named 'kermit.jpg' in the image_modifications directory, or"
        )
        print("2. Modify the 'image_path' variable to point to your image")
        print("\nExample usage patterns are shown below:")
        show_usage_examples()
        return

    try:
        # Initialize the ImageModifier
        print("🖼️  Initializing ImageModifier...")
        modifier = ImageModifier(image_path)

        # Display original image info
        info = modifier.get_image_info()
        print(
            f"📊 Image Info: {info['width']}x{info['height']} pixels, {info['channels']} channels"
        )

        # Example 1: Adjust luminosity
        print("\n✨ Example 1: Adjusting luminosity...")
        modifier.adjust_luminosity(25)  # Brighten by 25%
        modifier.display_comparison()

        # Reset and try darkening
        modifier.reset_to_original()
        modifier.adjust_luminosity(-30)  # Darken by 30%
        modifier.display_comparison()

        # Example 2: Apply color schemes
        print("\n🎨 Example 2: Applying color schemes...")
        modifier.reset_to_original()

        # Apply different color schemes
        color_schemes = ["sepia", "vintage", "cool", "warm", "high_contrast"]
        for scheme in color_schemes:
            modifier.reset_to_original()
            modifier.change_color_scheme(scheme)
            print(f"   ✅ Applied {scheme} color scheme")
            modifier.display_comparison()

        # Example 3: Zoom into region
        print("\n🔍 Example 3: Zooming into region...")
        modifier.reset_to_original()

        # Get image dimensions for safe zooming
        height, width = modifier.current_cv_image.shape[:2]

        # Zoom into center region (adjust coordinates based on your image)
        zoom_x = width // 4
        zoom_y = height // 4
        zoom_width = width // 2
        zoom_height = height // 2

        modifier.zoom_region(zoom_x, zoom_y, zoom_width, zoom_height, zoom_factor=1.5)
        modifier.display_comparison()

        # Example 4: Chain multiple modifications
        print("\n🔗 Example 4: Chaining multiple modifications...")
        modifier.reset_to_original()

        # Chain: brighten -> apply warm tone -> slight zoom
        (
            modifier.adjust_luminosity(15)
            .change_color_scheme("warm")
            .zoom_region(zoom_x, zoom_y, zoom_width, zoom_height, 1.2)
        )

        modifier.display_comparison()

        print("\n🎉 Demo completed!")

    except Exception as e:
        print(f"❌ Error during demo: {e}")


def show_usage_examples():
    """
    Show code examples without actually running them.
    """
    print("\n📝 Code Examples:")
    print("=" * 50)

    print("\n1️⃣  Basic Usage:")
    print("""
# Load an image
modifier = ImageModifier("your_image.jpg")

# Adjust brightness by 20%
modifier.adjust_luminosity(20)

# Save the result
modifier.save_image("bright_image.jpg")
""")

    print("\n2️⃣  Color Scheme Changes:")
    print("""
# Apply different color schemes
modifier.change_color_scheme('sepia')     # Vintage sepia tone
modifier.change_color_scheme('cool')      # Cool blue tones
modifier.change_color_scheme('warm')      # Warm red/yellow tones
modifier.change_color_scheme('vintage')   # Vintage effect
modifier.change_color_scheme('grayscale') # Black and white
""")

    print("\n3️⃣  Region Zooming:")
    print("""
# Zoom into a specific region
# zoom_region(x, y, width, height, zoom_factor)
modifier.zoom_region(100, 100, 200, 200, 2.0)  # 2x zoom
""")

    print("\n4️⃣  Method Chaining:")
    print("""
# Chain multiple operations
modifier.adjust_luminosity(15).change_color_scheme('warm').zoom_region(50, 50, 300, 300, 1.5)
""")

    print("\n5️⃣  Available Color Schemes:")
    schemes = [
        "grayscale",
        "sepia",
        "vintage",
        "cool",
        "warm",
        "high_contrast",
        "negative",
        "enhance_red",
        "enhance_green",
        "enhance_blue",
    ]
    for i, scheme in enumerate(schemes, 1):
        print(f"   {i:2d}. {scheme}")


if __name__ == "__main__":
    print("🚀 ImageModifier Demo")
    print("=" * 50)

    # Run the demonstration
    demonstrate_image_modifications()

    print("\n💡 Tips:")
    print("- Use luminosity adjustments between -50 and +50 for natural results")
    print("- Zoom factors between 1.2 and 3.0 work best for most images")
    print("- Try combining 'vintage' or 'sepia' with slight luminosity adjustments")
    print("- Use 'cool' or 'warm' color schemes for mood enhancement")
