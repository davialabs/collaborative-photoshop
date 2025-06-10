import cv2 as cv
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import matplotlib.pyplot as plt
from typing import Tuple, Union, Optional
import os


class ImageModifier:
    """
    A comprehensive image modification class inspired by top image processing libraries.

    This class provides functionality to:
    - Adjust image luminosity/brightness
    - Zoom into specific regions of an image
    - Apply various color scheme transformations

    Supports both OpenCV and PIL/Pillow workflows for maximum flexibility.
    """

    def __init__(self, image_path: str):
        """
        Initialize the ImageModifier with an image.

        Args:
            image_path (str): Path to the input image file
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        self.image_path = image_path
        self.original_cv_image = cv.imread(image_path)
        self.original_pil_image = Image.open(image_path)

        if self.original_cv_image is None:
            raise ValueError(f"Could not load image: {image_path}")

        self.current_cv_image = self.original_cv_image.copy()
        self.current_pil_image = self.original_pil_image.copy()

        print(f"Loaded image: {image_path}")
        print(f"Dimensions: {self.original_cv_image.shape}")

    def adjust_luminosity(self, percentage: float) -> "ImageModifier":
        """
        Adjust the luminosity/brightness of the image by a given percentage.

        Args:
            percentage (float): Percentage to adjust luminosity (-100 to 100)
                               Positive values brighten, negative values darken

        Returns:
            ImageModifier: Self for method chaining
        """
        if not -100 <= percentage <= 100:
            raise ValueError("Percentage must be between -100 and 100")

        # Method 1: Using PIL ImageEnhance (more precise for brightness)
        enhancer = ImageEnhance.Brightness(self.current_pil_image)
        # Convert percentage to enhancement factor (0.0 = black, 1.0 = original, 2.0 = twice as bright)
        factor = 1.0 + (percentage / 100.0)
        factor = max(0.0, factor)  # Ensure factor is not negative

        self.current_pil_image = enhancer.enhance(factor)

        # Method 2: Using OpenCV (HSV adjustment for more natural results)
        hsv = cv.cvtColor(self.current_cv_image, cv.COLOR_BGR2HSV)
        h, s, v = cv.split(hsv)

        # Adjust value (brightness) channel
        if percentage > 0:
            # Brighten
            v = cv.add(v, np.ones(v.shape, dtype=v.dtype) * (percentage * 2.55))
        else:
            # Darken
            v = cv.subtract(
                v, np.ones(v.shape, dtype=v.dtype) * (abs(percentage) * 2.55)
            )

        hsv_adjusted = cv.merge([h, s, v])
        self.current_cv_image = cv.cvtColor(hsv_adjusted, cv.COLOR_HSV2BGR)

        print(f"Adjusted luminosity by {percentage}%")
        return self

    def zoom_region(
        self, x: int, y: int, width: int, height: int, zoom_factor: float = 2.0
    ) -> "ImageModifier":
        """
        Zoom into a specific region of the image.

        Args:
            x (int): X coordinate of the top-left corner of the region
            y (int): Y coordinate of the top-left corner of the region
            width (int): Width of the region to zoom
            height (int): Height of the region to zoom
            zoom_factor (float): Factor by which to zoom (default: 2.0)

        Returns:
            ImageModifier: Self for method chaining
        """
        if zoom_factor <= 0:
            raise ValueError("Zoom factor must be positive")

        # Get image dimensions
        img_height, img_width = self.current_cv_image.shape[:2]

        # Validate coordinates
        if x < 0 or y < 0 or x + width > img_width or y + height > img_height:
            raise ValueError("Zoom region exceeds image boundaries")

        # Extract region of interest using OpenCV
        roi_cv = self.current_cv_image[y : y + height, x : x + width]

        # Calculate new dimensions
        new_width = int(width * zoom_factor)
        new_height = int(height * zoom_factor)

        # Resize the region using different interpolation methods based on zoom factor
        if zoom_factor > 1.0:
            # Zooming in - use cubic interpolation for better quality
            zoomed_roi = cv.resize(
                roi_cv, (new_width, new_height), interpolation=cv.INTER_CUBIC
            )
        else:
            # Zooming out - use area interpolation for better downsampling
            zoomed_roi = cv.resize(
                roi_cv, (new_width, new_height), interpolation=cv.INTER_AREA
            )

        # Create new image with zoomed region
        if zoom_factor > 1.0:
            # If zoomed region is larger than original, crop to fit
            crop_x = max(0, (new_width - img_width) // 2)
            crop_y = max(0, (new_height - img_height) // 2)
            end_x = min(new_width, crop_x + img_width)
            end_y = min(new_height, crop_y + img_height)

            self.current_cv_image = zoomed_roi[crop_y:end_y, crop_x:end_x]
        else:
            # If zoomed region is smaller, center it in the original image
            result = np.zeros_like(self.current_cv_image)
            start_x = (img_width - new_width) // 2
            start_y = (img_height - new_height) // 2
            result[start_y : start_y + new_height, start_x : start_x + new_width] = (
                zoomed_roi
            )
            self.current_cv_image = result

        # Update PIL image
        self.current_pil_image = Image.fromarray(
            cv.cvtColor(self.current_cv_image, cv.COLOR_BGR2RGB)
        )

        print(f"Zoomed region ({x}, {y}, {width}, {height}) by factor {zoom_factor}")
        return self

    def change_color_scheme(self, scheme: str) -> "ImageModifier":
        """
        Apply various color scheme transformations to the image.

        Args:
            scheme (str): Color scheme to apply. Options:
                         'grayscale', 'sepia', 'vintage', 'cool', 'warm',
                         'high_contrast', 'negative', 'enhance_red',
                         'enhance_green', 'enhance_blue'

        Returns:
            ImageModifier: Self for method chaining
        """
        scheme = scheme.lower()

        if scheme == "grayscale":
            # Convert to grayscale
            self.current_cv_image = cv.cvtColor(
                self.current_cv_image, cv.COLOR_BGR2GRAY
            )
            self.current_cv_image = cv.cvtColor(
                self.current_cv_image, cv.COLOR_GRAY2BGR
            )
            self.current_pil_image = self.current_pil_image.convert("L").convert("RGB")

        elif scheme == "sepia":
            # Apply sepia tone transformation
            sepia_filter = np.array(
                [[0.272, 0.534, 0.131], [0.349, 0.686, 0.168], [0.393, 0.769, 0.189]]
            )

            sepia_img = cv.transform(self.current_cv_image, sepia_filter)
            self.current_cv_image = np.clip(sepia_img, 0, 255).astype(np.uint8)
            self.current_pil_image = Image.fromarray(
                cv.cvtColor(self.current_cv_image, cv.COLOR_BGR2RGB)
            )

        elif scheme == "vintage":
            # Create vintage effect with reduced saturation and slight sepia
            hsv = cv.cvtColor(self.current_cv_image, cv.COLOR_BGR2HSV)
            hsv[:, :, 1] = hsv[:, :, 1] * 0.6  # Reduce saturation
            vintage_img = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

            # Apply slight sepia
            sepia_filter = np.array([[0.8, 0.6, 0.4], [0.9, 0.7, 0.5], [0.4, 0.3, 0.2]])
            vintage_img = cv.transform(vintage_img, sepia_filter)
            self.current_cv_image = np.clip(vintage_img, 0, 255).astype(np.uint8)
            self.current_pil_image = Image.fromarray(
                cv.cvtColor(self.current_cv_image, cv.COLOR_BGR2RGB)
            )

        elif scheme == "cool":
            # Apply cool tone (enhance blues, reduce reds)
            b, g, r = cv.split(self.current_cv_image)
            b = cv.add(b, 30)  # Enhance blue
            r = cv.subtract(r, 15)  # Reduce red
            self.current_cv_image = cv.merge([b, g, r])
            self.current_pil_image = Image.fromarray(
                cv.cvtColor(self.current_cv_image, cv.COLOR_BGR2RGB)
            )

        elif scheme == "warm":
            # Apply warm tone (enhance reds/yellows, reduce blues)
            b, g, r = cv.split(self.current_cv_image)
            r = cv.add(r, 30)  # Enhance red
            g = cv.add(g, 15)  # Enhance green (for yellow tint)
            b = cv.subtract(b, 20)  # Reduce blue
            self.current_cv_image = cv.merge([b, g, r])
            self.current_pil_image = Image.fromarray(
                cv.cvtColor(self.current_cv_image, cv.COLOR_BGR2RGB)
            )

        elif scheme == "high_contrast":
            # Increase contrast using PIL
            enhancer = ImageEnhance.Contrast(self.current_pil_image)
            self.current_pil_image = enhancer.enhance(2.0)
            self.current_cv_image = cv.cvtColor(
                np.array(self.current_pil_image), cv.COLOR_RGB2BGR
            )

        elif scheme == "negative":
            # Create negative effect
            self.current_cv_image = 255 - self.current_cv_image
            self.current_pil_image = Image.fromarray(
                cv.cvtColor(self.current_cv_image, cv.COLOR_BGR2RGB)
            )

        elif scheme == "enhance_red":
            # Enhance red channel
            b, g, r = cv.split(self.current_cv_image)
            r = cv.multiply(r, 1.3)
            self.current_cv_image = cv.merge([b, g, r])
            self.current_pil_image = Image.fromarray(
                cv.cvtColor(self.current_cv_image, cv.COLOR_BGR2RGB)
            )

        elif scheme == "enhance_green":
            # Enhance green channel
            b, g, r = cv.split(self.current_cv_image)
            g = cv.multiply(g, 1.3)
            self.current_cv_image = cv.merge([b, g, r])
            self.current_pil_image = Image.fromarray(
                cv.cvtColor(self.current_cv_image, cv.COLOR_BGR2RGB)
            )

        elif scheme == "enhance_blue":
            # Enhance blue channel
            b, g, r = cv.split(self.current_cv_image)
            b = cv.multiply(b, 1.3)
            self.current_cv_image = cv.merge([b, g, r])
            self.current_pil_image = Image.fromarray(
                cv.cvtColor(self.current_cv_image, cv.COLOR_BGR2RGB)
            )

        else:
            raise ValueError(
                f"Unknown color scheme: {scheme}. Available schemes: "
                "'grayscale', 'sepia', 'vintage', 'cool', 'warm', "
                "'high_contrast', 'negative', 'enhance_red', 'enhance_green', 'enhance_blue'"
            )

        print(f"Applied color scheme: {scheme}")
        return self

    def reset_to_original(self) -> "ImageModifier":
        """
        Reset the image to its original state.

        Returns:
            ImageModifier: Self for method chaining
        """
        self.current_cv_image = self.original_cv_image.copy()
        self.current_pil_image = self.original_pil_image.copy()
        print("Reset to original image")
        return self

    def save_image(self, output_path: str, quality: int = 95) -> None:
        """
        Save the current modified image.

        Args:
            output_path (str): Path where to save the modified image
            quality (int): JPEG quality (1-100, default: 95)
        """
        # Use PIL for saving to maintain quality
        if output_path.lower().endswith(".jpg") or output_path.lower().endswith(
            ".jpeg"
        ):
            self.current_pil_image.save(output_path, "JPEG", quality=quality)
        else:
            self.current_pil_image.save(output_path)

        print(f"Saved modified image to: {output_path}")

    def display_comparison(self, figsize: Tuple[int, int] = (15, 5)) -> None:
        """
        Display original and modified images side by side.

        Args:
            figsize (Tuple[int, int]): Figure size for the plot
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Display original image
        original_rgb = cv.cvtColor(self.original_cv_image, cv.COLOR_BGR2RGB)
        axes[0].imshow(original_rgb)
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        # Display modified image
        modified_rgb = cv.cvtColor(self.current_cv_image, cv.COLOR_BGR2RGB)
        axes[1].imshow(modified_rgb)
        axes[1].set_title("Modified Image")
        axes[1].axis("off")

        plt.tight_layout()
        plt.show()

    def get_image_info(self) -> dict:
        """
        Get information about the current image.

        Returns:
            dict: Dictionary containing image information
        """
        height, width, channels = self.current_cv_image.shape
        return {
            "width": width,
            "height": height,
            "channels": channels,
            "size": self.current_pil_image.size,
            "mode": self.current_pil_image.mode,
            "format": self.current_pil_image.format,
        }


# Example usage and demonstration
if __name__ == "__main__":
    # Example usage (uncomment to test with your own image)
    """
    # Initialize the image modifier
    modifier = ImageModifier("path/to/your/image.jpg")
    
    # Chain multiple modifications
    modifier.adjust_luminosity(20).change_color_scheme('warm').zoom_region(100, 100, 200, 200, 1.5)
    
    # Display comparison
    modifier.display_comparison()
    
    # Save the result
    modifier.save_image("modified_image.jpg")
    
    # Get image information
    info = modifier.get_image_info()
    print(f"Image info: {info}")
    """

    print("ImageModifier class loaded successfully!")
    print("\nAvailable methods:")
    print("- adjust_luminosity(percentage): Adjust brightness (-100 to 100)")
    print(
        "- zoom_region(x, y, width, height, zoom_factor): Zoom into a specific region"
    )
    print("- change_color_scheme(scheme): Apply color transformations")
    print("- reset_to_original(): Reset to original image")
    print("- save_image(path): Save the modified image")
    print("- display_comparison(): Show before/after comparison")
    print("- get_image_info(): Get image details")

    print("\nAvailable color schemes:")
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
    for scheme in schemes:
        print(f"  - {scheme}")
