import base64


def encode_image(image_bytes):
    """
    Encode an image to base64.

    Args:
        image_bytes (bytes): The image to encode.

    Returns:
        str: The base64 encoded image.
    """
    return base64.b64encode(image_bytes).decode("utf-8")
