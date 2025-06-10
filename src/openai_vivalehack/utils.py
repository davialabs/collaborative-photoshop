import base64


def encode_image(image_bytes: bytes) -> str:
    """
    Encode an image to base64.

    Args:
        image_bytes (bytes): The image to encode.

    Returns:
        str: The base64 encoded image.
    """
    return base64.b64encode(image_bytes).decode("utf-8")


def decode_image(base64_image: str) -> bytes:
    """
    Decode a base64 encoded image.

    Args:
        base64_image (str): The base64 encoded image.

    Returns:
        bytes: The decoded image.
    """
    return base64.b64decode(base64_image)
