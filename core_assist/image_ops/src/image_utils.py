import os
from io import BytesIO
from typing import Optional, Tuple, Union

import cv2
import numpy as np


# Load image
def load_img(img_path: str) -> Optional[np.ndarray]:
    """
    Loads an image from the specified path using OpenCV.

    Args:
        img_path (str): The path to the image file.

    Returns:
        Optional[np.ndarray]: The loaded image as a numpy array, or None if the image fails to load.
    """
    try:
        # Open the image using OpenCV
        # cv2.imread returns BGR by default, so we don't need to do any color conversions
        img = cv2.imread(img_path)
        if img is None:
            raise Exception("Failed to load image")
        return img
    except Exception as e:
        print(f"Failed to load image from path: {img_path}")
        print("Error:", e)
        return None


# Load RGB image
def load_rgb(img_path: str) -> Optional[np.ndarray]:
    """
    Loads an image from the specified path and converts it to RGB format.

    Args:
        img_path (str): The path to the image file.

    Returns:
        Optional[np.ndarray]: The loaded image in RGB format as a numpy array, or None if the image fails to load.
    """
    # Check if the image file exists at the provided path
    if not os.path.exists(img_path):
        print(f"Error: Image file not found at {img_path}")
        return None

    # Load the image using OpenCV (default is BGR format)
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Failed to load image from {img_path}")
        return None

    try:
        # Convert the BGR image to RGB format
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return rgb_img
    except Exception as e:
        print(f"Error converting image to RGB: {e}")
        return None


# Load BGR image
def load_bgr(img_path: str) -> Optional[np.ndarray]:
    """
    Loads an image and returns it in BGR format (OpenCV default).

    Args:
        img_path (str): The path to the image file.

    Returns:
        Optional[np.ndarray]: The loaded image in BGR format as a numpy array, or None if the image fails to load.
    """
    # Check if the image file exists at the provided path
    if not os.path.exists(img_path):
        print(f"Error: Image file not found at {img_path}")
        return None

    # Load the image using OpenCV (default is BGR format)
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Failed to load image from {img_path}")
        return None

    try:
        # Return the BGR image (OpenCV default format)
        return img
    except Exception as e:
        print(f"Error loading image in BGR format: {e}")
        return None


# Load Gray image
def load_gray(img_path: str) -> Optional[np.ndarray]:
    """
    Loads an image and converts it to grayscale using OpenCV.

    Args:
        img_path (str): The path to the image file.

    Returns:
        Optional[np.ndarray]: The loaded image in grayscale as a numpy array, or None if the image fails to load.
    """
    # Load the image using OpenCV
    img = load_img(img_path)

    try:
        if img is not None:
            # Convert the image to grayscale using OpenCV
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            return gray_img
    except Exception as e:
        # Handle any errors encountered during image loading and conversion
        print(f"Error loading image in grayscale: {e}")
        return None


# Load RGB image from buffer
def load_buffer_rgb(buffer_data: BytesIO):
    """
    Loads an image from buffer data and converts it to RGB mode.

    Args:
        buffer_data (bytes): The image data as a bytes object.

    Returns:
        numpy.ndarray: The loaded image in RGB mode as a numpy array, or None if the image fails to load.
    """
    try:
        # Read the image from buffer data (NumPy array)
        img_array = np.asarray(bytearray(buffer_data), dtype=np.uint8)
        img = cv2.imdecode(
            img_array, cv2.IMREAD_COLOR
        )  # Decode the image as color (BGR)

        if img is not None:
            # Convert from BGR to RGB
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return rgb_img
    except Exception as e:
        # Handle any errors encountered during image loading and conversion
        print(f"Error loading image from buffer in RGB format: {e}")
        return None


# Load BGR image from buffer
def load_buffer_bgr(buffer_data: BytesIO):
    """
    Loads an image from buffer data and returns it in BGR mode.

    Args:
        buffer_data (bytes): The image data as a bytes object.

    Returns:
        numpy.ndarray: The loaded image in BGR mode as a numpy array, or None if the image fails to load.
    """
    try:
        # Read the image from buffer data (NumPy array)
        img_array = np.asarray(bytearray(buffer_data), dtype=np.uint8)
        # Decode the image as color (BGR)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        return img
    except Exception as e:
        # Handle any errors encountered during image loading
        print(f"Error loading image from buffer in BGR format: {e}")
        return None


# Load gray image from buffer
def load_buffer_gray(buffer_data: BytesIO):
    """
    Loads an image from buffer data and converts it to grayscale.

    Args:
        buffer_data (bytes): The image data as a bytes object.

    Returns:
        numpy.ndarray: The loaded image in grayscale as a numpy array, or None if the image fails to load.
    """
    try:
        # Read the image from buffer data (NumPy array)
        img_array = np.asarray(bytearray(buffer_data), dtype=np.uint8)
        # Decode the image as grayscale
        img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)

        return img
    except Exception as e:
        # Handle any errors encountered during image loading and conversion
        print(f"Error loading image from buffer in grayscale format: {e}")
        return None


# Save Image (Format)
def save_image(image: np.ndarray, filename: str, file_format: str = ".jpg") -> None:
    """
    Saves the image to the specified file and format.

    Args:
        image (numpy.ndarray): Input image
        filename (str): Name of the file (without extension)
        file_format (str, optional): Image format (default is .jpg)

    Raises:
        ValueError: If the input image is None, filename is empty, or the format is invalid.
    """
    if image is None:
        raise ValueError("Input image is None. Cannot save.")
    if not filename:
        raise ValueError("Filename must be specified.")
    valid_formats = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
    if file_format not in valid_formats:
        raise ValueError(
            f"Invalid format '{file_format}'. Supported formats: {valid_formats}"
        )

    save_path = f"{filename}{file_format}"
    cv2.imwrite(save_path, image)


def resize(
    image: np.ndarray,
    size: Optional[Union[int, Tuple[int, int]]] = None,
    is_mask: bool = False,
) -> np.ndarray:
    """
    Resizes an image while optionally preserving the aspect ratio.

    Args:
        image (numpy.ndarray): The input image.
        size (Optional[Union[int, Tuple[int, int]]]):
            - If an integer is provided, the image is resized while maintaining the aspect ratio.
            - If a tuple `(width, height)` is provided, the image is resized to exact dimensions.
            - If None, the image remains unchanged.
        is_mask (bool, optional):
            - If True, uses nearest-neighbor interpolation (suitable for masks).
            - Otherwise, uses area interpolation (better for shrinking images smoothly).

    Returns:
        numpy.ndarray: The resized image.

    Raises:
        TypeError: If `size` is not an integer, tuple, or None.
    """
    if size is None:
        return image

    # Handle aspect ratio preservation when size is an integer
    if isinstance(size, int):
        # Calculate the target dimensions while preserving the aspect ratio
        original_height, original_width = image.shape[:2]
        aspect_ratio = original_width / original_height

        if aspect_ratio > 1:  # Landscape image
            # Calculate the target width
            target_width = size
            # Calculate the target height using the aspect ratio
            target_height = int(target_width / aspect_ratio)
        else:  # Portrait or square image
            # Calculate the target height
            target_height = size
            # Calculate the target width using the aspect ratio
            target_width = int(target_height * aspect_ratio)

    elif isinstance(size, tuple) and len(size) == 2:
        # Use the provided width and height
        target_width, target_height = size

    else:
        raise TypeError(
            "size must be either an integer, a tuple (width, height), or None."
        )

    # Choose interpolation method
    interpolation = cv2.INTER_NEAREST if is_mask else cv2.INTER_AREA

    # Resize the image
    return cv2.resize(image, (target_width, target_height), interpolation=interpolation)


# Crop Image
def crop(image: np.ndarray, x: int, y: int, width: int, height: int):
    """
    Crop the image to the specified region.

    Args:
        image (numpy.ndarray): Input image
        x (int): Starting x coordinate
        y (int): Starting y coordinate
        width (int): Width of the crop
        height (int): Height of the crop

    Returns:
        numpy.ndarray: Cropped image

    Raises:
        ValueError: If the input image is None, coordinates or dimensions are invalid, or the crop exceeds the image bounds
    """
    try:
        if image is None:
            raise ValueError("Input image is None. Cannot crop.")
        if x < 0 or y < 0 or width <= 0 or height <= 0:
            raise ValueError(
                "Invalid crop dimensions. Coordinates and dimensions must be positive."
            )
        if y + height > image.shape[0] or x + width > image.shape[1]:
            raise ValueError("Crop dimensions exceed image bounds.")

        # Extract the region of interest using NumPy array slicing
        cropped_image = image[y : y + height, x : x + width]
        return cropped_image
    except Exception as e:
        print(f"Error cropping image: {e}")


# Rotate Image
def rotate(image: np.ndarray, angle: int):
    """
    Rotate the image by the specified angle.

    Args:
        image (numpy.ndarray): Input image
        angle (int): Rotation angle in degrees

    Returns:
        numpy.ndarray: Rotated image

    Raises:
        ValueError: If the input image is None
        Exception: If an error occurs during rotation
    """
    try:
        if image is None:
            raise ValueError("Input image is None. Cannot rotate.")
        # Get the image dimensions
        (h, w) = image.shape[:2]
        # Calculate the center of the image
        center = (w // 2, h // 2)
        # Create the rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        # Rotate the image using the rotation matrix
        rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h))
        return rotated_image
    except Exception as e:
        print(f"Error rotating image: {e}")


# Flip Image
def flip(image: np.ndarray, flip_code: int):
    """
    Flip the image according to the specified flip code.

    Args:
        image (numpy.ndarray): Input image
        flip_code (int): Flip code (0 = vertical, 1 = horizontal, -1 = both)

    Returns:
        numpy.ndarray: Flipped image

    Raises:
        ValueError: If the input image is None or the flip code is invalid
        Exception: If an error occurs during the flip operation
    """
    try:
        if image is None:
            raise ValueError("Input image is None. Cannot flip.")
        if flip_code not in [0, 1, -1]:
            raise ValueError(
                "Invalid flip code. Use 0 (vertical), 1 (horizontal), or -1 (both)."
            )

        # Flip the image according to the specified flip code
        flipped_image = cv2.flip(image, flip_code)

        return flipped_image
    except Exception as e:
        print(f"Error flipping image: {e}")


# Color Space Conversion
def convert_color_space(image: np.ndarray, color_space: str):
    """
    Convert the image to the specified color space.

    Args:
        image (numpy.ndarray): Input image as a NumPy array (BGR format).
        color_space (str): Target color space (e.g., 'GRAY', 'RGB', 'HSV').

    Returns:
        numpy.ndarray: Converted image in the specified color space.

    Raises:
        ValueError: If the input image is None or the color space is unsupported.
        Exception: If an error occurs during the conversion.
    """
    try:
        if image is None:
            raise ValueError("Input image is None. Cannot convert color space.")
        if color_space == "GRAY":
            # Convert the image to grayscale
            converted_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif color_space == "RGB":
            # Convert the image to RGB format
            converted_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif color_space == "HSV":
            # Convert the image to HSV format
            converted_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        else:
            raise ValueError("Unsupported color space. Use 'GRAY', 'RGB', or 'HSV'.")

        return converted_image
    except Exception as e:
        print(f"Error converting color space: {e}")


def crop_with_pad(
    image: np.ndarray,
    left: int,
    top: int,
    right: int,
    bottom: int,
    pad: float = 0.0,
) -> np.ndarray:
    """
    Crop a region from the image while retaining the image area and applying padding strictly within the bounds of the image.

    Args:
        image (np.ndarray): The input image as a NumPy array (BGR format).
        left (int): The x-coordinate of the left edge of the crop rectangle.
        top (int): The y-coordinate of the top edge of the crop rectangle.
        right (int): The x-coordinate of the right edge of the crop rectangle.
        bottom (int): The y-coordinate of the bottom edge of the crop rectangle.
        pad (float): Padding percentage to add to the crop area (value between 0 and 1).

    Returns:
        np.ndarray: The cropped image with padding applied within the image bounds.

    Raises:
        ValueError: If the pad value is not between 0 and 1.
    """
    # Validate the padding value
    if not (0 <= pad <= 1):
        raise ValueError("Pad must be a value between 0 and 1.")

    # Get image dimensions
    img_height, img_width = image.shape[:2]

    # Calculate the dimensions of the crop area
    crop_width = right - left
    crop_height = bottom - top

    # Calculate padding in pixels
    pad_x = int(crop_width * pad)
    pad_y = int(crop_height * pad)

    # Adjust crop coordinates to include padding, ensuring they remain within image bounds
    left = max(0, left - pad_x)
    top = max(0, top - pad_y)
    right = min(img_width, right + pad_x)
    bottom = min(img_height, bottom + pad_y)

    # Crop the image within the adjusted coordinates
    cropped_image = image[top:bottom, left:right]

    return cropped_image


def ROI_blur(image: np.ndarray, roi: tuple, blur: int) -> np.ndarray:
    """
    Apply a blur to a specific region of interest (ROI) in the image.

    Args:
        image (np.ndarray): Input image (H, W, C).
        roi (tuple): The region of interest as (x, y, width, height).
        blur (int): Kernel size for the blur (single integer, will be adjusted to the nearest odd number).

    Returns:
        np.ndarray: Image with the specified ROI blurred.
    """
    # Ensure blur is an odd number
    # The blur kernel size should be odd to ensure symmetric blur
    if blur % 2 == 0:
        blur += 1

    # Convert blur into a tuple for the kernel size
    # The kernel size is specified as (width, height), so we duplicate the blur value
    blur_ksize = (blur, blur)

    # Extract the ROI coordinates from the tuple
    x, y, w, h = roi

    # Extract the ROI from the image using NumPy array slicing
    roi_region = image[y : y + h, x : x + w]

    # Apply the blur to the ROI using OpenCV's GaussianBlur
    blurred_roi = cv2.GaussianBlur(roi_region, blur_ksize, 0)

    # Copy the blurred ROI back to the original image using NumPy array assignment
    # We use the copy() method to ensure the original image is not modified
    result = image.copy()
    result[y : y + h, x : x + w] = blurred_roi

    return result
