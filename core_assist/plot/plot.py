import concurrent.futures
import hashlib
import random
from typing import List, Literal, Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np


def image(**image_dict):
    """
    Display multiple images side by side in a single row.

    Args:
        **image_dict: Keyword arguments where each key is the image title (str)
                      and each value is a NumPy array representing the image.

    Behavior:
    - Automatically determines the number of images to display.
    - Converts images from BGR to RGB if necessary.
    - Removes axis ticks for a cleaner visualization.
    - Titles are formatted by replacing underscores with spaces and capitalizing words.
    """
    n = len(image_dict)  # Number of images to display
    plt.figure(figsize=(7 * n, 5))  # Set figure size based on number of images

    for i, (name, img) in enumerate(image_dict.items()):
        plt.subplot(1, n, i + 1)  # Create subplot for each image
        plt.xticks([])  # Remove x-axis ticks
        plt.yticks([])  # Remove y-axis ticks
        plt.title(" ".join(name.split("_")).title())  # Format title

        # Display the image, converting from BGR to RGB if necessary
        if len(img.shape) == 3 and img.shape[2] == 3:
            plt.imshow(img)  # Image is already in RGB format
        else:
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # Convert to RGB

    plt.show()  # Display all images


def load_and_process_image(img: Union[str, np.ndarray], rgb_flag: bool) -> np.ndarray:
    """
    Load and process an image from a file path or NumPy array.

    Args:
        img (str or np.ndarray): Image file path or NumPy array.
        rgb_flag (bool): Whether to convert BGR images to RGB.

    Returns:
        np.ndarray: Processed image.
    """

    # Check if the image is a file path
    if isinstance(img, str):
        # Read the image using OpenCV
        img = cv2.imread(img, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Image file not found: {img}")

        # Convert from BGR to RGB if needed
        if len(img.shape) == 3 and rgb_flag:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Check if the image is a NumPy array
    elif isinstance(img, np.ndarray):
        # Check if the image has the correct dimensions
        if img.ndim not in [2, 3] or (img.ndim == 3 and img.shape[2] not in [1, 3]):
            raise ValueError("Each image must be a 2D grayscale or 3D RGB image.")
    else:
        # Raise an error if the image is of an unsupported type
        raise TypeError("Each image must be a file path (str) or a NumPy array.")

    return img


def display_multiple_images(
    images: List[Union[str, np.ndarray]],
    labels: List[str] = None,
    rows: int = 1,
    cols: int = 1,
    rgb_flag: bool = False,
) -> None:
    """
    Display multiple images in a grid layout with multi-threaded loading.

    Args:
        images (List[Union[str, np.ndarray]]): List of image file paths or NumPy arrays.
        labels (List[str], optional): List of labels for each image.
        rows (int): Number of rows in the grid.
        cols (int): Number of columns in the grid.
        rgb_flag (bool): If True, converts BGR images to RGB before displaying.

    Raises:
        ValueError: If the number of images does not match the number of labels.
        ValueError: If the grid size is too small for the number of images.

    Returns:
        None
    """
    if labels and len(images) != len(labels):
        raise ValueError("The number of images must match the number of labels.")

    n = len(images)
    if rows * cols < n:
        raise ValueError(
            f"Grid too small! {rows}x{cols} < {n} images. Adjust grid size."
        )

    # Use multi-threading for fast image loading
    with concurrent.futures.ThreadPoolExecutor() as executor:
        images = list(
            executor.map(lambda img: load_and_process_image(img, rgb_flag), images)
        )

    # Create figure
    plt.figure(figsize=(7 * cols, 5 * rows))

    for i, img in enumerate(images):
        plt.subplot(rows, cols, i + 1)
        plt.xticks([])  # Hide x-axis ticks
        plt.yticks([])  # Hide y-axis ticks
        plt.title(labels[i] if labels else "")  # Set title if labels are provided

        if img.ndim == 2:  # Grayscale
            plt.imshow(img, cmap="gray")  # Display grayscale image
        else:  # Color image
            plt.imshow(img)  # Display color image

    plt.tight_layout()  # Adjust layout so that images don't overlap
    plt.show()  # Display all images


def generate_color_for_text(text: str) -> tuple:
    """
    Generate a visually distinct and consistent color for a given text using hashing.

    This function takes a string and produces a color tuple in BGR format that is
    visually distinct from other generated colors. The color generation is based on
    the hashing of the input text, which ensures that the same input will always
    result in the same color.

    Args:
        text: The input text as a string.

    Returns:
        tuple: A distinct color tuple in BGR format with three integers (0-255).
    """
    hash_value = int(hashlib.md5(text.encode()).hexdigest(), 16)

    # Use hash to determine Hue (0-179 for OpenCV HSV)
    hue = hash_value % 180

    # Use shifts for Saturation & Value to keep colors vibrant
    saturation = 200 + (hash_value % 56)  # Range: 200-255
    value = 200 + ((hash_value >> 8) % 56)  # Range: 200-255

    # Convert HSV to BGR
    # Create a 3D NumPy array with a single value for Hue, Saturation, and Value
    color_hsv = np.uint8([[[hue, saturation, value]]])

    # Convert the HSV color to BGR (Blue-Green-Red) format
    color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0]

    # Return the BGR color as a tuple of three integers
    return tuple(int(c) for c in color_bgr)


def bbox(
    img: np.ndarray,
    bboxes: List[Tuple[int, int, int, int]],
    labels: Optional[List[str]] = None,
    confs: Optional[List[float]] = None,
    pad: float = 0.0,
    text_color: Tuple[int, int, int] = (255, 255, 255),
    thickness: int = 2,
    font_scale: float = 0.5,
    ret: bool = False,
) -> Optional[np.ndarray]:
    """
    Draw multiple bounding boxes with text on an image, ensuring consistent color for the same text.

    Args:
        img (np.ndarray): The input image.
        bboxes (List[Tuple[int, int, int, int]]): List of bounding boxes as (x, y, width, height).
        labels (Optional[List[str]]): List of labels corresponding to each bounding box. Defaults to None.
        confs (Optional[List[float]]): List of confidence values for each bounding box. Defaults to None.
        pad (float): Padding percentage (e.g., 0.1 for 10% padding). Defaults to 0 (no padding).
        text_color (Tuple[int, int, int]): Color of the text in BGR format. Defaults to white.
        thickness (int): Thickness of the bounding box lines. Defaults to 2.
        font_scale (float): Scale of the text. Defaults to 0.5.
        ret (bool): If true, returns the image with bounding boxes; otherwise, plots the original image and bounding box image.

    Returns:
        Optional[np.ndarray]: Image with bounding boxes if ret is True, otherwise None.
    """
    if labels and len(bboxes) != len(labels):
        raise ValueError(
            "The number of bounding boxes must match the number of labels."
        )
    if confs and len(bboxes) != len(confs):
        raise ValueError(
            "The number of bounding boxes must match the number of confidence values."
        )

    # Create a copy of the original image for drawing bounding boxes
    bbox_image = img.copy()

    # Iterate over each bounding box
    for i, bbox in enumerate(bboxes):
        x, y, w, h = bbox
        x, y, w, h = int(x), int(y), int(w), int(h)

        # Apply padding if specified
        if pad > 0:
            pad_w = int(w * pad)
            pad_h = int(h * pad)
            x = max(0, x - pad_w)
            y = max(0, y - pad_h)
            w = w + 2 * pad_w
            h = h + 2 * pad_h

        # Generate a random color if no label is provided
        label = labels[i] if labels else None
        if label is None:
            color = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255),
            )
        else:
            color = generate_color_for_text(label)

        # Draw the bounding box
        cv2.rectangle(bbox_image, (x, y), (x + w, y + h), color, thickness)

        if label:
            # Format text with confidence if provided
            conf = confs[i] if confs else None
            display_text = f"{label}: {conf:.2f}" if conf is not None else label

            # Get text size
            (text_width, text_height), baseline = cv2.getTextSize(
                display_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )

            # Draw a filled rectangle for the text background
            cv2.rectangle(
                bbox_image,
                (x, y - text_height - baseline),
                (x + text_width, y),
                color,
                -1,
            )

            # Put the text on the image
            cv2.putText(
                bbox_image,
                display_text,
                (x, y - baseline),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                text_color,
                thickness,
            )

    # Return the image with bounding boxes if ret is True
    if ret:
        return bbox_image

    # Otherwise, plot the original image and bounding box image
    else:
        image(image=img, Bounding_box=bbox_image)


def segment(
    img: Union[str, np.ndarray],
    masks: List[Union[str, np.ndarray]],
    mask_labels: List[str],
    confs: Optional[List[float]] = None,
    bbox_flag: bool = False,
    pad_bbox: float = 0.0,
    text_color: Tuple[int, int, int] = (255, 255, 255),
    thickness: int = 2,
    font_scale: float = 1,
    segment_type: Literal["both", "mask", "bbox"] = "both",
    ret: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform segmentation using multiple masks, combine them into a single mask with labels,
    and draw bounding boxes (if `bbox_flag=True`) by calculating them from the masks.

    Args:
        img (str or np.ndarray): Path to the input image or a numpy array (color image).
        masks (list of str or np.ndarray): List of paths to mask images or numpy arrays (grayscale masks).
        mask_labels (list of str): List of labels corresponding to each mask.
        confs (list of float, optional): List of confidence scores for each mask label. Defaults to None.
        bbox_flag (bool): If True, calculate bounding boxes from masks and overlay them.
        pad_bbox (float): Padding percentage for the bounding boxes. Defaults to 0 (no padding).
        text_color (tuple): Color of the text in BGR format. Defaults to white.
        thickness (int): Thickness of the bounding box lines. Defaults to 2.
        font_scale (float): Scale of the text. Defaults to 1.
        segment_type (str): Type of segmentation ("outline", "filled", "both"). Defaults to "both".
        ret (bool): Whether to return the processed images.

    Returns:
        If `ret` is True, returns (segmented image with bbox, combined mask).
    """

    # Validate segment_type
    assert segment_type in [
        "outline",
        "filled",
        "both",
    ], "Invalid segment_type. Choose from 'outline', 'filled', or 'both'."

    # Read the image
    if isinstance(img, str):
        img = cv2.imread(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif not isinstance(img, np.ndarray):
        raise ValueError("Invalid image input. Provide a file path or numpy array.")

    # Ensure masks and labels lengths match
    if len(masks) != len(mask_labels):
        raise ValueError("The number of masks must match the number of mask labels.")

    if confs and len(confs) != len(mask_labels):
        raise ValueError(
            "The number of confidence scores must match the number of mask labels."
        )

    # Load masks and compute their areas
    mask_areas = []
    loaded_masks = []

    for mask in masks:
        if isinstance(mask, str):
            mask = cv2.imread(mask, 0)  # Load as grayscale
        elif not isinstance(mask, np.ndarray):
            raise ValueError("Invalid mask input. Provide a file path or numpy array.")

        mask_area = np.sum(mask > 0)  # Count nonzero pixels
        mask_areas.append(mask_area)
        loaded_masks.append(mask)

    # Sort masks by area in descending order
    sorted_indices = np.argsort(mask_areas)[::-1]
    sorted_masks = [loaded_masks[i] for i in sorted_indices]
    sorted_labels = [mask_labels[i] for i in sorted_indices]
    sorted_confs = [
        confs[i] if confs else None for i in sorted_indices
    ]  # Sort confs if provided

    # Initialize the segmented image and combined mask
    segmented_img_with_bbox = img.copy()
    combined_mask = np.zeros_like(img)

    # Store calculated bounding boxes if bbox_flag is True
    calculated_bboxes = []
    calculated_bbox_labels = []

    # Process each sorted mask
    for i, mask in enumerate(sorted_masks):
        conf = sorted_confs[i]
        base_label = sorted_labels[i]  # Label without confidence
        mask_label = f"{base_label} ({conf:.2f})" if conf is not None else base_label

        # Generate a unique color using only the base label (without confidence)
        color = generate_color_for_text(base_label)

        # Darker version of the color for outline
        dark_color = tuple(max(0, c - 50) for c in color)  # Make the outline darker

        # Create a colored version of the mask for visualization
        color_mask = np.zeros_like(img)
        color_mask[mask > 0] = color

        # If segment_type is 'filled' or 'both', fill the mask
        if segment_type in ["filled", "both"]:
            combined_mask = cv2.addWeighted(combined_mask, 1, color_mask, 0.35, 0)
            segmented_img_with_bbox = cv2.addWeighted(
                segmented_img_with_bbox, 1, color_mask, 0.35, 0
            )

        # Find mask contours
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # If segment_type is 'outline' or 'both', draw the outline
        if segment_type in ["outline", "both"]:
            cv2.drawContours(
                segmented_img_with_bbox, contours, -1, dark_color, thickness
            )

        # Add text for the mask label on the segmented image
        mask_position = np.where(mask > 0)
        if len(mask_position[0]) > 0:
            text_x = mask_position[1].min()
            text_y = max(
                mask_position[0].min() - 10, 10
            )  # Ensure text stays within bounds
            cv2.putText(
                segmented_img_with_bbox,
                mask_label,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                dark_color,
                thickness,
            )

        # Calculate bounding box from the mask if bbox_flag is True
        if bbox_flag:
            x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))  # Get bbox from mask
            calculated_bboxes.append((x, y, w, h))
            calculated_bbox_labels.append(
                base_label
            )  # Store bbox label with confidence

    # Process bounding boxes
    if bbox_flag:
        segmented_img_with_bbox = bbox(
            segmented_img_with_bbox,
            calculated_bboxes,
            calculated_bbox_labels,
            confs=None,
            pad=pad_bbox,
            text_color=text_color,
            thickness=thickness,
            font_scale=font_scale,
            ret=True,
        )

    if ret:
        return segmented_img_with_bbox, combined_mask
    else:
        image(image=img, mask=combined_mask, segmented=segmented_img_with_bbox)
