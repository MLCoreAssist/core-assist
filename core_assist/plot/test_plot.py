import cv2
import numpy as np
import pytest
from core_assist.plot.plot import (bbox, display_multiple_images,
                                   generate_color_for_text, image, segment,
                                   segments)
from matplotlib import pyplot as plt


@pytest.fixture
def sample_image():
    """
    Fixture to create a sample image for testing.

    The image is a 100x100 black square with a white rectangle from (20, 20)
    to (80, 80).
    """
    # Create a simple image (a black square with a white rectangle)
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.rectangle(img, (20, 20), (80, 80), (255, 255, 255), -1)  # White rectangle
    return img


@pytest.fixture
def sample_masks():
    """
    Fixture to create sample binary masks for testing.

    This fixture generates a binary mask with a white rectangle
    on a black background for the purpose of testing image processing
    functions.
    """
    # Create a binary mask with a black background
    mask = np.zeros((100, 100), dtype=np.uint8)

    # Draw a white rectangle on the mask
    cv2.rectangle(mask, (20, 20), (80, 80), 1, -1)

    # Return the mask as a list
    return [mask]


@pytest.fixture
def sample_bboxes():
    """
    Fixture to create sample bounding boxes for testing.

    This fixture provides a list of tuples, where each tuple represents a
    bounding box in the format (x, y, width, height) for the purpose of
    testing bounding box plotting functions.

    The fixture currently provides a single bounding box with the
    coordinates (20, 20, 60, 60), which is a rectangle with the top-left
    corner at (20, 20) and a width and height of 60 pixels.
    """
    # Create bounding box (x, y, width, height)
    return [(20, 20, 60, 60)]  # A single bounding box


@pytest.fixture
def sample_labels():
    """
    Fixture to create sample labels for testing.

    This fixture provides a list containing a single label, "Object", for
    the purpose of testing bounding box and mask plotting functions.
    """
    return ["Object"]


def test_generate_color_for_text():
    """
    Test that the generate_color_for_text function generates consistent
    and visually distinct colors for different text inputs.

    This test checks that the same text input results in the same color
    output, and that different text inputs result in different color
    outputs.
    """
    text1 = "Object1"
    text2 = "Object2"

    color1 = generate_color_for_text(text1)
    color2 = generate_color_for_text(text2)

    # Check that the colors are consistent for the same text
    assert generate_color_for_text(text1) == color1
    assert generate_color_for_text(text2) == color2
    # Check that different text results in different colors
    assert color1 != color2


def test_image_plot(sample_image):
    """
    Test the image function to display a single image.

    This test checks that the image function can display a single image
    using Matplotlib.

    Args:
        sample_image (np.ndarray): Sample image to display.
    """
    # Call the image function with the sample image
    image(img=sample_image)


def test_display_multiple_images(sample_image):
    """
    Test the display_multiple_images function.

    This test checks that multiple images can be displayed with
    corresponding labels, arranged in specified rows and columns.

    Args:
        sample_image (np.ndarray): Sample image to be displayed.

    """
    # Display two copies of the sample image with labels
    # in a single row and two columns
    display_multiple_images(
        [sample_image, sample_image], labels=["Image 1", "Image 2"], rows=1, cols=2
    )


def test_bbox_plot(sample_image, sample_bboxes, sample_labels):
    """
    Test the bbox function with a sample image.

    This test overlays bounding boxes on the image and checks that the
    bounding boxes are displayed correctly.

    Args:
        sample_image (np.ndarray): Sample image to overlay bounding boxes.
        sample_bboxes (list): List of sample bounding boxes.
        sample_labels (list): List of sample labels for the bounding boxes.
    """
    bbox(sample_image, sample_bboxes, labels=sample_labels)


def test_segment_plot(sample_image, sample_masks, sample_labels, sample_bboxes):
    """
    Test the segment function to overlay masks and bounding boxes.

    Args:
        sample_image (np.ndarray): Sample image to overlay masks and bounding boxes.
        sample_masks (list): List of sample binary masks.
        sample_labels (list): List of sample labels for the masks.
        sample_bboxes (list): List of sample bounding boxes.

    """
    # Call the segment function with the sample data
    segment(
        sample_image,
        sample_masks,
        sample_labels,
        bboxes=sample_bboxes,
        bbox_labels=sample_labels,
    )


def test_segments_plot(sample_image, sample_masks, sample_labels, sample_bboxes):
    """
    Test the segments function with multiple images, masks, and bounding boxes.

    Args:
        sample_image: Fixture providing a sample image.
        sample_masks: Fixture providing sample binary masks.
        sample_labels: Fixture providing sample labels for the masks.
        sample_bboxes: Fixture providing sample bounding boxes.
    """
    # Call the segments function with the sample data
    segments(
        [sample_image],  # List of images to process
        [sample_masks],  # List of masks corresponding to the images
        [sample_labels],  # List of labels for each mask
        bboxes=[sample_bboxes],  # List of bounding boxes for the images
        bbox_labels=[sample_labels],  # List of labels for each bounding box
    )
