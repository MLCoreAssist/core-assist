import hashlib
import os
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError


def extension(
    file_paths: List[str], valid_formats: List[str] = [".jpg", ".png", ".jpeg"]
) -> List[bool]:
    """
    Validate that the file has a valid image format.

    Args:
        file_paths (List[str]): List of paths to the files to check.
        valid_formats (List[str], optional): List of valid file extensions. Defaults to ['.jpg', '.png', '.jpeg'].

    Returns:
        List[bool]: List of bools indicating if the file has a valid format.
    """
    res = []
    try:
        for file_path in file_paths:
            _, ext = os.path.splitext(file_path)
            res.append(ext.lower() in valid_formats)

        return res
    except Exception as e:
        print(f"Error validating file extension: {e}")
        return False


def path(file_path: str) -> bool:
    """
    Validate if the file path exists and is a file.

    Args:
        file_path (str): The path to the file to check.

    Returns:
        bool: True if the file path exists and is a file, False otherwise.
    """
    try:
        return os.path.isfile(file_path)
    except Exception as e:
        print(f"Error validating file path: {e}")
        return False


def image(file_path: str) -> bool:
    """
    Check if the image is corrupt or invalid.

    Args:
        file_path (str): The path to the image file.

    Returns:
        bool: True if the image is valid, False otherwise.
    """
    try:
        image = cv2.imread(file_path)
        return image is not None
    except Exception as e:
        print(f"Error validating image: {e}")
        return False


def hash_file(file_path: str) -> Optional[str]:
    """
    Generate a hash for the image file to detect duplicates.

    Args:
        file_path (str): The path to the image file.

    Returns:
        Optional[str]: The hash of the file as a string, or None if an error occurs.
    """
    try:
        with open(file_path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    except Exception as e:
        print(f"Error generating hash: {e}")
        return None


def find_content_duplicate(dir: str) -> list:
    """
    Detect duplicate images in the directory using hash comparison.

    Args:
        dir (str): The directory path to search for duplicate images.

    Returns:
        list: A list of duplicate file names.
    """
    try:
        hashes = {}
        duplicates = []
        for file_name in os.listdir(dir):
            file_path = os.path.join(dir, file_name)
            if os.path.isfile(file_path):
                file_hash = hash_file(file_path)
                if file_hash is None:
                    continue
                if file_hash in hashes:
                    duplicates.append(file_name)  # Append the duplicate file name
                else:
                    hashes[file_hash] = (
                        file_name  # Store the hash and its corresponding file
                    )
        return duplicates
    except Exception as e:
        print(f"Error finding content duplicates: {e}")
        return []


from typing import List


def find_dir_common_files(dir1: str, dir2: str) -> List[str]:
    """
    Find common file names between two directories.

    Args:
        dir1 (str): Path to the first directory.
        dir2 (str): Path to the second directory.

    Returns:
        List[str]: List of common file names.
    """
    try:
        if not os.path.isdir(dir1) or not os.path.isdir(dir2):
            raise ValueError("One or both directories do not exist or are invalid.")

        # Get list of file names in both directories
        files_in_dir1 = set(os.listdir(dir1))
        files_in_dir2 = set(os.listdir(dir2))

        # Find common files
        common_files = list(files_in_dir1.intersection(files_in_dir2))

        return common_files

    except Exception as e:
        print(f"Error finding common files: {e}")
        return []


def image_size_and_dim(
    file_path: str, min_dim: Tuple[int, int] = (50, 50), max_size_mb: float = 5
) -> bool:
    """
    Check if the image meets the required dimensions and file size without fully loading the image.

    Args:
        file_path (str): The path to the image file.
        min_dim (Tuple[int, int], optional): Minimum dimensions (width, height) required for the image. Defaults to (50, 50).
        max_size_mb (float, optional): Maximum file size allowed in megabytes. Defaults to 5.

    Returns:
        bool: True if the image meets the required dimensions and size, False otherwise.
    """
    try:
        with Image.open(file_path) as image:
            width, height = image.size
        size_mb = os.path.getsize(file_path) / (1024 * 1024)  # Get file size in MB
        return width >= min_dim[0] and height >= min_dim[1] and size_mb <= max_size_mb
    except UnidentifiedImageError:
        print(f"Error: File at {file_path} is not a valid image.")
    except Exception as e:
        print(f"Error checking image dimensions and size: {e}")
    return False


def match_mask_size(image_path: str, mask_path: str) -> bool:
    """
    Check if the mask size matches the image size for segmentation without fully loading the image and mask.

    Args:
        image_path (str): The path to the image file.
        mask_path (str): The path to the mask file.

    Returns:
        bool: True if the mask size matches the image size, False otherwise.
    """
    try:
        with Image.open(image_path) as image, Image.open(mask_path) as mask:
            image.verify()
            mask.verify()
            return image.size == mask.size
    except UnidentifiedImageError:
        print(f"Error: Either {image_path} or {mask_path} is not a valid image.")
    except Exception as e:
        print(f"Error matching mask size: {e}")
    return False


def search_dir(dir: str, filename: str) -> bool:
    """
    Check if a file exists in the specified directory.

    Args:
        dir (str): The directory path to search in.
        filename (str): The name of the file to search for.

    Returns:
        bool: True if the file exists in the directory, False otherwise.
    """
    try:
        return os.path.isfile(os.path.join(dir, filename))
    except Exception as e:
        print(f"Error searching for file: {e}")
        return False


def search_csv(csv_path: str, column: str, to_search: str) -> bool:
    """
    Check if a value exists in a specific column of a CSV file.

    Args:
        csv_path (str): Path to the CSV file.
        column (str): The column name to search in.
        to_search (str): The value to search for in the specified column.

    Returns:
        bool: True if the value exists in the specified column, False otherwise.
    """
    try:
        df = pd.read_csv(csv_path)
        return (
            to_search in df[column].values
        )  # Check if value exists in specified column
    except FileNotFoundError:
        print(f"Error: CSV file at {csv_path} not found.")
    except KeyError:
        print(f"Error: Column '{column}' does not exist in the CSV file.")
    except Exception as e:
        print(f"Error searching CSV: {e}")
    return False


# def rectify_csv(df, img_paths):
#     """
#     Remove rows from the DataFrame where the 'img_path' column matches any path in the given list of image paths.

#     :param df: pandas.DataFrame - The input DataFrame containing a column named 'img_path'.
#     :param img_paths: list - A list of image paths to be removed from the DataFrame.
#     :return: pandas.DataFrame - A new DataFrame with the specified rows removed.
#     """
#     if "img_path" not in df.columns:
#         raise ValueError("The DataFrame must contain a column named 'img_path'.")

#     # Filter the DataFrame to exclude rows where 'img_path' matches any path in img_paths
#     filtered_df = df[~df["img_path"].isin(img_paths)]
#     return filtered_df
