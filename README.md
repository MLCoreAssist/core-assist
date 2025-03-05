<div align="center">

![Core Assist](assets/core_assist.png)

</div>

**Core Assist** is tipically designed to generalize and simplify the work of computer vision team by providing reusable, optimized and modular functions for common computer vision tasks.

<br/>

## Table of Contents

1. [Why Use Core Assist?](#why-use-core_assist)
2. [Installation](#installation)
3. [Features](#features)
4. [Usage](#usage)
5. [Documentation](#documentation)
6. [License](#license)

## Why Use core_assist?

**core_assist** simplifies repetitive tasks in computer vision workflows. It focuses on image processing  , plotting , validation and metric functions that often come up in various projects.

Instead of implementing these tasks from scratch for each project, simply use **core_assist** for clean and efficient code.

## Installation
Please check [install.md](install.md) for installation steps.

## Features 
| **Dataset** | **Metrics** | **Plot** | **Validate** | **Image Ops** |
|------------|------------|------------|------------|----------------------------|
| **Segmentation** | **Segmentation** | ✔ image | ✔ mask_size | ✔ load_rgb |
| ✔ SegDataset Class | ✔ Image Class | ✔ display_multiple_image | ✔ image_size_and_dim | ✔ load_bgr |
|   - export |   - plot | ✔ bbox | ✔ find_name_duplicate | ✔ load_grey |
|   - visualizer |   - get_stats | ✔ segment | ✔ find_content_duplicate | ✔ load_buffer_rgb |
|     -- show_image | ✔ DatasetStats |  | ✔ extension | ✔ load_buffer_bgr |
|     -- show_batch |   - get_stats |  | ✔ path | ✔ load_buffer_grey |
|     -- show_video |   - filter |  | ✔ search_csv | ✔ save |
|   - train_test_split |   - plot |  | ✔ search_dir | ✔ resize |
|   - bbox_scatter |   - plot_metric |  |  | ✔ crop |
|   - show_distribution |   - get_class_wise_result |  |  | ✔ rotate |
|   - label_df |   - get_dataset_result |  |  | ✔ flip |
|  |   - get_mAP_result |  |  | ✔ convert_color_space |
| **Detection** | **Detection** |  |  | ✔ blend |
| ✔ DetDataset Class | ✔ Image Class |  |  | ✔ add_padding |
|   - export |   - plot |  |  | ✔ detect_edges |
|   - visualizer |   - get_stats |  |  | ✔ extract_contours |
|     -- show_image | ✔ DatasetStats |  |  | ✔ blend_using_mask |
|     -- show_batch |   - get_stats |  |  |  |
|     -- show_video |   - filter |  |  |  |
|   - train_test_split |   - plot |  |  |  |
|   - bbox_scatter |   - plot_metric |  |  |  |
|   - show_distribution |   - get_class_wise_result |  |  |  |
|   - label_df |   - get_dataset_result |  |  |  |
|  |   - get_mAP_result |  |  |  |
| | **Classification** |  |  |  |
|  | ✔ Image Class |  |  |  |
|  |   - plot |  |  |  |
|  | ✔ DatasetStats |  |  |  |
|  |   - get_stats |  |  |  |
|  |   - filter |  |  |  |
|  |   - plot |  |  |  |
|  |   - plot_metric |  |  |  |
|  |   - get_class_wise_result |  |  |  |
|  |   - get_dataset_result |  |  |  |
|  |   - get_mAP_result |  |  |  |


## Usage

Here’s a quick overview of how to use **core_assist** in your project.

- **Image Operations**: For image-related operations, refer to the notebook [image_ops.ipynb](demos/image_ops.ipynb).
- **Plotting**: To explore various plotting techniques, check out [plot.ipynb](demos/plot.ipynb).
- **Validation**: For validation methods and practices, see [validate.ipynb](demos/validate.ipynb).
- **Metrics**: To understand and compute different metrics, refer to [metrics.ipynb](demos/metrics.ipynb).
- **Dataset**: For dataset-related tasks and manipulations, explore [dataset.ipynb](demos/Dataset.ipynb).

## Documentation

## License