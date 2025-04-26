# Satellite Image Normalizer

[GUI Screenshot](screenshot.png) *Example screenshot of the application*

A GUI application for normalizing satellite images' brightness and contrast, with special features for academic competitions.

## Features

- **Brightness Normalization**: Adjusts images to target brightness level
- **Histogram Matching**: Ensures consistent tonal distribution across images
- **Competition Mode**: Special processing for academic challenges with strict requirements
- **Image Comparison**: Side-by-side before/after comparison with metrics
- **Single Image Adjustment**: Fine-tune individual images post-normalization
- **ZIP Processing**: Directly process competition ZIP files

## Installation

1. **Prerequisites**:
   - Python 3.8+
   - pip package manager

2. **Install dependencies**:
   pip install -r requirements.txt

FILE STRUCTURE:

   satellite_image_normalizer/
│── main_GUI.py            # Main application script
│── background123.jpg      # Optional background image
├── output/                # Default output folder
├── modified_images/       # Stores manually adjusted images
└── competition_output/    # GLOBAL-BALANCED-SPECIFIC outputs