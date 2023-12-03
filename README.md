# Person Attribute Recognition with Deep Learning

## Overview

This repository contains a PyTorch implementation of a person attribute recognition model. The model has been trained to recognize various attributes such as age, gender, hair length, upper body features, lower body features, and accessories.

## Model Details

The model is trained to recognize the following attributes:

- Age
  - Young
  - Adult
  - Old

- Gender
  - Female

- Hair Length
  - Short
  - Long
  - Bald

- Upper Body Features
  - Length
    - Short
  - Color
    - Black
    - Blue
    - Brown
    - Green
    - Grey
    - Orange
    - Pink
    - Purple
    - Red
    - White
    - Yellow
    - Other

- Lower Body Features
  - Length
    - Short
  - Color
    - Black
    - Blue
    - Brown
    - Green
    - Grey
    - Orange
    - Pink
    - Purple
    - Red
    - White
    - Yellow
    - Other
  - Type
    - Trousers & Shorts
    - Skirt & Dress

- Accessories
  - Backpack
  - Bag
  - Glasses
    - Normal
    - Sun
  - Hat

## Usage

### 1. Installation

Clone the repository and navigate to the project directory:

```bash
git clone https://github.com/dsabarinathan/attribute-recognition.git
cd attribute-recognition
```

### 2. Download Pre-trained Model

Download the pre-trained model weights file from the releases section of this repository and place it in the `models/` directory.

### 3. Install Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

### 4. Run Inference

Use the provided script to perform attribute recognition on an input image:

```bash
python inference.py --image_path path/to/your/image.jpg
```

Replace `path/to/your/image.jpg` with the path to the image you want to analyze.

## Contributing

We welcome contributions! If you find any issues or have suggestions for improvements, please create an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

