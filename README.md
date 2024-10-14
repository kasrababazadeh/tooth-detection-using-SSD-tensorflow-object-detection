# Tooth Detection using TensorFlow Object Detection API

## Overview

This project implements a tooth detection model using the TensorFlow Object Detection API. The goal is to detect and classify teeth in images, facilitating automatic analysis in dental diagnostics. The model is trained and tested using a small dataset comprising 24 bitewing images, which contain around 180 teeth.

## Table of Contents

- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Training the Model](#training-the-model)
- [Evaluating the Model](#evaluating-the-model)
- [Testing the Model](#testing-the-model)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Installation

To get started, clone this repository and install the required dependencies. This project was developed in a Google Colab environment but can be run locally with minor adjustments.

### Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY
```

### Install TensorFlow and Dependencies

This project requires TensorFlow 2.x and other libraries. You can install them using:

```bash
pip install tensorflow==2.13.0
pip install matplotlib
```

## Dataset Preparation

The dataset consists of 24 bitewing images containing approximately 180 teeth, along with corresponding labels in CSV format.

1. Place your dataset in the following structure:

```bash
object_detection_dataset/
├── train/
│   └── [image files]
└── test/
    └── [image files]
```

2. Create the TFRecord files using the provided generate_tfrecord.py script:

```bash
python generate_tfrecord.py --csv_input=/path/to/train.csv --image_dir=/path/to/train --output_path=/path/to/train.record
python generate_tfrecord.py --csv_input=/path/to/test.csv --image_dir=/path/to/test --output_path=/path/to/test.record
```

## Training the Model

To train the model, you need to configure the training parameters. Modify the configuration file according to your dataset and model requirements. The configuration file can be found in the configs folder of the TensorFlow models repository.

### Training Command

Run the training process using the following command:

```bash
python models/research/object_detection/model_main_tf2.py \
    --pipeline_config_path=/path/to/model_config.config \
    --model_dir=/path/to/model_directory \
    --num_train_steps=8000 \
    --num_eval_steps=1000
```

## Evaluating the Model

To evaluate the model, use the following command:

```bash
python models/research/object_detection/model_main_tf2.py \
    --pipeline_config_path=/path/to/model_config.config \
    --model_dir=/path/to/model_directory \
    --checkpoint_dir=/path/to/checkpoint_directory \
    --eval_once
```

## Testing the Model

You can test the trained model on new images using the inference script. Ensure to load the model and visualize the results:

```bash
# Load your image and run inference
image_path = 'path/to/test/image.jpg'
image_np = load_image_into_numpy_array(image_path)
output_dict = run_inference_for_single_image(model, image_np)
```

## Results

The model achieves the following metrics on the test dataset:
- Average Precision (AP) at IoU=0.50:0.95: 0.864
- Average Recall (AR) at IoU=0.50:0.95: 0.892
For more detailed results, refer to the output generated during evaluation.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
