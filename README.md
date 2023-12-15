# Facial Emotion Classification

## Overview

This repository contains the code and documentation for a Facial Emotion Classification project. The project explores the classification of human emotions using a unique Convolutional Neural Network (CNN) architecture. Departing from pre-trained models, the CNN is designed for enhanced precision and insights, achieving a 64.02% accuracy on unseen data.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Methods and Theory](#methods--theory)
- [Model Architecture](#model-architecture)
- [Implementation](#implementation)
- [Results](#results)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Introduction

Emotions are a crucial aspect of human expression, and this project aims to classify seven different emotions based on facial expressions using a bespoke CNN architecture. Unlike traditional transfer learning approaches, the model is constructed from the ground up, allowing for fine-tuning and experimentation.

## Dataset

The project utilizes the FER 2013 dataset, a freely available collection of grayscale images with seven emotion classes: anger, happiness, neutrality, surprise, fear, sadness, and disgust. Meticulous dataset partitioning and data augmentation techniques were implemented to enhance the model's exposure to real-world scenarios.

## Methods and Theory

The project employs various deep learning methodologies, including convolutional layers, pooling layers, batch normalization, ReLU activation, dropout layers, and fully connected layers. The model's architecture is experimented with different batch sizes and epochs to achieve optimal performance.

## Model Architecture

The CNN architecture comprises 38 layers, organized into convolutional blocks A and B, and a fully connected block. The model is designed to accurately classify facial expressions while addressing challenges in emotion classification.

## Implementation

The complete model is implemented in Python using the Keras API for the TensorFlow library. Google Cloud GPU-enabled Colab platform is utilized to accelerate the training process, reducing the training time to 30 minutes for 60 epochs.

## Results

The model achieves a 64.02% accuracy on unseen data, as evaluated through accuracy metrics, loss curves, confusion matrix, and heat maps. The results highlight the model's strengths in classifying certain emotions and areas for potential improvement.

## Usage

To use the code, follow these steps:

1. Clone the repository: `git clone https://github.com/your-username/facial-emotion-classification.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the main script: `python main.py`

## Contributing

Contributions are welcome! Please follow the [contribution guidelines](CONTRIBUTING.md) when submitting pull requests.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgements

- [Keras API](https://keras.io/api/)
- [Google Cloud Colab](https://colab.research.google.com/notebooks/gpu.ipynb)
- [FER 2013 Dataset](https://www.kaggle.com/datasets/msambare/fer2013)
- [References](#references) (See the References section in the project report for the complete list)
