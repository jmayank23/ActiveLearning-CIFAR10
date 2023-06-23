# Active Learning with PyTorch and CIFAR-10

![collect_labels](https://github.com/jmayank23/ActiveLearning_CIFAR10/assets/27727185/de1e9c8d-7f09-40b3-86e1-e7cbbd0fa86a)
![end_of_active_learning](https://github.com/jmayank23/ActiveLearning_CIFAR10/assets/27727185/bfc188f1-0ce8-46cf-bb89-c988fa1cbc15)\
*Note: In the real setting, the ground truth label would not be displayed when collecting labels for fresh data.*

This repository contains an example of active learning using PyTorch and the CIFAR-10 dataset. It demonstrates how to use a simple active learning approach to improve the performance of a model by iteratively requesting new labels for instances where the model is least confident, then retraining the model with the newly labeled data.

## Overview

The active learning process is implemented in a loop, where each iteration consists of the following steps:

1. Train the model on the currently labeled set of data.
2. Use the trained model to make predictions on the unlabeled set.
3. Select instances from the unlabeled set where the model's prediction confidence is lowest.
4. Request new labels for these instances.
5. Add the newly labeled instances to the labeled set and remove them from the unlabeled set.
6. Repeat the process for a predetermined number of annotation rounds.

The model used for this process is a pre-trained ResNet-18 network from torchvision, which is trained using the cross-entropy loss and stochastic gradient descent (SGD) optimizer.

## Code Description

### Parameters for Active Learning

At the beginning, we define parameters for the active learning process:

- `least_confident_instances`: The top k instances where the model's prediction confidence is lowest will be selected for relabeling.
- `num_annotation_rounds`: The number of re-labeling and re-training rounds.

### Dataset Loading and Splitting

The CIFAR-10 dataset is loaded and transformed to tensors and normalized. The dataset is then split into an initial labeled set and an unlabeled set.

### Model Initialization

A ResNet-18 model from torchvision is initialized with 10 output classes. A cross-entropy loss function and a SGD optimizer are also defined.

### Training Function

The `train_model` function is defined for training the model on a given dataset using the defined loss function and optimizer.

### Active Learning Helper Functions

Several helper functions are defined for the active learning process:

- `transform_image`: Transforms an image tensor from the range -1 to 1 to the range 0 to 1 for display.
- `request_new_labels`: Requests new labels for selected instances from a human annotator.
- `LabeledDataset`: A custom PyTorch Dataset class to be able to add the newly collected labeled data.

### Active Learning Loop

The active learning process is implemented in a loop that iterates for a number of rounds defined by `num_annotation_rounds`. In each round, the following steps are executed:

1. The model is trained on the currently labeled set of data.
2. The trained model is used to make predictions on the unlabeled set.
3. Instances where the model's prediction confidence is lowest are selected from the unlabeled set.
4. New labels for these instances are requested from a human annotator.
5. The newly labeled instances are added to the labeled set and removed from the unlabeled set.

After all annotation rounds are completed, the model is trained one final time on the final labeled set.

## Usage

To run the active learning process, simply execute the provided script. During the process, the script will print out information about each annotation round and request new labels for selected instances. After each label request, enter the correct label for the displayed image. The process will repeat for the number of annotation rounds specified at the beginning of the script. After all rounds are completed, the model is trained on the final labeled set and the active learning process is finished.
