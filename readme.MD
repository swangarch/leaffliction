# Leaffliction

**Leaffliction** is a computer vision project designed to detect plant leaf diseases for applications in **precision agriculture**.
The project is structured as a tightly integrated pipeline—each module naturally builds upon the previous one, forming a coherent workflow.
It leverages libraries such as OpenCV, PlantCV, and PyTorch to create a practical, fully functional solution.

The system consists of five core components:

- Dataset distribution analysis
- Data augmentation and class balancing
- Image transformations and feature exploration
- Model training (CNN and ResNet-based CNN)
- Image prediction (single or batch)

## Setup

```bash
bash venv.sh
# Setup virtual environment and install dependencies

source venv/bin/activate
# Activate virtual environment
```

## Dataset

The dataset contains two plant species—grape and apple—each with four health conditions, forming a total of 8 classes.

Each subfolder inside the dataset directory represents one class, and the folder name corresponds directly to the species and its condition. This makes the data automatically interpretable.

## 1.Distribution Anaylsis

This module analyzes the dataset by scanning subdirectories, counts images per class, and generates:
- Pie charts
- Bar charts

Labels are derived automatically from folder names.

```python
python Distribution.py leaves
```

We observe that initial distribution results reveal significant class imbalance..

![Distribution](./visualize/Grape_before.jpg)

## 2.Data Autmentation

To address imbalance, this module applies controlled augmentation until all classes reach similar sample sizes.

Six augmentation operations are provided:

 - Flip
 - Rotate
 - Skew
 - Shear
 - Crop
 - Distortion

```python
python Augmentation.py leaves
```
After augmentation, the dataset becomes balanced and ready for model training.

![Distribution](./visualize/Grape_after.jpg)


## 3.Transformation

This module explores and visualizes key image features using classical computer vision techniques. It supports both single-image mode and directory-level processing, saving all outputs automatically.

Included transformations:

- Gaussian blur
- Mask generation
- ROI detection
- Object analysis
- Pseudolandmarks
- Color histogram
- Segmentation


```python
python Transformation.py path 
```

![Transformation](./visualize/Transformation.png)
This stage bridges raw imagery and deep learning by revealing underlying structure and patterns.

## 4.Training

The training module builds a leaf disease classifier using **deep learning** to automatically detects the type and the health state of each leaf.
GPU is detected automatically when available.
The project supports multiple model architectures for comparing their performance, between:

- **CNN**
- **ResNet-based CNN**

```bash
python train.py leaves -m CNN
# or
python train.py leaves -m RESNET
```

Training metrics:

![learningcurve](./visualize/Metric.jpg)

This module:

- loads and preprocesses images
- splits them into training and validation sets
- trains the selected model
- saves trained weights

## 5.Predict

```bash
python predict.py path -l weights.pth
```

### Prediction on 1 image

When the provided path points to a single image, the prediction module automatically loads and displays the image, generates its segmentation result and overlays the predicted label directly on the visualization.

![prediction](./visualize/Prediction1.png)
![prediction](./visualize/Prediction2.png)
![prediction](./visualize/Prediction3.png)

### Batch prediction example

When path points to a directory, the folder must follow the same structure as the training dataset: each class (ground truth) should be stored inside a subfolder named after its label. This allows the script to automatically read true labels and evaluate performance.

During batch prediction, the module will: iterate through all images in their respective class folders, generate predictions, compare them with ground truth labels and calculate accuracy. Finally it saves results into a CSV file in the format:

```csv
path,prediction
leaves_test/Apple_Black_rot/image (106).JPG,Apple_Black_rot
leaves_test/Apple_Black_rot/image (149).JPG,Apple_Black_rot
...
```

The model achieves 95–97% accuracy across all 8 classes, demonstrating both reliability and practical usability.

## Summary

Leaffliction is a complete, interconnected pipeline for plant leaf disease detection. From dataset analysis to augmentation, transformation, training, and prediction, every stage is designed to support and enhance the next, forming a seamless end-to-end system.

By automating leaf disease recognition, this project helps:

- reduce manual inspection
- detect issues earlier
- prevent large-scale crop losses
- support smart farming decisions
- enable drone or greenhouse monitoring platforms

With its unified workflow and strong performance, Leaffliction shows how computer vision can be effectively applied to build scalable, impactful tools for modern agriculture.