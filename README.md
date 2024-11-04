Cat and Dog Classification with YOLOv8
This project implements a real-time pet classification model using YOLOv8 to distinguish between images of cats and dogs. The model is trained on the Oxford Pets dataset and can reliably identify the two classes, achieving robust performance across various image conditions.

Project Overview
The goal of this project is to develop a computer vision model that can accurately classify pets in images as either "cat" or "dog." Using YOLOv8, the project focuses on real-time classification and detection, showcasing a practical application of deep learning in image processing.

Features
Real-Time Classification: The YOLOv8 model enables rapid classification of cats and dogs in images.
Pretrained Model Fine-Tuning: Fine-tuned on the Oxford Pets dataset for enhanced accuracy in classifying pets.
Data Augmentation: Utilizes image augmentation techniques to improve model generalization.
Hyperparameter Optimization: Key hyperparameters were optimized to achieve optimal performance.
Installation
To set up this project locally, ensure you have the following dependencies:

Python 3.8 or later
YOLOv8 framework
NumPy
PyTorch
Install the dependencies with the following command:

bash
Copy code
pip install ultralytics opencv-python-headless numpy torch
Dataset
The model was trained on the Oxford Pets Dataset, which includes labeled images of cats and dogs. Download the dataset and place it in the data/ directory as follows:

plaintext
Copy code
data/
└── OxfordPets/
    ├── images/
    └── labels/
Training the Model
To train the model, run:

bash
Copy code
yolo task=detect mode=train model=yolov8n.pt data=OxfordPets.yaml epochs=50 imgsz=640
Adjust the hyperparameters as needed for your specific use case.

Testing and Inference
To test the trained model on new images, run:

bash
Copy code
yolo task=detect mode=predict model=path/to/best.pt source=path/to/images
This will output the results in real-time, with bounding boxes around detected pets.

Results
The model has achieved reliable accuracy in distinguishing between cats and dogs in various conditions, making it suitable for real-time pet recognition tasks.

Future Improvements
Implement additional classes or species.
Improve performance by testing alternative data augmentation techniques.
Explore integration with edge devices for mobile or IoT applications.
Acknowledgments
YOLOv8 by Ultralytics for the model architecture.
Oxford Pets Dataset for labeled cat and dog images.
