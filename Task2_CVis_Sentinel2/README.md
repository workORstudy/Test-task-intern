# Task2_CVis_Sentinel2: Satellite Image Keypoint Matching

 
## Project description

This project implements an algorithm for detecting key points and their correspondence in Sentinel-2 satellite images. The solution includes data preparation, model training, evaluation of results, and demonstration.

Project contains:

● Jupyter notebook that explains the process of the dataset creation>>two files with comments: data_preprocessing.py & data_postprocessing.py

● Link to the dataset (given Dataset from Kaggle).

● Link to model weights: 

    - Model download from git clone https://github.com/magicleap/SuperPointPretrainedNetwork.git

    - Usage superpoint_v1.pth

    - Script superpoint.py to customise and adapt the model architecture to our data

● Python script (.py) for model training or algorithm creation: train.model.py

● Python script (.py) for model inference: inference.py

● Jupyter notebook with demo: inference_demo.py

The project uses the **SuperPoint** model adapted for satellite imagery tasks. The main stages include:
- Image pre-processing.
- Training of the neural network.
- Inference and visualisation of results.
- Evaluation of the model's performance.



 
## Step-by-step instructions

1. Step 1: Install the dependencies.
   
   In the terminal, run:
   
   pip install -r requirements.txt

2. Step 2: Prepare the data

Manually upload & unzip the dataset to the data/raw/ folder. Then start the data processing:

python scripts/data_preprocessing.py
python scripts/data_postprocessing.py

    Step 3: Training the model

To train the model, run:

python scripts/train_model.py

The saved weights will be placed in the folder models/checkpoints/.

    Step 4: Model inversion

For inference on the processed images:

python scripts/inference.py

The results will be saved in the folder results/.

    Step 5: Evaluation

To evaluate the correspondence of key points:

python scripts/evaluation.py

    Step 6: Demonstration

Launch notebooks/inference_demo.ipynb to describe the project steps for the logic of identifying keypoints and descriptors for image pairs and visualising the work.(does not meet the requirements of the assignment, the problem could not be solved)






