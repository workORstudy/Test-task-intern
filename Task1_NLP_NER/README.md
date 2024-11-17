# Task 1. Natural Language Processing. Named entity recognition

The project focused on training NER model for detection mountain names in unstructured text data/sentences.

**The output for this task contain:**

    ● Jupyter notebook that explains the process of the dataset creation: data_generation.ipynb

    ● Dataset including all artifacts it consists of.(Preproccesed BIO-labeled sentences)

        Train data:

            CSV-file with generated for training sentences and BIO-labels: annotated_mountain_sentences.csv

            CSV-file with final-proccesed generated data with annotation ready for training :
            annotated_data.csv

        Test data:

            CSV-file with generated for testing sentences and BIO-labels: test_annotated_mountain_sentences.csv

            CSV-file with final-proccesed generated data with annotation ready for testing :
            test_annotated_data.csv

    ● Link to model weights: located in model_save folder (via URL https://drive.google.com/drive/folders/1rBPJNtzwLI3zABvZ9CwE6zmIHugtMQiC?usp=drive_link )
        also folder with result(checkpoints via URL https://drive.google.com/drive/folders/1LS8ecROZAr8fiC2cLEZ2LxGTlQo6OS3F?usp=drive_link )

    ● Python script (.py) for model training: train_model.py (realisation of training & fast evaluation procceses)

    ● Python script (.py) for model inference: inference.py (checking model's work on 3 samples)

    ● Jupyter notebook with demo: project.ipynb (full description for all project's blocks & instruction fot its usage)

    ● Text file with requirements for project: requirements.txt (libraries version & tools used during project realisation)

    ● Additional py-script for evaluationg trained model : evaluate_model.py (evaluation & analysis of trained model)