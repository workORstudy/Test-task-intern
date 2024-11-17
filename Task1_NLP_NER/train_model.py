import warnings
warnings.filterwarnings("ignore") # Suppresses warnings to keep the output clean.

import pandas as pd
import numpy as np
import ast
from sklearn.model_selection import train_test_split
from transformers import BertTokenizerFast, BertForTokenClassification, Trainer, TrainingArguments
import torch

# Load and preprocess the dataset
df = pd.read_csv('./Task1_NLP_NER/data/annotated_data.csv') # Load the annotated data file.

# Convert the 'sentence' and 'annotation' columns from strings to lists using ast.literal_eval.
df['sentence'] = df['sentence'].apply(ast.literal_eval)
df['annotation'] = df['annotation'].apply(ast.literal_eval)

# Define label encoding for NER
tag2id = {'O': 0, 'B-MOUNTAIN': 1, 'I-MOUNTAIN': 2} # Map each label to a unique ID.
id2tag = {v: k for k, v in tag2id.items()} # Create a reverse mapping for decoding labe

# Splitting on train & test sets
train_sentences, test_sentences, train_annotations, test_annotations = train_test_split(
    df['sentence'], df['annotation'], test_size=0.2, random_state=42 # 80% train, 20% test.
)
# Create DataFrames for training and testing data.
train_df = pd.DataFrame({'sentence': train_sentences, 'annotation': train_annotations})
test_df = pd.DataFrame({'sentence': test_sentences, 'annotation': test_annotations})

# Initialize the BERT tokenizer and model
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased') # Load a pretrained BERT tokenizer.
model = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=len(tag2id)) # Specify the number of unique labels.

# Helper function to align labels with tokenized words
def align_labels_with_tokens(labels, word_ids):
    """
    Align labels with tokenized words. 
    Labels are adjusted to match tokenized inputs, using -100 for padding tokens.
    """
    new_labels = [-100] * len(word_ids) # Initialize all labels as -100 (ignored during training).
    label_index = 0 # Index to track the original labels.
    for i, word_id in enumerate(word_ids):
        if word_id is not None: # Ignore special tokens.
            if label_index < len(labels): # Ensure we don't exceed the label length.
                new_labels[i] = labels[label_index]
            if i == 0 or word_id != word_ids[i - 1]: # Increment label index at word boundaries.
                label_index += 1
    return new_labels

# Function to tokenize data and align labels
def tokenize_and_align_labels(sentences, annotations):
    """
    Tokenize input sentences and align the labels to the tokenized words.
    """
    tokenized_inputs = tokenizer(
        sentences.tolist(), # Convert sentences to a list.
        is_split_into_words=True, # Indicate the input is word-tokenized.
        padding=True, # Pad sequences to the same length.
        truncation=True, # Truncate sequences longer than max_length.
        max_length=512, # Maximum sequence length.
        return_offsets_mapping=True) # Return mapping of tokens to original text.
    labels_aligned = [] # Store aligned labels.
    for i in range(len(sentences)):
        word_ids = tokenized_inputs.word_ids(batch_index=i) # Get word IDs for each token.
        labels = [tag2id[tag] for tag in annotations.iloc[i]] # Map annotation tags to IDs.
        labels_aligned.append(align_labels_with_tokens(labels, word_ids)) # Align labels.
    tokenized_inputs['labels'] = labels_aligned # Add labels to tokenized inputs.
    return tokenized_inputs

# Tokenize and align labels for training and testing data
train_encodings = tokenize_and_align_labels(train_df['sentence'], train_df['annotation'])
test_encodings = tokenize_and_align_labels(test_df['sentence'], test_df['annotation'])

# Create a PyTorch Dataset class for NER
class NERDataset(torch.utils.data.Dataset):
    """
    A custom PyTorch Dataset for handling tokenized NER data.
    """
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        # Return tokenized data as tensors for a given index.
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        # Return the number of examples in the dataset.
        return len(self.encodings['input_ids'])

# Create datasets for training and testing
train_dataset = NERDataset(train_encodings)
test_dataset = NERDataset(test_encodings)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./Task1_NLP_NER/results',
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500, # Number of warmup steps for learning rate scheduler.
    weight_decay=0.01, # Weight decay (L2 regularization).
    logging_dir='./Task1_NLP_NER/logs',
    logging_steps=100,
    evaluation_strategy="epoch", # Evaluate the model at the end of each epoch
    save_strategy="epoch", # Save the model at the end of each epoch.
    load_best_model_at_end=True, # Load the best model based on evaluation metrics.
    save_total_limit=3, # Keep only the 3 most recent model checkpoints.
)

# Initialize the Trainer
trainer = Trainer(
    model=model, # The model to be trained.
    args=training_args, # Training arguments defined above.
    train_dataset=train_dataset, # Training data.
    eval_dataset=test_dataset, # Testing data for evaluation during training.
)

# Train the model
trainer.train()

# Save the trained model and tokenizer
output_dir = "./Task1_NLP_NER/model_save"
model.save_pretrained(output_dir) # Save model weights.
tokenizer.save_pretrained(output_dir) # Save tokenizer.

print("Model saved at:", output_dir)

# Evaluate the model on the test dataset
from sklearn.metrics import classification_report
# Evaluate the model and get performance metrics
eval_results = trainer.evaluate()
print("Evaluation results on test dataset:")
print(eval_results)

# Generate a detailed classification report
def get_classification_report(trainer, test_dataset):
    
    predictions, label_ids, _ = trainer.predict(test_dataset) # Predict on test data.
    predictions = np.argmax(predictions, axis=2) # Get predicted labels.
    
    # Map predicted and true labels to their tag names.
    true_labels = [
        [id2tag[label] for label in label_row if label != -100]
        for label_row in label_ids
    ]
    pred_labels = [
        [id2tag[pred] for pred in pred_row if pred != -100]
        for pred_row in predictions
    ]
    
    # Flatten lists for comparison.
    true_flat = [label for sublist in true_labels for label in sublist if label != 'O']
    pred_flat = [label for sublist in pred_labels for label in sublist if label != 'O']
    
    # Print the classification report for the test dataset
    return classification_report(true_flat, pred_flat)

"""recived evaluation on a test sample:
{'eval_loss': 3.99416676373221e-05, 'eval_runtime': 17.5977, 
'eval_samples_per_second': 56.826, 'eval_steps_per_second': 3.58, 'epoch': 5.0}

### Analysing the evaluation on the test sample

1. `eval_loss`: 3.99416676373221e-05
- This metric shows the average loss on the test data. A low value indicates that 
the model has adapted well to the test set and can accurately predict the correct labels.
- Analysis: A value close to zero (≈ 0.00004) indicates that the model makes almost 
no errors in its predictions for the test data. This is an indicator of high model 
accuracy, but requires additional validation on real data to avoid possible overfitting.

2. `eval_runtime`: 17.5977 seconds.
- Meaning: Time spent on model evaluation on the test sample.
- Analysis: The runtime indicates that the model is fast even for a large amount of data. 
This indicates an optimised model architecture, but the speed may vary depending on the 
hardware.

3. `eval_samples_per_second`: 56.826
- Meaning: The number of test samples processed by the model per second.
- Analysis: Processing 56.826 samples per second is an indicator of high performance. 
This means that the model is suitable for real-time or large data sets.

4. `eval_steps_per_second`: 3.58
- Meaning: The number of processed steps (batches) per second during the evaluation.
- Analysis: The value indicates that the model processes the data in batches 
at an optimal speed. If a batch contains, for example, 16 examples, it means processing 
about 57 examples per second.

5. `epoch`: 5.0
- What it means: The number of training epochs used to evaluate the model.
- Analysis: The training is completed at epoch 5, which is usually a good choice 
for balancing between under- and overfitting. It is important to check for overfitting 
by comparing the results on the validation and test samples.

---

### Overall conclusion.
- Model accuracy: The low loss on the test set indicates that the model has been trained 
effectively and is able to generalise the data.
- Performance: The high speed of processing examples and steps indicates that the model 
can be used for large amounts of data or in real-time scenarios.
- Risks: Despite the good results, the model should be further evaluated on real 
or previously unseen data to ensure that it is not overtrained.

Recommendation: To check the accuracy, completeness and F1-measure metrics for a 
deeper analysis of the model performance. 

! Deeper analysis is done by script evaluate_model.py
"""