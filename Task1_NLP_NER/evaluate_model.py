import torch
from transformers import BertTokenizerFast, BertForTokenClassification, Trainer
from sklearn.metrics import classification_report, precision_recall_fscore_support
import numpy as np
import pandas as pd
import ast


# Load the saved model and tokenizer from a specified directory
model_path = "./Task1_NLP_NER/model_save"
model = BertForTokenClassification.from_pretrained(model_path) # Load the trained model
tokenizer = BertTokenizerFast.from_pretrained(model_path) # Load the corresponding tokenizer

# Define label mappings for Named Entity Recognition (NER)
tag2id = {'O': 0, 'B-MOUNTAIN': 1, 'I-MOUNTAIN': 2} # Map tag names to IDs
id2tag = {v: k for k, v in tag2id.items()} # Reverse mapping from IDs to tag names

# Load the test dataset from a CSV file
df = pd.read_csv('./Task1_NLP_NER/data/test_annotated_data.csv')
df['sentence'] = df['sentence'].apply(ast.literal_eval) # Convert stringified lists to actual lists
df['annotation'] = df['annotation'].apply(ast.literal_eval) # Convert annotation lists similarly

# Extract sentences and annotations for testing
test_sentences = df['sentence']
test_annotations = df['annotation']
test_df = pd.DataFrame({'sentence': test_sentences, 'annotation': test_annotations})

# Align labels with tokenized words to match the BERT tokenization output
def align_labels_with_tokens(labels, word_ids):
    """
    Aligns entity labels with tokenized word IDs.
    Tokens that do not align with a word are set to -100 (ignored during evaluation).
    """
    new_labels = [-100] * len(word_ids) # Initialize with ignored labels
    label_index = 0
    for i, word_id in enumerate(word_ids):
        if word_id is not None: # Check if the token corresponds to a word
            if label_index < len(labels):
                new_labels[i] = labels[label_index] # Assign label to the token
            if i == 0 or word_id != word_ids[i - 1]: # Move to the next label only for the next word
                label_index += 1
    return new_labels

# Tokenize input sentences and align labels with tokenized tokens
def tokenize_and_align_labels(sentences, annotations):
    """
    Tokenizes sentences and aligns corresponding entity labels.
    """
    tokenized_inputs = tokenizer(
        sentences.tolist(), # List of sentences
        is_split_into_words=True, # Treat input as word-tokenized
        padding=True, # Pad to the maximum sentence length
        truncation=True, # Truncate to fit the max length
        max_length=512, # Maximum sequence length for BERT
        return_offsets_mapping=True) # Return mapping of tokens to characters
    labels_aligned = []
    for i in range(len(sentences)):
        word_ids = tokenized_inputs.word_ids(batch_index=i) # Map tokens to words
        labels = [tag2id[tag] for tag in annotations.iloc[i]] # Convert tags to IDs
        labels_aligned.append(align_labels_with_tokens(labels, word_ids))
    tokenized_inputs['labels'] = labels_aligned # Add aligned labels to tokenized inputs
    return tokenized_inputs

# Prepare test encodings
test_encodings = tokenize_and_align_labels(test_df['sentence'], test_df['annotation'])

# Define a PyTorch dataset class for NER
class NERDataset(torch.utils.data.Dataset):
    """
    Custom dataset for NER tasks, storing tokenized inputs and labels.
    """
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        # Returns the tokenized inputs and labels as tensors
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        # Returns the size of the dataset
        return len(self.encodings['input_ids'])

# Create a dataset instance for testing
test_dataset = NERDataset(test_encodings)

# Initialize Hugging Face Trainer for evaluation
trainer = Trainer(
    model=model, # Use the pre-trained model
    eval_dataset=test_dataset, # Provide the test dataset
    tokenizer=tokenizer # Use the corresponding tokenizer
)

# Evaluate the model and map predictions to human-readable labels
def evaluate_model(trainer, test_dataset, id2tag):
    """
    Evaluates the model on a test dataset and returns true and predicted labels.
    """
    predictions, labels, _ = trainer.predict(test_dataset) # Get predictions and true labels
    preds = np.argmax(predictions, axis=2) # Convert logits to predicted label indices

    true_labels = []
    pred_labels = []
    for i in range(len(labels)):
        for j in range(len(labels[i])):
            if labels[i][j] != -100:  # Ignore padding tokens during evaluation
                true_labels.append(id2tag[labels[i][j]]) # Map label ID to its name
                pred_labels.append(id2tag[preds[i][j]]) # Map predicted ID to its name
    return true_labels, pred_labels

# Get true and predicted labels
true_labels, pred_labels = evaluate_model(trainer, test_dataset, id2tag)

# Generate a classification report (precision, recall, F1-score)
report = classification_report(
    true_labels, 
    pred_labels, 
    labels=["B-MOUNTAIN", "I-MOUNTAIN"], # Only evaluate entity labels
    target_names=["B-MOUNTAIN", "I-MOUNTAIN"], # Human-readable names for labels
    zero_division=0 # Handle cases with zero predictions
)

# Print the classification report
print("Classification Report:\n")
print(report)

# Calculate additional weighted metrics
precision, recall, f1, _ = precision_recall_fscore_support(
    true_labels, 
    pred_labels, 
    labels=["B-MOUNTAIN", "I-MOUNTAIN"], 
    average='weighted', 
    zero_division=0
)

# Print additional metrics
print(f"Weighted Precision: {precision:.4f}")
print(f"Weighted Recall: {recall:.4f}")
print(f"Weighted F1-Score: {f1:.4f}")

"""
### Recived evaluation metrics:

Classification Report:

              precision    recall  f1-score   support

  B-MOUNTAIN       0.98      1.00      0.99       978
  I-MOUNTAIN       1.00      1.00      1.00      1138

   micro avg       0.99      1.00      1.00      2116
   macro avg       0.99      1.00      0.99      2116
weighted avg       0.99      1.00      1.00      2116

Weighted Precision: 0.9903
Weighted Recall: 1.0000
Weighted F1-Score: 0.9951

### Analysis of model performance:

1. Precision, Recall, F1-Score, and Support (Per Class)

    # B-MOUNTAIN  
- Precision: 0.98  
  Out of all tokens predicted as `B-MOUNTAIN` (beginning of mountain entity), 
  98% were correct.  
- High precision suggests that the model rarely makes false positive predictions 
  for this class.  

- Recall: 1.00  
  All actual `B-MOUNTAIN` tokens in the dataset were correctly identified by the model.  
- This indicates zero false negatives for this class.  

- F1-Score: 0.99  
  The harmonic mean of precision and recall. A value of 0.99 reflects a balance between 
  the high precision and perfect recall.  

- Support: 978  
  This is the total number of `B-MOUNTAIN` tokens in the test set.  

  
    # I-MOUNTAIN  
- Precision: 1.00  
  All tokens predicted as `I-MOUNTAIN` (continuation of mountain entity) were correct.  
- This indicates zero false positives for this class.  

- Recall: 1.00  
  All actual `I-MOUNTAIN` tokens in the dataset were correctly identified by the model.  
- This indicates zero false negatives for this class.  

- F1-Score: 1.00  
  Perfect F1-score due to both perfect precision and recall.  

- Support: 1138  
  This is the total number of `I-MOUNTAIN` tokens in the test set.  

---

2. Aggregated Metrics

# Micro Average  
- Precision, Recall, F1-Score: 0.99, 1.00, 1.00  
  Micro averaging aggregates contributions of all classes based on their individual support.  
- The near-perfect scores here indicate that across all tokens, the model is both 
  highly precise and exhaustive in its predictions.

# Macro Average  
- Precision, Recall, F1-Score: 0.99, 1.00, 0.99  
  Macro averaging computes the unweighted mean of precision, recall, 
  and F1 across all classes.  
- Slight differences between micro and macro averages occur when class distributions 
  are imbalanced.  

# Weighted Average  
- Weighted Precision: 0.99, Weighted Recall: 1.00, Weighted F1-Score: 0.9951  
  Weighted averaging adjusts metrics by the support (number of instances) of each class.  
- These values reflect the overall performance, with the slightly lower precision 
  in `B-MOUNTAIN` class reducing the weighted precision slightly.  


---

### Conclusion
The model demonstrates excellent performance in identifying mountain names, 
with near-perfect precision, recall, and F1-scores. Minor precision losses 
in the `B-MOUNTAIN` class suggest room for improvement in reducing false positives. 
However, given the test data's metrics, the model is highly reliable and accurate for 
the NER task.
"""