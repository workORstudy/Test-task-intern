import warnings
warnings.filterwarnings("ignore") # Suppress any warnings for cleaner output

from transformers import BertTokenizerFast, BertForTokenClassification
import torch

# Load the tokenizer and the pre-trained model for NER
model_path = "./Task1_NLP_NER/model_save"
tokenizer = BertTokenizerFast.from_pretrained(model_path) # Tokenizer processes text into model-ready tokens
model = BertForTokenClassification.from_pretrained(model_path) # Load the fine-tuned BERT model

# Map from label IDs (integers) to their corresponding tags (strings)
id2tag = {0: 'O', 1: 'B-MOUNTAIN', 2: 'I-MOUNTAIN'}

# Function to merge subtokens (those starting with "##") back into full words
def merge_subtokens(tokens, labels):
    """
    Merge BERT subtokens into full tokens and align corresponding labels.
    
    Args:
        tokens (list of str): Tokenized text (including subtokens).
        labels (list of str): Predicted labels for each token.

    Returns:
        merged_tokens (list of str): Merged full tokens.
        merged_labels (list of str): Labels aligned with merged tokens.
    """

    merged_tokens = []
    merged_labels = []
    
    current_token = "" # Store the current token being merged
    current_label = None # Store the label for the current token
    
    for token, label in zip(tokens, labels):
        if token.startswith("##"):  # Indicates a subtoken
            current_token += token[2:]  # Append the subtoken part (without "##")
        else:
            if current_token:  # If a full token is already being built, save it
                merged_tokens.append(current_token)
                merged_labels.append(current_label)
            current_token = token  # Start a new token
            current_label = label
    if current_token:  # Add the last token to the list
        merged_tokens.append(current_token)
        merged_labels.append(current_label)
    
    return merged_tokens, merged_labels

# Function to make predictions for a given text
def predict(text):
    """
    Predict NER labels for a given text using the loaded model.
    
    Args:
        text (str): Input text to process.

    Returns:
        results (list of tuples): List of (token, label) pairs after processing.
    """

    # Tokenize the input text (split into words and process into model-friendly format)
    tokens = tokenizer(text.split(), is_split_into_words=True, return_tensors="pt", truncation=True)
    
    # Perform inference without gradient computation (saves memory)
    with torch.no_grad():
        outputs = model(**tokens) # Forward pass through the model
    logits = outputs.logits # Extract raw predictions (logits)
    predictions = torch.argmax(logits, dim=2) # Convert logits to class IDs (highest probability)
    
    
    # Convert token IDs back to strings and map predicted IDs to tag names
    tokens_list = tokenizer.convert_ids_to_tokens(tokens['input_ids'][0]) # Original token strings
    labels = [id2tag[label_id] for label_id in predictions[0].tolist()] # Predicted labels as strings
    
    # Merge subtokens back into full words and align their labels
    merged_tokens, merged_labels = merge_subtokens(tokens_list, labels)
    
    # Filter out special tokens ([CLS], [SEP])
    results = [(token, label) for token, label in zip(merged_tokens, merged_labels) if token not in ["[CLS]", "[SEP]"]]
    return results


# Example sentences to test the model
example_sentences = [
    "Mount Everest and K2 are among the most challenging peaks for climbers.",
    "Many adventurers dream of visiting Mount Kilimanjaro someday.",
    "Mount Fuji in Japan is a symbol of beauty and tradition."
]

# Perform predictions for each example sentence
for i, sentence in enumerate(example_sentences, 1):
    print(f"Example {i}: {sentence}")
    predictions = predict(sentence) # Predict labels for the sentence
    print("Predictions:")
    for token, label in predictions:
        print(f"{token}: {label}") # Print each token and its predicted label
    print("-" * 50) # Separator for readability
