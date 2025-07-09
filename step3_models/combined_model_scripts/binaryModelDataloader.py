import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import torch
from torch.nn import CrossEntropyLoss, BCELoss, BCEWithLogitsLoss
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler, TensorDataset
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import pandas as pd
import hdbscan
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

#        #correct += (outputs.argmax(dim=1) == quality_label).sum().item()  # Adjust this based on your model output


class Transformer(nn.Module):
    def __init__(self, method_embedding_dim=768, reduced_method_dim=32, 
                 category_dim=64, duration_dim=64, num_heads=4, 
                 num_layers=1, num_classes=1, dropout_rate=0.2, max_seq_len=32): 
        super(Transformer, self).__init__()

        # Nonlinear dimensionality reduction for method embeddings
        self.method_reducer = nn.Sequential(
            nn.Linear(method_embedding_dim, method_embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(method_embedding_dim // 2, reduced_method_dim),
            nn.ReLU()
        )

        # Total embedding dimension for the transformer input
        total_embedding_dim = reduced_method_dim + category_dim + duration_dim

        # Positional encoding
        self.positional_encoding = nn.Parameter(
            torch.zeros((1, max_seq_len + 1, total_embedding_dim))  # Add 1 for sequence aggregator token
        )

        # Special header token (e.g., [CLS])
        self.header_token = nn.Parameter(torch.zeros(1, 1, total_embedding_dim))

        # Transformer Encoder Layer
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=total_embedding_dim,
            nhead=num_heads,
            dropout=dropout_rate,
            #batch_first=True,  # Ensure it operates as batch first

        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_encoder_layer,
            num_layers=num_layers
        )

        # Dropout layer after embedding concatenation
        self.dropout = nn.Dropout(dropout_rate)

        # Fully connected layers for the output
        self.fc_layer1 = nn.Linear(total_embedding_dim, total_embedding_dim // 2)
        self.fc_layer2 = nn.Linear(total_embedding_dim // 2, num_classes)

        # Dropout after the fully connected layers
        self.fc_dropout = nn.Dropout(dropout_rate)

    def forward(self, method_embeddings, category_embeddings, duration_embeddings, attention_mask=None):
        # Reduce method embedding dimensions
        method_embeddings = self.method_reducer(method_embeddings)

        # Expand embeddings to match the sequence length (batch_size, seq_len, feature_dim)
        method_embeddings = method_embeddings.squeeze(1)  # Remove any unnecessary dimensions
        category_embeddings = category_embeddings.squeeze(1)  # Same for category_embeddings
        duration_embeddings = duration_embeddings.squeeze(1)  # Same for duration_embeddings

        # Concatenate embeddings along the feature dimension
        combined_embeddings = torch.cat(
            (method_embeddings, category_embeddings, duration_embeddings), dim=2
        )  # Shape: (batch_size, seq_len, total_embedding_dim)

        # Apply dropout after concatenation
        combined_embeddings = self.dropout(combined_embeddings)

        # Add special header token to the beginning of the sequence
        header_token_repeated = self.header_token.expand(combined_embeddings.size(0), 1, -1)
        combined_embeddings = torch.cat((header_token_repeated, combined_embeddings), dim=1)

        # Add positional encodings
        seq_length = combined_embeddings.size(1)
        positional_encodings = self.positional_encoding[:, :seq_length, :]
        combined_embeddings = combined_embeddings + positional_encodings

        # Reshape for transformer: (seq_len, batch_size, total_embedding_dim)
        combined_embeddings = combined_embeddings.permute(1, 0, 2)
    
        if attention_mask is not None:
            # Apply attention mask to the transformer (padding tokens will be ignored)
            attention_mask = attention_mask.squeeze(1)  # Remove unnecessary dimensions
            attention_mask = attention_mask[:, :seq_length + 1]  # Ensure it has the correct length

            # Convert padding tokens to large negative values so they won't be attended to
            attention_mask = (attention_mask == 0).to(dtype=torch.float32)
            attention_mask = (1.0 - attention_mask) * -10000.0  # Large negative value for padding
            
        # Pass through transformer encoder
        transformer_output = self.transformer_encoder(
            combined_embeddings, src_key_padding_mask=attention_mask # Get attention weights
        )

        # transformer_output, attention_weights = self.transformer_encoder(
        #     combined_embeddings, src_key_padding_mask=attention_mask, need_weights=True  # Get attention weights
        # )

        # Output corresponding to the special header token (first token)
        header_token_output = transformer_output[0, :, :]

        # Fully connected layers for classification
        projected_output = self.fc_layer1(header_token_output)
        
        # Apply dropout after the first fully connected layer
        projected_output = self.fc_dropout(projected_output)
        
        final_output = self.fc_layer2(projected_output)

        print("Transformer Output:", transformer_output.shape)
        print("Final Output:", final_output.shape)

        return final_output #, attention_weights

class CustomDataset(torch.utils.data.Dataset):
   

    def __len__(self):
        return len(self.labels)




    def __init__(self, embeddings, labels, participant_ids):
        self.embeddings = embeddings  # This contains tuples of embeddings (method_embeddings, category_embeddings, duration_embeddings)
        self.labels = labels  # Quality labels
        self.participant_ids = participant_ids  # Participant IDs

    def __getitem__(self, idx):
        # Extract individual embeddings
        method_embeddings, category_embeddings, duration_embeddings = self.embeddings[idx]
        
        # Ensure embeddings are of the correct type after stacking
        method_embeddings = torch.stack(method_embeddings).to(dtype=torch.float32) if isinstance(method_embeddings, (list, tuple)) else method_embeddings
        category_embeddings = torch.stack(category_embeddings).to(dtype=torch.float32) if isinstance(category_embeddings, (list, tuple)) else category_embeddings
        duration_embeddings = torch.stack(duration_embeddings).to(dtype=torch.float32) if isinstance(duration_embeddings, (list, tuple)) else duration_embeddings

        # Extract quality label, participant_id, and method for this sample
        quality_label = self.labels[idx]
        participant_id = self.participant_ids[idx]

        # Return stacked embeddings and other details
        return method_embeddings, category_embeddings, duration_embeddings, quality_label, participant_id

def train_confidenceInterval(model, train_loader, loss_fn, optimizer, device):
    model.train()  # Set model to training mode
    total_loss = 0
    correct_preds = 0
    total_preds = 0
    confidences = []  # To store confidence values for predictions

    for i, (method_embeddings, category_embeddings, duration_embeddings, quality_label, participant_id) in enumerate(train_loader):
        # Convert lists to tensors before moving to device
        method_embeddings = torch.tensor(method_embeddings).to(device)
        category_embeddings = torch.tensor(category_embeddings).to(device)
        duration_embeddings = torch.tensor(duration_embeddings).to(device)
        quality_label = torch.tensor(quality_label, dtype=torch.float32).to(device)

        quality_label = quality_label.view(-1, 1)  # Reshapes to (batch_size, 1)

        # Forward pass
        outputs = model(method_embeddings, category_embeddings, duration_embeddings)

        # Calculate loss
        loss = loss_fn(outputs, quality_label)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Convert outputs to probabilities using sigmoid
        probabilities = torch.sigmoid(outputs)

        # Convert probabilities to binary predictions
        predicted = torch.round(probabilities)

        # Collect confidence for predictions
        batch_confidences = probabilities.squeeze(1).cpu().detach().numpy()
        confidences.extend(batch_confidences)

        # Accumulate correct predictions and total predictions
        correct_preds += (predicted == quality_label).sum().item()
        total_preds += quality_label.size(0)

        # Accumulate total loss
        total_loss += loss.item()

        # Print predictions and confidence for every 10th batch
        if i % 10 == 0:  # Adjust the frequency of printing as necessary
            print(f"Batch {i}/{len(train_loader)}:")
            print(f"Predictions: {predicted.squeeze(1).cpu().detach().numpy()}")
            print(f"Ground Truth: {quality_label.squeeze(1).cpu().detach().numpy()}")
            print(f"Confidences: {batch_confidences}")
            print(f"Train Loss: {loss.item():.4f}")
            print(f"Train Accuracy: {100 * correct_preds / total_preds:.2f}%")

    # Calculate average loss and accuracy after the entire epoch
    avg_loss = total_loss / len(train_loader)
    accuracy = 100 * correct_preds / total_preds

    # Summarize confidence levels
    avg_confidence = np.mean(confidences)
    print(f"Train Loss: {avg_loss:.4f}, Train Accuracy: {accuracy:.2f}%")
    print(f"Average Confidence: {avg_confidence:.4f}")

    return avg_loss, accuracy, avg_confidence

def train(model, train_loader, loss_fn, optimizer, device):
    model.train()  # Set model to training mode
    total_loss = 0
    correct_preds = 0
    total_preds = 0

    for i, (method_embeddings, category_embeddings, duration_embeddings, quality_label, participant_id) in enumerate(train_loader):
        # Convert lists to tensors before moving to device
        method_embeddings = torch.tensor(method_embeddings).to(device)
        category_embeddings = torch.tensor(category_embeddings).to(device)
        duration_embeddings = torch.tensor(duration_embeddings).to(device)
        quality_label = torch.tensor(quality_label, dtype=torch.float32).to(device)

        quality_label = quality_label.view(-1, 1)  # Reshapes to (32, 1)
        # Forward pass
        outputs = model(method_embeddings, category_embeddings, duration_embeddings)


        # Calculate loss
        loss = loss_fn(outputs, quality_label)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Convert outputs to binary predictions
        predicted = torch.round(torch.sigmoid(outputs))

        # Accumulate correct predictions and total predictions
        correct_preds += (predicted == quality_label).sum().item()
        total_preds += quality_label.size(0)

        # Accumulate total loss
        total_loss += loss.item()

        # Print the prediction, ground truth, and participant info for every 10th batch
        if i % 10 == 0:  # Adjust the frequency of printing as necessary
            print(f"Batch {i}/{len(train_loader)}:")
            print(f"Predictions: {predicted.squeeze(1).cpu().detach().numpy()}")
            print(f"Ground Truth: {quality_label.squeeze(1).cpu().detach().numpy()}")
            print(f"Participant ID: {participant_id.cpu().detach().numpy()}")  # Print participant_id
            print(f"Train Loss: {loss.item():.4f}")
            print(f"Train Accuracy: {100 * correct_preds / total_preds:.2f}%")

    # Calculate average loss and accuracy after the entire epoch
    avg_loss = total_loss / len(train_loader)
    accuracy = 100 * correct_preds / total_preds
    print(f"Train Loss: {avg_loss:.4f}, Train Accuracy: {accuracy:.2f}%")

    return avg_loss, accuracy

def validate(model, valid_loader, loss_fn, device):
    model.eval()  # Set model to evaluation mode
    total_loss = 0
    correct_preds = 0
    total_preds = 0
    #attention_weights_list = []  # To store attention weights for visualization


    with torch.no_grad():  # Disable gradient calculation for validation
        for i, (method_embeddings, category_embeddings, duration_embeddings, quality_label, participant_id) in enumerate(valid_loader):
            # Convert lists to tensors before moving to device
            method_embeddings = torch.tensor(method_embeddings).to(device)
            category_embeddings = torch.tensor(category_embeddings).to(device)
            duration_embeddings = torch.tensor(duration_embeddings).to(device)
            quality_label = torch.tensor(quality_label, dtype=torch.float32).to(device)

            quality_label = quality_label.view(-1, 1)  # Reshapes to (batch_size, 1)

            # Forward pass
            outputs = model(method_embeddings, category_embeddings, duration_embeddings)
            #outputs, attention_weights = model(method_embeddings, category_embeddings, duration_embeddings)
            #attention_weights_list.append(attention_weights.cpu())  # Store on CPU for visualization

            # Calculate loss
            loss = loss_fn(outputs, quality_label)

            # Convert outputs to binary predictions
            predicted = torch.round(torch.sigmoid(outputs))

            # Accumulate correct predictions and total predictions
            correct_preds += (predicted == quality_label).sum().item()
            total_preds += quality_label.size(0)

            # Accumulate total loss
            total_loss += loss.item()

    # Calculate average loss and accuracy after the entire validation phase
    avg_loss = total_loss / len(valid_loader)
    accuracy = 100 * correct_preds / total_preds
    return avg_loss, accuracy#, attention_weights_list

def test(model, test_loader, loss_fn, device):
    model.eval()  # Set model to evaluation mode
    total_loss = 0
    correct_preds = 0
    total_preds = 0

    with torch.no_grad():  # No need to compute gradients for testing
        for i, (method_embeddings, category_embeddings, duration_embeddings, quality_label, participant_id) in enumerate(test_loader):
            # Convert lists to tensors before moving to device
            method_embeddings = torch.tensor(method_embeddings).to(device)
            category_embeddings = torch.tensor(category_embeddings).to(device)
            duration_embeddings = torch.tensor(duration_embeddings).to(device)
            quality_label = torch.tensor(quality_label, dtype=torch.float32).to(device)

            quality_label = quality_label.view(-1, 1)  # Reshapes to (32, 1)

            # Forward pass
            outputs = model(method_embeddings, category_embeddings, duration_embeddings)

            # Calculate loss
            loss = loss_fn(outputs, quality_label)

            # Convert outputs to binary predictions
            predicted = torch.round(torch.sigmoid(outputs))

            # Accumulate correct predictions and total predictions
            correct_preds += (predicted == quality_label).sum().item()
            total_preds += quality_label.size(0)

            # Accumulate total loss
            total_loss += loss.item()

    # Calculate average loss and accuracy after the entire epoch
    avg_loss = total_loss / len(test_loader)
    accuracy = 100 * correct_preds / total_preds
    return avg_loss, accuracy




def train_withoutPID(model, train_loader, loss_fn, optimizer, device):
    model.train()  # Set model to training mode
    total_loss = 0
    correct_preds = 0
    total_preds = 0

    for i, (method_embeddings, category_embeddings, duration_embeddings, quality_label) in enumerate(train_loader):
        # Move tensors to device
        method_embeddings, category_embeddings, duration_embeddings, quality_label = method_embeddings.to(device), category_embeddings.to(device), duration_embeddings.to(device), quality_label.to(device)
       
        # Forward pass
        outputs = model(method_embeddings, category_embeddings, duration_embeddings)

        # Calculate loss
        loss = loss_fn(outputs, quality_label)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Convert outputs to binary predictions
        predicted = torch.round(torch.sigmoid(outputs))

        # Accumulate correct predictions and total predictions
        correct_preds += (predicted == quality_label).sum().item()
        total_preds += quality_label.size(0)

        # Accumulate total loss
        total_loss += loss.item()

        # Print the prediction and ground truth for every batch (every 10th batch)
        if i % 10 == 0:  # Adjust the frequency of printing as necessary
            print(f"Batch {i}/{len(train_loader)}:")
            print(f"Predictions: {predicted.squeeze(1).cpu().detach().numpy()}")
            print(f"Ground Truth: {quality_label.squeeze(1).cpu().detach().numpy()}")
            print(f"Train Loss: {loss.item():.4f}")
            print(f"Train Accuracy: {100 * correct_preds / total_preds:.2f}%")

    # Calculate average loss and accuracy after the entire epoch
    avg_loss = total_loss / len(train_loader)
    accuracy = 100 * correct_preds / total_preds
    #print(f"Train Loss: {avg_loss:.4f}, Train Accuracy: {accuracy:.2f}%")

    return avg_loss, accuracy

def tokenize_data_input_embeddings(data_tuple, tokenizer, model, category_vocab, category_dim=32, duration_dim=32):
    """
    Tokenize and embed data from the scanpath, keeping embeddings separate for method tokens,
    semantic categories, and fixation duration bins.

    Args:
        data_tuple: Tuple containing participant data in the format:
                    (ParticipantID, QualitySummary, {MethodName: [methodToken, semanticCategory, DurationBin]}).
        tokenizer: Pretrained tokenizer (e.g., CodeBERT tokenizer).
        model: Pretrained model (e.g., CodeBERT model).
        category_vocab: Vocabulary mapping semantic categories to unique IDs.
        category_dim: Embedding dimension for semantic categories.
        duration_dim: Embedding dimension for fixation duration bins.

    Returns:
        List of tuples containing tokenized and embedded data.
    """
    # Initialize embedding layers
    category_embedding_layer = nn.Embedding(num_embeddings=len(category_vocab), embedding_dim=category_dim)
    duration_embedding_layer = nn.Embedding(num_embeddings=6, embedding_dim=duration_dim)  # Fixed 6 bins

    tokenized_data = []

    for entry in data_tuple:
       
        participant_id = entry[0]  # Unique identifier
        quality_summary = entry[1]  # Label we want
        scanpath = data_tuple[entry]  # {MethodName: [methodToken, semanticCategory, DurationBin]}
        
        tokenized_entry = []

        # Direct iteration over method_data keys
        for token in scanpath:
            method_token = token[0]
            semantic_category = token[1]
            duration_bin = token[2]

            # Method Token Embedding
            inputs = tokenizer(method_token, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                outputs = model(**inputs)
                method_token_embedding = outputs.last_hidden_state.mean(dim=1).squeeze().detach()

            # Semantic Category Embedding
            category_idx = torch.tensor([category_vocab[semantic_category]])
            category_embedding = category_embedding_layer(category_idx).squeeze().detach()

            # Fixation Duration Bin Embedding
            duration_idx = torch.tensor([int(duration_bin.split()[1]) - 1])  # Map "Bin 1" to 0
            duration_embedding = duration_embedding_layer(duration_idx).squeeze().detach()

            tokenized_entry.append((method_token_embedding, category_embedding, duration_embedding))

        tokenized_data.append((participant_id, quality_summary, tokenized_entry))

    

    return tokenized_data


def flatten_data(data):
    # Open the pickle file in binary mode
    with open(data, 'rb') as f:
        data = pickle.load(f)

    # Ensure the loaded data is a dictionary
    if not isinstance(data, dict):
        raise ValueError("Expected a dictionary, but got something else.")
    
    flat_map = {}
    for key, entries in data.items():
        participant_id, quality, method_name = key
        flattened_entries = []
        for entry in entries:
            for token, sem_dict in entry.items():
                for semantic_category, fixation_duration in sem_dict.items():
                    # Remove numeric suffixes from semantic category (e.g., 'function declaration.1')
                    #stripped_semantic_category = semantic_category.split('.')[0]
                    flattened_entries.append((token, semantic_category, fixation_duration))
        flat_map[key] = flattened_entries
        #print(flat_map)
    
    return flat_map


def create_fixation_durations_fixed_bins(data_tuple):
    """
    Function to fit fixation durations into 10 fixed bins.
    
    Args:
        data_tuple: The normalized data dictionary containing fixation durations.
    """
    countOutliers = 0
    total_bins = {'fixed_bins': 0}
    
    fixed_bin_ranges = [ 
        (0.0, 0.01),   # Bin 1: 1332 samples
        (0.01, 0.02),  # Bin 2: 1145 samples
        (0.02, 0.04),  # Bin 3: 1190 samples (merged 0.02 to 0.04)
        (0.04, 0.06),  # Bin 4: 259 samples (merged 0.04 to 0.06)
        (0.06, 0.1),   # Bin 5: 75 samples (merged 0.06 to 0.1)
        (0.1, 0.3)     # Bin 6: 29 samples (merged 0.1 to 0.3)
    ]
    
    bin_names = ["Bin 1", "Bin 2", "Bin 3", "Bin 4", "Bin 5", "Bin 6"]
    
    # Initialize overall bins count
    bins_count = [0] * len(fixed_bin_ranges)

    for entry in data_tuple:
        # Collect durations for the current entry
        durations = [token[2] for token in data_tuple[entry] if isinstance(token[2], (int, float))]
        
        #print(f"Durations for entry {entry}: {durations}")  # Debugging
        
        N = len(durations)
        if N == 0:
            #print(f"No valid data for entry {entry}")
            continue

        # Rewriting durations into bin names
        rewritten_durations = []
        for duration in durations:
            if duration > 0.26:  # Check if the duration is an outlier (adjust if necessary)
                countOutliers += 1
                #print(f"Outlier duration for entry {entry}: {duration}")  # Debugging
                rewritten_durations.append("Outlier")
                continue
            
            # Map each duration into the corresponding bin
            bin_found = False
            for i, (lower, upper) in enumerate(fixed_bin_ranges):
                if lower <= duration < upper:
                    bins_count[i] += 1
                    rewritten_durations.append(bin_names[i])  # Replace duration with bin name
                    bin_found = True
                    break
            
            if not bin_found:
                rewritten_durations.append("Outlier")

        # Replace the original durations with the rewritten bin names
        data_tuple[entry] = [(token[0], token[1], bin_name) for token, bin_name in zip(data_tuple[entry], rewritten_durations)]
    #print(data_tuple)
    total_bins['fixed_bins'] = sum(bins_count)

    # Print counts for each bin
    #for i, (lower, upper) in enumerate(fixed_bin_ranges):
        #print(f"Bin {i + 1} ({lower} to {upper}): {bins_count[i]}")

    # Print outliers and total counts
    print(f"Total outliers (durations > 0.26): {countOutliers}")
    #print(f"Total Counts across all entries: {total_bins['fixed_bins']}")
    return data_tuple

def create_vocab(data_tuple):
    """
    Extracts unique semantic categories (with suffixes) and maps them to unique IDs.
    """
    unique_categories = set()
    for entry in data_tuple:
        for token in data_tuple[entry]:
            #print(token)
            unique_categories.add(token[1])  # token[1] is the semantic category
    vocab = {category: idx for idx, category in enumerate(unique_categories)}
    return vocab




def preprocessing_tokenized_data(data_tuple, tokenizer, model, category_vocab, category_dim=64, duration_dim=64, max_sequence_length=None):
    """
    Tokenize and embed data from the scanpath, keeping embeddings separate for method tokens,
    semantic categories, and fixation duration bins.

    Args:
        data_tuple: Tuple containing participant data in the format:
                    (ParticipantID, QualitySummary, {MethodName: [methodToken, semanticCategory, DurationBin]}).
        tokenizer: Pretrained tokenizer (e.g., CodeBERT tokenizer).
        model: Pretrained model (e.g., CodeBERT model).
        category_vocab: Vocabulary mapping semantic categories to unique IDs.
        category_dim: Embedding dimension for semantic categories.
        duration_dim: Embedding dimension for fixation duration bins.
        max_sequence_length: Optional max sequence length for padding/truncation.

    Returns:
        Padded and tokenized data, attention masks, and max sequence length.
    """
    # Initialize embedding layers
    category_embedding_layer = nn.Embedding(num_embeddings=len(category_vocab), embedding_dim=category_dim)
    duration_embedding_layer = nn.Embedding(num_embeddings=6, embedding_dim=duration_dim)  # Fixed 6 bins

    tokenized_data = []
    calculated_max_length = 0  # To dynamically find max sequence length if not provided

    for entry in data_tuple:
        participant_id = entry[0]  # Unique identifier
        quality_summary = entry[1]  # Label we want
        method_name = entry[2]
        scanpath = data_tuple[entry]  # {MethodName: [methodToken, semanticCategory, DurationBin]}
        
        tokenized_entry = []
        scan_path_length = len(scanpath)  # Length of the current scanpath

        # Update calculated max length dynamically
        calculated_max_length = max(calculated_max_length, scan_path_length)

        # Process each token in the scanpath
        for token in scanpath:
            method_token = token[0]
            semantic_category = token[1]
            duration_bin = token[2]

            # Method Token Embedding
            inputs = tokenizer(method_token, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                outputs = model(**inputs)
                method_token_embedding = outputs.last_hidden_state.mean(dim=1).squeeze().detach()

            # Semantic Category Embedding
            category_idx = torch.tensor([category_vocab[semantic_category]])
            category_embedding = category_embedding_layer(category_idx).squeeze().detach()

            # Fixation Duration Bin Embedding
            duration_idx = torch.tensor([int(duration_bin.split()[1]) - 1])  # Map "Bin 1" to 0
            duration_embedding = duration_embedding_layer(duration_idx).squeeze().detach()

            tokenized_entry.append((method_token_embedding, category_embedding, duration_embedding))

        tokenized_data.append((participant_id, quality_summary, method_name, tokenized_entry))

    # Determine final max sequence length
    max_sequence_length = max_sequence_length or calculated_max_length

    # Pad sequences to max_sequence_length and create attention masks
    padded_data = []
    attention_masks = []
    for entry in tokenized_data:
        participant_id, quality_summary, method_name, scanpath_embeddings = entry

        # Pad or truncate scanpath embeddings
        padding_needed = max_sequence_length - len(scanpath_embeddings)
        attention_mask = [1] * len(scanpath_embeddings)  # Real tokens

        if padding_needed > 0:
            pad_vector = (torch.zeros_like(method_token_embedding),
                          torch.zeros_like(category_embedding),
                          torch.zeros_like(duration_embedding))
            scanpath_embeddings.extend([pad_vector] * padding_needed)
            attention_mask.extend([0] * padding_needed)  # Padding tokens
        else:
            scanpath_embeddings = scanpath_embeddings[:max_sequence_length]
            attention_mask = attention_mask[:max_sequence_length]

        padded_data.append((participant_id, quality_summary, method_name, scanpath_embeddings))
        attention_masks.append(attention_mask)

    return padded_data, attention_masks, max_sequence_length

# Function to save data using pickle
def save_data(filename, data):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)

# Function to load data using pickle
def load_data(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

class EarlyStopping:
    def __init__(self, patience=3, delta=0.01):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_model_state = None

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.best_model_state = model.state_dict()
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_state = model.state_dict()
            self.counter = 0

    def load_best_model(self, model):
        model.load_state_dict(self.best_model_state)
def visualize_attention_weights(attention_weights, seq_len, layer_idx, head_idx, category_vocab=None):
    """
    Visualizes attention weights for a specific layer and head.

    Args:
        attention_weights (torch.Tensor or np.ndarray): Shape (num_layers, num_heads, seq_len, seq_len)
        seq_len (int): Sequence length.
        layer_idx (int): Layer index to visualize.
        head_idx (int): Head index to visualize.
        category_vocab (dict, optional): Category vocabulary for labeling.
    """
    # Convert attention_weights to numpy if it's a PyTorch tensor
    if isinstance(attention_weights, torch.Tensor):
        attention_weights = attention_weights.numpy()
    
    # Extract the attention weights for the specified layer and head
    layer_attention = attention_weights[layer_idx]  # Shape: (num_heads, seq_len, seq_len)
    head_attention = layer_attention[head_idx]  # Shape: (seq_len, seq_len)

    # Generate labels for axes
    labels = create_labels(seq_len, category_vocab)

    # Create the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        head_attention,
        xticklabels=labels,
        yticklabels=labels,
        cmap="viridis",
        cbar=True,
        square=True,
        linewidths=0.5
    )
    plt.title(f"Attention Weights - Layer {layer_idx}, Head {head_idx}")
    plt.xlabel("Keys")
    plt.ylabel("Queries")
    plt.show()

def create_labels(seq_len):
    """
    Generate labels dynamically for sequence positions: Method Token, Semantic Category, Duration.
    """
    labels = []
    for i in range(seq_len):
        if i % 3 == 0:
            labels.append(f"Method Token {i // 3}")
        elif i % 3 == 1:
            labels.append(f"Semantic Category {i // 3}")
        else:
            labels.append(f"Duration {i // 3}")
    return labels


if __name__ == '__main__':

    # Load and preprocess your data
    

    data_tuple = flatten_data('model_map.pkl')

    #normalized_data_tuple = normalize_fixation_durations(data_tuple)

    #print_fixation_durations(data_tuple, 5)
    data_tuple = create_fixation_durations_fixed_bins(data_tuple) # fixed bins of 6 

    # Create vocabulary for semantic categories
    category_vocab = create_vocab(data_tuple)

    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    model = AutoModel.from_pretrained("microsoft/codebert-base")

    # Tokenize and embed data
    padded_data, attention_masks, max_sequence_length = preprocessing_tokenized_data(
    data_tuple,
    tokenizer,
    model,
    category_vocab,
    category_dim=64,
    duration_dim=64
    )

    # # # Save preprocessed data
    save_data("padded_data.pkl", padded_data)
    save_data("attention_masks.pkl", attention_masks)

    # print(max_sequence_length)



    # Train and validate the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    padded_data = load_data('padded_data.pkl')
    attention_masks = load_data('attention_masks.pkl')

    # # Train and validate the model
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # # Model parameters
    # # method_embedding_dim = 768
    # # reduced_method_dim = 32
    # # category_dim = 64
    # # duration_dim = 64
    # # num_heads = 4
    # # num_layers = 1
    # # num_classes = 1  # Binary classification
    # # dropout_rate = 0.1
    # max_seq_len = 32  # You can set this based on your data

    # # Initialize the model, loss function, and optimizer
    # model = Transformer(
    #     method_embedding_dim=768,  # Example embedding dimension
    #     reduced_method_dim=32,
    #     category_dim=64,
    #     duration_dim=64,
    #     num_heads=4,
    #     num_layers=1,
    #     num_classes=1,  # Binary classification
    #     dropout_rate=0.2,
    #     max_seq_len=max_seq_len
    # ).to(device)  # Use GPU if available, device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # # Loss function and optimizer
    # loss_fn = nn.BCEWithLogitsLoss()  # Binary cross-entropy with logits
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)


    # random_seeds = [0, 1, 42, 123, 12345]
    # results = []

    # for seed in random_seeds:
    #     # Split your data into training, validation, and test sets
    #     train_data, temp_data = train_test_split(padded_data, test_size=0.2, random_state=42)
    #     valid_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

    #     # Extract the quality_summary labels from your padded data (now just the second element of each tuple)
    #     train_labels = [quality_summary for _, quality_summary, _ in train_data]
    #     print(train_labels)
    #     valid_labels = [quality_summary for _, quality_summary, _ in valid_data]
    #     test_labels = [quality_summary for _, quality_summary, _ in test_data]

    #     # Initialize lists to hold embeddings and associated information
    #     train_embeddings = []
    #     train_participant_ids = []

    #     for participant_id, quality_summary, scanpath_embeddings in train_data:
    #         method_embeddings, category_embeddings, duration_embeddings = zip(*scanpath_embeddings)
    #         train_embeddings.append((method_embeddings, category_embeddings, duration_embeddings))
    #         train_participant_ids.append(participant_id)  # Store participant_id

    #     valid_embeddings = []
    #     valid_participant_ids = []


    #     for participant_id, quality_summary, scanpath_embeddings in valid_data:
    #         method_embeddings, category_embeddings, duration_embeddings = zip(*scanpath_embeddings)
    #         valid_embeddings.append((method_embeddings, category_embeddings, duration_embeddings))
    #         valid_participant_ids.append(participant_id)

    #     test_embeddings = []
    #     test_participant_ids = []
        

    #     for participant_id, quality_summary, scanpath_embeddings in test_data:
    #         method_embeddings, category_embeddings, duration_embeddings = zip(*scanpath_embeddings)
    #         test_embeddings.append((method_embeddings, category_embeddings, duration_embeddings))
    #         test_participant_ids.append(participant_id)

    #     # Now assuming `CustomDataset` expects embeddings, labels, participant_ids, and methods
    #     train_dataset = CustomDataset(train_embeddings, train_labels, train_participant_ids)
    #     valid_dataset = CustomDataset(valid_embeddings, valid_labels, valid_participant_ids)
    #     test_dataset = CustomDataset(test_embeddings, test_labels, test_participant_ids)

    #     # Create DataLoader instances
    #     train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    #     valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
    #     test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


    #     early_stopping = EarlyStopping(patience=3, delta=0.01)

    # # Training loop with EarlyStopping
    #     for epoch in range(10):  # Set the number of epochs
    #         print(f"Epoch {epoch+1}/10")
            
    #         # Train phase
    #         train_loss, train_accuracy = train(model, train_loader, loss_fn, optimizer, device)
    #         print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")

    #         # Validation phase
    #         valid_loss, valid_accuracy = validate(model, valid_loader, loss_fn, device)
    #         print(f"Validation Loss: {valid_loss:.4f}, Validation Accuracy: {valid_accuracy:.4f}")

    #         # Check for early stopping
    #         early_stopping(valid_loss, model)
    #         if early_stopping.early_stop:
    #             print(f"Early stopping at epoch: {epoch}")
    #             break

    #     test_loss, test_accuracy = test(model, test_loader, loss_fn, device)
    #     print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    #     results.append((train_loss, train_accuracy, valid_loss, valid_accuracy, test_loss, test_accuracy))

    #     # Compute mean and std of results
    # results = np.array(results)  # Shape: (num_seeds, metrics_per_seed)
    # metrics_mean = np.mean(results, axis=0)
    # metrics_std = np.std(results, axis=0)

    # print("\n=== Final Results Across Seeds ===")
    # print(f"Train Loss: {metrics_mean[0]:.4f} ± {metrics_std[0]:.4f}")
    # print(f"Train Accuracy: {metrics_mean[1]:.4f} ± {metrics_std[1]:.4f}")
    # print(f"Validation Loss: {metrics_mean[2]:.4f} ± {metrics_std[2]:.4f}")
    # print(f"Validation Accuracy: {metrics_mean[3]:.4f} ± {metrics_std[3]:.4f}")
    # print(f"Test Loss: {metrics_mean[4]:.4f} ± {metrics_std[4]:.4f}")
    # print(f"Test Accuracy: {metrics_mean[5]:.4f} ± {metrics_std[5]:.4f}")

    #     # seq_len = max_seq_len  # Use the fixed max sequence length from your data
    #     # layer_idx = 0  # Visualize the first layer (or choose dynamically)
    #     # head_idx = 0  # Visualize the first head (or choose dynamically)

    #     # for i, attention_weights in enumerate(test_attention_weights[:3]):  # Visualize first 3 samples
    #     #     visualize_attention_weights(attention_weights, seq_len, layer_idx, head_idx)
