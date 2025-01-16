
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
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder


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
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence


class CustomTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, *args, **kwargs):
        super(CustomTransformerEncoderLayer, self).__init__(*args, **kwargs)
        self.attn_weights = None  # To store attention weights

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        # print(f"Input src shape: {src.shape}")

        # Perform self-attention and get the attention weights
        src2, attn_weights = self.self_attn(
            src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )
        # print(f"Attn weights shape: {attn_weights.shape}")  # Debug line
        # print(f"Padding mask: {src_key_padding_mask}")

        # print(f"After self-attention src2 shape: {src2.shape}")

        # Save the attention weights
        self.attn_weights = attn_weights
        # Residual connection and layer norm
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # Feed-forward network with residual
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        # Add the attention output (src2) to the input (src) using residual connection
        # src = src + src2
        # print(f"After residual connection src shape: {src.shape}")

        return src, attn_weights
    
class Transformer(nn.Module):
    def __init__(self, method_embedding_dim=768, reduced_method_dim=32, 
                num_heads=4, num_layers=1, num_classes=1,  dropout_rate=0.2, max_seq_len=32): 
        super(Transformer, self).__init__()

        # Nonlinear dimensionality reduction for method embeddings
        self.method_reducer = nn.Sequential(
            nn.Linear(method_embedding_dim, method_embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(method_embedding_dim // 2, reduced_method_dim),
            nn.ReLU()
        )

        # Total embedding dimension for the transformer input
        total_embedding_dim = reduced_method_dim 

        # Positional encoding
        self.positional_encoding = nn.Parameter(
            torch.zeros((1, max_seq_len + 1, total_embedding_dim))  # Add 1 for sequence aggregator token
        )

        # Special header token (e.g., [CLS])
        self.header_token = nn.Parameter(torch.zeros(1, 1, total_embedding_dim))

        # Transformer Encoder Layer
        self.transformer_encoder_layer = CustomTransformerEncoderLayer(
            d_model=total_embedding_dim,
            nhead=num_heads,
            dropout=dropout_rate,
            # batch_first=True,  # Ensure it operates as batch first
            
        )
      
        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_encoder_layer,
            num_layers=num_layers,
            
        )
        # Dropout layer after embedding concatenation
        self.dropout = nn.Dropout(dropout_rate)

        # Fully connected layers for the output
        self.fc_layer1 = nn.Linear(total_embedding_dim, total_embedding_dim // 2)
        self.fc_layer2 = nn.Linear(total_embedding_dim // 2, num_classes) 

        # Dropout after the fully connected layers
        self.fc_dropout = nn.Dropout(dropout_rate)


    def forward(self, method_embeddings, attention_mask=None):
        # Reduce method embedding dimensions
        method_embeddings = self.method_reducer(method_embeddings)

        # Expand embeddings to match the sequence length (batch_size, seq_len, feature_dim)
        
        method_embeddings = method_embeddings.squeeze(1)  # Remove any unnecessary dimensions
    
        
        # Add special header token to the beginning of the sequence
        header_token_repeated = self.header_token.expand(method_embeddings.size(0), 1, -1)
        method_embeddings = torch.cat((header_token_repeated, method_embeddings), dim=1)

        # Add positional encodings
        seq_length = method_embeddings.size(1)
        positional_encodings = self.positional_encoding[:, :seq_length, :]
        method_embeddings = method_embeddings + positional_encodings

        # Reshape for transformer: (seq_len, batch_size, total_embedding_dim)
        method_embeddings = method_embeddings.permute(1, 0, 2)
        # print(f"After permute: {method_embeddings.shape}")

    
        if attention_mask is not None:
            # Apply attention mask to the transformer (padding tokens will be ignored)
            if attention_mask.ndimension() == 1:
        # If it's 1D, we need to expand it to 2D to match [batch_size, seq_length]
                attention_mask = attention_mask.unsqueeze(1).expand(-1, seq_length)  # [batch_size, seq_len, seq_len]
            attention_mask = torch.cat((torch.ones(attention_mask.size(0), 1).to(attention_mask.device), attention_mask), dim=1)

            attention_mask = (attention_mask == 0).to(dtype=torch.float32)  # 1 for padding tokens, 0 for non-padding tokens
            attention_mask = attention_mask * -10000.0  # Set padding positions to a large negative value
            attention_mask[attention_mask == 0] = 1.0  # Keep non-padding tokens as 0
            # print(f"Attention Mask Shape: {attention_mask.shape}")
        else:
            print("No attention mask provided; using default masking.")
        # Pass through transformer encoder
        transformer_output, attn_weights = self.transformer_encoder(
            method_embeddings, src_key_padding_mask=attention_mask
        )
        # The transformer output after the last layer
        print(f"transformer_output shape: {transformer_output.shape}")

        # Output corresponding to the special header token (first token)
        header_token_output = transformer_output[0, :, :]

        # Fully connected layers for classification
        projected_output = self.fc_layer1(header_token_output)
        projected_output = self.fc_dropout(projected_output)  # Apply dropout after the first FC layer
        final_output = self.fc_layer2(projected_output)

        print("Method Embeddings:", method_embeddings.shape)
        print("Transformer Output:", transformer_output.shape)
        print("Final Output:", final_output.shape)

        return final_output, [attn_weights]

class CustomDataset(torch.utils.data.Dataset):
    def __len__(self):
        return len(self.labels)

    def __init__(self, embeddings, labels, participant_ids, method_names, attention_masks):
        self.embeddings = embeddings  # This contains tuples of embeddings (method_embeddings)
        self.labels = labels  # Quality labels
        self.participant_ids = participant_ids  # Participant IDs
        self.method_names = method_names  # Participant IDs
        self.attention_masks = attention_masks

    def __getitem__(self, idx):
        # Extract individual embeddings
        method_embeddings = self.embeddings[idx]
        
        # Ensure embeddings are of the correct type after stacking
        method_embeddings = torch.stack(method_embeddings).to(dtype=torch.float32) if isinstance(method_embeddings, (list, tuple)) else method_embeddings
        # Check if attention_masks is a list and stack if so
        
        quality_label = self.labels[idx]
        participant_id = self.participant_ids[idx]
        method_name = self.method_names[idx]
        attention_mask = torch.tensor(self.attention_masks[idx], dtype=torch.float32)  # Ensure attention_mask is a tensor

        # Return stacked embeddings and other details
        return method_embeddings,quality_label, participant_id, method_name, attention_mask
    
def custom_collate_fn(batch, max_len=32):
    """
    Custom collate function to handle variable-length sequences in batches.

    Args:
        batch (list): List of tuples from the dataset __getitem__ method.
                      Each tuple contains: (method_embeddings, quality_label, participant_id, attention_mask)
        max_len (int): Maximum sequence length for padding.

    Returns:
        tuple: Batched tensors for method_embeddings, attention_masks, labels, and participant_ids.
    """
    # Unpack the batch
    method_embeddings, quality_label, participant_ids, method_names, attention_masks = zip(*batch)

    # Feature dimension
    feature_dim = method_embeddings[0].shape[1]  # Assuming all embeddings have the same feature dimension
    batch_size = len(batch)

    # Initialize padded tensors for embeddings and masks
    padded_embeddings = torch.zeros((batch_size, max_len, feature_dim))
    padded_attention_masks = torch.zeros((batch_size, max_len), dtype=torch.long)

    # Copy embeddings and masks for each sequence into the padded tensors
    for i, (embedding, attention_mask) in enumerate(zip(method_embeddings, attention_masks)):
        seq_len = embedding.shape[0]
        padded_embeddings[i, :seq_len, :] = embedding  # Copy original embeddings
        padded_attention_masks[i, :seq_len] = attention_mask  # Copy original attention mask


    # Convert quality labels to a tensor (no padding needed, as they are per-sequence labels)
    quality_label = torch.tensor(quality_label, dtype=torch.float32).view(-1, 1)  # Shape: [batch_size, 1]

    # Convert participant IDs to a tensor
    participant_ids = torch.tensor(participant_ids, dtype=torch.int64)  # Shape: [batch_size]
    method_names = torch.tensor(method_names)  # Shape: [batch_size]


    # Move everything to the correct device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    padded_embeddings = padded_embeddings.to(device)
    padded_quality_labels = quality_label.to(device)
    padded_attention_masks = padded_attention_masks.to(device)
    participant_ids = participant_ids.to(device)
    method_names = method_names.to(device)

    # # Debug prints
    # print(f"Padded embeddings shape: {padded_embeddings.shape}")  # [batch_size, max_len, feature_dim]
    # print(f"Quality labels shape: {padded_quality_labels.shape}")  # [batch_size, 1]
    # print(f"Attention masks shape: {padded_attention_masks.shape}")  # [batch_size, max_len]
    # print(f"Participant IDs shape: {participant_ids.shape}")  # [batch_size]

    return padded_embeddings, padded_quality_labels, participant_ids, method_names, padded_attention_masks


def custom_collate_fn_old(batch, max_len=32):
    """
    Custom collate function to handle variable-length sequences in batches.

    Args:
        batch (list): List of tuples from the dataset __getitem__ method.
                      Each tuple contains: (method_embeddings, quality_label, participant_id, attention_mask)
        max_len (int): Maximum sequence length for padding.

    Returns:
        tuple: Batched tensors for method_embeddings, attention_masks, labels, and participant_ids.
    """
    # Unpack the batch
    method_embeddings, quality_label, participant_ids, attention_masks = zip(*batch)
    # print(method_embeddings, quality_label, participant_ids, attention_masks)
    # Convert lists to tensors
    method_embeddings = torch.stack(method_embeddings)  # (batch_size, seq_len, feature_dim)
    quality_label = torch.tensor(quality_label, dtype=torch.float32).view(-1, 1)  # (batch_size, 1)
    participant_ids = torch.tensor(participant_ids, dtype=torch.int64)  # (batch_size,)
    attention_masks = torch.stack(attention_masks)  # (batch_size, seq_len)

    # Get current batch size and sequence length
    current_batch_size = method_embeddings.shape[0]
    current_seq_len = method_embeddings.shape[1]

    # print(f"method_embeddings shape before permute: {method_embeddings.shape}")
    # print(f"quality_label shape: {quality_label.shape}")
    # print(f"attention_masks shape: {attention_masks.shape}")

    method_embeddings = method_embeddings.permute(1, 0, 2)  # [seq_len, batch_size, feature_dim] -> [batch_size, seq_len, feature_dim]

    # print(f"method_embeddings shape after permute: {method_embeddings.shape}")


    # Ensure shapes are consistent
    assert method_embeddings.shape[0] == quality_label.shape[0] == attention_masks.shape[0], \
        f"Mismatch in batch size: {method_embeddings.shape[0]} vs {quality_label.shape[0]} vs {attention_masks.shape[0]}"

    # Move everything to the correct device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    method_embeddings = method_embeddings.to(device)
    quality_label = quality_label.to(device)
    attention_masks = attention_masks.to(device)
    participant_ids = participant_ids.to(device)

    # Debug prints
    print(f"Method embeddings shape: {method_embeddings.shape}")
    print(f"Quality labels shape: {quality_label.shape}")
    print(f"Attention masks shape: {attention_masks.shape}")
    print(f"Participant IDs shape: {participant_ids.shape}")

    return method_embeddings, quality_label, participant_ids, attention_masks

def train_confidenceInterval(model, train_loader, loss_fn, optimizer, device):
    model.train()  # Set model to training mode
    total_loss = 0
    correct_preds = 0
    total_preds = 0
    confidences = []  # To store confidence values for predictions

    for i, (method_embeddings, _, _, quality_label, participant_id) in enumerate(train_loader):
        # Convert lists to tensors before moving to device
        method_embeddings = torch.tensor(method_embeddings).to(device)
       
        quality_label = torch.tensor(quality_label, dtype=torch.float32).to(device)

        quality_label = quality_label.view(-1, 1)  # Reshapes to (batch_size, 1)

        # Forward pass
        outputs, _ = model(method_embeddings)

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

    for i, (method_embeddings, quality_label, participant_id, _, attention_masks) in enumerate(train_loader):
        # Convert lists to tensors before moving to device
        # method_embeddings = torch.tensor(method_embeddings).to(device)
        # print(type(attention_masks))
        #print(attention_masks)

        # # Check if attention_masks is a list and stack if so
        # if isinstance(attention_masks, list):
        #     attention_masks = torch.stack(attention_masks).to(dtype=torch.int64).to(device)
        # else:
        #     attention_masks = attention_masks.to(device)  # In case it's already a tensor

        print(f"Batch {i+1} shape: {method_embeddings.shape}, participant id, {participant_id}, quality labels Ids: {(quality_label)},  attention mask shape: {attention_masks.shape}")
        # print(method_embeddings, attention_masks)
        # print(len(method_embeddings), len(attention_masks))

        # quality_label = torch.tensor(quality_label, dtype=torch.float32).to(device)

        quality_label = quality_label.view(-1, 1)  # Reshapes to (32, 1)
        # print(f"quality_label {quality_label.shape}")

        # assert quality_label.shape == 1, f"Expected labels of shape [32, 1], but got {quality_label.shape}"

        
        # Forward pass
        outputs, _ = model(method_embeddings, attention_masks)


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

    attention_weights_list = []

    with torch.no_grad():  # Disable gradient calculation for validation
        for i, (method_embeddings, quality_label, participant_id, _, attention_masks) in enumerate(valid_loader):
            # Convert lists to tensors before moving to device
            # method_embeddings = torch.tensor(method_embeddings).to(device)
            
            # quality_label = torch.tensor(quality_label, dtype=torch.float32).to(device)

            quality_label = quality_label.view(-1, 1)  # Reshapes to (batch_size, 1)
            print(f"method tokens {method_embeddings[0]}")
            # Forward pass
            outputs, attention_weights = model(method_embeddings, attention_masks)
            attention_weights_list.append(attention_weights)
            print((attention_weights[0])) # one participant summary  [1, 33, 33]
            print(f"attention mask {attention_masks[0]}")
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
    return avg_loss, accuracy, attention_weights_list

def test(model, test_loader, loss_fn, device):
    model.eval()  # Set model to evaluation mode
    total_loss = 0
    correct_preds = 0
    total_preds = 0

    all_labels = []
    all_preds = []

    with torch.no_grad():  # No need to compute gradients for testing
        for i, (method_embeddings, quality_label, participant_id, _, attention_masks) in enumerate(test_loader):
            # Convert lists to tensors before moving to device
            method_embeddings = torch.tensor(method_embeddings).to(device)
            quality_label = torch.tensor(quality_label, dtype=torch.float32).to(device)

            quality_label = quality_label.view(-1, 1)  # Reshapes to (32, 1)

            # Forward pass
            outputs,_ = model(method_embeddings, attention_masks)

            print(f"Outputs: {outputs}")


            # Calculate loss
            loss = loss_fn(outputs, quality_label)

            # Convert outputs to binary predictions
            predicted = torch.round(torch.sigmoid(outputs))

            # Accumulate correct predictions and total predictions
            correct_preds += (predicted == quality_label).sum().item()
            total_preds += quality_label.size(0)

            # Store labels and predictions for metrics computation
            all_labels.extend(quality_label.cpu().numpy().flatten())
            all_preds.extend(predicted.cpu().numpy().flatten())
            print(all_labels)
            print(all_preds)

            # Accumulate total loss
            total_loss += loss.item()

    # Calculate average loss and accuracy after the entire epoch
    avg_loss = total_loss / len(test_loader)
    accuracy = 100 * correct_preds / total_preds

     # Compute precision, recall, and F1-score
    test_precision = precision_score(all_labels, all_preds, average='binary', zero_division=1)
    test_recall = recall_score(all_labels, all_preds, average='binary', zero_division=1)
    test_f1 = f1_score(all_labels, all_preds, average='binary', zero_division=1)

    print(f"Test Prediction: {test_precision:.4f}, Test Recall: {test_recall:.4f}, Test F1: {test_f1:.4f}")

    
    return avg_loss, accuracy, test_precision, test_recall, test_f1


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


# def plot_attention(attention_weights_list, titles):
#     print(f"Starting plot_attention with {len(attention_weights_list)} entries.")
    
#     for idx, attention_weights in enumerate(attention_weights_list):
#         print(f"Processing entry {idx} with title '{titles[idx % len(titles)]}'")
#         if attention_weights is None:
#             print(f"Skipping entry {idx}: None detected.")
#             continue

#         if isinstance(attention_weights, list):
#             print(f"Stacking list of tensors at entry {idx}.")
#             attention_weights = torch.stack(attention_weights, dim=0)

#         attention_weights = attention_weights.detach().cpu().numpy()

#         if len(attention_weights.shape) == 4:
#             print(f"Averaging attention heads for entry {idx}.")
#             attention_weights = attention_weights.mean(axis=1)
#             attention_weights = attention_weights[0]

#         if len(attention_weights.shape) != 2:
#             raise ValueError(f"Entry {idx}: Expected 2D attention weights, got {attention_weights.shape}")

#         print(f"Attention Weights Matrix (after normalization):\n{attention_weights}")
        
#         plt.figure(figsize=(10, 8))
#         sns.heatmap(attention_weights, cmap="Purples", square=True, annot=False)
#         plt.title(f"Attention Weights: {titles}")
#         plt.xlabel("Key Positions")
#         plt.ylabel("Query Positions")
#         # Tight layout to ensure spacing is appropriate
#         plt.tight_layout()
#         plt.show()
        

def find_boundary(attention_weights):
    """
    Find the index where the repetition starts in the attention matrix based on query-key pairs.

    Parameters:
    - attention_weights: 2D numpy array of attention weights.

    Returns:
    - index of the boundary where repetition starts.
    """
    
    num_queries, num_keys = attention_weights.shape
    # print("Query-Key Values (q*v):")

    # Iterate over all query and key pairs
    for i in range(1, num_queries):
        for j in range(num_keys):
            q_v_value = attention_weights[i, j]
            # print(f"Query {i}, Key {j}, Value: {q_v_value:.5f}")
    
    # # Check for repetition
    # for i in range(1, num_queries):  # Start at 1 to skip the first row
    #     for j in range(i):  # Compare with all previous rows
    #         if np.allclose(attention_weights[i], attention_weights[j]):
    #             # print(f"Repetition detected between rows {i} and {j}")
    #             return i  # Return the index where repetition starts
    #         # else:
    #         #     # Debugging: Print differences when rows do not match
    #         #     diff = np.abs(attention_weights[i] - attention_weights[j])
    #         #     # print(f"Row {i} vs Row {j} differences: {diff}")
    
    # return attention_weights.shape[0]  # If no repetition, return the size of the matrix
    # Initialize variables to track the index where repetition is found
    query_repetition_index = None
    key_repetition_index = None

    # Check for repetition in queries
    for i in range(1, num_queries):  # Start at 1 to skip the first row
        for j in range(i):  # Compare with all previous rows
            if np.allclose(attention_weights[i], attention_weights[j]):
                query_repetition_index = i
                print(f"Repetition detected between rows (queries) {i} and {j}")
                break
        if query_repetition_index is not None:
            break  # Exit outer loop once query repetition is found
    
    # Check for repetition in keys (columns of the attention matrix)
    for j in range(1, num_keys):  # Start at 1 to skip the first column
        for k in range(j):  # Compare with all previous columns
            if np.allclose(attention_weights[:, j], attention_weights[:, k]):
                key_repetition_index = j
                print(f"Repetition detected between columns (keys) {j} and {k}")
                break
        if key_repetition_index is not None:
            break  # Exit outer loop once key repetition is found
    
    # Determine which index to return
    if query_repetition_index is None and key_repetition_index is None:
        return attention_weights.shape[0]  # If no repetition, return the size of the matrix
    elif query_repetition_index is None:
        return key_repetition_index
    elif key_repetition_index is None:
        return query_repetition_index
    else:
        # Return the lower index of the two
        return min(query_repetition_index, key_repetition_index)

def find_scanpathLength_boundary(index):
    scanpath_dict = {
    0: 3, 1: 12, 2: 5, 3: 10, 4: 8, 5: 6, 6: 13, 7: 20, 8: 12, 9: 9,
    10: 10, 11: 17, 12: 17, 13: 6, 14: 8, 15: 8, 16: 8, 17: 16, 18: 9,
    19: 8, 20: 9, 21: 7, 22: 0, 23: 4, 24: 4, 25: 2, 26: 2, 27: 3,
    28: 7, 29: 1, 30: 3, 31: 0, 32: 5, 33: 2, 34: 1, 35: 8, 36: 16
    }

    if index in scanpath_dict:
        return scanpath_dict[index]
    else:
        print(f"Index {index} not found in the dictionary.")
        return None

def plot_attention(attention_weights_list, titles, specific_index, pad_token_idx=0):
    """
    Plots the attention weights heatmap for a specific index, excluding padded tokens.

    Parameters:
    - attention_weights_list: List of lists containing attention tensors.
    - titles: Title for the plot (string or list).
    - specific_index: Index specifying which attention matrix to visualize.
    - pad_token_idx: Index of the padding token (default is 0).
    """
    attention_weights = None

    # Validate the specific_index
    if specific_index < 0 or specific_index > 37:
        print("Invalid index. Please provide an index between 0 and 37.")
        return

    # Iterate through attention_weights_list
    for list_idx, tensor_list in enumerate(attention_weights_list):
        print(f"Processing list {list_idx}:")

        # Check if the tensor_list is valid
        if tensor_list is None or len(tensor_list) == 0:
            continue

        # Iterate through each batch tensor in the list
        for batch_idx, batch_tensor in enumerate(tensor_list):
            # Iterate through each sample in the batch
            for sample_idx in range(batch_tensor.size(0)):  # Iterate over the batch dimension
                sample = batch_tensor[sample_idx]  # Shape: (33, 33)

                # If the sample matches the specific index, select it
                if sample_idx == specific_index:
                    print(f"    Processing List {list_idx}, Batch {batch_idx}, Sample {sample_idx}, Shape: {sample.shape}")
                    attention_weights = sample
                    break
            if attention_weights is not None:
                break
        if attention_weights is not None:
            break

    # Validate if attention_weights was found
    if attention_weights is None:
        print(f"No attention weights found for specific index {specific_index}.")
        return

    # Convert attention weights to numpy if it's a tensor
    if hasattr(attention_weights, 'detach'):
        attention_weights = attention_weights.detach().cpu().numpy()

    # Check the shape and ensure it's 2D
    if attention_weights.ndim == 3:  # Shape: (batch, seq, seq)
        attention_weights = attention_weights[0]  # Remove batch dimension
    elif attention_weights.ndim != 2:
        print(f"Invalid attention weights shape: {attention_weights.shape}")
        return
    
    # Find boundary where the repetition starts
    # boundary = find_boundary(attention_weights)
    boundary = find_scanpathLength_boundary(specific_index)
    print(attention_weights.shape[0])
    print(f"boundary {boundary}")
    # Find boundary where the repetition starts based on query-key pairs
    if boundary < attention_weights.shape[0]:
        print(f"Repetition detected at index {boundary}, excluding repeated data.")
        attention_weights = attention_weights[:boundary, :boundary]  # Crop both query and key values

    # print("\nQuery-Key Attention Weights:")
    # for i in range(attention_weights.shape[0]):
    #     for j in range(attention_weights.shape[1]):
    #         print(f"Query {i}, Key {j}: {attention_weights[i, j]:.4f}")
    # sub_matrix = attention_weights[0:10, 0:10]  # This slices the first 10 queries and keys
    # print(f"Attention Weights for Query 0-9, Key 0-9:\n{sub_matrix}")
    
    # Create a mask for the padded tokens (assuming padding token is at index 0)
    # mask = (attention_weights == pad_token_idx)
    # # Set the attention weights for padded positions to NaN
    # attention_weights[mask] = np.nan
    # Prepare labels, setting the first position to "CLS"
        
    min_boundary = 5 # manually change this depdening on what the boundary is between base and combined model
    # Exclude CLS token by slicing the matrix
    attention_weights = attention_weights[1:, 1:]  # Exclude the first row and column
    # if (min_boundary < boundary):
    #     attention_weights = attention_weights[:min_boundary, :min_boundary]

    
    # Update labels starting from 0 after CLS
    num_tokens = attention_weights.shape[0]
    labels = [str(i) for i in range(num_tokens)]  # Start labels from 0

    # Plot the attention heatmap
    plt.figure(figsize=(10, 8))
    custom_cmap = sns.light_palette("#7c70adff", as_cmap=True)
    ax = sns.heatmap(attention_weights, cmap=custom_cmap, square=True, annot=False)
    
    # Update axis labels
    ax.set_xticks(np.arange(num_tokens) + 0.5)
    ax.set_yticks(np.arange(num_tokens) + 0.5)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels, rotation=0)

    plt.title(f"{titles if isinstance(titles, str) else 'Attention Weights Visual'}")
    plt.xlabel("Key")
    plt.ylabel("Query")
    plt.tight_layout()
    plt.show() 
    

if __name__ == '__main__':
    padded_data = load_data('padded_data.pkl')
    attention_masks = load_data('attention_masks.pkl')

    # Train and validate the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    max_seq_len = 32  # You can set this based on your data

    # Initialize the model, loss function, and optimizer
    model = Transformer(
        method_embedding_dim=768,  # Example embedding dimension
        reduced_method_dim=32,
        num_heads=4,
        num_layers=1,
        num_classes=1,  # Binary classification
        dropout_rate=0.2,
        max_seq_len=max_seq_len
    ).to(device)  # Use GPU if available, device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Loss function and optimizer
    loss_fn = nn.BCEWithLogitsLoss()  # Binary cross-entropy with logits
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # Define random seeds
    random_seeds = [0, 1, 42, 123, 12345]
    results = []

    # Extract unique method names for encoding
    unique_method_names = list({method_name for _, _, method_name, _ in padded_data})
    # print(unique_method_names)

    # Initialize and fit the label encoder
    label_encoder = LabelEncoder()
    label_encoder.fit(unique_method_names)

    # Save the label mapping for future use
    label_mapping = {idx: label for idx, label in enumerate(label_encoder.classes_)}
    with open("label_mapping.pkl", "wb") as f:
        pickle.dump(label_mapping, f)

    # Encode method names for the entire dataset
    encoded_data = [
        (pid, qual, label_encoder.transform([method_name])[0], scanpath)
        for pid, qual, method_name, scanpath in padded_data
    ]

    # print(encoded_data[2])

    for seed in random_seeds:
        print(f"\n=== Starting run with random seed: {seed} ===")
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Split data with the current random seed
        train_data, temp_data = train_test_split(encoded_data, test_size=0.2, random_state=seed)
        valid_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=seed)

        # Extract the quality_summary labels from your padded data (now just the second element of each tuple)
        train_labels = [quality_summary for _, quality_summary, _, _ in train_data]
        # print(train_labels)
        valid_labels = [quality_summary for _, quality_summary,_, _ in valid_data]
        test_labels = [quality_summary for _, quality_summary,_, _ in test_data]

        # Prepare datasets
        train_embeddings, train_participant_ids, train_method_names = [], [], []
        for participant_id, _, method_name, scanpath_embeddings in train_data:
            method_embeddings, _, _ = zip(*scanpath_embeddings)
            train_embeddings.append(method_embeddings)
            train_participant_ids.append(participant_id)
            train_method_names.append(method_name)


        valid_embeddings, valid_participant_ids, valid_method_names = [], [], [] 
        for participant_id, _, method_name, scanpath_embeddings in valid_data:
            method_embeddings, _, _ = zip(*scanpath_embeddings)
            valid_embeddings.append(method_embeddings)
            valid_participant_ids.append(participant_id)
            valid_method_names.append(method_name)

        test_embeddings, test_participant_ids, test_method_names = [], [], []
        for participant_id, _, method_name, scanpath_embeddings in test_data:
            method_embeddings, _, _ = zip(*scanpath_embeddings)
            test_embeddings.append(method_embeddings)
            test_participant_ids.append(participant_id)
            test_method_names.append(method_name)

        train_attention_masks = attention_masks[:len(train_data)]
        valid_attention_masks = attention_masks[len(train_data):len(train_data) + len(valid_data)]
        test_attention_masks = attention_masks[len(train_data) + len(valid_data):]

        # Create datasets
        train_dataset = CustomDataset(train_embeddings, train_labels, train_participant_ids, train_method_names, train_attention_masks)
        valid_dataset = CustomDataset(valid_embeddings, valid_labels, valid_participant_ids, train_method_names, valid_attention_masks)
        test_dataset = CustomDataset(test_embeddings, test_labels, test_participant_ids, train_method_names, test_attention_masks)

        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=custom_collate_fn) 
        valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, collate_fn=custom_collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=custom_collate_fn)

        # Initialize model, loss function, and optimizer for this run
        model = Transformer(
            method_embedding_dim=768,
            reduced_method_dim=32,
            num_heads=4,
            num_layers=1,
            num_classes=1,
            dropout_rate=0.2,
            max_seq_len=32
        ).to(device)

        loss_fn = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        early_stopping = EarlyStopping(patience=3, delta=0.01)

        # Training loop
        for epoch in range(10):  # Adjust the number of epochs as needed
            print(f"Epoch {epoch+1}/10")
            
            
            # Train phase
            train_loss, train_accuracy = train(model, train_loader, loss_fn, optimizer, device)
            print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")

            # Validation phase
            valid_loss, valid_accuracy, attention_weights = validate(model, valid_loader, loss_fn, device)
            print(f"Validation Loss: {valid_loss:.4f}, Validation Accuracy: {valid_accuracy:.4f}")

            # Early stopping
            early_stopping(valid_loss, model)
            if early_stopping.early_stop:
                print(f"Early stopping at epoch: {epoch}")
                break

        # Test phase
        # test_loss, test_accuracy = test(model, test_loader, loss_fn, device)
        test_loss, test_accuracy, test_precision, test_recall, test_f1 = test(model, test_loader, loss_fn, device)

        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

        # Save results for this seed
        results.append((train_loss, train_accuracy, valid_loss, valid_accuracy, test_loss, test_accuracy, test_precision, test_recall, test_f1))

    # Compute mean and std of results
    results = np.array(results)  # Shape: (num_seeds, metrics_per_seed)
    metrics_mean = np.mean(results, axis=0)
    metrics_std = np.std(results, axis=0)

    print("\n=== Final Results Across Seeds ===")
    print(f"Train Loss: {metrics_mean[0]:.4f} ± {metrics_std[0]:.4f}")
    print(f"Train Accuracy: {metrics_mean[1]:.4f} ± {metrics_std[1]:.4f}")
    print(f"Validation Loss: {metrics_mean[2]:.4f} ± {metrics_std[2]:.4f}")
    print(f"Validation Accuracy: {metrics_mean[3]:.4f} ± {metrics_std[3]:.4f}")
    print(f"Test Loss: {metrics_mean[4]:.4f} ± {metrics_std[4]:.4f}")
    print(f"Test Accuracy: {metrics_mean[5]:.4f} ± {metrics_std[5]:.4f}")
    print(f"Test Precision: {metrics_mean[6]:.4f} ± {metrics_std[6]:.4f}")
    print(f"Test Recall: {metrics_mean[7]:.4f} ± {metrics_std[7]:.4f}")
    print(f"Test F1: {metrics_mean[8]:.4f} ± {metrics_std[8]:.4f}")

    # Plot the attention weights
    plot_attention(
        attention_weights,
        titles= "Base Model: Method Tokens", specific_index=28)
