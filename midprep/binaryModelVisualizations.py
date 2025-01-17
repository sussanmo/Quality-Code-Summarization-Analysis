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
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from matplotlib.ticker import MaxNLocator


import pandas as pd
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

import torch
import torch.nn as nn

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
        # print(f"Padding mask: {src_key_padding_mask}")

    
        # Save the attention weights
        self.attn_weights = attn_weights
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
        # total_embedding_dim = reduced_method_dim + category_dim # feature isolation for individual heatmaps
        total_embedding_dim = reduced_method_dim + duration_dim # feature isolation for individual heatmaps

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
        # self.transformer_encoder_layer = nn.TransformerEncoderLayer(
        #     d_model=total_embedding_dim,
        #     nhead=num_heads,
        #     dropout=dropout_rate
        # )
        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_encoder_layer,
            num_layers=num_layers,
            
        )
        

        # # Custom Transformer Encoder Layers
        # self.layers = nn.ModuleList([
        #     CustomTransformerEncoderLayer(
        #         d_model=total_embedding_dim,
        #         nhead=num_heads,
        #         dropout=dropout_rate,
        #         batch_first=True,
        #     )
            
        # ])

        # Dropout layer after embedding concatenation
        self.dropout = nn.Dropout(dropout_rate)

        # Fully connected layers for the output
        self.fc_layer1 = nn.Linear(total_embedding_dim, total_embedding_dim // 2)
        self.fc_layer2 = nn.Linear(total_embedding_dim // 2, num_classes)

        # Dropout after the fully connected layers
        self.fc_dropout = nn.Dropout(dropout_rate)

    def forward(self, method_embeddings, category_embeddings, duration_embeddings, attention_mask):
        # Reduce method embedding dimensions
        method_embeddings = self.method_reducer(method_embeddings)

        # Expand embeddings to match the sequence length (batch_size, seq_len, feature_dim)
        method_embeddings = method_embeddings.squeeze(1)  # Remove any unnecessary dimensions
        category_embeddings = category_embeddings.squeeze(1)  # Same for category_embeddings
        duration_embeddings = duration_embeddings.squeeze(1)  # Same for duration_embeddings

        # Concatenate embeddings along the feature dimension
        # combined_embeddings = torch.cat(
        #     (method_embeddings, category_embeddings, duration_embeddings), dim=2
        # )  # Shape: (batch_size, seq_len, total_embedding_dim)

        # combined_embeddings = torch.cat(
        #     (method_embeddings, category_embeddings), dim=2
        # )  # Shape: (batch_size, seq_len, total_embedding_dim)

        combined_embeddings = torch.cat(
            (method_embeddings, duration_embeddings), dim=2
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
            if attention_mask.ndimension() == 1:
        # If it's 1D, we need to expand it to 2D to match [batch_size, seq_length]
                attention_mask = attention_mask.unsqueeze(1).expand(-1, seq_length)  # [batch_size, seq_len, seq_len]
            attention_mask = torch.cat((torch.ones(attention_mask.size(0), 1).to(attention_mask.device), attention_mask), dim=1)

            # Convert padding tokens to large negative values so they won't be attended to
            # attention_mask = (attention_mask == 0).to(dtype=torch.float32)
            # attention_mask = (1.0 - attention_mask) * -10000.0  # Large negative value for padding
            attention_mask = (attention_mask == 0).to(dtype=torch.float32)  # 1 for padding tokens, 0 for non-padding tokens
            attention_mask = attention_mask * -10000.0  # Set padding positions to a large negative value
            attention_mask[attention_mask == 0] = 1.0  # Keep non-padding tokens as 0

            # print(f"Attention Mask Shape: {attention_mask.shape}")
        else:
            print("No attention mask provided; using default masking.")
            
        # Pass through transformer encoder
        # transformer_output = self.transformer_encoder(
        #     combined_embeddings, src_key_padding_mask=attention_mask # Get attention weights
        # )
            
        transformer_output, attention_weights = self.transformer_encoder(
        combined_embeddings, src_key_padding_mask=attention_mask # Get attention weights
        )
        
        # transformer_output  = self.transformer_encoder(
        #     combined_embeddings, src_key_padding_mask=attention_mask
        # )

        # attention_weights = self.transformer_encoder_layer.attn_weights
        # attention_weights = self.transformer_encoder_layer.attn_weights  # This should now be correctly assigned

        # Initialize attention weights list
        all_attention_weights = []

        # Pass through each transformer encoder layer
        # transformer_output = combined_embeddings  # Initialize with the input embeddings

        # for layer in self.layers:
        #     transformer_output, attn_weights = layer(
        #         transformer_output, src_mask=attention_mask
        #     )
        #     all_attention_weights.append(attn_weights)

        # for i, layer in enumerate(self.layers):
        #     transformer_output, attn_weights = layer(combined_embeddings, src_mask=attention_mask)
        #     print(f"After Layer {i + 1}, transformer_output shape: {transformer_output.shape}")
        #     all_attention_weights.append(attn_weights)
       

        # transformer_output, attn_weights = self.layers[0](combined_embeddings, src_mask=attention_mask)

        # The transformer output after the last layer
        # print(f"transformer_output shape: {transformer_output.shape}")

        # Output corresponding to the special header token (first token)
        header_token_output = transformer_output[0, :, :]

        # Fully connected layers for classification
        projected_output = self.fc_layer1(header_token_output)
        
        # Apply dropout after the first fully connected layer
        projected_output = self.fc_dropout(projected_output)
        
        final_output = self.fc_layer2(projected_output)

        # print("Method Embeddings:", method_embeddings.shape)
        # print("Transformer Output:", transformer_output.shape)
        # print("Final Output:", final_output.shape)

        return final_output, [attention_weights]




class CustomDataset(torch.utils.data.Dataset):
    def __len__(self):
        return len(self.labels)

    def __init__(self, embeddings, labels, participant_ids, method_names, attention_masks):
        self.embeddings = embeddings  # This contains tuples of embeddings (method_embeddings, category_embeddings, duration_embeddings)
        self.labels = labels  # Quality labels
        self.participant_ids = participant_ids  # Participant IDs
        self.method_names = method_names  # Participant IDs
        self.attention_masks = attention_masks # List of attention masks

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
        method_name = self.method_names[idx]
        # attention_masks = self.attention_masks[idx]
        attention_mask = torch.tensor(self.attention_masks[idx], dtype=torch.float32)  # Ensure attention_mask is a tensor


        # Return stacked embeddings and other details
        return method_embeddings, category_embeddings, duration_embeddings, quality_label, participant_id, method_name, attention_mask

  
def custom_collate_fn(batch, max_len=32):
    """
    Custom collate function to handle variable-length sequences in batches.

    Args:
        batch (list): List of tuples from the dataset __getitem__ method.
                      Each tuple contains: (method_embeddings, category_embeddings, duration_embeddings, quality_label, participant_id, attention_mask)
        max_len (int): Maximum sequence length for padding.

    Returns:
        tuple: Batched tensors for method_embeddings, category_embeddings, duration_embeddings, attention_masks, labels, and participant_ids.
    """
    # Unpack the batch
    method_embeddings, category_embeddings, duration_embeddings, quality_label, participant_ids, method_names, attention_masks = zip(*batch)

    # Feature dimensions for each embedding
    feature_dim_method = method_embeddings[0].shape[1]  # Assuming all embeddings have the same feature dimension
    feature_dim_category_duration = category_embeddings[0].shape[1]
    batch_size = len(batch)

    # Initialize padded tensors for embeddings and masks
    padded_method_embeddings = torch.zeros((batch_size, max_len, feature_dim_method))
    padded_category_embeddings = torch.zeros((batch_size, max_len, feature_dim_category_duration))
    padded_duration_embeddings = torch.zeros((batch_size, max_len, feature_dim_category_duration))  # Assuming same dim for category and duration
    padded_attention_masks = torch.zeros((batch_size, max_len), dtype=torch.long)

    # Copy embeddings and masks for each sequence into the padded tensors
    for i, (method_embedding, category_embedding, duration_embedding, attention_mask) in enumerate(zip(method_embeddings, category_embeddings, duration_embeddings, attention_masks)):
        seq_len = method_embedding.shape[0]
        padded_method_embeddings[i, :seq_len, :] = method_embedding  # Copy original method embeddings
        padded_category_embeddings[i, :seq_len, :] = category_embedding  # Copy original category embeddings
        padded_duration_embeddings[i, :seq_len, :] = duration_embedding  # Copy original duration embeddings
        padded_attention_masks[i, :seq_len] = attention_mask  # Copy original attention mask

    # Convert quality labels to a tensor (no padding needed, as they are per-sequence labels)
    quality_label = torch.tensor(quality_label, dtype=torch.float32).view(-1, 1)  # Shape: [batch_size, 1]

    # Convert participant IDs to a tensor
    participant_ids = torch.tensor(participant_ids, dtype=torch.int64)  # Shape: [batch_size]

    # label_encoder = LabelEncoder()
    # method_names_encoded = label_encoder.fit_transform(method_names)

    # # Optionally, save just the mapping for decoding
    # label_mapping = {idx: label for idx, label in enumerate(label_encoder.classes_)}  # Use label_encoder.classes_
    # with open("label_mapping.pkl", "wb") as f:
    #     pickle.dump(label_mapping, f)

    # Now you can convert the encoded integers to a tensor
    method_names_tensor = torch.tensor(method_names)

    # Move everything to the correct device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    padded_method_embeddings = padded_method_embeddings.to(device)
    padded_category_embeddings = padded_category_embeddings.to(device)
    padded_duration_embeddings = padded_duration_embeddings.to(device)
    padded_quality_labels = quality_label.to(device)
    padded_attention_masks = padded_attention_masks.to(device)
    participant_ids = participant_ids.to(device)
    method_names = method_names_tensor.to(device)

    # Debug prints
    # print(f"Padded method embeddings shape: {padded_method_embeddings.shape}")  # [batch_size, max_len, feature_dim]
    # print(f"Padded category embeddings shape: {padded_category_embeddings.shape}")  # [batch_size, max_len, feature_dim]
    # print(f"Padded duration embeddings shape: {padded_duration_embeddings.shape}")  # [batch_size, max_len, feature_dim]
    # print(f"Quality labels shape: {padded_quality_labels.shape}")  # [batch_size, 1]
    # print(f"Attention masks shape: {padded_attention_masks.shape}")  # [batch_size, max_len]
    # print(f"Participant IDs shape: {participant_ids.shape}")  # [batch_size]

    return padded_method_embeddings, padded_category_embeddings, padded_duration_embeddings, padded_quality_labels, participant_ids, method_names, padded_attention_masks



def train(model, train_loader, loss_fn, optimizer, device):
    model.train()  # Set model to training mode
    total_loss = 0
    correct_preds = 0
    total_preds = 0

    for i, (method_embeddings, category_embeddings, duration_embeddings, quality_label, participant_id, method_names, attention_masks) in enumerate(train_loader):
        # Convert lists to tensors before moving to device
        method_embeddings = torch.tensor(method_embeddings).to(device)
        category_embeddings = torch.tensor(category_embeddings).to(device)
        duration_embeddings = torch.tensor(duration_embeddings).to(device)
        
        # print(f"Method embeddings: {method_embeddings[0]}")
        # print(f"category embeddings: {category_embeddings[0]}")
        # print(f"duration embeddings: {duration_embeddings[0]}")

        # print(f"Attention masks: {attention_masks[0]}")

        # # Check if attention_masks is a list and stack if so
        # if isinstance(attention_masks, list):
        #     attention_masks = torch.stack(attention_masks).to(dtype=torch.int64).to(device)
        # else:
        #     attention_masks = attention_masks.to(device)  # In case it's already a tensor

        # attention_masks = attention_masks.unsqueeze(1)

        # quality_label = torch.tensor(quality_label, dtype=torch.float32).to(device)


        quality_label = quality_label.view(-1, 1)  # Reshapes to (32, 1)
        # Forward pass
        outputs, _= model(method_embeddings, category_embeddings, duration_embeddings, attention_masks)

        # print(f"Type of outputs: {type(outputs)}")
        # if isinstance(outputs, tuple):
        #     print(f"Length of tuple outputs: {len(outputs)}")
        #     for i, out in enumerate(outputs):
        #         print(f"Shape of outputs[{i}]: {type(out)}")
        # else:
        #     print(f"Shape of outputs: {outputs.shape}")

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
        # if i % 10 == 0:  # Adjust the frequency of printing as necessary
        #     print(f"Batch {i}/{len(train_loader)}:")
        #     print(f"Predictions: {predicted.squeeze(1).cpu().detach().numpy()}")
        #     print(f"Ground Truth: {quality_label.squeeze(1).cpu().detach().numpy()}")
        #     print(f"Participant ID: {participant_id.cpu().detach().numpy()}")  # Print participant_id
        #     print(f"Train Loss: {loss.item():.4f}")
        #     print(f"Train Accuracy: {100 * correct_preds / total_preds:.2f}%")

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
    attention_weights_list = []  # To store attention weights for visualization
    
    with open("label_mapping.pkl", "rb") as f:
        label_mapping = pickle.load(f)

    

    with torch.no_grad():  # Disable gradient calculation for validation
        for i, (method_embeddings, category_embeddings, duration_embeddings, quality_label, participant_id, method_names_encoded, attention_masks) in enumerate(valid_loader):
            print(f"Batch {i+1} size: {method_embeddings.shape[0]}")  # Print batch size (number of samples)

            scanpath_lengths = attention_masks.sum(dim=1)  # Sum along the token axis (dim=1)

            decoded_method_names = [label_mapping[encoded.item()] for encoded in method_names_encoded]

            scanpath_lengths = attention_masks.sum(dim=1)  # Sum along the token axis (dim=1)

            for idx, (scanpath_length, pid, quality, decoded_method_name) in enumerate(zip(scanpath_lengths, participant_id, quality_label, decoded_method_names)):
                global_index = i * method_embeddings.shape[0] + idx
                print(f"Sample {global_index}: Participant ID = {pid}, Quality = {quality}, Method = {decoded_method_name}, Scanpath Length = {scanpath_length.item()}")

            # method_embeddings = torch.tensor(method_embeddings).to(device)
            # category_embeddings = torch.tensor(category_embeddings).to(device)
            # duration_embeddings = torch.tensor(duration_embeddings).to(device)
            # attention_mask = attention_mask.to(device)

            # quality_label = torch.tensor(quality_label, dtype=torch.float32).to(device)

            quality_label = quality_label.view(-1, 1)  # Reshapes to (batch_size, 1)

            # Forward pass
            outputs, attention_weights = model(method_embeddings, category_embeddings, duration_embeddings, attention_masks)

            # print((attention_weights[0])) # one participant summary  [1, 33, 33]
            # print(attention_masks[0])
            # print(f"Outputs: {outputs}")
            # print(attention_weights)  # all attention weights for all batches: [32, 33, 33]
            # print("Attention Weights for Padded Tokens:")
            # padded_indices = (attention_masks == 0)
            padded_indices = (attention_masks[0] == 0)
            
            # # print("Inspecting Attention Weights for Padded Tokens (Batch 0):")
            # for idx, token_attention in enumerate(attention_weights[0]):  # Iterate through each token's attention weights
            #     for head_idx, token in enumerate(token_attention):  # Iterate through each attention head's score for the token
            #         if padded_indices[idx]:  # Check if this token is padded
            #             print(f"Padding Token at Index {idx}: {token_attention}")  # Print the entire token_attention for padded token
            #         else:
            #             print(f"Attention for Token {idx}, Head {head_idx}: {token}")  # Print the individual attention score for each head

            # print("Inspecting Attention Weights for Padded Tokens (Batch 0):")
            # for idx, token_attention in enumerate(attention_weights[0]):  # Iterate through each token's attention weights
            #     for head_idx, token in enumerate(token_attention):  # Iterate through each attention head's score for the token
            #         if padded_indices[0, idx]:  # Check if this token is padded
            #             print(f"0:  {token_attention}")  # Print the entire token_attention for padded token
            #         else: 
            #             print(f"1:  {token}")  # Print the individual attention score for each head
                
            
            # Store attention weights if available
            attention_weights_list.append(attention_weights if attention_weights is not None else print("NONE"))

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

    all_labels = []  # To store all true labels
    all_preds = []   # To store all predicted labels
    with torch.no_grad():  # No need to compute gradients for testing
        for i, (method_embeddings, category_embeddings, duration_embeddings, quality_label, participant_id, method_name, attention_masks) in enumerate(test_loader):
            # Convert lists to tensors before moving to device
            # method_embeddings = torch.tensor(method_embeddings).to(device)
            # category_embeddings = torch.tensor(category_embeddings).to(device)
            # duration_embeddings = torch.tensor(duration_embeddings).to(device)
            # attention_mask = attention_mask.to(device)

            # quality_label = torch.tensor(quality_label, dtype=torch.float32).to(device)

            quality_label = quality_label.view(-1, 1)  # Reshapes to (32, 1)

            # Forward pass
            outputs, _ = model(method_embeddings, category_embeddings, duration_embeddings, attention_masks)

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
    test_precision = precision_score(all_labels, all_preds, zero_division=0)
    test_recall = recall_score(all_labels, all_preds, zero_division=0)
    test_f1 = f1_score(all_labels, all_preds, zero_division=0)

    # print(f"Test Prediction: {test_precision:.4f}, Test Recall: {test_recall:.4f}, Test F1: {test_f1:.4f}")

    return avg_loss, accuracy, test_precision, test_recall, test_f1

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

# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt

# def plot_attention(attention_weights_list, titles, specific_index):
#     """
#     Plots the attention weights heatmap for a specific index, highlighting the boundary if needed.

#     Parameters:
#     - attention_weights_list: List of lists containing attention tensors.
#     - titles: Title for the plot (string or list).
#     - specific_index: Index specifying which attention matrix to visualize.
#     """
#     attention_weights = None

#     # Validate the specific_index
#     if specific_index < 0 or specific_index > 37:
#         print("Invalid index. Please provide an index between 0 and 37.")
#         return

#     # Iterate through attention_weights_list
#     for list_idx, tensor_list in enumerate(attention_weights_list):
#         print(f"Processing list {list_idx}:")

#         # Check if the tensor_list is valid
#         if tensor_list is None or len(tensor_list) == 0:
#             continue

#         # Iterate through each batch tensor in the list
#         for batch_idx, batch_tensor in enumerate(tensor_list):
#             # Iterate through each sample in the batch
#             for sample_idx in range(batch_tensor.size(0)):  # Iterate over the batch dimension
#                 sample = batch_tensor[sample_idx]  # Shape: (33, 33)

#                 # If the sample matches the specific index, select it
#                 if sample_idx == specific_index:
#                     print(f"    Processing List {list_idx}, Batch {batch_idx}, Sample {sample_idx}, Shape: {sample.shape}")
#                     attention_weights = sample
#                     break
#             if attention_weights is not None:
#                 break
#         if attention_weights is not None:
#             break

#     # Validate if attention_weights was found
#     if attention_weights is None:
#         print(f"No attention weights found for specific index {specific_index}.")
#         return

#     # Convert attention weights to numpy if it's a tensor
#     if hasattr(attention_weights, 'detach'):
#         attention_weights = attention_weights.detach().cpu().numpy()

#     # Check the shape and ensure it's 2D
#     if attention_weights.ndim == 3:  # Shape: (batch, seq, seq)
#         attention_weights = attention_weights[0]  # Remove batch dimension
#     elif attention_weights.ndim != 2:
#         print(f"Invalid attention weights shape: {attention_weights.shape}")
#         return

#     print(f"Attention Matrix Shape: {attention_weights.shape}")

#     # Apply find_boundary if necessary
#     if 'find_boundary' in globals():
#         boundary_index = find_boundary(attention_weights)
#         print(f"Boundary index: {boundary_index}")
#     else:
#         print("Warning: 'find_boundary' function not defined.")

#     # Plot the attention heatmap
#     plt.figure(figsize=(10, 8))
    
#     # Highlight the boundary if necessary
#     if boundary_index < attention_weights.shape[0]:
#         # Set the part of the matrix after the boundary to a different color, e.g., zero out the values after the boundary
#         attention_weights[boundary_index:] = np.nan  # Set the values after the boundary to NaN (not plotted)
#         sns.heatmap(attention_weights, cmap="Blues", square=True, annot=False, cbar_kws={'label': 'Attention Weight'})
#         plt.axvline(x=boundary_index, color='r', linestyle='--', label='Boundary')  # Mark the boundary
#         plt.axhline(y=boundary_index, color='r', linestyle='--')
#     else:
#         sns.heatmap(attention_weights, cmap="Blues", square=True, annot=False, cbar_kws={'label': 'Attention Weight'})

#     plt.title(f"Attention Weights: {titles if isinstance(titles, str) else 'Heatmap'}")
#     plt.xlabel("Key Positions")
#     plt.ylabel("Query Positions")
#     plt.tight_layout()
#     plt.legend(loc="upper right")
#     plt.show()



# def find_boundary(arrays):
#     last_array = arrays[0]
#     for idx, current_array in enumerate(arrays[1:], start=1):  # Start enumeration from 1, since we have already considered the 0-th element as the last_array.
#         if np.allclose(current_array, last_array):  # Use np.allclose to check if all elements are approximately equal, considering possible floating-point errors.
#             return idx  # Return the index where the repetition starts.
#         last_array = current_array
#     return len(arrays)  # Return the length of arrays if no repetition is found.

# def find_boundar(attention_matrix):
#     """
#     Finds the boundary index in the attention matrix where attention values exceed a dynamic threshold.

#     Parameters:
#     - attention_matrix: A 2D numpy array of attention weights.

#     Returns:
#     - boundary_index: The first row or column index where the attention exceeds the threshold.
#     """
#     # Validate the matrix dimensions
#     if attention_matrix.ndim != 2 or attention_matrix.shape[0] != attention_matrix.shape[1]:
#         print(f"Invalid attention matrix shape: {attention_matrix.shape}. Must be square (N x N).")
#         return None

#     # Calculate the dynamic threshold (using percentile of row sums)
#     row_sums = np.sum(attention_matrix, axis=1)
#     threshold_percentile = 90  # Use the 90th percentile of row sums as a threshold
#     threshold = np.percentile(row_sums, threshold_percentile)
#     print(f"Row Sums: {row_sums}")
#     print(f"90th Percentile Threshold: {threshold}")

#     # Check each row for boundary detection relative to this percentile threshold
#     for i in range(attention_matrix.shape[0]):
#         row_sum = row_sums[i]
#         print(f"Row {i} Sum: {row_sum} (Threshold: {threshold})")
        
#         if row_sum >= threshold:  # Compare with percentile threshold
#             print(f"Boundary found at index {i} with row sum {row_sum}.")
#             return i

#     # If no boundary is found, return the last index as the default
#     print("No boundary exceeded the threshold. Returning the last index.")
#     return attention_matrix.shape[0] - 1

def find_boundary(attention_weights):
    """
    Find the index where the repetition starts in the attention matrix based on query-key pairs.
    
    Parameters:
    - attention_weights: 2D numpy array of attention weights.
    
    Returns:
    - index of the boundary where repetition starts, or the size of the matrix if no repetition is found.
    """
    
    num_queries, num_keys = attention_weights.shape
    
    # Set a tolerance level for comparison to handle small differences in floating-point values
    tolerance = 1e-4  # You can adjust this tolerance as needed
    
    # Initialize variables to track the index where repetition is found
    query_repetition_index = None
    key_repetition_index = None
    
    # Check for repetition in rows (queries)
    for i in range(1, num_queries):  # Start at 1 to skip the first row
        for j in range(i):  # Compare with all previous rows
            # Compare the rows at index i and j
            if np.allclose(attention_weights[i], attention_weights[j], atol=tolerance):
                query_repetition_index = i
                print(f"Repetition detected between rows (queries) {i} and {j}")
                print(f"Row {i}: {attention_weights[i]}")
                print(f"Row {j}: {attention_weights[j]}")
                print(f"Row 9: {attention_weights[9]}")

                break  # Exit inner loop once repetition is found
        if query_repetition_index is not None:
            break  # Exit outer loop once query repetition is found
    
    # Check for repetition in columns (keys)
    for j in range(1, num_keys):  # Start at 1 to skip the first column
        for k in range(j):  # Compare with all previous columns
            # Compare the columns at index j and k
            if np.allclose(attention_weights[:, j], attention_weights[:, k], atol=tolerance):
                key_repetition_index = j
                print(f"Repetition detected between columns (keys) {j} and {k}")
                print(f"Column {j}: {attention_weights[:, j]}")
                print(f"Column {k}: {attention_weights[:, k]}")
                print(f"Column 9: {attention_weights[:, 9]}")
                break  # Exit inner loop once repetition is found
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
        # Return the lower index of the two if both repetitions were found
        print(f"Query Repetition Index: {query_repetition_index}, Key Repetition Index: {key_repetition_index}")
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

    
# def find_boundary_(attention_weights):
#     """
#     Find the index where the repetition starts in the attention matrix based on query-key pairs.

#     Parameters:
#     - attention_weights: 2D numpy array of attention weights.

#     Returns:
#     - index of the boundary where repetition starts.
#     """
    
#     num_queries, num_keys = attention_weights.shape
#     print("Query-Key Values (q*v):")

#     # Iterate over all query and key pairs
#     for i in range(1, num_queries):
#         for j in range(num_keys):
#             q_v_value = attention_weights[i, j]
#             #print(f"Query {i}, Key {j}, Value: {q_v_value:.5f}")
    
#     # Check for repetition
#     for i in range(1, num_queries):  # Start at 1 to skip the first row
#         for j in range(i):  # Compare with all previous rows
#             if np.allclose(attention_weights[i], attention_weights[j]):
#                 print(f"Repetition detected between rows {i} and {j}")
#                 print("Attention weights up to the boundary:")
#                 for k in range(i + 1):  # Include the row where repetition is found
#                     print(f"Row {k}: {attention_weights[k]}")
                
#                 return i  # Return the index where repetition starts
#             else:
                
#                 # Debugging: Print differences when rows do not match
#                 diff = np.abs(attention_weights[i] - attention_weights[j])
#                 print(f"Row {i} vs Row {j} differences: {diff}")
#     return attention_weights.shape[0]  # If no repetition, return the size of the matrix

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
    if (boundary-1) < attention_weights.shape[0]:
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
    # Exclude CLS token by slicing the matrix
    min_boundary = 5 # manually change this depdening on what the boundary is between base and combined model
    # Exclude CLS token by slicing the matrix
    attention_weights = attention_weights[1:, 1:]  # Exclude the first row and column
    # if (min_boundary < boundary):
    #     attention_weights = attention_weights[:min_boundary, :min_boundary]

    
    # Update labels starting from 0 after CLS
    num_tokens = attention_weights.shape[0]
    labels = [str(i) for i in range(num_tokens)]  # Start labels from 0

    # Plot the attention heatmap ff943dff, d45f5fff, 659fd5ff
    plt.figure(figsize=(10, 8))
    custom_cmap = sns.light_palette("#d45f5fff", as_cmap=True)

    ax = sns.heatmap(attention_weights, cmap=custom_cmap, square=True, annot=False)
    
    # Update axis labels
    ax.set_xticks(np.arange(num_tokens) + 0.5)
    ax.set_yticks(np.arange(num_tokens) + 0.5)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=30, weight='bold')
    ax.set_yticklabels(labels, rotation=0, fontsize=30, weight='bold')
    cbar = ax.collections[0].colorbar

    cbar.ax.tick_params(labelsize=30, width=2)  # Set tick size and line width

    cbar.locator = MaxNLocator(nbins=8)  # Adjust number of ticks as needed
    cbar.ax.yaxis.set_major_locator(cbar.locator)
    cbar.ax.tick_params(labelsize=30)

    
    # Explicitly apply bold formatting to each colorbar tick label
    for label in cbar.ax.yaxis.get_ticklabels():
        label.set_fontsize(30)  # Set font size
        label.set_weight('bold')  # Set font weight to bold

    
    # plt.title(f"{titles if isinstance(titles, str) else 'Attention Weights Visual'}")
    # plt.xlabel("Key")
    # plt.ylabel("Query")
    plt.tight_layout()
    plt.show() 
    # Plot the attention heatmap
    # plt.figure(figsize=(10, 8))
    
    # sns.heatmap(attention_weights, cmap="Blues", square=True, annot=False, cbar_kws={'label': 'Attention Weight'})
    
    # plt.title(f"Attention Weights: {titles if isinstance(titles, str) else 'Heatmap'}")
    # plt.xlabel("Key Positions")
    # plt.ylabel("Query Positions")
    # plt.tight_layout()
    # plt.show()
def getValidationSetScanpathData(model_map, specific_index):
    """
    Returns the scanpath data for a specific index in the validation set (293-293+36).
    
    Parameters:
    - model_map: Dictionary loaded from `model_map.pkl` containing scanpath information.
    - specific_index: The specific index to retrieve the scanpath data for.

    Returns:
    - A tuple of (pid, quality, method_name, scanpaths) for the given index.
    """
    adjusted_index = 293 + specific_index
    indexCount = 0
    for (pid, quality, method_name), scanpaths in model_map.items():
        indexCount += 1
        if indexCount == adjusted_index:
            print(f"Found data for pid {pid} (method: {method_name}, quality: {quality})")
            return pid, quality, method_name, scanpaths
    return None


def plot_attention_with_context(attention_weights_list, titles, specific_index, model_map, pad_token_idx=0):
    """
    Plots the attention weights heatmap for a specific index with contextual semantic information.

    Parameters:
    - attention_weights_list: List of lists containing attention tensors.
    - titles: Title for the plot (string or list).
    - specific_index: Index specifying which attention matrix to visualize.
    - model_map: Dictionary loaded from `model_map.pkl` containing scanpath information.
    - pad_token_idx: Index of the padding token (default is 0).
    """
    # Retrieve the specific scanpath data for the given index
    scanpath_data = getValidationSetScanpathData(model_map, specific_index)
    if not scanpath_data:
        print(f"Scanpath data not found for specific index {specific_index}.")
        return

    pid, quality, method_name, scanpaths = scanpath_data

    # # Define the participant quality map for validation
    # participant_quality_map = {
    #     111: 1,
    #     139: 1,
    #     311: 1,
    #     117: 0,
    #     310: 1,
    #     186: 0,
    #     191: 0,
    #     166: 0,
    #     133: 1,
    #     314: 1,
    #     315: 0,
    #     313: 1,
    #     312: 0,
    #     168: 1,
    #     143: 1,
    #     147: 1,
    #     319: 0,
    #     106: 1,
    #     155: 0,
    #     136: 1,
    #     189: 0
    # }

    # # Verify if the quality matches the expected quality from participant_quality_map
    # if participant_quality_map.get(pid) != quality:
    #     print(f"Mismatch in participant quality for pid {pid}. Expected {participant_quality_map.get(pid)} but got {quality}.")
    #     return

    # Extract semantic category names from scanpath
    semantic_labels = []
    for scanpath in scanpaths:
        for token, category_dict in scanpath.items():
            for category, fixd in category_dict.items():
                semantic_labels.append(category.split('.')[0])  # Extracting semantic categories

    # Get the attention weights for the specific index
    attention_weights = None
    for tensor_list in attention_weights_list:
        for batch_tensor in tensor_list:
            if batch_tensor is not None and specific_index < batch_tensor.size(0):
                attention_weights = batch_tensor[specific_index]
                break
        if attention_weights is not None:
            break

    if attention_weights is None:
        print(f"No attention weights found for specific index {specific_index}.")
        return

    # Convert attention weights to numpy if it's a tensor
    if hasattr(attention_weights, 'detach'):
        attention_weights = attention_weights.detach().cpu().numpy()

    boundary = find_scanpathLength_boundary(specific_index)
    print(attention_weights.shape[0])
    print(f"boundary {boundary}")
    # Find boundary where the repetition starts based on query-key pairs
    if (boundary-1) < attention_weights.shape[0]:
        print(f"Repetition detected at index {boundary}, excluding repeated data.")
        attention_weights = attention_weights[:boundary, :boundary]  # Crop both query and key values
    # Exclude padded tokens
    attention_weights = attention_weights[1:, 1:]

    # Ensure labels match the attention weights
    num_tokens = min(len(semantic_labels), attention_weights.shape[0])
    semantic_labels = semantic_labels[:num_tokens]

    # Plot the attention heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        attention_weights[:num_tokens, :num_tokens],
        cmap='Blues',
        square=True,
        annot=False,
        xticklabels=semantic_labels,
        yticklabels=semantic_labels
    )
    plt.title(f"{titles if isinstance(titles, str) else 'Attention Weights'}")
    plt.xlabel("Key")
    plt.ylabel("Query")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

# def getAllValidSetScanpathData(model_map, start_index=293, end_index=293+36):
#     """
#     Returns the scanpath data for all indices in the validation set within a given range.

#     Parameters:
#     - model_map: Dictionary loaded from `model_map.pkl` containing scanpath information.
#     - start_index: The starting index of the range (default is 293).
#     - end_index: The ending index of the range (default is 293+36).

#     Returns:
#     - A list of tuples, where each tuple contains (pid, quality, method_name, scanpaths) for each index in the range.
#     """
#     scanpath_data_list = []
#     indexCount = 0
#     for (pid, quality, method_name), scanpaths in model_map.items():
#         indexCount += 1
#         if start_index <= indexCount <= end_index:
#             print(f"Found data for pid {pid} (method: {method_name}, quality: {quality}) at index {indexCount}")
#             scanpath_data_list.append((pid, quality, method_name, scanpaths))
#     if scanpath_data_list:
#         print(len(scanpath_data_list))
#         print((scanpath_data_list))
#         return scanpath_data_list
#     else:
#         print(f"No scanpath data found in the range {start_index}-{end_index}.")
#         return None
    
def getAllValidSetScanpathData(model_map):
    """
    Returns the scanpath data for all indices in the validation set with correct participant ID, 
    quality, and matching scanpath length.

    Parameters:
    - model_map: Dictionary loaded from `model_map.pkl` containing scanpath information.

    Returns:
    - A list of tuples, where each tuple contains (pid, quality, method_name, scanpaths) for valid participants.
    """

    participant_quality_map = {
        111: ["go"],
        139: ["messageSent", "testNegativeParseCases", "getMenuAdministracion"],
        311: ["appendDeclarations", "getImageWithSource", "BFSdist", "testSetExample"],
        117: ["updateSchema"],
        310: ["updateGain"],
        186: ["updateSchema", "actionPerformed"],
        191: ["addPKColumn", "getUserNameFromCookie"],
        166: ["messageSent", "testNegativeParseCases", "getBackCommand2"],
        133: ["testNegativeParseCases", "equals"],
        314: ["updateGain", "testInvoke"],
        315: ["getTargetServiceName", "swapItems", "setGenJarDir", "setPhoto"],
        313: ["compareTo"],
        312: ["addRotation", "go"],
        168: ["exportXVRL"],
        143: ["getNAGString", "getImageWithSource"],
        147: ["setGenJarDir"],
        319: ["getBackCommand2"],
        106: ["searchRecipe"],
        155: ["addRotation"],
        136: ["swapItems"],
        189: ["handleHalt"]
    }


    print(len(participant_quality_map))

    scanpath_data_list = []

    for participant, methodList in participant_quality_map.items():
        for method in methodList:
            print(f"Processing: {participant} - {method}")
        # Loop through the model_map to extract scanpaths for each valid participant
            # Loop through the model_map to extract scanpaths for each valid participant
            for (pid, quality, method_name), scanpaths in model_map.items():
                # Check if the pid exists in participant_quality_map and the method matches
                if pid == participant and method_name.strip() == method.strip():  # Use .strip() to handle potential spaces
                    print(f"Found valid data for pid {pid} (method: {method_name}, quality: {quality})")
                    scanpath_data_list.append((pid, quality, method_name, scanpaths))
                    break  # Exit inner loop after finding the match
            
            # else:
            #     # Optionally, print this if you want to see what's skipped
            #     print(f"Skipping data for pid {participant}: method: {method}")
    # Return the scanpath data if valid data was found, otherwise return None
    if scanpath_data_list:
        print(f"Total valid scanpath data found: {len(scanpath_data_list)}")
        return scanpath_data_list
    else:
        print("No valid scanpath data found.")
        return None

def plot_all_attention_with_context(attention_weights_list, titles, model_map, pad_token_idx=0, start_index=293, end_index=293+36):
    """
    Plots the attention weights heatmap for all indices in the validation set with contextual semantic information.

    Parameters:
    - attention_weights_list: List of lists containing attention tensors.
    - titles: Title for the plot (string or list).
    - model_map: Dictionary loaded from `model_map.pkl` containing scanpath information.
    - pad_token_idx: Index of the padding token (default is 0).
    - start_index: The starting index of the range for the scanpaths (default is 293).
    - end_index: The ending index of the range for the scanpaths (default is 293+36).
    """
    # Get the validation set scanpath data for the specified range
    scanpath_data_list = getAllValidSetScanpathData(model_map, start_index, end_index)
    if not scanpath_data_list:
        print("No scanpath data found for the validation set.")
        return

    # Initialize a list to store all semantic categories across all indices
    all_semantic_labels = []
    all_attention_weights = []

    # Iterate over each validation set scanpath data
    for scanpath_data in scanpath_data_list:
        pid, quality, method_name, scanpaths = scanpath_data

        # Extract semantic category names from scanpath
        semantic_labels = []
        for scanpath in scanpaths:
            for token, category_dict in scanpath.items():
                for category, fixd in category_dict.items():
                    semantic_labels.append(category.split('.')[0])  # Extracting semantic categories
        all_semantic_labels.append(semantic_labels)

        # Get the attention weights for the current scanpath
        attention_weights = None
        for tensor_list in attention_weights_list:
            for batch_tensor in tensor_list:
                if batch_tensor is not None and scanpath_data_list.index(scanpath_data) < batch_tensor.size(0):
                    attention_weights = batch_tensor[scanpath_data_list.index(scanpath_data)]
                    break
            if attention_weights is not None:
                break

        if attention_weights is None:
            print(f"No attention weights found for specific index {scanpath_data_list.index(scanpath_data)}.")
            continue

        # Convert attention weights to numpy if it's a tensor
        if hasattr(attention_weights, 'detach'):
            attention_weights = attention_weights.detach().cpu().numpy()

        # Exclude padded tokens
        attention_weights = attention_weights[1:, 1:]

        all_attention_weights.append(attention_weights)

    # Aggregate all attention weights for plotting
    avg_attention_weights = np.mean(np.array(all_attention_weights), axis=0)

    # Flatten all semantic labels to ensure they align with the aggregated attention weights
    flattened_semantic_labels = [label for sublist in all_semantic_labels for label in sublist]
    num_tokens = min(len(flattened_semantic_labels), avg_attention_weights.shape[0])
    flattened_semantic_labels = flattened_semantic_labels[:num_tokens]

    # Plot the aggregated attention heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        avg_attention_weights[:num_tokens, :num_tokens],
        cmap='Blues',
        square=True,
        annot=False,
        xticklabels=flattened_semantic_labels,
        yticklabels=flattened_semantic_labels
    )
    plt.title(f"{titles if isinstance(titles, str) else 'Attention Weights'}")
    plt.xlabel("Key")
    plt.ylabel("Query")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


def plot_aggregate_attention_with_context_(attention_weights_list, titles, model_map, pad_token_idx=0):
    """
    Plots the aggregated attention weights heatmap for all indices in the validation set with contextual semantic information.

    Parameters:
    - attention_weights_list: List of lists containing attention tensors.
    - titles: Title for the plot (string or list).
    - model_map: Dictionary loaded from `model_map.pkl` containing scanpath information.
    - pad_token_idx: Index of the padding token (default is 0).
    """
    combined_attention_weights = []
    
    # Combine batches into a single batch of size 37
    for list_idx, tensor_list in enumerate(attention_weights_list):
        for batch_idx, batch_tensor in enumerate(tensor_list):
            # Flatten the batch tensors into one list
            for sample in batch_tensor:
                combined_attention_weights.append(sample)

    print(f"Combined attention weights length: {len(combined_attention_weights)}")
    
    if len(combined_attention_weights) == 37:
        attention_weights_list = combined_attention_weights
    else:
        print(f"Unexpected combined length: {len(combined_attention_weights)}")
        return

    # Get the validation set scanpath data
    # scanpath_data_list = getAllValidSetScanpathData(model_map, 293, 293 + 36)
    scanpath_data_list = getAllValidSetScanpathData(model_map)
    if not scanpath_data_list:
        print("No scanpath data found for the validation set.")
        return

    # Initialize a list to store all aggregated semantic category attention weights
    aggregated_attention_weights = []

    # Iterate over each validation set scanpath data
    for scanpath_data in scanpath_data_list:
        pid, quality, method_name, scanpaths = scanpath_data

        # Initialize a dictionary to store aggregated attention weights by base category
        category_attention_map = {}

        # Extract attention weights for the current scanpath
        attention_weights = None
        index_in_combined = scanpath_data_list.index(scanpath_data)

        if 0 <= index_in_combined < len(combined_attention_weights):
            attention_weights = combined_attention_weights[index_in_combined]
        else:
            print(f"No attention weights found for specific index {index_in_combined}.")
            continue

        # Convert attention weights to numpy if it's a tensor
        if hasattr(attention_weights, 'detach'):
            attention_weights = attention_weights.detach().cpu().numpy()

        # Ensure attention_weights is 2D before slicing
        if attention_weights.ndim > 1:
            attention_weights = attention_weights[1:, 1:]  # Exclude padded tokens
        else:
            print(f"Attention weights for {scanpath_data} is not 2D, skipping.")
            continue

        # Aggregate attention weights for each unique category
        for scanpath in scanpaths:
            for token, category_dict in scanpath.items():
                for category, fixd in category_dict.items():
                    # Remove suffix from category (e.g., '.0', '.1', etc.)
                    base_category = category.split('.')[0]

                    # Add attention weight to the corresponding base category
                    if base_category not in category_attention_map:
                        category_attention_map[base_category] = []

                    category_attention_map[base_category].append(fixd)

        # Calculate average attention weights for each category
        aggregated_attention_map = {
            category: np.mean(attention_weights) for category, attention_weights in category_attention_map.items()
        }

        aggregated_attention_weights.append(aggregated_attention_map)

    # Now you can use `aggregated_attention_weights` to plot the heatmap
    # Flatten the aggregated attention weights from all scanpaths
    all_categories = set()
    all_attention_values = []

    for attention_map in aggregated_attention_weights:
        for category, attention_weight in attention_map.items():
            all_categories.add(category)
            all_attention_values.append(attention_weight)

    # Sort the categories
    sorted_categories = sorted(all_categories)
    modified_categories = [category.replace("function", "method").title() for category in sorted_categories]


    # sorted_categories = sorted([category.replace("function", "method") for category in all_categories])

    # Prepare the matrix for plotting
    attention_matrix = np.zeros((len(sorted_categories), len(sorted_categories)))

    # Fill the matrix (you can decide how to compute this, for example, averaging or summing)
    for i, category1 in enumerate(sorted_categories):
        for j, category2 in enumerate(sorted_categories):
            # Here we sum the attention weights between category pairs
            attention_matrix[i, j] = np.mean([attention_map.get(category1, 0) * attention_map.get(category2, 0)
                                              for attention_map in aggregated_attention_weights])

    # Plot the aggregated attention heatmap
            
    plt.figure(figsize=(10, 8))
    custom_cmap = sns.light_palette("#567d59ff", as_cmap=True)
    sns.heatmap(
        attention_matrix,
        cmap=custom_cmap,
        square=True,
        annot=False,
        xticklabels=modified_categories,
        yticklabels=modified_categories
    )
    plt.title(f"{titles if isinstance(titles, str) else 'Aggregated Attention Weights'}")
    
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.tight_layout()
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

    
    # Train and validate the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    padded_data = load_data('padded_data.pkl')
    attention_masks = load_data('attention_masks.pkl')

    # Train and validate the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    max_seq_len = 32  # You can set this based on your data

    # Initialize the model, loss function, and optimizer
    model = Transformer(
        method_embedding_dim=768,  # Example embedding dimension
        reduced_method_dim=32,
        category_dim=64,
        duration_dim=64,
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

    for seed in random_seeds:
        print(f"\n=== Starting run with random seed: {seed} ===")
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Split your data into training, validation, and test sets
        # train_data, temp_data = train_test_split(padded_data, test_size=0.2, random_state=42)
        train_data, temp_data = train_test_split(encoded_data, test_size=0.2, random_state=42)
        valid_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

        # Extract the quality_summary labels from your padded data (now just the second element of each tuple)
        train_labels = [quality_summary for _, quality_summary, _, _ in train_data]
        # print(train_labels)
        valid_labels = [quality_summary for _, quality_summary,_, _ in valid_data]
        test_labels = [quality_summary for _, quality_summary,_, _ in test_data]

        # Initialize lists to hold embeddings and associated information
        train_embeddings = []
        train_participant_ids = []
        train_method_names = []
        for participant_id, quality_summary, method_name, scanpath_embeddings in train_data:
            method_embeddings, category_embeddings, duration_embeddings = zip(*scanpath_embeddings)
            train_embeddings.append((method_embeddings, category_embeddings, duration_embeddings))
            train_participant_ids.append(participant_id)  # Store participant_id
            train_method_names.append(method_name)

        valid_embeddings = []
        valid_participant_ids = []
        valid_method_names = []


        for participant_id, quality_summary, method_name,scanpath_embeddings in valid_data:
            method_embeddings, category_embeddings, duration_embeddings = zip(*scanpath_embeddings)
            valid_embeddings.append((method_embeddings, category_embeddings, duration_embeddings))
            valid_participant_ids.append(participant_id)
            valid_method_names.append(method_name)

            

        test_embeddings = []
        test_participant_ids = []
        test_method_names = []
        

        for participant_id, quality_summary, method_name, scanpath_embeddings in test_data:
            method_embeddings, category_embeddings, duration_embeddings = zip(*scanpath_embeddings)
            test_embeddings.append((method_embeddings, category_embeddings, duration_embeddings))
            test_participant_ids.append(participant_id)
            test_method_names.append(method_name)

        train_attention_masks = attention_masks[:len(train_data)]
        valid_attention_masks = attention_masks[len(train_data):len(train_data) + len(valid_data)]
        test_attention_masks = attention_masks[len(train_data) + len(valid_data):]

        # Now assuming `CustomDataset` expects embeddings, labels, participant_ids, and methods
        train_dataset = CustomDataset(train_embeddings, train_labels, train_participant_ids, train_method_names, train_attention_masks)
        valid_dataset = CustomDataset(valid_embeddings, valid_labels, valid_participant_ids, valid_method_names, valid_attention_masks)
        test_dataset = CustomDataset(test_embeddings, test_labels, test_participant_ids, test_method_names, test_attention_masks)

        # Create DataLoader instances
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=custom_collate_fn)
        valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, collate_fn=custom_collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=custom_collate_fn)


        early_stopping = EarlyStopping(patience=3, delta=0.01)

    # Training loop with EarlyStopping
        for epoch in range(10):  # Set the number of epochs
            print(f"Epoch {epoch+1}/10")
            
            # Train phase
            train_loss, train_accuracy = train(model, train_loader, loss_fn, optimizer, device)
            print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")

            # Validation phase
            valid_loss, valid_accuracy, attention_weights = validate(model, valid_loader, loss_fn, device)
            print(f"Validation Loss: {valid_loss:.4f}, Validation Accuracy: {valid_accuracy:.4f}")

            # Check for early stopping
            early_stopping(valid_loss, model)
            if early_stopping.early_stop:
                print(f"Early stopping at epoch: {epoch}")
                break

        
        # test_loss, test_accuracy = test(model, test_loader, loss_fn, device)
        test_loss, test_accuracy, test_precision, test_recall, test_f1 = test(model, test_loader, loss_fn, device)

        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

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

    print(f"Length of Padded Data: {len(padded_data)}")

        # Plot the attention weights
    # plot_attention(
    #     attention_weights,
    #     titles= "Combined Feature Model", specific_index=28)

    plot_attention(
        attention_weights,
        titles= "Combined Feature Model: Semantic Category", specific_index=28)
    modelmap = load_data('model_map.pkl')

    

    # def plot_attention_with_context(attention_weights_list, titles, specific_index, model_map, pad_token_idx=0):
    # Single heatmap for semanticCategory
    # plot_attention_with_context(
    #     attention_weights_list=attention_weights,
    #     titles="Semantic Categories",
    #     specific_index=28,
    #     model_map=modelmap,
    #     pad_token_idx=0
    # )       

    # plot_all_attention_with_context(
    #     attention_weights_list=attention_weights,
    #     titles="Semantic Categories",
    #     model_map=modelmap,
    #     pad_token_idx=0)

    plot_aggregate_attention_with_context_(
        attention_weights_list=attention_weights,
        titles="Aggregate Semantic Categories Attention",
        model_map=modelmap,
        pad_token_idx=0)
    

