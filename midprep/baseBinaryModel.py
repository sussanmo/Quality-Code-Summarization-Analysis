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

import torch
import torch.nn as nn
import torch.optim as optim

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
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=total_embedding_dim,
            nhead=num_heads,
            dropout=dropout_rate
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
        sequence_length = 32 # Use the actual sequence length if different
        
        method_embeddings = method_embeddings.squeeze(1)  # Remove any unnecessary dimensions
        

        # # Concatenate embeddings along the feature dimension
        # combined_embeddings = torch.cat(
        #     (method_embeddings), dim=2
        # )  # Shape: (batch_size, seq_len, total_embedding_dim)
        
        # Add special header token to the beginning of the sequence
        header_token_repeated = self.header_token.expand(method_embeddings.size(0), 1, -1)
        method_embeddings = torch.cat((header_token_repeated, method_embeddings), dim=1)

        # Add positional encodings
        seq_length = method_embeddings.size(1)
        positional_encodings = self.positional_encoding[:, :seq_length, :]
        method_embeddings = method_embeddings + positional_encodings

        # Reshape for transformer: (seq_len, batch_size, total_embedding_dim)
        method_embeddings = method_embeddings.permute(1, 0, 2)
    
        if attention_mask is not None:
            # Apply attention mask to the transformer (padding tokens will be ignored)
            attention_mask = attention_mask.squeeze(1)  # Remove unnecessary dimensions
            attention_mask = attention_mask[:, :seq_length + 1]  # Ensure it has the correct length

            # Convert padding tokens to large negative values so they won't be attended to
            attention_mask = (attention_mask == 0).to(dtype=torch.float32)
            attention_mask = (1.0 - attention_mask) * -10000.0  # Large negative value for padding

            print(f"Attention Mask Shape: {attention_mask.shape}")
        else:
            print("No attention mask provided; using default masking.")

        
            
        # Pass through transformer encoder
        transformer_output = self.transformer_encoder(
            method_embeddings, src_key_padding_mask=attention_mask
        )

        # Output corresponding to the special header token (first token)
        header_token_output = transformer_output[0, :, :]

        # Fully connected layers for classification
        projected_output = self.fc_layer1(header_token_output)
        projected_output = self.fc_dropout(projected_output)  # Apply dropout after the first FC layer
        final_output = self.fc_layer2(projected_output)

        # print("Method Embeddings:", method_embeddings.shape)
        print("Transformer Output:", transformer_output.shape)
        print("Final Output:", final_output.shape)

        return final_output

class CustomDataset(torch.utils.data.Dataset):

    def __len__(self):
        return len(self.labels)

    def __init__(self, embeddings, labels, participant_ids):
        self.embeddings = embeddings  # This contains tuples of embeddings (method_embeddings)
        self.labels = labels  # Quality labels
        self.participant_ids = participant_ids  # Participant IDs

    def __getitem__(self, idx):
        # Extract individual embeddings
        method_embeddings = self.embeddings[idx]
        
        # Ensure embeddings are of the correct type after stacking
        method_embeddings = torch.stack(method_embeddings).to(dtype=torch.float32) if isinstance(method_embeddings, (list, tuple)) else method_embeddings
        

        # Extract quality label, participant_id, and method for this sample
        quality_label = self.labels[idx]
        participant_id = self.participant_ids[idx]

        # Return stacked embeddings and other details
        return method_embeddings,quality_label, participant_id
    

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
        outputs = model(method_embeddings)

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

    for i, (method_embeddings, quality_label, participant_id) in enumerate(train_loader):
        # Convert lists to tensors before moving to device
        method_embeddings = torch.tensor(method_embeddings).to(device)
        
        quality_label = torch.tensor(quality_label, dtype=torch.float32).to(device)

        quality_label = quality_label.view(-1, 1)  # Reshapes to (32, 1)
        # Forward pass
        outputs = model(method_embeddings)


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

    with torch.no_grad():  # Disable gradient calculation for validation
        for i, (method_embeddings, quality_label, participant_id) in enumerate(valid_loader):
            # Convert lists to tensors before moving to device
            method_embeddings = torch.tensor(method_embeddings).to(device)
            
            quality_label = torch.tensor(quality_label, dtype=torch.float32).to(device)

            quality_label = quality_label.view(-1, 1)  # Reshapes to (batch_size, 1)

            # Forward pass
            outputs = model(method_embeddings)

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
    return avg_loss, accuracy

def test(model, test_loader, loss_fn, device):
    model.eval()  # Set model to evaluation mode
    total_loss = 0
    correct_preds = 0
    total_preds = 0

    with torch.no_grad():  # No need to compute gradients for testing
        for i, (method_embeddings, quality_label, participant_id) in enumerate(test_loader):
            # Convert lists to tensors before moving to device
            method_embeddings = torch.tensor(method_embeddings).to(device)
            quality_label = torch.tensor(quality_label, dtype=torch.float32).to(device)

            quality_label = quality_label.view(-1, 1)  # Reshapes to (32, 1)

            # Forward pass
            outputs = model(method_embeddings)

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

    for seed in random_seeds:
        print(f"\n=== Starting run with random seed: {seed} ===")
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Split data with the current random seed
        train_data, temp_data = train_test_split(padded_data, test_size=0.2, random_state=seed)
        valid_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=seed)

        # Extract the quality_summary labels from your padded data
        train_labels = [quality_summary for _, quality_summary, _ in train_data]
        valid_labels = [quality_summary for _, quality_summary, _ in valid_data]
        test_labels = [quality_summary for _, quality_summary, _ in test_data]

        # Prepare datasets
        train_embeddings, train_participant_ids = [], []
        for participant_id, _, scanpath_embeddings in train_data:
            method_embeddings, _, _ = zip(*scanpath_embeddings)
            train_embeddings.append(method_embeddings)
            train_participant_ids.append(participant_id)

        valid_embeddings, valid_participant_ids = [], []
        for participant_id, _, scanpath_embeddings in valid_data:
            method_embeddings, _, _ = zip(*scanpath_embeddings)
            valid_embeddings.append(method_embeddings)
            valid_participant_ids.append(participant_id)

        test_embeddings, test_participant_ids = [], []
        for participant_id, _, scanpath_embeddings in test_data:
            method_embeddings, _, _ = zip(*scanpath_embeddings)
            test_embeddings.append(method_embeddings)
            test_participant_ids.append(participant_id)

        # Create datasets
        train_dataset = CustomDataset(train_embeddings, train_labels, train_participant_ids)
        valid_dataset = CustomDataset(valid_embeddings, valid_labels, valid_participant_ids)
        test_dataset = CustomDataset(test_embeddings, test_labels, test_participant_ids)

        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        # Initialize model, loss function, and optimizer for this run
        model = Transformer(
            method_embedding_dim=768,
            reduced_method_dim=32,
            num_heads=4,
            num_layers=1,
            num_classes=1,
            dropout_rate=0.1,
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
            valid_loss, valid_accuracy = validate(model, valid_loader, loss_fn, device)
            print(f"Validation Loss: {valid_loss:.4f}, Validation Accuracy: {valid_accuracy:.4f}")

            # Early stopping
            early_stopping(valid_loss, model)
            if early_stopping.early_stop:
                print(f"Early stopping at epoch: {epoch}")
                break

        # Test phase
        test_loss, test_accuracy = test(model, test_loader, loss_fn, device)
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

        # Save results for this seed
        results.append((train_loss, train_accuracy, valid_loss, valid_accuracy, test_loss, test_accuracy))

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
