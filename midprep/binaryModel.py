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




class Transformer(nn.Module):
    #def __init__(self, input_dim, embedding_dim, num_heads, num_layers=1, num_classes=1, dropout_rate=0.1):

    def __init__(self, method_embedding_dim=768, reduced_method_dim=32, 
                 category_dim=16, duration_dim=16, num_heads=4, 
                 num_layers=1, num_classes=1, dropout_rate=0.1,  max_seq_len=32): # look into batch size
        super(Transformer, self).__init__()

        # Nonlinear dimensionality reduction for method embeddings
        self.method_reducer = nn.Sequential(
            nn.Linear(method_embedding_dim, method_embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(method_embedding_dim // 2, reduced_method_dim),
            nn.ReLU()
        )

        # Transformer encoder input dimension
        total_embedding_dim = reduced_method_dim + category_dim + duration_dim

        
        # Positional encoding to add information about token positions
        self.positional_encoding = nn.Parameter(
            torch.zeros((1, max_seq_len + 1, total_embedding_dim)) #### add 1 for sequence aggregator for later
        )  # Assuming max sequence length of 32

        # Special header token (e.g., [CLS])
        self.header_token = nn.Parameter(torch.zeros(1, 1, total_embedding_dim))

        # Transformer Encoder
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=total_embedding_dim,
            nhead=num_heads,
            dropout=dropout_rate
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_encoder_layer,
            num_layers=num_layers
        )

        # Fully connected layers
        self.fc_layer1 = nn.Linear(total_embedding_dim, total_embedding_dim // 2)
        self.fc_layer2 = nn.Linear(total_embedding_dim // 2, num_classes) 

    def forward(self, method_embeddings, category_embeddings, duration_embeddings, attention_mask=None):
        """
        Forward pass through the transformer.

        Args:
            method_embeddings (Tensor): Method token embeddings (batch_size, seq_len, reduced_method_dim)
            category_embeddings (Tensor): Semantic category embeddings (batch_size, seq_len, category_dim)
            duration_embeddings (Tensor): Duration bin embeddings (batch_size, seq_len, duration_dim)
            attention_mask (Tensor): Attention mask indicating padding (batch_size, seq_len)
        
        Returns:
            final_output (Tensor): The output of the transformer, passed through the fully connected layers
        """

        # # Debugging: Check the shapes of inputs
        # print(f"method_embeddings shape: {method_embeddings.shape}")
        # print(f"category_embeddings shape: {category_embeddings.shape}")
        # print(f"duration_embeddings shape: {duration_embeddings.shape}")

        # Reduce method embedding dimensions
        method_embeddings = self.method_reducer(method_embeddings)

        # print(f"Reduced method_embeddings shape: {method_embeddings.shape}")

        
        # Expanding tensor to match a fixed sequence length, e.g., 32
        sequence_length = 32
        method_embeddings = method_embeddings.unsqueeze(1).expand(-1, sequence_length, -1)  # Expands to (batch_size, sequence_length, feature_dim)
        category_embeddings = category_embeddings.unsqueeze(1).expand(-1, sequence_length, -1)  # Similarly for other embeddings
        duration_embeddings = duration_embeddings.unsqueeze(1).expand(-1, sequence_length, -1)

        # Concatenate embeddings along the feature dimension
        combined_embeddings = torch.cat(
            (method_embeddings, category_embeddings, duration_embeddings), dim=2
        )  # Shape: (batch_size, seq_len, total_embedding_dim)

        # print(f"combined_embeddings shape after concat: {combined_embeddings.shape}")


        
        # Add special header token to the beginning of the sequence
        header_token_repeated = self.header_token.expand(combined_embeddings.size(0), 1, -1)
        combined_embeddings = torch.cat((header_token_repeated, combined_embeddings), dim=1)

        # print(f"combined_embeddings shape after adding header token: {combined_embeddings.shape}")



        # Add positional encodings
        seq_length = combined_embeddings.size(1)
        positional_encodings = self.positional_encoding[:, :seq_length, :]
        combined_embeddings = combined_embeddings + positional_encodings

        # print(f"combined_embeddings shape after positional encoding: {combined_embeddings.shape}")


        # Reshape for transformer: (seq_len, batch_size, total_embedding_dim)
        combined_embeddings = combined_embeddings.permute(1, 0, 2)

        # print(f"combined_embeddings shape after permute: {combined_embeddings.shape}")


        # # If an attention mask is provided, modify it for transformer compatibility
        # if attention_mask is not None:
        #     # Transformer attention mask requires the mask to be False (0) for padded tokens and True (1) for actual tokens
        #     # Convert padding tokens to large negative values so they won't be attended to
        #     attention_mask = (attention_mask == 0).squeeze(1).squeeze(1)  # (batch_size, 1, 1, seq_len)
        #     attention_mask = attention_mask.to(dtype=torch.float32)  # Convert to float for masking

        #     # Apply attention mask: The padding tokens will be given a large negative value (-inf) in the attention scores
        #     attention_mask = (1.0 - attention_mask) * -10000.0

        #     print(f"attention_mask shape: {attention_mask.shape}")

        if attention_mask is not None:
            # Ensure attention mask has shape (batch_size, seq_len)
            # seq_len should be 33 because of the header token
            attention_mask = attention_mask.squeeze(1)  # Remove unnecessary dimensions
            attention_mask = attention_mask[:, :33]  # Ensure it's length 33
            
            # Convert padding tokens to large negative values so they won't be attended to
            attention_mask = (attention_mask == 0).to(dtype=torch.float32)  # (batch_size, seq_len)
            
            # Apply attention mask: The padding tokens will be given a large negative value (-inf) in the attention scores
            attention_mask = (1.0 - attention_mask) * -10000.0

            # print(f"attention_mask shape after modification: {attention_mask.shape}")

        # Pass through transformer encoder
        transformer_output = self.transformer_encoder(combined_embeddings, src_key_padding_mask=attention_mask)

        # print(f"transformer_output shape: {transformer_output.shape}")

        # Output corresponding to the special header token (first token)
        header_token_output = transformer_output[0, :, :]

        # print(f"header_token_output shape: {header_token_output.shape}")


        # Pooling over sequence dimension - averages 
        #pooled_output = torch.mean(transformer_output, dim=0)  # Shape: (batch_size, total_embedding_dim)
        #projected_output = self.fc_layer1(pooled_output)
       
        # Fully connected layers
        projected_output = self.fc_layer1(header_token_output)
        
        final_output = self.fc_layer2(projected_output)

        # new layer should just be a argmax into 2 neurons  
        # act fun of last layer - softmax 

        # print(f"projected_output shape: {projected_output.shape}")

        #print(f"final_output shape: {final_output.shape}")

        #final_output = final_output.squeeze()  # Remove the singleton dimension, resulting in shape [32]
        print(f"final_output shape: {final_output.shape}")

        return final_output


def train(model, train_data, loss_fn, optimizer, device):
    model.train()  # Set model to training mode
    total_loss = 0
    correct = 0
    total = 0

    for participant_id, quality_summary, scanpath_embeddings in train_data:
       
        #print(participant_id, quality_summary, len(scanpath_embeddings))
        # Prepare your inputs and labels
        method_embeddings, category_embeddings, duration_embeddings = zip(*scanpath_embeddings)
        

        # Convert lists of embeddings to tensors
        method_embeddings = torch.stack(method_embeddings).to(device)
        category_embeddings = torch.stack(category_embeddings).to(device)
        duration_embeddings = torch.stack(duration_embeddings).to(device)
        attention_mask = torch.ones(method_embeddings.size(0), method_embeddings.size(1)).to(device)  # Assuming no padding initially

        print(participant_id,quality_summary )
        print(method_embeddings.shape)
        print(category_embeddings.shape)
        print(duration_embeddings.shape) 

        # Ground truth label
        quality_label = torch.tensor([quality_summary]).to(device).view(-1, 1)
        print(f"quality_label shape: {quality_label.shape}")
        #quality_label = torch.tensor([quality_summary] * method_embeddings.size(0)).to(device)  # Ensure the label has the same batch size as the inpute
        # 
        #quality_label = quality_label.view(-1, 32)  # Ensure target shape is (batch_size, 1)
        # print(f"quality_label shape: {quality_label.shape}")

        # Forward pass
        optimizer.zero_grad()
        output = model(method_embeddings, category_embeddings, duration_embeddings, attention_mask)
        
        
        # Calculate loss
        loss = loss_fn(output, quality_label.float())
        total_loss += loss.item()

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # For metrics, assuming binary classification for simplicity
        predicted = torch.round(torch.sigmoid(output))
        print(predicted.shape)
        #quality_label_cpu = quality_label.squeeze().cpu().numpy()  # Convert to NumPy for easy handling

        #print(f"Model output: {predicted}")
        correct += (predicted.squeeze() == quality_label).sum().item()
        total += quality_label.size(0)
        # for pred, correct in zip(predicted, quality_label):
        #     print(f"Predicted: {pred.item()}, Correct: {correct.item()}")

    accuracy = correct / total
    avg_loss = total_loss / len(train_data)
    print(f"Train Loss: {avg_loss:.4f}, Train Accuracy: {accuracy:.4f}")
    
    return avg_loss, accuracy


def validate(model, val_data, loss_fn, device):
    model.eval()  # Set model to evaluation mode
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():  # Disable gradient calculations
        for participant_id, quality_summary, scanpath_embeddings in val_data:
            # Prepare inputs and labels
            method_embeddings, category_embeddings, duration_embeddings = zip(*scanpath_embeddings)

            # Convert lists of embeddings to tensors
            method_embeddings = torch.stack(method_embeddings).to(device)
            category_embeddings = torch.stack(category_embeddings).to(device)
            duration_embeddings = torch.stack(duration_embeddings).to(device)
            attention_mask = torch.ones(method_embeddings.size(0), method_embeddings.size(1)).to(device)

            quality_label = torch.tensor([quality_summary] * method_embeddings.size(0)).to(device)
            quality_label = quality_label.view(-1, 1)

            # Forward pass
            output = model(method_embeddings, category_embeddings, duration_embeddings, attention_mask)
            
            # Calculate loss
            loss = loss_fn(output, quality_label.float())
            total_loss += loss.item()

            # Calculate accuracy
            predicted = torch.round(torch.sigmoid(output))
            correct += (predicted.squeeze() == quality_label).sum().item()
            total += quality_label.size(0)

    accuracy = correct / total
    avg_loss = total_loss / len(val_data)
    print(f"Validation Loss: {avg_loss:.4f}, Validation Accuracy: {accuracy:.4f}")
    
    return avg_loss, accuracy



def test(model, test_data, loss_fn, device):
    model.eval()  # Set model to evaluation mode
    total_loss = 0
    correct = 0
    total = 0

    all_predictions = []  # Store predictions for analysis
    all_labels = []  # Store ground truth labels

    with torch.no_grad():  # Disable gradient calculations
        for participant_id, quality_summary, scanpath_embeddings in test_data:
            # Prepare inputs and labels
            method_embeddings, category_embeddings, duration_embeddings = zip(*scanpath_embeddings)

            # Convert lists of embeddings to tensors
            method_embeddings = torch.stack(method_embeddings).to(device)
            category_embeddings = torch.stack(category_embeddings).to(device)
            duration_embeddings = torch.stack(duration_embeddings).to(device)
            attention_mask = torch.ones(method_embeddings.size(0), method_embeddings.size(1)).to(device)

            quality_label = torch.tensor([quality_summary] * method_embeddings.size(0)).to(device)
            quality_label = quality_label.view(-1, 1)

            # Forward pass
            output = model(method_embeddings, category_embeddings, duration_embeddings, attention_mask)
            
            # Calculate loss
            loss = loss_fn(output, quality_label.float())
            total_loss += loss.item()

            # Calculate accuracy
            predicted = torch.round(torch.sigmoid(output))
            correct += (predicted.squeeze() == quality_label).sum().item()
            total += quality_label.size(0)

            # Store predictions and labels
            all_predictions.append(predicted.cpu().numpy())
            all_labels.append(quality_label.cpu().numpy())

    accuracy = correct / total
    avg_loss = total_loss / len(test_data)
    #print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.4f}")
    #precision = precision_score(all_labels, all_predictions, average='weighted')
    #recall = recall_score(all_labels, all_predictions, average='weighted')
    #f1 = f1_score(all_labels, all_predictions, average='weighted')

    print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.4f}")
    #print(f" Recall: {recall:.4f}, F1 Score: {f1:.4f}")
    return avg_loss, accuracy, all_predictions, all_labels






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

def normalize_fixation_durations(data_tuple):
    # Lists to hold fixation durations and original entries
    fixation_durations = []
    entries = []  # To store original entries along with their durations
    
    for entry in data_tuple:
        # Get the fixation duration tuples for the current entry
        durations = data_tuple.get(entry)

        if durations:  # Ensure there are durations available
            for duration_entry in durations:  # Accessing the list of tuples
                fixation_durations.append(duration_entry[2])  # entry[2] is the fixation duration
                # Store original entry and corresponding fixation entry
                entries.append((entry[0], entry[1], entry[2], duration_entry[0], duration_entry[1], duration_entry[2]))

    # Convert the list to a numpy array for normalization
    fixation_durations = np.array(fixation_durations).reshape(-1, 1)
    

    # Initialize the scaler and fit to the fixation durations
    scaler = MinMaxScaler()
    normalized_durations = scaler.fit_transform(fixation_durations)

    # Create a new dictionary to store the normalized data in the desired format
    normalized_data = {}

    # Iterate through the original entries to construct the normalized data
    for i, original_entry in enumerate(entries):
        # Prepare the normalized entry for the dictionary
        normalized_entry = [original_entry[3], original_entry[4], float(normalized_durations[i][0])]  # token, semantic, normalized duration
        
        # Create the key as (pid, quality, method)
        pid_quality_method = (original_entry[0], original_entry[1], original_entry[2])  # (pid, quality, method)
        
        # Add the normalized entry to the dictionary
        if pid_quality_method not in normalized_data:
            normalized_data[pid_quality_method] = []
        normalized_data[pid_quality_method].append(normalized_entry)

    print(normalized_data)
    return normalized_data






# convert scanpath into sequence
# Function to clean the category and combine with fixation duration
def process_scanpath(scanpath):
    sequence = []
    for item in scanpath:
        category = item[0].split('.')[0]  # Remove suffix from category
        duration = item[2]  # Fixation duration
        # Combine the cleaned category and the duration
        sequence.append(f"{category} {duration}")
    return ' '.join(sequence)  # Return as a single string (like a sentence)

def print_fixation_durations(data_tuple, num_bins=5):
    """
    Function to calculate and print the number of bins across all methods.
    
    Args:
        data_tuple: The normalized data dictionary containing fixation durations.
        num_bins: The desired number of bins for aggregation.
    """
    
    total_bins = {'sqrt': 0, 'sturges': 0, 'rice': 0}

    for entry in data_tuple:
        # Collect durations for the current entry
        durations = [token[2] for token in data_tuple[entry] if isinstance(token[2], (int, float))]
        print (durations)
        N = len(durations)

        if N == 0:
            print(f"No valid data for entry {entry}")
            continue

        # Calculate bins with adjustments
        sqrt_bins = max(1, int(np.sqrt(N) / 2))  # Fewer bins
        sturges_bins = min(int(np.log2(N)) + 1, num_bins)  # Limit to num_bins
        rice_bins = min(int(2 * (N ** (1/3))), num_bins)  # Limit to num_bins

        # Update total bins
        total_bins['sqrt'] += sqrt_bins
        total_bins['sturges'] += sturges_bins
        total_bins['rice'] += rice_bins

        # Print bins for the current entry
        print(f"Entry: {entry}, Bins - sqrt: {sqrt_bins}, sturges: {sturges_bins}, rice: {rice_bins}")

    # Print final total bins across all entries
    for method, bins in total_bins.items():
        print(f"Total Bins using {method}: {bins}")
   
def print_fixation_durations_fixed_bins(data_tuple):
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





def tokenize_data(data_tuple, tokenizer, model, embedding_dim=16):
    # Initialize dictionaries for dynamic semantic category mapping and embedding layer
    vocab = {}
    embedding_layer = nn.Embedding(num_embeddings=1, embedding_dim=embedding_dim)  # Temporary; will be resized as we add categories

    # Prepare tokenized data
    tokenized_data = []

    for entry in data_tuple:
        entry_id = entry[:3]  # Unique identifier
        print(f"\nProcessing entry ID: {entry_id}")
        tokenized_entry = []

        for token in data_tuple[entry]:
            method_token, semantic_category, duration_bin = token
            print(f"Processing token: method_token='{method_token}', semantic_category='{semantic_category}', duration_bin='{duration_bin}'")

            # Dynamically add new semantic categories (with suffixes) to the vocab
            if semantic_category not in vocab:
                vocab[semantic_category] = len(vocab)
                # Resize embedding layer to accommodate new category
                embedding_layer = nn.Embedding(num_embeddings=len(vocab), embedding_dim=embedding_dim)
                print(f"Added new semantic category '{semantic_category}' to vocab with ID {vocab[semantic_category]}")
                print(f"Updated embedding layer to handle {len(vocab)} categories")

            # Tokenize the method token using the pretrained tokenizer
            inputs = tokenizer(method_token, return_tensors="pt", padding=True, truncation=True) 

            # Get token embeddings from the pretrained model
            with torch.no_grad(): # disables gradience
                outputs = model(**inputs)
                method_token_embedding = outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()  # Detach before converting
            print(f"Method token embedding shape: {method_token_embedding.shape}")

            # Get the embedding for the semantic category with suffix
            category_idx = torch.tensor([vocab[semantic_category]])  # Create tensor for the category index
            semantic_category_embedding = embedding_layer(category_idx).squeeze().detach().numpy()  # Detach before converting
            print(f"Semantic category embedding for '{semantic_category}': {semantic_category_embedding}")

            # Tokenize the fixation duration bin
            duration_bin_token = int(duration_bin.split()[1]) - 1  # Convert "Bin 1" to 0, etc.
            print(f"Duration bin token: {duration_bin_token}")

            # Append the tokenized/embedded values
            tokenized_entry.append((method_token_embedding, semantic_category_embedding, duration_bin_token))

        tokenized_data.append((entry_id, tokenized_entry))
        print(f"Finished processing entry ID: {entry_id}")

    print("\nFinal vocabulary mapping (including suffixes):", vocab)
    return tokenized_data

def tokenize_data_reduced(data_tuple, tokenizer, model, embedding_dim=128):  # Changed embedding_dim to 128
    # Initialize dictionaries for dynamic semantic category mapping and embedding layer
    vocab = {}
    embedding_layer = nn.Embedding(num_embeddings=1, embedding_dim=embedding_dim)  # Temporary; will be resized as we add categories
    duration_embedding_layer = nn.Embedding(num_embeddings=6, embedding_dim=embedding_dim)  # Assuming up to 6 duration bins


    # Prepare tokenized data
    tokenized_data = []

    for entry in data_tuple:
        
        entry_id = entry[:3]  # Unique identifier
        print(f"\nProcessing entry ID: {entry_id}")
        tokenized_entry = []

        for token in data_tuple[entry]:
            method_token, semantic_category, duration_bin = token
            #print(f"Processing token: method_token='{method_token}', semantic_category='{semantic_category}', duration_bin='{duration_bin}'")

            # Dynamically add new semantic categories (with suffixes) to the vocab
            if semantic_category not in vocab:
                vocab[semantic_category] = len(vocab)
                # Resize embedding layer to accommodate new category
                embedding_layer = nn.Embedding(num_embeddings=len(vocab), embedding_dim=embedding_dim)
                #print(f"Added new semantic category '{semantic_category}' to vocab with ID {vocab[semantic_category]}")
                #print(f"Updated embedding layer to handle {len(vocab)} categories")

            # Tokenize the method token using the pretrained tokenizer
            inputs = tokenizer(method_token, return_tensors="pt", padding=True, truncation=True)

            # Get token embeddings from the pretrained model
            with torch.no_grad():  # disables gradient computation
                outputs = model(**inputs)
                method_token_embedding = outputs.last_hidden_state.mean(dim=1).squeeze().detach()  # No need to convert to numpy yet
                method_token_embedding = method_token_embedding[:embedding_dim]  # Ensure it has the desired embedding size
            #print(f"Method token embedding shape: {method_token_embedding.shape}")

            # Get the embedding for the semantic category with suffix
            category_idx = torch.tensor([vocab[semantic_category]])  # Create tensor for the category index
            semantic_category_embedding = embedding_layer(category_idx).squeeze().detach().numpy()  # Detach before converting
            #print(f"Semantic category embedding for '{semantic_category}': {semantic_category_embedding}")

            # Tokenize the fixation duration bin
            print(duration_bin)
            duration_bin_token = int(duration_bin.split()[1]) - 1  # Convert "Bin 1" to 0, etc.
            # Append the tokenized/embedded values as integers
            tokenized_entry.append((method_token_embedding.numpy(), semantic_category_embedding, duration_bin_token))

        tokenized_data.append((entry_id, tokenized_entry))
        print(f"Finished processing entry ID: {entry_id}")

    #print("\nFinal vocabulary mapping (including suffixes):", vocab)
    return tokenized_data, vocab

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




def preprocessing_tokenized_data(data_tuple, tokenizer, model, category_vocab, category_dim=32, duration_dim=32, max_sequence_length=None):
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

        tokenized_data.append((participant_id, quality_summary, tokenized_entry))

    # Determine final max sequence length
    max_sequence_length = max_sequence_length or calculated_max_length

    # Pad sequences to max_sequence_length and create attention masks
    padded_data = []
    attention_masks = []
    for entry in tokenized_data:
        participant_id, quality_summary, scanpath_embeddings = entry

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

        padded_data.append((participant_id, quality_summary, scanpath_embeddings))
        attention_masks.append(attention_mask)

    return padded_data, attention_masks, max_sequence_length

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







# Function to save data using pickle
def save_data(filename, data):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)

# Function to load data using pickle
def load_data(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)
    
if __name__ == '__main__':

    # Load and preprocess your data
    # Load scanpath data from pickle file
    #with open('model_map.pkl', 'rb') as scanpathfile:
        #scanpaths = pickle.load(scanpathfile)

    data_tuple = flatten_data('model_map.pkl')

    #normalized_data_tuple = normalize_fixation_durations(data_tuple)

    # Apply HDBScCAN clustering to the dataset
    #clustered_data = hdbscan_fixation_clustering(normalized_data, min_cluster_size=12)

    # Output the clustered data
    #for key, value in clustered_data.items():
        #print(key, value)

    #print_fixation_durations(data_tuple, 5)
    data_tuple = print_fixation_durations_fixed_bins(data_tuple) # fixed bins of 6 

    # Create vocabulary for semantic categories
    category_vocab = create_vocab(data_tuple)

    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    model = AutoModel.from_pretrained("microsoft/codebert-base")

    # # Tokenize data and obtain embeddings
    # tokenized_data = tokenize_data_input_embeddings(
    #     data_tuple,
    #     tokenizer,
    #     model,
    #     category_vocab,
    #     category_dim=16,
    #     duration_dim=16
    # )

    # # Tokenize and embed data
    # padded_data, attention_masks, max_sequence_length = preprocessing_tokenized_data(
    # data_tuple,
    # tokenizer,
    # model,
    # category_vocab,
    # category_dim=16,
    # duration_dim=16
    # )

    # # # Save preprocessed data
    # save_data("padded_data.pkl", padded_data)
    # save_data("attention_masks.pkl", attention_masks)

    # print(max_sequence_length)


    padded_data = load_data('padded_data.pkl')
    attention_masks = load_data('attention_masks.pkl')

    # # Train and validate the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

   # Model parameters
    method_embedding_dim = 768
    reduced_method_dim = 32
    category_dim = 16
    duration_dim = 16
    num_heads = 4
    num_layers = 1
    num_classes = 1  # Binary classification
    dropout_rate = 0.1
    #max_seq_len = padded_data.shape[1]  # Automatically determine sequence length

    # Step 2: Initialize the model, loss function, and optimizer
    model = Transformer(
        method_embedding_dim=768,  # Example embedding dimension
        reduced_method_dim=32,
        category_dim=16,
        duration_dim=16,
        num_heads=4,
        num_layers=1,
        num_classes=1,  # Binary classification (adjust for your use case)
        dropout_rate=0.1,
        max_seq_len=32
    ).to(device)  # Use GPU if available, device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Loss function and optimizer
    loss_fn = nn.BCEWithLogitsLoss()  # Binary cross-entropy with logits
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # Move model and data to device
    

     # Step 2: Split the data into training, validation, and test sets
    train_data, temp_data = train_test_split(padded_data, test_size=0.2, random_state=42)
    valid_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

    #print(train_data)


    # Step 4: Train the model
    for epoch in range(10):  # Set the number of epochs you want
        print(f"Epoch {epoch+1}/{10}")
        # Training phase
        avg_train_loss, train_accuracy = train(model, train_data, loss_fn, optimizer, device)
        validate(model, valid_data, loss_fn, device) # def validate(model, val_data, loss_fn, device):
    test(model, test_data, loss_fn, device) # def test(model, test_data, loss_fn, device):




    #tokenize semantic categories and tokens
    #Load the pretrained model and tokenizer (CodeBERT in this example)
    

    #tokenize_data(data_tuple, tokenizer, model) embedding difference using code bert (758 v. just 16 for semantic)
    #tokenized_data, vocab = tokenize_data_revisions(data_tuple, tokenizer, model)
    #tokenized_data, vocab = tokenize_data_reduced(data_tuple, tokenizer, model)
    #save_data('tokenized_data_file.pkl', tokenized_data)
    #save_data('vocab_file.pkl', vocab)
    #print(tokenized_data)
    
    
    


    



    #print(len(padded_data))
    # Define model parameters
    # input_dim = len(vocab)  # Number of unique tokens in your vocabulary
    # embedding_dim = 128
    # num_heads = 8  # Adjust based on your requirements
    # num_layers = 1  # As per your requirement for 1 hidden layer
    # num_classes = 10  # Based on your task (adjust as needed)
    # dropout = 0.1  # Common dropout value

    # # Initialize the transformer model
    # tformer = Transformer(input_dim, embedding_dim, num_heads, num_layers, num_classes, dropout)

    # # Define loss function and optimizer
    # #loss_fn = nn.CrossEntropyLoss() #multiple classifcaiton 
    # loss_fn = nn.BCEWithLogitsLoss()

    # optimizer = torch.optim.Adam(tformer.parameters(), lr=0.0001)

    # # Split the data into train, validation, and test sets
    # train_data, temp_data = train_test_split(tokenized_data, test_size=0.2, random_state=42)
    # valid_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

    # # Train and validate the model
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # tformer.to(device)

    # epochs = 10  # Adjust as needed
    # for epoch in range(epochs):
    #     print(f"\nEpoch {epoch + 1}/{epochs}")
    #     #def train_model(model, train_data, optimizer, criterion, device):
    #     #train_model(tformer, train_data, optimizer, loss_fn, device)

    #     tformer.train_model(train_data, optimizer, loss_fn, device)
    #     tformer.valid_model(valid_data, loss_fn, device)

    # tformer.test_model(test_data, loss_fn, device)