import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from sklearn.model_selection import train_test_split
from collections import defaultdict
import random
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

# Custom collate function for handling variable-length sequences
def collate_fn(batch):
    # Sort the batch by sequence length in descending order
    batch.sort(key=lambda x: x[0].shape[0], reverse=True)
    
    # Separate sequences and labels
    sequences, labels = zip(*batch)
    
    # Get lengths of each sequence for packing
    lengths = [seq.shape[0] for seq in sequences]
    
    # Pad sequences
    padded_seqs = pad_sequence(sequences, batch_first=True)
    
    # Convert labels to tensor
    labels = torch.stack(labels)
    
    return padded_seqs, labels, lengths

class TimeSeriesDataset(Dataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths
        
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        data = np.load(self.file_paths[idx])
        x = torch.FloatTensor(data['array'])  # Shape: [sequence_len, 3, dims]
        
        # Reshape to flatten the sensor readings
        # From [sequence_len, 3, dims] to [sequence_len, 3 * dims]
        x = x.reshape(x.shape[0], -1)  # Now shape is [sequence_len, 3 * dims]
        
        y = torch.tensor(data['label']).long()
        return x, y

class GRUClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(GRUClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x, lengths):
        # Pack the padded sequences
        packed_x = pack_padded_sequence(x, lengths, batch_first=True)
        
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate GRU
        packed_out, _ = self.gru(packed_x, h0)
        
        # Unpack the sequence
        out, _ = pad_packed_sequence(packed_out, batch_first=True)
        
        # Get the output of the last time step for each sequence
        batch_size = out.size(0)
        # Use the actual lengths to get the last output for each sequence
        idx = (torch.tensor(lengths) - 1).view(-1, 1).expand(-1, self.hidden_size)
        idx = idx.unsqueeze(1).to(out.device)
        last_out = out.gather(1, idx).squeeze(1)
        
        # Decode the hidden state
        out = self.fc(last_out)
        return out

def prepare_data(data_dir, test_samples_per_class=3):
    # Collect all .npz files
    class_files = defaultdict(list)
    
    for file in os.listdir(data_dir):
        if file.endswith('.npz'):
            class_name = file.split('_')[0]
            file_path = os.path.join(data_dir, file)
            class_files[class_name].append(file_path)
    
    # Split into train and test ensuring equal class distribution
    train_files = []
    test_files = []
    
    for class_name, files in class_files.items():
        test_samples = random.sample(files, test_samples_per_class)
        train_samples = [f for f in files if f not in test_samples]
        
        test_files.extend(test_samples)
        train_files.extend(train_samples)
    
    return train_files, test_files

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    best_val_acc = 0
    train_losses = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_idx, (data, target, lengths) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            outputs = model(data, lengths)
            loss = criterion(outputs, target)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target, lengths in val_loader:
                data, target = data.to(device), target.to(device)
                outputs = model(data, lengths)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        val_accuracy = 100 * correct / total
        avg_loss = total_loss / len(train_loader)
        
        train_losses.append(avg_loss)
        val_accuracies.append(val_accuracy)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')
        
        # Save best model
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            torch.save(model.state_dict(), 'best_model.pth')
    
    return train_losses, val_accuracies

def main():
    # Hyperparameters
    dims = 64
    num_sensors = 3
    input_size = dims * num_sensors  # Now 64 * 3 = 192
    hidden_size = 128
    num_layers = 2
    num_classes = 3
    num_epochs = 50
    batch_size = 32
    learning_rate = 0.001
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Prepare datasets
    train_files, test_files = prepare_data('preprocessed_data')
    
    # Create datasets
    train_dataset = TimeSeriesDataset(train_files)
    test_dataset = TimeSeriesDataset(test_files)
    
    # Create data loaders with custom collate function
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                            collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                           collate_fn=collate_fn)
    
    # Initialize the model
    model = GRUClassifier(input_size, hidden_size, num_layers, num_classes).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Train the model
    train_losses, val_accuracies = train_model(
        model, train_loader, test_loader, criterion, optimizer, num_epochs, device
    )
    
    # Load best model and evaluate on test set
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target, lengths in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data, lengths)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    print(f'Test Accuracy: {100 * correct / total:.2f}%')

if __name__ == '__main__':
    main()