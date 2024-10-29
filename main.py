import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from sklearn.model_selection import train_test_split
from collections import defaultdict
import random
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Mapping for class names
idx_to_gesture = {0: 'comehere', 1: 'spin', 2: 'stop'}
gesture_to_idx = {'comehere': 0, 'spin': 1, 'stop': 2}

# Previous collate_fn, TimeSeriesDataset, and GRUClassifier classes remain the same
def collate_fn(batch):
    batch.sort(key=lambda x: x[0].shape[0], reverse=True)
    sequences, labels = zip(*batch)
    lengths = [seq.shape[0] for seq in sequences]
    padded_seqs = pad_sequence(sequences, batch_first=True)
    labels = torch.stack(labels)
    return padded_seqs, labels, lengths

class TimeSeriesDataset(Dataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths
        
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        data = np.load(self.file_paths[idx])
        x = torch.FloatTensor(data['array'])
        x = x.reshape(x.shape[0], -1)
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
        packed_x = pack_padded_sequence(x, lengths, batch_first=True)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        packed_out, _ = self.gru(packed_x, h0)
        out, _ = pad_packed_sequence(packed_out, batch_first=True)
        batch_size = out.size(0)
        idx = (torch.tensor(lengths) - 1).view(-1, 1).expand(-1, self.hidden_size)
        idx = idx.unsqueeze(1).to(out.device)
        last_out = out.gather(1, idx).squeeze(1)
        out = self.fc(last_out)
        return out

def prepare_data(data_dir, test_samples_per_class=3):
    class_files = defaultdict(list)
    
    for file in os.listdir(data_dir):
        if file.endswith('.npz'):
            class_name = file.split('_')[0]
            file_path = os.path.join(data_dir, file)
            class_files[class_name].append(file_path)
    
    train_files = []
    test_files = []
    
    for class_name, files in class_files.items():
        test_samples = random.sample(files, test_samples_per_class)
        train_samples = [f for f in files if f not in test_samples]
        
        test_files.extend(test_samples)
        train_files.extend(train_samples)
    
    return train_files, test_files

def plot_training_metrics(train_losses, train_accuracies, val_accuracies):
    plt.figure(figsize=(12, 5))
    
    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss per Epoch')
    plt.legend()
    
    # Plot accuracies
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy per Epoch')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.close()

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[idx_to_gesture[i] for i in range(len(idx_to_gesture))],
                yticklabels=[idx_to_gesture[i] for i in range(len(idx_to_gesture))])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    best_val_acc = 0
    train_losses = []
    train_accuracies = []
    val_accuracies = []
    best_predictions = None
    best_true_labels = None
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_loss = 0
        correct_train = 0
        total_train = 0
        
        for batch_idx, (data, target, lengths) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            outputs = model(data, lengths)
            loss = criterion(outputs, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total_train += target.size(0)
            correct_train += (predicted == target).sum().item()
        
        # Calculate training accuracy
        train_accuracy = 100 * correct_train / total_train
        train_accuracies.append(train_accuracy)
        
        # Validation phase
        model.eval()
        correct = 0
        total = 0
        all_predictions = []
        all_true_labels = []
        
        with torch.no_grad():
            for data, target, lengths in val_loader:
                data, target = data.to(device), target.to(device)
                outputs = model(data, lengths)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_true_labels.extend(target.cpu().numpy())
        
        val_accuracy = 100 * correct / total
        avg_loss = total_loss / len(train_loader)
        
        train_losses.append(avg_loss)
        val_accuracies.append(val_accuracy)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, '
              f'Training Accuracy: {train_accuracy:.2f}%, '
              f'Validation Accuracy: {val_accuracy:.2f}%')
        
        # Save best model and predictions
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            torch.save(model.state_dict(), 'best_model.pth')
            best_predictions = all_predictions
            best_true_labels = all_true_labels
    
    # Plot training metrics
    plot_training_metrics(train_losses, train_accuracies, val_accuracies)
    
    # Plot confusion matrix for best epoch
    plot_confusion_matrix(best_true_labels, best_predictions)
    
    return train_losses, val_accuracies

def main():
    # Hyperparameters
    dims = 64
    num_sensors = 3
    input_size = dims * num_sensors
    hidden_size = 128
    num_layers = 2
    num_classes = 3
    num_epochs = 50
    batch_size = 32
    learning_rate = 0.001
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_files, test_files = prepare_data('preprocessed_data')
    
    train_dataset = TimeSeriesDataset(train_files)
    test_dataset = TimeSeriesDataset(test_files)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                            collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                           collate_fn=collate_fn)
    
    model = GRUClassifier(input_size, hidden_size, num_layers, num_classes).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    train_losses, val_accuracies = train_model(
        model, train_loader, test_loader, criterion, optimizer, num_epochs, device
    )
    
    # Final evaluation
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    
    correct = 0
    total = 0
    all_predictions = []
    all_true_labels = []
    
    with torch.no_grad():
        for data, target, lengths in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data, lengths)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_true_labels.extend(target.cpu().numpy())
    
    test_accuracy = 100 * correct / total
    print(f'Test Accuracy: {test_accuracy:.2f}%')
    
    # Plot final test set confusion matrix
    plot_confusion_matrix(all_true_labels, all_predictions)

if __name__ == '__main__':
    main()