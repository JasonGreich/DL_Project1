import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
import numpy as np

class TextDataset(Dataset):
    def __init__(self, texts, labels, encoder, class_to_idx):
        self.texts = texts
        self.labels = labels
        self.encoder = encoder
        self.class_to_idx = class_to_idx
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        embedding = self.encoder.encode(text, convert_to_tensor=True)
        label_idx = self.class_to_idx[label]
        
        return embedding, torch.tensor(label_idx, dtype=torch.long)


class ClassificationHead(nn.Module):
    def __init__(self, embedding_dim=384, num_classes=5, hidden_dim=128):
        super(ClassificationHead, self).__init__()
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
    
    def forward(self, embeddings):
        x = self.relu(self.fc1(embeddings))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def train_sentence_transformers_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("[INFO] Loading pre-trained Sentence Transformer encoder...")
    encoder = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    encoder.eval()
    
    print("[INFO] Loading training data...")
    with open("nlp_training_data.json", "r") as f:
        data = json.load(f)
    
    texts = [item["text"] for item in data]
    labels = [item["label"] for item in data]
    
    classes = sorted(list(set(labels)))
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}
    
    print(f"Classes: {classes}")
    print(f"Total samples: {len(texts)}")
    
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    train_dataset = TextDataset(train_texts, train_labels, encoder, class_to_idx)
    test_dataset = TextDataset(test_texts, test_labels, encoder, class_to_idx)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    embedding_dim = encoder.get_sentence_embedding_dimension()
    num_classes = len(classes)
    
    head = ClassificationHead(embedding_dim=embedding_dim, num_classes=num_classes, hidden_dim=128)
    head.to(device)
    
    optimizer = optim.Adam(head.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    epochs = 50
    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0
    
    print(f"\nTraining classification head for {epochs} epochs...\n")
    
    for epoch in range(epochs):
        head.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for embeddings, labels_batch in train_loader:
            embeddings, labels_batch = embeddings.to(device), labels_batch.to(device)
            
            optimizer.zero_grad()
            outputs = head(embeddings)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels_batch).sum().item()
            total += labels_batch.size(0)
        
        head.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for embeddings, labels_batch in test_loader:
                embeddings, labels_batch = embeddings.to(device), labels_batch.to(device)
                outputs = head(embeddings)
                loss = criterion(outputs, labels_batch)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_correct += (predicted == labels_batch).sum().item()
                val_total += labels_batch.size(0)
        
        train_acc = 100 * correct / total
        val_acc = 100 * val_correct / val_total
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(test_loader)
        
        print(f"Epoch {epoch+1:3d}/{epochs} | Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(head.state_dict(), "sentence_transformers_head.pth")
            print(f"  -> Saved best model")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
    
    head.load_state_dict(torch.load("sentence_transformers_head.pth"))
    
    import pickle
    with open("sentence_transformers_classes.pkl", "wb") as f:
        pickle.dump(classes, f)
    
    print("\nModel training completed and saved!")
    print(f"Saved files:")
    print(f"  - sentence_transformers_head.pth")
    print(f"  - sentence_transformers_classes.pkl")
    print(f"\nEncoder: all-MiniLM-L6-v2 (automatically downloaded)")


if __name__ == "__main__":
    train_sentence_transformers_model()
