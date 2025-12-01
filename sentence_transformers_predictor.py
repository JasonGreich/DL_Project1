import torch
import torch.nn as nn
import pickle
from sentence_transformers import SentenceTransformer


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


class SentenceTransformersPredictor:
    def __init__(self, head_path, classes_path, device="cpu", model_name="all-MiniLM-L6-v2"):
        self.device = torch.device(device)
        
        print("[INFO] Loading Sentence Transformer encoder...")
        self.encoder = SentenceTransformer(model_name, device=self.device)
        self.encoder.eval()
        
        with open(classes_path, "rb") as f:
            self.classes = pickle.load(f)
        
        self.idx_to_class = {i: cls for i, cls in enumerate(self.classes)}
        
        embedding_dim = self.encoder.get_sentence_embedding_dimension()
        num_classes = len(self.classes)
        
        self.head = ClassificationHead(embedding_dim=embedding_dim, num_classes=num_classes, hidden_dim=128)
        self.head.load_state_dict(torch.load(head_path, map_location=self.device))
        self.head.to(self.device)
        self.head.eval()
    
    def predict(self, text):
        embedding = self.encoder.encode(text, convert_to_tensor=True)
        embedding = embedding.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.head(embedding)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, class_idx = torch.max(probabilities, 1)
        
        predicted_class = self.idx_to_class[class_idx.item()]
        confidence_score = confidence.item()
        
        return predicted_class, confidence_score
    
    def predict_with_probs(self, text):
        embedding = self.encoder.encode(text, convert_to_tensor=True)
        embedding = embedding.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.head(embedding)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, class_idx = torch.max(probabilities, 1)
        
        predicted_class = self.idx_to_class[class_idx.item()]
        confidence_score = confidence.item()
        
        all_probs = {self.idx_to_class[i]: probabilities[0, i].item() for i in range(len(self.classes))}
        
        return predicted_class, confidence_score, all_probs
