import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter


class WordTokenizer:
    def __init__(self, vocab_size=5000):
        self.vocab_size = vocab_size
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}  # Special tokens
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}

    def build_vocab(self, texts):
        word_freq = Counter()
        for text in texts:
            words = self.tokenize(text)
            word_freq.update(words)

        # Take most common words up to vocab_size (excluding special tokens)
        most_common = word_freq.most_common(self.vocab_size - len(self.word2idx))
        for idx, (word, _) in enumerate(most_common, start=len(self.word2idx)):
            self.word2idx[word] = idx
            self.idx2word[idx] = word

    def tokenize(self, text):
        return re.findall(r"\b\w+\b", text.lower())  # Simple word-based tokenizer

    def encode(self, text, max_length=32, padding=True, truncation=True):
        words = self.tokenize(text)
        input_ids = [self.word2idx.get(word, self.word2idx["<UNK>"]) for word in words]

        # Apply truncation
        if truncation and len(input_ids) > max_length:
            input_ids = input_ids[:max_length]

        # Apply padding
        if padding:
            while len(input_ids) < max_length:
                input_ids.append(self.word2idx["<PAD>"])

        attention_mask = [1 if i != self.word2idx["<PAD>"] else 0 for i in input_ids]
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def decode(self, token_ids):
        return " ".join([self.idx2word.get(i, "<UNK>") for i in token_ids])


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=32):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoded = self.tokenizer.encode(self.texts[idx], max_length=self.max_length)
        return (
            torch.tensor(encoded["input_ids"], dtype=torch.long),
            torch.tensor(encoded["attention_mask"], dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )


class SimpleGPTClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, num_classes=2):
        super(SimpleGPTClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_ids, attention_mask):
        embedded = self.embedding(input_ids)  # (batch, seq_len, embed_dim)
        _, hidden = self.rnn(embedded)       # (1, batch, hidden_dim)
        logits = self.fc(hidden.squeeze(0))  # (batch, num_classes)
        return logits


# Sample dataset
texts = [
    "The movie was a masterpiece!",
    "I hated the film. It was terrible.",
    "An excellent and thrilling experience.",
    "Not worth watching at all.",
    "The cinematography was stunning!"
]
labels = [1, 0, 1, 0, 1]  # 1 = Positive, 0 = Negative

# Initialize and train tokenizer
tokenizer = WordTokenizer(vocab_size=50)
tokenizer.build_vocab(texts)

# Create dataset and dataloader
dataset = TextDataset(texts, labels, tokenizer, max_length=16)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleGPTClassifier(vocab_size=len(tokenizer.word2idx)).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for input_ids, attention_mask, labels in dataloader:
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss:.4f}")

def classify_text(text):
    model.eval()
    encoded = tokenizer.encode(text, max_length=16)
    input_ids = torch.tensor(encoded["input_ids"], dtype=torch.long).unsqueeze(0).to(device)
    attention_mask = torch.tensor(encoded["attention_mask"], dtype=torch.long).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_ids, attention_mask)

    predicted_class = torch.argmax(logits, dim=1).item()
    return "Positive" if predicted_class == 1 else "Negative"

# Test classification
# test_text = "The Dark Knight was a masterpiece!"
# print(f"Test Text: '{test_text}' => Prediction: {classify_text(test_text)}")

torch.save(model.state_dict(), "model.pth")
