import torch
import torch.nn as nn
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam



def load_text_frame(path):
    df = pd.read_csv(path)
    if 'text' not in df.columns:
        df = pd.read_csv(path, header=None, names=['text'])
    return df[['text']]


df = load_text_frame("train.csv")
print(df.head())
print(df.shape)


# We get Train and text data set in different files so we dont split the train dataset inro 2 parts.


def tokenize(text):
    text = text.lower()
    text = text.replace("\r", " ")
    text = text.replace("\n", " ")
    text = text.replace("?", "")
    text = text.replace("'", "")
    text = text.replace(".", "")
    text = text.replace(";", "")
    text = text.replace(",", "")
    text = text.replace(":", "")
    return text.split()

MAX_SEQUENCE_LENGTH = 512

vocab = {"<UNK>": 0}

# Create a vocabulary from the text data
for text in df['text']:
    tokens = tokenize(text)
    for token in tokens:
        if token not in vocab:
            vocab[token] = len(vocab)


print("Vocabulary size:", len(vocab))

inverse_vocab = {index: token for token, index in vocab.items()}

# Get the index of the tokens in vocab
def index_tokens(tokens, vocab):
    return [vocab.get(token, vocab["<UNK>"]) for token in tokens]


# Dataset class 
class textDataset(Dataset):
    def __init__(self, df, vocab):
        self.df = df
        self.vocab = vocab

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        text = self.df.iloc[index]['text']
        tokens = tokenize(text)
        indexed_tokens = index_tokens(tokens, self.vocab)
        indexed_tokens = indexed_tokens[:MAX_SEQUENCE_LENGTH]
        if len(indexed_tokens) < 2:
            indexed_tokens = indexed_tokens + [self.vocab["<UNK>"]]
        input_tokens = torch.tensor(indexed_tokens[:-1], dtype=torch.long)
        target_tokens = torch.tensor(indexed_tokens[1:], dtype=torch.long)
        return input_tokens, target_tokens

# use copilot to create a collate function to pad the sequences in the batch I just forget this part and error comes.
def collate_batch(batch):
    inputs, targets = zip(*batch)
    inputs = pad_sequence(inputs, batch_first=True, padding_value=vocab["<UNK>"])
    targets = pad_sequence(targets, batch_first=True, padding_value=vocab["<UNK>"])
    return inputs, targets

# Create Dataset object
dataset = textDataset(df, vocab)


# Create Dataloader object
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_batch)


# Define the model 
class model(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.lstm(embedded)
        output = self.linear(output)
        return output


# Set hyperparameters
learning_rate = 0.001
epochs = 50

model1 = model(vocab_size=len(vocab), embedding_dim=100, hidden_dim=128, output_dim=len(vocab))

loss_fuction = nn.CrossEntropyLoss(ignore_index=vocab["<UNK>"])
optimizer = Adam(model1.parameters(), lr=learning_rate)


# Training Loop
for epoch in range(epochs):
    for batch in dataloader:
        inputs, targets = batch
        # Clear the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model1(inputs)

        # Compute the loss
        Loss = loss_fuction(outputs.reshape(-1, len(vocab)), targets.reshape(-1))

        # backward pass
        Loss.backward()

        # Update the parameters
        optimizer.step()

        print(f"Epoch [{epoch+1}], Loss: {Loss.item():.4f}")



model1.eval()
df_test = load_text_frame("test.csv")



dataset_test = textDataset(df_test, vocab)
dataloader_test = DataLoader(dataset_test, batch_size=32, shuffle=False, collate_fn=collate_batch)

# Evaluate the model on the test dataset
with torch.no_grad():
    for batch in dataloader_test:
        inputs, _ = batch
        outputs = model1(inputs)
        predictions = torch.argmax(outputs, dim=2)
        print("Predictions:", predictions.tolist())

print("Model evaluation completed.")


def predict_next_token(input_text):
    model1.eval()
    tokens = tokenize(input_text)
    indexed_tokens = index_tokens(tokens, vocab)
    if not indexed_tokens:
        indexed_tokens = [vocab["<UNK>"]]
    input_tensor = torch.tensor([indexed_tokens[:MAX_SEQUENCE_LENGTH]], dtype=torch.long)

    with torch.no_grad():
        outputs = model1(input_tensor)
        last_token_logits = outputs[0, len(input_tensor[0]) - 1]
        top_prediction = torch.argmax(last_token_logits).item()

    return inverse_vocab.get(top_prediction, "<UNK>")


validate_model = pd.read_csv("validation.csv")
results = []
for text in validate_model:
    next_token = predict_next_token(text)
    print("Predicted next tokens for validation set:", next_token)











    

