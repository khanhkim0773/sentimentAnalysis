import torch
from torch.nn import Linear
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.data import Data
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from gensim.models import KeyedVectors
import pandas as pd
from torch_geometric.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
import re
from sklearn.model_selection import train_test_split
import networkx as nx
import matplotlib.pyplot as plt

df = pd.read_csv('model\\data\\text.csv', nrows= 100)
path_to_model = "model\\archive\\GoogleNews-vectors-negative300.bin"

df.drop(columns='Unnamed: 0', inplace=True)
df = df.drop_duplicates()
df['text'] = df['text'].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', x))
df['text'] = df['text'].str.lower()
df['text'] = df['text'].str.replace(r'http\S+', '', regex=True)
df['text'] = df['text'].str.replace(r'\s+', ' ', regex=True)
df['text'] = df['text'].str.replace(r'[^\w\s]', '', regex=True)

print(df)
word2vec_model = KeyedVectors.load_word2vec_format(path_to_model, binary=True)

# sentence = "i love natural language processing"
# data = preprocess_data(sentence)

# G = nx.Graph()
# for i, word in enumerate(sentence.split()):
#     G.add_node(i, label=word)
# edge_index = data.edge_index.numpy()
# for i in range(edge_index.shape[1]):
#     G.add_edge(edge_index[0, i], edge_index[1, i])
# pos = nx.spring_layout(G)
# labels = nx.get_node_attributes(G, 'label')
# nx.draw_networkx_nodes(G, pos)
# nx.draw_networkx_labels(G, pos, labels)
# nx.draw_networkx_edges(G, pos)
# plt.show()

def preprocess_data(text):
    words = text.split()
    embeddings = []
    for word in words:
        if word in word2vec_model:
            embeddings.append(word2vec_model[word])

        else:
            embeddings.append(np.zeros(word2vec_model.vector_size))
    
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform([' '.join(words)])
    tfidf_weights = tfidf_matrix.toarray()[0]
    
    edge_index = []
    edge_attr = []
    for i in range(len(words)):
        for j in range(i+1, len(words)):
            if i < len(tfidf_weights) and j < len(tfidf_weights):
                edge_index.append([i, j])
                edge_attr.append(tfidf_weights[i] * tfidf_weights[j]) 
    
    x = torch.tensor(embeddings, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float).view(-1, 1) 
    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    return data

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
train_data_list = []
for text in train_df['text']:
    processed_data = preprocess_data(text)
    train_data_list.append(processed_data)

test_data_list = []
for text in test_df['text']:
    processed_data = preprocess_data(text)
    test_data_list.append(processed_data)

train_labels = torch.tensor(train_df['label'].values, dtype=torch.long)
for data, label in zip(train_data_list, train_labels):
    data.y = label

test_labels = torch.tensor(test_df['label'].values, dtype=torch.long)
for data, label in zip(test_data_list, test_labels):
    data.y = label

train_loader = DataLoader(train_data_list, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data_list, batch_size=32, shuffle=False)

class GNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.classifier = Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = global_mean_pool(x, batch) 
        x = self.classifier(x)
        return x
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GNN(input_dim=300, hidden_dim=64, output_dim=6).to(device)
optimizer = Adam(model.parameters(), lr=0.01)
criterion = CrossEntropyLoss()

best_accuracy = 0.0
for epoch in range(200):
    total_loss = 0
    model.train()
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch)
            pred = out.argmax(dim=1)
            correct += (pred == batch.y).sum().item()
            total += batch.y.size(0)
    
    accuracy = correct / total

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(model.state_dict(), 'my_gnn_model.pth')
    
    print(f'Epoch: {epoch+1}, Loss: {total_loss/len(train_loader)}, Accuracy: {accuracy}')

print(f'Best Accuracy: {best_accuracy}')