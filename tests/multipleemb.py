import numpy as np
import pandas as pd
import plotly.express as px

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA

import torch
import torch.nn as nn
import torch.optim as optim

# Load the dataset (assuming the subset has been saved as 'subset_dataset.csv')
df = pd.read_csv('data/name_cmc_type_dataset.csv')


# Tokenization
mana_vocab = {mana: idx for idx, mana in enumerate(df['mana_cost'].unique())}
type_vocab = {type_: idx for idx, type_ in enumerate(df['type_line'].unique())}
df['mana_token'] = df['mana_cost'].map(mana_vocab)
df['type_token'] = df['type_line'].map(type_vocab)

# Define the CombinedEmbedding model
class CombinedEmbedding(nn.Module):
    def __init__(self, embedding_dim, mana_vocab_size, type_vocab_size):
        super(CombinedEmbedding, self).__init__()
        self.mana_embeddings = nn.Embedding(mana_vocab_size, embedding_dim)
        self.type_embeddings = nn.Embedding(type_vocab_size, embedding_dim)
        self.fc = nn.Linear(embedding_dim * 2, embedding_dim)

    def forward(self, mana_token, type_token):
        mana_emb = self.mana_embeddings(mana_token)
        type_emb = self.type_embeddings(type_token)
        combined = torch.cat((mana_emb, type_emb), dim=1)
        return self.fc(combined)

# Initialize the model
embedding_dim = 50
mana_vocab_size = len(mana_vocab)
type_vocab_size = len(type_vocab)
model = CombinedEmbedding(embedding_dim, mana_vocab_size, type_vocab_size)

# Define the training process
def train_model(model, df, epochs=10, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CosineEmbeddingLoss()

    for epoch in range(epochs):
        total_loss = 0
        for _, row in df.iterrows():
            # Get the current card's tokens
            mana_token = torch.tensor([row['mana_token']])
            type_token = torch.tensor([row['type_token']])

            # Generate the embedding
            embedding = model(mana_token, type_token)

            # Select a positive example (same mana_cost, same type_line)
            positive_examples = df[(df['mana_cost'] == row['mana_cost']) & (df['type_line'] == row['type_line'])]
            if len(positive_examples) > 1:
                pos_row = positive_examples.sample().iloc[0]
                pos_mana_token = torch.tensor([pos_row['mana_token']])
                pos_type_token = torch.tensor([pos_row['type_token']])
                pos_embedding = model(pos_mana_token, pos_type_token)
            else:
                continue

            # Select a negative example (different mana_cost or type_line)
            negative_examples = df[(df['mana_cost'] != row['mana_cost']) | (df['type_line'] != row['type_line'])]
            neg_row = negative_examples.sample().iloc[0]
            neg_mana_token = torch.tensor([neg_row['mana_token']])
            neg_type_token = torch.tensor([neg_row['type_token']])
            neg_embedding = model(neg_mana_token, neg_type_token)

            # Prepare labels for loss calculation
            pos_label = torch.tensor([1.0])  # Positive example
            neg_label = torch.tensor([-1.0])  # Negative example

            # Calculate the loss for positive and negative pairs
            pos_loss = criterion(embedding, pos_embedding, pos_label)
            neg_loss = criterion(embedding, neg_embedding, neg_label)

            loss = pos_loss + neg_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {total_loss}")

# Train the model
train_model(model, df)

# Generate embeddings for the entire dataset after training
embeddings = []
for _, row in df.iterrows():
    mana_token = torch.tensor([row['mana_token']])
    type_token = torch.tensor([row['type_token']])
    embedding = model(mana_token, type_token).detach().numpy().flatten()
    embeddings.append(embedding)


df['embedding'] = embeddings

# Function to find the closest cards based on a card name
def find_closest_cards(card_name, df, top_n=5):
    target_embedding = df[df['name'] == card_name]['embedding'].values[0]
    similarities = cosine_similarity([target_embedding], df['embedding'].tolist())[0]
    df['similarity'] = similarities
    closest_cards = df.sort_values(by='similarity', ascending=False).head(top_n + 1)
    return closest_cards[closest_cards['name'] != card_name][['name', 'mana_cost', 'type_line', 'similarity']]

# Example usage
closest_cards = find_closest_cards("Natural Order", df)
print(closest_cards)

embeddings = np.vstack(df['embedding'].values)

pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(embeddings)

# Add the reduced embeddings to the DataFrame
df['x'] = reduced_embeddings[:, 0]
df['y'] = reduced_embeddings[:, 1]

# Create a DataFrame for the plot
plot_data = pd.DataFrame({
    'name': df['name'],
    'x': df['x'],
    'y': df['y']
})

# Create an interactive scatter plot
fig = px.scatter(
    plot_data,
    x='x',
    y='y',
    hover_name='name',
    labels={'x': 'Embedding Dimension 1', 'y': 'Embedding Dimension 2'},
    title="Card Embeddings Visualization",
    template="plotly_white"
)

# Customize plot appearance
fig.update_traces(marker=dict(size=10, opacity=0.7), selector=dict(mode='markers'))
fig.update_layout(clickmode='event+select')

# Show the plot
fig.show()