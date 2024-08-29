# inference.py
import torch
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from model.embedding_cmc_type import CombinedEmbedding

import numpy as np
import pandas as pd
import plotly.express as px

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA

# Load the trained model
model = CombinedEmbedding.load_from_checkpoint("checkpoints/final_combined_embedding_model.ckpt").to('cpu')
model.eval()  # Set the model to evaluation mode

# Load the dataset
df = pd.read_csv('data/name_cmc_type_dataset.csv')
# Tokenize the mana_cost and type_line columns
mana_vocab = {mana: idx for idx, mana in enumerate(df['mana_cost'].unique())}
type_vocab = {type_: idx for idx, type_ in enumerate(df['type_line'].unique())}
df['mana_token'] = df['mana_cost'].map(mana_vocab)
df['type_token'] = df['type_line'].map(type_vocab)

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
    if card_name not in df['name'].values:
        print(f"Card name '{card_name}' not found in the dataset.")
        return pd.DataFrame()  # Return an empty DataFrame

    target_embedding = df[df['name'] == card_name]['embedding'].values[0]
    similarities = cosine_similarity([target_embedding], df['embedding'].tolist())[0]
    df['similarity'] = similarities
    closest_cards = df.sort_values(by='similarity', ascending=False).head(top_n + 1)
    return closest_cards[closest_cards['name'] != card_name][['name', 'mana_cost', 'type_line', 'similarity']]

# Example usage
closest_cards = find_closest_cards("Natural Order", df)
if not closest_cards.empty:
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