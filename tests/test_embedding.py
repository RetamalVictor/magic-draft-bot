import plotly.express as px
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Example Card2Vec model
class Card2Vec(nn.Module):
    def __init__(self, embedding_dim, vocab_size):
        super(Card2Vec, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, input_card):
        return self.embeddings(input_card)

# Fake data: 10 cards and their contexts
cards = [
    {"input_card": "Fireball", "contexts": {"deck": ["Lightning Bolt", "Shock"], "type": ["Pyroblast"], "functional": ["Flame Slash"]}},
    {"input_card": "Counterspell", "contexts": {"deck": ["Mana Leak", "Daze"], "type": ["Negate"], "functional": ["Force of Will"]}},
    {"input_card": "Giant Growth", "contexts": {"deck": ["Llanowar Elves", "Rancor"], "type": ["Wildsize"], "functional": ["Titanic Growth"]}},
    {"input_card": "Llanowar Elves", "contexts": {"deck": ["Elvish Mystic", "Giant Growth"], "type": ["Elvish Visionary"], "functional": ["Fyndhorn Elves"]}},
    {"input_card": "Shock", "contexts": {"deck": ["Fireball", "Lightning Bolt"], "type": ["Lava Spike"], "functional": ["Incinerate"]}},
    {"input_card": "Serra Angel", "contexts": {"deck": ["Sword to Plowshares", "Wrath of God"], "type": ["Baneslayer Angel"], "functional": ["Archangel of Thune"]}},
    {"input_card": "Wrath of God", "contexts": {"deck": ["Serra Angel", "Sword to Plowshares"], "type": ["Day of Judgment"], "functional": ["Damnation"]}},
    {"input_card": "Rancor", "contexts": {"deck": ["Giant Growth", "Llanowar Elves"], "type": ["Briar Shield"], "functional": ["Aspect of Hydra"]}},
    {"input_card": "Lightning Bolt", "contexts": {"deck": ["Fireball", "Shock"], "type": ["Lava Spike"], "functional": ["Chain Lightning"]}},
    {"input_card": "Sword to Plowshares", "contexts": {"deck": ["Serra Angel", "Wrath of God"], "type": ["Path to Exile"], "functional": ["Condemn"]}}
]

# Tokenization example
vocab = {card["input_card"]: idx for idx, card in enumerate(cards)}
for card in cards:
    for context_type, context_list in card["contexts"].items():
        for context in context_list:
            if context not in vocab:
                vocab[context] = len(vocab)

def tokenize_card(card_name):
    return vocab[card_name]

# Initialize the Card2Vec model
vocab_size = len(vocab)
embedding_dim = 50  # You can adjust this value
card2vec = Card2Vec(embedding_dim, vocab_size)

# Training loop with the fake data
def train_card2vec_multicontext(model, training_data, epochs=500):
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        total_loss = 0
        for data in training_data:
            input_card = torch.tensor([tokenize_card(data["input_card"])], dtype=torch.long)
            context_cards = [torch.tensor([tokenize_card(card)], dtype=torch.long) for context_type, cards in data["contexts"].items() for card in cards]
            
            input_embedding = model(input_card)
            losses = []

            for context in context_cards:
                context_embedding = model(context)
                # Calculate similarity (e.g., cosine similarity) and use it to define the loss
                similarity = F.cosine_similarity(input_embedding, context_embedding)
                # Maximize similarity (bring embeddings closer)
                loss = 1 - similarity.mean()  # Want similarity to be 1, so minimize 1 - similarity
                losses.append(loss)

            loss = torch.stack(losses).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {total_loss}")

# Example usage
train_card2vec_multicontext(card2vec, cards)

# Suppose after training, you have the embeddings and card names
card_names = [card["input_card"] for card in cards]
card_embeddings = torch.stack([card2vec(torch.tensor([tokenize_card(card_name)], dtype=torch.long)) for card_name in card_names])
card_embeddings = card_embeddings.detach().numpy().squeeze(1)

print(card_embeddings.shape)
# Creating a DataFrame for the plot
data = pd.DataFrame({
    'name': card_names,
    'x': card_embeddings[:, 0],  # Embedding dimension 1
    'y': card_embeddings[:, 1],  # Embedding dimension 2
})


# Create an interactive scatter plot
fig = px.scatter(
    data,
    x='x',
    y='y',
    hover_name='name',
    labels={'x': 'Embedding Dimension 1', 'y': 'Embedding Dimension 2'},
    title="Card2Vec Embeddings Visualization",
    template="plotly_white"
)

# Customize plot appearance
fig.update_traces(marker=dict(size=10, opacity=0.7), selector=dict(mode='markers'))
fig.update_layout(clickmode='event+select')

# Show the plot
fig.show()