# model.py
import torch
import torch.nn as nn
import pandas as pd
import torch.optim as optim
import pytorch_lightning as pl


class CombinedEmbedding(pl.LightningModule):
    def __init__(self, embedding_dim, mana_vocab_size, type_vocab_size, lr=0.001):
        super(CombinedEmbedding, self).__init__()
        self.save_hyperparameters()  # This line will automatically save the init arguments
        self.mana_embeddings = nn.Embedding(mana_vocab_size, embedding_dim)
        self.type_embeddings = nn.Embedding(type_vocab_size, embedding_dim)
        self.fc = nn.Linear(embedding_dim * 2, embedding_dim)
        self.lr = lr
        self.criterion = nn.CosineEmbeddingLoss()

    def forward(self, mana_token, type_token):
        mana_emb = self.mana_embeddings(mana_token).squeeze(1)
        type_emb = self.type_embeddings(type_token).squeeze(1)
        combined = torch.cat((mana_emb, type_emb), dim=1)
        return self.fc(combined)

    def training_step(self, batch, batch_idx):
        mana_token, type_token, pos_token, half_pos_token, neg_token = batch
        
        # Compute embeddings for the input tokens
        embedding = self(mana_token, type_token)
        pos_embedding = self(pos_token[:, 0], pos_token[:, 1])
        half_pos_embedding = self(half_pos_token[:, 0], half_pos_token[:, 1])
        neg_embedding = self(neg_token[:, 0], neg_token[:, 1])

        # Labels for loss calculation
        pos_label = torch.tensor([1.0], device=self.device)  # Positive example
        half_pos_label = torch.tensor([0.5], device=self.device)  # Half-positive example
        neg_label = torch.tensor([-1.0], device=self.device)  # Negative example

        # Calculate losses
        pos_loss = self.criterion(embedding, pos_embedding, pos_label)
        half_pos_loss = self.criterion(embedding, half_pos_embedding, half_pos_label)
        neg_loss = self.criterion(embedding, neg_embedding, neg_label)

        # Combine losses
        loss = pos_loss + half_pos_loss + neg_loss
        self.log('train_loss', loss)
        
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)
