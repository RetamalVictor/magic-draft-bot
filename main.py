import pandas as pd
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from model.embedding_cmc_type import CombinedEmbedding
from data_training.utils import prepare_training_data


# from visuals.cluster import PCALoggingCallback

# Load the dataset
df = pd.read_csv('data/name_cmc_type_dataset.csv')

# Tokenize the mana_cost and type_line columns
mana_vocab = {mana: idx for idx, mana in enumerate(df['mana_cost'].unique())}
type_vocab = {type_: idx for idx, type_ in enumerate(df['type_line'].unique())}
df['mana_token'] = df['mana_cost'].map(mana_vocab)
df['type_token'] = df['type_line'].map(type_vocab)

# Prepare training data
training_data = prepare_training_data(df)
train_loader = DataLoader(training_data, batch_size=32, shuffle=True)

# Initialize the model
embedding_dim = 50
mana_vocab_size = len(mana_vocab)
type_vocab_size = len(type_vocab)
model = CombinedEmbedding(embedding_dim, mana_vocab_size, type_vocab_size)

# Define callbacks
# Initialize the PCA logging callback
# pca_logging_callback = PCALoggingCallback(df, log_dir="tb_logs/combined_embedding", every_n_epochs=10)


checkpoint_callback = ModelCheckpoint(
    monitor='train_loss',
    dirpath='checkpoints',
    filename='combined_embedding-{epoch:02d}-{train_loss:.2f}',
    save_top_k=3,
    mode='min',
)

early_stopping_callback = EarlyStopping(
    monitor='train_loss',
    patience=30,
    mode='min'
)

# Set up loggers
tensorboard_logger = TensorBoardLogger("tb_logs", name="combined_embedding")
# wandb_logger = WandbLogger(project="combined_embedding_project")

# Train the model with PyTorch Lightning Trainer
trainer = pl.Trainer(
    max_epochs=1000,
    callbacks=[checkpoint_callback, early_stopping_callback],
    # logger=[tensorboard_logger, wandb_logger]
    logger=[tensorboard_logger]
)
trainer.fit(model, train_loader)

# Save the final model checkpoint
trainer.save_checkpoint("checkpoints/final_combined_embedding_model.ckpt")
