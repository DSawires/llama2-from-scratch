import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from sentencepiece import SentencePieceProcessor
import numpy as np
from tqdm import tqdm
from config import ModelArgs
import wandb
import os
from torch.optim import AdamW
from torch.nn import functional as F

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, context_length):
        self.tokenizer = tokenizer
        self.context_length = context_length
        
        # Tokenize all texts and concatenate
        tokens = []
        for text in tqdm(texts, desc="Tokenizing texts"):
            if isinstance(text, str) and text.strip():
                text_tokens = self.tokenizer.EncodeAsIds(text)
                tokens.extend(text_tokens)
        
        # Create sequences of context_length + 1 (input + target)
        self.sequences = []
        for i in range(0, len(tokens) - context_length):
            sequence = tokens[i:i + context_length + 1]
            if len(sequence) == context_length + 1:
                self.sequences.append(sequence)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        return {
            "input_ids": torch.tensor(sequence[:-1], dtype=torch.long),
            "labels": torch.tensor(sequence[1:], dtype=torch.long)
        }

def create_dataloaders(batch_size, context_length, split_sizes=(0.8, 0.1, 0.1)):
    """Create train, validation, and test dataloaders from a HuggingFace dataset."""
    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    
    # Load the SentencePiece tokenizer
    print("Loading tokenizer...")
    sp_model = SentencePieceProcessor()
    sp_model.Load("tokenizer.model")
    
    # Create datasets
    print("Creating train dataset...")
    train_dataset = TextDataset(
        dataset["train"]["text"],
        sp_model,
        context_length
    )
    
    print("Creating validation dataset...")
    val_dataset = TextDataset(
        dataset["validation"]["text"],
        sp_model,
        context_length
    )
    
    print("Creating test dataset...")
    test_dataset = TextDataset(
        dataset["test"]["text"],
        sp_model,
        context_length
    )
    
    print(f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True
    )

    return train_dataloader, val_dataloader, test_dataloader

def train_epoch(model, dataloader, optimizer, scheduler, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    for batch in progress_bar:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        
        optimizer.zero_grad()
        
        outputs, loss = model(
            tokens=input_ids,
            targets=labels,
            start_pos=0
        )
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Add gradient clipping
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
            
        total_loss += loss.item()
        progress_bar.set_postfix({"loss": loss.item()})
        
    return total_loss / len(dataloader)

def validate(model, dataloader, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            
            outputs, loss = model(
                tokens=input_ids,
                targets=labels,
                start_pos=0
            )
            
            total_loss += loss.item()
            
    return total_loss / len(dataloader)

def main():
    # Initialize wandb
    wandb.init(project="transformer-training")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load tokenizer and get vocab size
    sp_model = SentencePieceProcessor()
    sp_model.Load("tokenizer.model")
    vocab_size = sp_model.GetPieceSize()
    print(f"Vocabulary size: {vocab_size}")
    
    # Model configuration
    args = ModelArgs(
        dim=512,
        n_layers=8,
        n_heads=8,
        n_kv_heads=8,
        vocab_size=vocab_size,
        multiple_of=32,
        max_seq_length=1024,
        mode='train',
        device=device
    )
    
    # Initialize model
    from transformer import Transformer
    model = Transformer(args).to(device)
    print("Model initialized")
    
    # Create dataloaders
    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(
        batch_size=32,
        context_length=args.max_seq_length - 1  # -1 to account for the target token
    )
    
    # Initialize optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    total_steps = len(train_dataloader) * 10  # 10 epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps
    )
    
    # Training loop
    best_val_loss = float('inf')
    num_epochs = 10
    
    print("Starting training...")
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # Train
        train_loss = train_epoch(model, train_dataloader, optimizer, scheduler, device)
        
        # Validate
        val_loss = validate(model, val_dataloader, device)
        
        # Log metrics
        wandb.log({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "epoch": epoch,
            "learning_rate": scheduler.get_last_lr()[0]
        })
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
            }, "best_model.pt")
            print(f"Saved new best model with validation loss: {val_loss:.4f}")
        
        print(f"Train loss: {train_loss:.4f}")
        print(f"Validation loss: {val_loss:.4f}")
    
    # Test final model
    test_loss = validate(model, test_dataloader, device)
    print(f"\nFinal test loss: {test_loss:.4f}")
    wandb.log({"test_loss": test_loss})
    
    wandb.finish()

if __name__ == "__main__":
    main()
