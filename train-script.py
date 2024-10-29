import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from sentencepiece import SentencePieceProcessor
import numpy as np
from tqdm import tqdm
from config import ModelArgs
import wandb
import os
from torch.optim import AdamW
from torch.nn import functional as F

def create_dataloaders(batch_size, context_length, split_sizes=(0.8, 0.1, 0.1)):
    """Create train, validation, and test dataloaders from a HuggingFace dataset."""
    # Load dataset (you can replace 'wikitext' with your preferred dataset)
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    
    # Load the SentencePiece tokenizer
    sp_model = SentencePieceProcessor()
    sp_model.Load("tokenizer.model")
    
    def tokenize_function(examples):
        # Tokenize all texts and concatenate them
        tokenized = []
        for text in examples["text"]:
            if text.strip():  # Skip empty lines
                tokens = sp_model.EncodeAsIds(text)
                tokenized.extend(tokens)
        return {"input_ids": tokenized}

    # Tokenize the dataset
    tokenized_datasets = dataset.map(
        tokenize_function,
        remove_columns=dataset["train"].column_names,
        batched=True
    )

    def group_texts(examples):
        # Concatenate all texts and split them into chunks of context_length
        concatenated = examples["input_ids"]
        total_length = len(concatenated)
        
        # Drop the last incomplete chunk if necessary
        total_length = (total_length // context_length) * context_length
        
        result = {
            "input_ids": [concatenated[i:i + context_length] for i in range(0, total_length, context_length)]
        }
        
        # Create targets (shifted input_ids)
        result["labels"] = [ids[1:] + [0] for ids in result["input_ids"]]
        result["input_ids"] = [ids[:-1] for ids in result["input_ids"]]
        
        return result

    # Group texts into chunks of context_length
    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True
    )

    # Convert to PyTorch tensors
    lm_datasets.set_format(type="torch")

    # Create dataloaders
    train_dataloader = DataLoader(
        lm_datasets["train"],
        batch_size=batch_size,
        shuffle=True
    )
    
    val_dataloader = DataLoader(
        lm_datasets["validation"],
        batch_size=batch_size
    )
    
    test_dataloader = DataLoader(
        lm_datasets["test"],
        batch_size=batch_size
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
    
    # Load tokenizer and get vocab size
    sp_model = SentencePieceProcessor()
    sp_model.Load("tokens.model")
    vocab_size = sp_model.GetPieceSize()
    
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
    
    # Create dataloaders
    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(
        batch_size=32,
        context_length=args.max_seq_length
    )
    
    # Initialize optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=len(train_dataloader) * 10  # 10 epochs
    )
    
    # Training loop
    best_val_loss = float('inf')
    num_epochs = 10
    
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
            "epoch": epoch
        })
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pt")
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
