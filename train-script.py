import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset
from datasets import load_dataset
from sentencepiece import SentencePieceProcessor
import numpy as np
from tqdm import tqdm
from config import ModelArgs
import wandb
import os
from torch.optim import AdamW
from torch.nn import functional as F
import gc

class StreamingTextDataset(IterableDataset):
    def __init__(self, texts, tokenizer, context_length, chunk_size=1000):
        self.texts = texts
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.chunk_size = chunk_size
        
    def __iter__(self):
        buffer = []
        
        for text in self.texts:
            if isinstance(text, str) and text.strip():
                # Tokenize the text
                tokens = self.tokenizer.EncodeAsIds(text.strip())
                buffer.extend(tokens)
                
                # Process when buffer is large enough
                while len(buffer) >= self.context_length + 1:
                    sequence = buffer[:self.context_length + 1]
                    yield {
                        "input_ids": torch.tensor(sequence[:-1], dtype=torch.long),
                        "labels": torch.tensor(sequence[1:], dtype=torch.long)
                    }
                    buffer = buffer[self.context_length:]  # Overlap by keeping the last token
            
            # Clear buffer periodically to prevent memory buildup
            if len(buffer) > self.chunk_size:
                buffer = buffer[-self.context_length:]
                gc.collect()  # Force garbage collection

def create_dataloaders(batch_size, context_length):
    """Create train, validation, and test dataloaders from a HuggingFace dataset."""
    print("Loading dataset...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    
    print("Loading tokenizer...")
    sp_model = SentencePieceProcessor()
    sp_model.Load("tokenizer.model")
    
    # Create streaming datasets
    print("Creating datasets...")
    train_dataset = StreamingTextDataset(
        dataset["train"]["text"],
        sp_model,
        context_length
    )
    
    val_dataset = StreamingTextDataset(
        dataset["validation"]["text"],
        sp_model,
        context_length
    )
    
    test_dataset = StreamingTextDataset(
        dataset["test"]["text"],
        sp_model,
        context_length
    )
    
    # Create dataloaders with prefetch factor
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=2,  # Reduced number of workers
        pin_memory=True,
        prefetch_factor=2
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=2,
        pin_memory=True,
        prefetch_factor=2
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=2,
        pin_memory=True,
        prefetch_factor=2
    )

    return train_dataloader, val_dataloader, test_dataloader

def train_epoch(model, dataloader, optimizer, scheduler, device, max_steps=None):
    """Train for one epoch or max_steps."""
    model.train()
    total_loss = 0
    steps = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    for batch in progress_bar:
        # Move batch to device
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        
        optimizer.zero_grad()
        
        outputs, loss = model(
            tokens=input_ids,
            targets=labels,
            start_pos=0
        )
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
            
        total_loss += loss.item()
        steps += 1
        
        # Update progress bar
        progress_bar.set_postfix({
            "loss": loss.item(),
            "avg_loss": total_loss / steps,
            "lr": scheduler.get_last_lr()[0] if scheduler else "N/A"
        })
        
        # Memory management
        del outputs, loss
        torch.cuda.empty_cache()
        
        if max_steps and steps >= max_steps:
            break
    
    return total_loss / steps

def validate(model, dataloader, device, max_steps=None):
    """Validate the model."""
    model.eval()
    total_loss = 0
    steps = 0
    
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
            steps += 1
            
            # Clean up memory
            del outputs, loss
            torch.cuda.empty_cache()
            
            if max_steps and steps >= max_steps:
                break
    
    return total_loss / steps

def main():
    # Set memory efficient settings
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
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
        max_seq_length=512,  # Reduced sequence length
        mode='train',
        device=device
    )
    
    # Initialize model
    from model import Transformer
    model = Transformer(args).to(device)
    print("Model initialized")
    
    # Create dataloaders
    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(
        batch_size=16,  # Reduced batch size
        context_length=args.max_seq_length - 1
    )
    
    # Initialize optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    steps_per_epoch = 1000  # Limit steps per epoch
    total_steps = steps_per_epoch * 10  # 10 epochs
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
        train_loss = train_epoch(
            model, train_dataloader, optimizer, scheduler, device,
            max_steps=steps_per_epoch
        )
        
        # Validate
        val_loss = validate(
            model, val_dataloader, device,
            max_steps=steps_per_epoch // 5
        )
        
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
        
        # Force garbage collection between epochs
        gc.collect()
        torch.cuda.empty_cache()
    
    # Test final model
    test_loss = validate(model, test_dataloader, device, max_steps=steps_per_epoch // 5)
    print(f"\nFinal test loss: {test_loss:.4f}")
    wandb.log({"test_loss": test_loss})
    
    wandb.finish()

if __name__ == "__main__":
    main()
