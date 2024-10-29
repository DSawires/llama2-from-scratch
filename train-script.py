import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from tokenizers import Tokenizer
from tqdm import tqdm
import datetime

class TrainingLogger:
    def __init__(self):
        self.start_time = time.time()
        self.best_val_loss = float('inf')
        
    def log_epoch_start(self, epoch, total_epochs):
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print("\n" + "="*80)
        print(f"Epoch {epoch}/{total_epochs} - Started at {current_time}")
        print("-"*80)
        
    def log_training_step(self, epoch, batch_idx, total_batches, loss, lr):
        elapsed = time.time() - self.start_time
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        
        print(f"\rBatch [{batch_idx}/{total_batches}] "
              f"| Loss: {loss:.4f} "
              f"| LR: {lr:.2e} "
              f"| Time: {hours:02d}:{minutes:02d} ", end="")
              
    def log_epoch_metrics(self, epoch, train_loss, val_loss, val_perplexity):
        is_best = val_loss < self.best_val_loss
        self.best_val_loss = min(val_loss, self.best_val_loss)
        
        print("\n" + "-"*80)
        print(f"Epoch {epoch} Summary:")
        print(f"  Training Loss:     {train_loss:.4f}")
        print(f"  Validation Loss:   {val_loss:.4f}")
        print(f"  Val Perplexity:    {val_perplexity:.2f}")
        print(f"  Best Val Loss:     {self.best_val_loss:.4f}")
        if is_best:
            print("  📈 New best validation loss achieved!")
        print("-"*80)

def load_tokenizer(tokenizer_path):
    """Load the tokenizer from the specified path"""
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer file not found at {tokenizer_path}")
    return Tokenizer.from_file(tokenizer_path)

def create_dataloaders(dataset_name, tokenizer, batch_size, seq_length, split_ratio=0.9):
    """Create training and validation dataloaders from a HuggingFace dataset"""
    print(f"\nLoading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name)
    
    # If dataset only has 'train' split, create validation split
    if 'validation' not in dataset:
        print("Creating validation split...")
        dataset = dataset['train'].train_test_split(train_size=split_ratio)
        train_dataset = dataset['train']
        val_dataset = dataset['test']
    else:
        train_dataset = dataset['train']
        val_dataset = dataset['validation']
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    def tokenize_function(examples):
        texts = examples['text']
        tokenized = [tokenizer.encode(text).ids for text in texts]
        return {'input_ids': tokenized}

    print("\nTokenizing datasets...")
    train_tokenized = train_dataset.map(
        tokenize_function,
        remove_columns=train_dataset.column_names,
        batched=True,
        desc="Tokenizing training data"
    )
    val_tokenized = val_dataset.map(
        tokenize_function,
        remove_columns=val_dataset.column_names,
        batched=True,
        desc="Tokenizing validation data"
    )

    def create_sequences(examples):
        concatenated = []
        for sequence in examples['input_ids']:
            concatenated.extend(sequence)
        
        sequences = [concatenated[i:i + seq_length + 1] 
                    for i in range(0, len(concatenated) - seq_length, seq_length)]
        
        result = {
            'input_ids': [seq[:-1] for seq in sequences],
            'labels': [seq[1:] for seq in sequences]
        }
        return result

    print("\nCreating sequences...")
    train_sequences = train_tokenized.map(
        create_sequences,
        batched=True,
        remove_columns=train_tokenized.column_names,
        desc="Processing training sequences"
    )
    val_sequences = val_tokenized.map(
        create_sequences,
        batched=True,
        remove_columns=val_tokenized.column_names,
        desc="Processing validation sequences"
    )

    print("\nCreating dataloaders...")
    train_loader = DataLoader(
        train_sequences,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )
    val_loader = DataLoader(
        val_sequences,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )

    return train_loader, val_loader

def train_epoch(model, train_loader, optimizer, device, logger, epoch, total_epochs):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    logger.log_epoch_start(epoch, total_epochs)
    
    for batch_idx, batch in enumerate(train_loader):
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        outputs, loss = model(input_ids, start_pos=0, targets=labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Log progress
        if batch_idx % 10 == 0:  # Log every 10 batches
            current_lr = optimizer.param_groups[0]['lr']
            logger.log_training_step(
                epoch, batch_idx, len(train_loader),
                loss.item(), current_lr
            )
    
    return total_loss / len(train_loader)

def validate(model, val_loader, device):
    """Evaluate the model on the validation set"""
    model.eval()
    total_loss = 0
    
    print("\nRunning validation...")
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            outputs, loss = model(input_ids, start_pos=0, targets=labels)
            total_loss += loss.item()
    
    avg_loss = total_loss / len(val_loader)
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    return avg_loss, perplexity

def main():
    # Print system information
    print("\n" + "="*80)
    print("TRAINING SETUP")
    print("="*80)
    
    # Training parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    epochs = 10
    batch_size = 32
    learning_rate = 3e-4
    seq_length = 128
    dataset_name = "wikitext"
    tokenizer_path = "tokens.model"
    
    print(f"\nTraining Parameters:")
    print(f"Epochs: {epochs}")
    print(f"Batch Size: {batch_size}")
    print(f"Learning Rate: {learning_rate}")
    print(f"Sequence Length: {seq_length}")
    print(f"Dataset: {dataset_name}")
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = load_tokenizer(tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()
    print(f"Vocabulary size: {vocab_size:,}")
    
    # Initialize model arguments
    print("\nInitializing model...")
    model_args = ModelArgs(
        dim=512,
        n_layers=8,
        n_heads=8,
        vocab_size=vocab_size,
        max_seq_length=seq_length,
        batch_size=batch_size,
        device=device,
        mode='train'
    )
    
    # Initialize model
    from transformer import Transformer
    model = Transformer(model_args).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        dataset_name,
        tokenizer,
        batch_size,
        seq_length
    )
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Initialize logger
    logger = TrainingLogger()
    
    print("\n" + "="*80)
    print("STARTING TRAINING")
    print("="*80)
    
    # Training loop
    for epoch in range(1, epochs + 1):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device, logger, epoch, epochs)
        
        # Validate
        val_loss, val_perplexity = validate(model, val_loader, device)
        
        # Log metrics
        logger.log_epoch_metrics(epoch, train_loss, val_loss, val_perplexity)
        
        # Save checkpoint if best validation loss
        if val_loss < logger.best_val_loss:
            print("\nSaving checkpoint...")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, 'best_model.pt')
    
    print("\n" + "="*80)
    print("TRAINING COMPLETED")
    print("="*80)
    print(f"Best validation loss: {logger.best_val_loss:.4f}")
    print(f"Model saved as: best_model.pt")
    print("="*80)

if __name__ == "__main__":
    main()
