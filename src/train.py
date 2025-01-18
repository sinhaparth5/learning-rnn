import torch
import torch.nn as nn
import torch.optim as optim
import os
from tqdm import tqdm
import sys
from pathlib import Path
import pickle
import warnings

def save_model_state(model, filepath):
    """Save only the model's state dictionary"""
    try:
        # Save only the model's state dictionary
        torch.save(model.state_dict(), filepath, _use_new_zipfile_serialization=False)
        return True
    except Exception as e:
        print(f"Error saving model state: {str(e)}")
        return False

def save_checkpoint(model, optimizer, epoch, loss, filepath):
    """Attempt to save full checkpoint with fallback options"""
    try:
        # First try: Save with old serialization
        state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }
        
        # Test if state is pickle-able
        pickle.dumps(state)
        
        # If pickle test passes, save the state
        torch.save(state, filepath, _use_new_zipfile_serialization=False)
        return True
    except Exception as e:
        print(f"Warning: Full checkpoint save failed: {str(e)}")
        print("Falling back to model state-only save...")
        return save_model_state(model, filepath)

def train_model(model, train_loader, num_epochs, learning_rate, device, save_dir):
    """Train the RNN model"""
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Create save directory if it doesn't exist
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for batch_idx, (data, target) in enumerate(progress_bar):
            try:
                data, target = data.to(device), target.to(device)
                
                # Forward pass
                optimizer.zero_grad()
                output, _ = model(data)
                loss = criterion(output, target)
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                # Update total loss
                total_loss += loss.item()
                
                # Update progress bar
                progress_bar.set_postfix({'loss': loss.item()})
            
            except Exception as e:
                print(f"Error in training batch {batch_idx}: {str(e)}")
                continue
        
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1}: Average loss = {avg_loss:.6f}')
        
        # Save if best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path = os.path.join(save_dir, 'best_model.pt')
            if save_checkpoint(model, optimizer, epoch, best_loss, save_path):
                print(f"Saved best model with loss: {best_loss:.6f}")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            save_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pt')
            if save_checkpoint(model, optimizer, epoch, avg_loss, save_path):
                print(f"Saved checkpoint at epoch {epoch+1}")

def main():
    # Parameters
    input_size = 128  # MIDI note range
    hidden_size = 256
    num_layers = 2
    sequence_length = 50
    batch_size = 32
    num_epochs = 100
    learning_rate = 0.001
    
    # Directories
    data_dir = "midi_dataset/processed"
    save_dir = "models"
    
    # Create directories
    os.makedirs(save_dir, exist_ok=True)
    
    # Check if data directory exists and has files
    if not os.path.exists(data_dir):
        print(f"Error: Directory '{data_dir}' does not exist")
        sys.exit(1)
    
    midi_files = [f for f in os.listdir(data_dir) if f.endswith(('.mid', '.midi'))]
    if not midi_files:
        print(f"Error: No MIDI files found in '{data_dir}'")
        sys.exit(1)
        
    print(f"Found {len(midi_files)} MIDI files in {data_dir}")
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        # Load data
        print("Loading MIDI data...")
        from preprocessing import load_midi_data
        train_loader = load_midi_data(data_dir, sequence_length, batch_size)
        print("Data loading complete!")
        
        # Initialize model
        from model import MusicRNN
        model = MusicRNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers
        ).to(device)
        
        # Train model
        print("Starting training...")
        train_model(model, train_loader, num_epochs, learning_rate, device, save_dir)
        print("Training complete!")
        
        # Generate samples
        print("\nGenerating music samples...")
        from generation import generate_multiple_samples
        output_dir = "output"
        generate_multiple_samples(
            model=model,
            train_loader=train_loader,
            output_dir=output_dir,
            num_samples=3,
            steps=1000,  # Generate longer sequences
            temperatures=[0.8, 1.0, 1.2]  # Different creativity levels
        )
        
    except Exception as e:
        print(f"Error during execution: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()