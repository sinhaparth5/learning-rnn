import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import mido
from tqdm import tqdm

class MidiPreprocessor:
    def __init__(self, sequence_length=50):
        self.sequence_length = sequence_length
        self.note_range = 128  # MIDI note range
        
    def midi_to_array(self, midi_path):
        """Convert MIDI file to piano roll numpy array"""
        try:
            mid = mido.MidiFile(midi_path)
            
            # Calculate total time in seconds
            total_time = sum(msg.time for track in mid.tracks for msg in track)
            
            # Convert to a reasonable number of time steps (downsample)
            time_steps = min(int(total_time * 100), 1000)  # Max 1000 time steps
            piano_roll = np.zeros((time_steps, self.note_range))
            
            current_time = 0
            for track in mid.tracks:
                for msg in track:
                    current_time += msg.time
                    # Convert time to index
                    time_idx = min(int(current_time * time_steps / total_time), time_steps - 1)
                    
                    if msg.type == 'note_on' and msg.velocity > 0:
                        piano_roll[time_idx:, msg.note] = 1
                    elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                        piano_roll[time_idx:, msg.note] = 0
            
            return piano_roll
        except Exception as e:
            print(f"Error processing {midi_path}: {str(e)}")
            return None

    def create_sequences(self, piano_roll):
        """Create sequences for training"""
        sequences = []
        targets = []
        
        for i in range(0, len(piano_roll) - self.sequence_length - 1, self.sequence_length // 2):
            seq = piano_roll[i:i + self.sequence_length]
            target = piano_roll[i + 1:i + self.sequence_length + 1]
            sequences.append(seq)
            targets.append(target)
            
        return np.array(sequences), np.array(targets)

class MidiDataset(Dataset):
    def __init__(self, sequences, targets):
        # Convert to tensors during initialization
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]
    
    def __getstate__(self):
        # Convert tensors to numpy arrays for pickling
        return {
            'sequences': self.sequences.numpy(),
            'targets': self.targets.numpy()
        }
    
    def __setstate__(self, state):
        # Convert back to tensors after unpickling
        self.sequences = torch.FloatTensor(state['sequences'])
        self.targets = torch.FloatTensor(state['targets'])

def load_midi_data(data_dir, sequence_length=50, batch_size=32, num_workers=0):
    """Load and preprocess MIDI files from directory"""
    preprocessor = MidiPreprocessor(sequence_length=sequence_length)
    all_sequences = []
    all_targets = []
    
    # Get list of MIDI files
    midi_files = [f for f in os.listdir(data_dir) if f.endswith(('.mid', '.midi'))]
    print(f"Found {len(midi_files)} MIDI files")
    
    # Process each MIDI file with progress bar
    for filename in tqdm(midi_files, desc="Loading MIDI files"):
        midi_path = os.path.join(data_dir, filename)
        piano_roll = preprocessor.midi_to_array(midi_path)
        
        if piano_roll is not None and len(piano_roll) > sequence_length + 1:
            sequences, targets = preprocessor.create_sequences(piano_roll)
            all_sequences.extend(sequences)
            all_targets.extend(targets)
            
    if not all_sequences:
        raise ValueError("No valid sequences were generated from the MIDI files")
    
    print(f"Created {len(all_sequences)} sequences from {len(midi_files)} files")
    
    # Convert lists to numpy arrays
    all_sequences = np.array(all_sequences)
    all_targets = np.array(all_targets)
    
    # Create dataset and dataloader
    dataset = MidiDataset(all_sequences, all_targets)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=False  # Set to False to avoid pickling issues
    )
    
    return dataloader