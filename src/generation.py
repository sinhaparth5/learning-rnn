import torch
import mido
import numpy as np
import os

def generate_music(model, seed_sequence, output_file, steps=500, temperature=1.0):
    """Generate MIDI file from trained model"""
    print(f"Generating music with temperature {temperature}...")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Move seed sequence to model's device
    device = next(model.parameters()).device
    seed_sequence = seed_sequence.to(device)
    
    try:
        # Generate sequence
        generated = model.generate(seed_sequence.unsqueeze(0), steps, temperature)
        generated = generated.squeeze(0).cpu().numpy()
        
        # Create MIDI file
        mid = mido.MidiFile()
        track = mido.MidiTrack()
        mid.tracks.append(track)
        
        # Set tempo (120 BPM)
        tempo = mido.bpm2tempo(120)
        track.append(mido.MetaMessage('set_tempo', tempo=tempo))
        
        # Convert piano roll to MIDI messages
        current_state = np.zeros(128, dtype=bool)
        time_per_step = 30  # Adjust this to change the speed of the music
        
        for step, new_state in enumerate(generated):
            new_state = new_state > 0.5  # Convert probabilities to binary
            
            # Find notes that changed
            notes_on = np.where((~current_state) & new_state)[0]
            notes_off = np.where(current_state & (~new_state))[0]
            
            # Add note off messages
            for note in notes_off:
                track.append(mido.Message('note_off', note=int(note), velocity=64, time=0))
            
            # Add note on messages
            for note in notes_on:
                track.append(mido.Message('note_on', note=int(note), velocity=64, time=0))
            
            # Update current state
            current_state = new_state
            
            # Add time gap between steps (if not the last step)
            if step < len(generated) - 1:
                track.append(mido.Message('note_on', note=0, velocity=0, time=time_per_step))
        
        # Turn off any remaining notes
        for note in np.where(current_state)[0]:
            track.append(mido.Message('note_off', note=int(note), velocity=64, time=0))
        
        # Save MIDI file
        mid.save(output_file)
        print(f"Successfully generated MIDI file: {output_file}")
        
    except Exception as e:
        print(f"Error generating music: {str(e)}")
        import traceback
        traceback.print_exc()

def generate_multiple_samples(model, train_loader, output_dir, num_samples=3, 
                            steps=500, temperatures=[0.8, 1.0, 1.2]):
    """Generate multiple music samples with different temperatures"""
    os.makedirs(output_dir, exist_ok=True)
    print("\nStarting music generation...")
    
    try:
        # Get a seed sequence from the training data
        seed_sequence = next(iter(train_loader))[0][0]
        
        # Generate samples with different temperatures
        for i, temp in enumerate(temperatures):
            output_file = os.path.join(output_dir, f'generated_sample_{i+1}_temp_{temp}.mid')
            generate_music(model, seed_sequence, output_file, steps=steps, temperature=temp)
            
        print(f"\nGenerated {len(temperatures)} samples in {output_dir}")
        
    except Exception as e:
        print(f"Error in generate_multiple_samples: {str(e)}")
        import traceback
        traceback.print_exc()