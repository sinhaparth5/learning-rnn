import os
import mido
from midiutil import MIDIFile
from shutil import copy2

def verify_midi_files(filepath):
    """ Verify if MIDI file is valid and contains actual music data """
    try:
        midi_file = mido.MidiFile(filepath)
        
        # Check if file has any notes
        has_notes = False
        total_time = 0
        
        for track in midi_file.tracks:
            for msg in track:
                # Add up message times to get total duration
                total_time += msg.time
                if msg.type == 'note_on' and msg.velocity > 0:
                    has_notes = True
        
        if not has_notes:
            return False, "No notes found in MIDI file"
        
        # Convert ticks to seconds
        if midi_file.ticks_per_beat:
            tempo = 500000  # Default tempo (120 BPM)
            seconds = total_time * (tempo / midi_file.ticks_per_beat) / 1000000
        else:
            seconds = 0
            
        # Check duration (should be at least 10 seconds)
        if seconds < 10:
            return False, f"MIDI file too short ({seconds:.2f} seconds)"
            
        return True, f"Valid MIDI file ({seconds:.2f} seconds)"
    except Exception as e:
        return False, str(e)


def process_midi_directory(input_dir, output_dir):
    """ Process all MIDI files in directory and copy valid nones to output """
    valid_files = []
    invalid_files = []

    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.mid', '.midi')):
                filepath = os.path.join(root, file)
                is_valid, message = verify_midi_files(filepath)

            if is_valid:
                copy2(filepath, output_dir)
                valid_files.append(file)
                print(f"✓ Valid: {file}")
            else:
                invalid_files.append((file, message))
                print(f"✗ Invalid: {file} - {message}")
    return valid_files, invalid_files

if __name__ == "__main__":
    raw_dir = "midi_dataset/raw"
    processed_dir = "midi_dataset/processed"


    # Processed files
    print("Processing MIDI files...")
    valid, invalid = process_midi_directory(raw_dir, processed_dir)

    # Print summary
    print("\nProcessing Complete!")
    print(f"Valid files: {len(valid)}")
    print(f"Invalid files: {len(invalid)}")

    # Save report
    with open("midi_dataset/processing_report.txt", "w") as f:
        f.write("Valid MIDI Files:\n")
        for file in valid:
            f.write(f"{file}\n")
        
        f.write("Invalid MIDI Files:\n")
        for file, reason in invalid:
            f.write(f"X {file} - {reason}\n")

