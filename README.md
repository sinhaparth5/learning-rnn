# MIDI Music Generation using RNN/LSTM

This project uses a Recurrent Neural Network (RNN) with LSTM to generate Indian devotional music. The model learns patterns from existing MIDI files and generates new, similar musical pieces.

## Prerequisites

- Python version: 3.12.x or 3.13.1+ (Note: Python 3.13.0 has a known pickling issue that affects model saving)
- uv package manager

## Installation

```bash
# Create and activate virtual environment using uv
uv venv
source .venv/bin/activate  # On Unix/macOS
# Or on Windows:
# .venv\Scripts\activate

# Install dependencies
uv pip install torch tqdm mido numpy
```

## Project Structure

```
midi_music/
├── data/
│   ├── raw/         # Original MIDI files
│   └── processed/   # Verified MIDI files
├── models/          # Saved model checkpoints
├── output/          # Generated music
└── src/
    ├── preprocessing.py  # MIDI data preprocessing
    ├── model.py         # LSTM model definition
    ├── train.py         # Training script
    └── generation.py    # Music generation code
```

## Usage

1. Place your MIDI files in `data/raw/` directory

2. Verify and process MIDI files:
```bash
python src/verify_midi.py
```

3. Train the model:
```bash
python src/train.py
```

4. Generate music:
The training process will automatically generate sample outputs. Generated MIDI files will be saved in the `output/` directory.

## Model Architecture

- Input size: 128 (MIDI note range)
- Hidden size: 256
- LSTM layers: 2
- Dropout: 0.5

## Features

- MIDI file preprocessing and verification
- LSTM-based sequence learning
- Temperature-based music generation
- Progress tracking with tqdm
- Checkpoint saving/loading
- Multiple sample generation with different creativity levels

## Known Issues

- Python 3.13.0 has a pickling issue that prevents model saving. Use Python 3.12.x or 3.13.1+ instead.
- Make sure your MIDI files are valid and contain actual musical data.

## License

MIT License

## Acknowledgments

- Training data: Indian devotional MIDI files
- Libraries: PyTorch, mido, tqdm, numpy
