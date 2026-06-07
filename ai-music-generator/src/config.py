import os
from pathlib import Path

# Base Directories
BASE_DIR = Path(__file__).resolve().parent.parent

# Subfolders
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"

# Hyperparameters for Preprocessing and Training
SEQUENCE_LENGTH = 64        # Number of historical notes/chords to consider
VALIDATION_SPLIT = 0.1      # Fraction of data to use for validation
BATCH_SIZE = 64             # Batch size for training
EPOCHS = 60                 # Number of training epochs
LEARNING_RATE = 0.001       # Adam optimizer learning rate

# LSTM Architecture Configuration
LSTM_UNITS = [256, 256]     # Units in stacked LSTM layers
DROPOUT_RATE = 0.3          # Dropout rate to prevent overfitting

# Generation Default Settings
DEFAULT_GENERATION_LENGTH = 200  # Number of notes to generate
DEFAULT_TEMPERATURE = 0.8        # Randomness factor (0.2 = conservative, 1.2 = highly creative)
DEFAULT_TEMPO = 120              # Beats per minute for output MIDI
DEFAULT_INSTRUMENT = "Acoustic Grand Piano"

# Output/Checkpoint filenames
NOTES_CACHE_FILE = PROCESSED_DIR / "notes_processed.pkl"
MODEL_SAVE_FILE = MODELS_DIR / "lstm_music_model.keras"
CHECKPOINT_PREFIX = MODELS_DIR / "lstm_music_checkpoint"
HISTORY_PLOT_FILE = MODELS_DIR / "training_history.png"

def setup_directories():
    """Ensure all required project subdirectories exist."""
    for directory in [DATA_DIR, RAW_DIR, PROCESSED_DIR, MODELS_DIR, OUTPUTS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
