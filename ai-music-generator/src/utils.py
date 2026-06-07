import os
import sys
import logging
import pickle
from pathlib import Path
from src import config

def setup_logging(name="ai_music_gen"):
    """Set up structured logging with formatting."""
    config.setup_directories()
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Avoid duplicate handlers
    if not logger.handlers:
        formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console Handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
        # File Handler
        log_file = config.BASE_DIR / "app.log"
        fh = logging.FileHandler(log_file, encoding='utf-8')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
    return logger

logger = setup_logging()

def check_gpu():
    """Verify if a GPU is available for TensorFlow acceleration."""
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            logger.info(f"GPU Support Detected! Available GPUs: {gpus}")
            # Enable memory growth to avoid blocking all GPU memory
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logger.info("GPU Memory Growth enabled.")
            except Exception as e:
                logger.warning(f"Could not configure GPU memory growth: {e}")
            return True
        else:
            logger.info("No GPU detected. TensorFlow will run on CPU.")
            return False
    except ImportError:
        logger.warning("TensorFlow is not installed. Running in Simulation Mode (CPU).")
        return False

def populate_sample_dataset(num_files=15):
    """
    Populates data/raw/ with sample classical MIDI files extracted
    directly from music21's built-in corpus.
    This provides an offline-safe, high-quality starting dataset.
    """
    import music21
    
    raw_dir = config.RAW_DIR
    config.setup_directories()
    
    # Check if there are already MIDI files in RAW_DIR
    existing_files = list(raw_dir.glob("*.mid")) + list(raw_dir.glob("*.midi"))
    if existing_files:
        logger.info(f"Using {len(existing_files)} existing MIDI files found in {raw_dir}")
        return existing_files
    
    logger.info("No MIDI files found in data/raw. Extracting classical chorales from music21 corpus...")
    
    # Search for Bach chorales in the corpus
    bach_bundle = music21.corpus.search('bach', fileExtensions='xml')
    if not bach_bundle:
        logger.error("Could not find music21 corpus files. Ensure music21 is fully installed.")
        return []
    
    extracted = []
    # Take a selection of Bach chorales
    for i, work in enumerate(bach_bundle[:num_files]):
        try:
            # Parse the piece from the corpus
            score = work.parse()
            
            # Write it as a MIDI file in our raw directory
            output_filename = raw_dir / f"bach_chorale_{i+1:02d}.mid"
            
            # We filter out non-keyboard parts or flatten to make it easier to parse
            # Let's save the whole score as a MIDI file
            mf = music21.midi.translate.music21ObjectToMidiFile(score)
            mf.open(str(output_filename), 'wb')
            mf.write()
            mf.close()
            
            extracted.append(output_filename)
            logger.info(f"Saved {output_filename.name} to data/raw/")
        except Exception as e:
            logger.warning(f"Could not parse/save corpus file {work}: {e}")
            
    logger.info(f"Successfully extracted {len(extracted)} MIDI files for training.")
    return extracted

def save_pickle(obj, filepath):
    """Helper to save an object as a pickle file."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)
    logger.info(f"Saved cache to {filepath}")

def load_pickle(filepath):
    """Helper to load an object from a pickle file."""
    filepath = Path(filepath)
    if not filepath.exists():
        logger.warning(f"File not found: {filepath}")
        return None
    with open(filepath, 'rb') as f:
        obj = pickle.load(f)
    logger.info(f"Loaded cache from {filepath}")
    return obj
