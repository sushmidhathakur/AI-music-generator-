import numpy as np
try:
    from tensorflow.keras.utils import to_categorical
except ImportError:
    def to_categorical(y, num_classes):
        """Native NumPy replacement for Keras to_categorical if TensorFlow is missing."""
        return np.eye(num_classes)[y]
from src.utils import logger
from src import config

def prepare_sequences(notes, pitchnames, sequence_length=None):
    """
    Preprocesses the list of note tokens into input-output sequence pairs for LSTM training.
    
    Args:
        notes (list): List of note/chord string tokens.
        pitchnames (list): List of unique note/chord/rest tokens (vocabulary).
        sequence_length (int, optional): Sequence history length. Defaults to config.SEQUENCE_LENGTH.
        
    Returns:
        tuple: (network_input, network_output, note_to_int, int_to_note)
    """
    if sequence_length is None:
        sequence_length = config.SEQUENCE_LENGTH
        
    n_vocab = len(pitchnames)
    if n_vocab == 0:
        logger.error("Empty vocabulary. Cannot prepare sequences.")
        return None, None, {}, {}

    # Map notes/chords to integers
    note_to_int = {note: number for number, note in enumerate(pitchnames)}
    int_to_note = {number: note for number, note in enumerate(pitchnames)}
    
    # Check if we have enough data
    if len(notes) <= sequence_length:
        logger.error(f"Dataset too small. Number of parsed notes ({len(notes)}) is less than sequence length ({sequence_length}).")
        return None, None, note_to_int, int_to_note

    network_input = []
    network_output = []

    # Create sliding window sequences
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)
    logger.info(f"Created {n_patterns} input-output training patterns.")

    # Reshape input for LSTM: (n_patterns, sequence_length, 1)
    # Normalizing inputs between 0 and 1 relative to vocab size
    network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
    network_input = network_input / float(n_vocab)

    # One-hot encode output: (n_patterns, n_vocab)
    network_output = to_categorical(network_output, num_classes=n_vocab)

    logger.info(f"Network input shape: {network_input.shape}")
    logger.info(f"Network output shape: {network_output.shape}")

    return network_input, network_output, note_to_int, int_to_note
