import os
import numpy as np
from src.utils import logger
from src import config

# Try to import TensorFlow/Keras
HAS_TENSORFLOW = True
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Activation
    from tensorflow.keras.optimizers import Adam
except ImportError:
    HAS_TENSORFLOW = False
    logger.warning("TensorFlow/Keras is not installed. System will use Simulated Model (Markov Chain transition model).")

class SimulatedModel:
    """
    A lightweight, dependency-free first-order Markov Chain model.
    Bypasses TensorFlow if it is not available.
    """
    def __init__(self, n_vocab):
        self.n_vocab = n_vocab
        self.transitions = {} # format: {last_note_idx: {next_note_idx: frequency}}
        self.is_simulated = True

    def train_on_patterns(self, network_input, network_output):
        """
        Reconstructs state transition frequencies from processed sequences.
        """
        logger.info("Training Markov Chain simulated transition model...")
        self.transitions = {}
        n_patterns = len(network_input)
        
        for i in range(n_patterns):
            # network_input[i] shape: (sequence_length, 1)
            # Get the last note in the sequence
            last_val = network_input[i][-1][0]
            last_idx = int(round(last_val * self.n_vocab))
            
            # network_output[i] is a one-hot encoded vector
            next_idx = int(np.argmax(network_output[i]))
            
            if last_idx not in self.transitions:
                self.transitions[last_idx] = {}
            
            self.transitions[last_idx][next_idx] = self.transitions[last_idx].get(next_idx, 0) + 1
            
        logger.info(f"Markov Chain trained. Recorded transition states for {len(self.transitions)} unique notes.")

    def predict(self, input_pattern, verbose=0):
        """
        Predicts next note probabilities based on the last note in the input pattern.
        """
        # input_pattern shape: (1, sequence_length, 1)
        last_val = input_pattern[0][-1][0]
        last_idx = int(round(last_val * self.n_vocab))
        
        probs = np.zeros((1, self.n_vocab))
        
        if last_idx in self.transitions and self.transitions[last_idx]:
            total = sum(self.transitions[last_idx].values())
            for idx, count in self.transitions[last_idx].items():
                if idx < self.n_vocab:
                    probs[0][idx] = count / total
        else:
            # Fallback: uniform distribution if state is unseen
            probs[0] = np.ones(self.n_vocab) / self.n_vocab
            
        return probs

    def save(self, filepath):
        """Saves transition mapping using pickle."""
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        logger.info(f"Saved simulated model pickle to {filepath}")


def create_lstm_model(input_shape, n_vocab):
    """
    Creates and compiles an LSTM model, or returns SimulatedModel if TensorFlow is missing.
    """
    if not HAS_TENSORFLOW:
        logger.info("Creating Simulated Markov Chain model (TensorFlow not available).")
        return SimulatedModel(n_vocab)
        
    logger.info(f"Creating LSTM model. Input shape: {input_shape}, Vocabulary size: {n_vocab}")
    try:
        model = Sequential()
        
        model.add(LSTM(
            config.LSTM_UNITS[0],
            input_shape=input_shape,
            return_sequences=True
        ))
        model.add(Dropout(config.DROPOUT_RATE))
        model.add(BatchNormalization())
        
        model.add(LSTM(
            config.LSTM_UNITS[1],
            return_sequences=False
        ))
        model.add(Dropout(config.DROPOUT_RATE))
        model.add(BatchNormalization())
        
        model.add(Dense(n_vocab))
        model.add(Activation('softmax'))
        
        optimizer = Adam(learning_rate=config.LEARNING_RATE)
        model.compile(
            loss='categorical_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy']
        )
        
        logger.info("LSTM Model compiled successfully.")
        return model
    except Exception as e:
        logger.warning(f"Failed to create TensorFlow model: {e}. Falling back to SimulatedModel.")
        return SimulatedModel(n_vocab)


def load_trained_model(filepath):
    """
    Loads a saved model. Detects if it is a Keras model or pickled SimulatedModel.
    """
    logger.info(f"Loading model from {filepath}")
    
    if not HAS_TENSORFLOW:
        logger.info("TensorFlow missing. Attempting to load as Simulated Model pickle...")
        import pickle
        with open(filepath, 'rb') as f:
            obj = pickle.load(f)
            logger.info("Simulated Model loaded successfully.")
            return obj
            
    # If TensorFlow is present, attempt loading as Keras first
    try:
        model = load_model(filepath)
        logger.info("Model loaded successfully using TensorFlow.")
        return model
    except Exception as e:
        logger.warning(f"Could not load using TensorFlow load_model (file may be pickled): {e}")
        logger.info("Attempting to load file as Pickled Simulated Model...")
        try:
            import pickle
            with open(filepath, 'rb') as f:
                obj = pickle.load(f)
                logger.info("Simulated Model loaded successfully from pickle.")
                return obj
        except Exception as pe:
            logger.error(f"Failed to load file as pickle: {pe}")
            raise pe
