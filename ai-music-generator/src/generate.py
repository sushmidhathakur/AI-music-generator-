import os
import random
import pickle
import numpy as np
from pathlib import Path
from music21 import stream, note, chord, tempo, instrument

from src import config
from src import utils
from src import model
from src.utils import logger

def sample_with_temperature(preds, temperature=1.0):
    """
    Helper function to sample an index from a probability array.
    
    Args:
        preds (numpy.ndarray): softmax probability distribution.
        temperature (float): randomness control (0.0 < temp). Low temp makes 
                             predictions conservative; high temp makes them creative.
    """
    # Prevent divide by zero or extreme values
    temperature = max(0.01, temperature)
    
    preds = np.asarray(preds).astype('float64')
    # Add a small epsilon to avoid log(0)
    preds = np.log(preds + 1e-7) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    
    # Draw a sample from the multinomial distribution
    probabilities = np.random.multinomial(1, preds, 1)
    return np.argmax(probabilities)

def generate_notes_list(lstm_model, note_to_int, int_to_note, pitchnames, 
                        seed_sequence=None, generate_len=None, temperature=None):
    """
    Generates a list of notes/chords/rests strings using the model.
    """
    n_vocab = len(pitchnames)
    if generate_len is None:
        generate_len = config.DEFAULT_GENERATION_LENGTH
    if temperature is None:
        temperature = config.DEFAULT_TEMPERATURE

    # 1. Prepare the seed sequence
    if seed_sequence is None or len(seed_sequence) < config.SEQUENCE_LENGTH:
        # Fallback: try to load from notes cache for a real seed
        if config.NOTES_CACHE_FILE.exists():
            logger.info("No seed sequence provided. Loading random seed from training notes cache...")
            with open(config.NOTES_CACHE_FILE, 'rb') as f:
                cache = pickle.load(f)
                cached_notes = cache['notes']
                start = random.randint(0, len(cached_notes) - config.SEQUENCE_LENGTH - 1)
                seed_sequence = cached_notes[start : start + config.SEQUENCE_LENGTH]
        else:
            logger.info("No cache file found. Generating a synthetic seed sequence from vocabulary...")
            # If no cache file, generate a random sequence from vocab
            seed_sequence = [random.choice(pitchnames) for _ in range(config.SEQUENCE_LENGTH)]
            
    # Map seed sequence to integers
    pattern = [note_to_int[char] for char in seed_sequence]
    prediction_output = []

    logger.info(f"Generating {generate_len} notes using temperature {temperature}...")

    # 2. Iteratively predict the next note
    for note_index in range(generate_len):
        # Format input: (1, sequence_length, 1) and normalize
        prediction_input = np.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(n_vocab)

        # Get prediction distribution
        prediction = lstm_model.predict(prediction_input, verbose=0)
        
        # Sample with temperature
        index = sample_with_temperature(prediction[0], temperature)
        result = int_to_note[index]
        prediction_output.append(result)

        # Shift sequence window: append the predicted note, drop the first note
        pattern.append(index)
        pattern = pattern[1:]

    return prediction_output

def notes_to_midi(notes_list, output_filename=None, bpm=None):
    """
    Converts a list of note/chord/rest tokens into a music21 Stream and writes it to a MIDI file.
    
    Args:
        notes_list (list): List of note/chord/rest tokens.
        output_filename (str/Path, optional): Target save path. Defaults to outputs/generated.mid.
        bpm (int, optional): Tempo in Beats Per Minute. Defaults to config.DEFAULT_TEMPO.
        
    Returns:
        Path: Absolute path to the saved MIDI file.
    """
    if output_filename is None:
        config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        # Find unique output name
        idx = 1
        while (config.OUTPUTS_DIR / f"generated_music_{idx:02d}.mid").exists():
            idx += 1
        output_filename = config.OUTPUTS_DIR / f"generated_music_{idx:02d}.mid"
    else:
        output_filename = Path(output_filename)
        output_filename.parent.mkdir(parents=True, exist_ok=True)

    if bpm is None:
        bpm = config.DEFAULT_TEMPO

    logger.info(f"Converting notes list (length {len(notes_list)}) to MIDI stream at {bpm} BPM...")
    
    midi_stream = stream.Stream()
    
    # Add instrument and tempo
    midi_stream.append(instrument.Piano())
    midi_stream.append(tempo.MetronomeMark(number=bpm))

    # We shift by a constant duration (e.g. 0.5 beat = eighth note) for each token
    # to create rhythmic pacing.
    offset = 0.0
    
    for pattern in notes_list:
        # Pattern is a Rest
        if pattern == "REST":
            new_rest = note.Rest()
            new_rest.offset = offset
            # Let's make rests quarter notes (1.0 duration) or eighth notes (0.5 duration)
            new_rest.quarterLength = 0.5
            midi_stream.append(new_rest)
            offset += 0.5
            
        # Pattern is a Chord (notes separated by dots, e.g. C4.E4.G4)
        elif "." in pattern:
            notes_in_chord = pattern.split('.')
            chord_notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(current_note)
                new_note.storedInstrument = instrument.Piano()
                chord_notes.append(new_note)
            new_chord = chord.Chord(chord_notes)
            new_chord.offset = offset
            new_chord.quarterLength = 0.5
            midi_stream.append(new_chord)
            offset += 0.5
            
        # Pattern is a single note
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.quarterLength = 0.5
            new_note.storedInstrument = instrument.Piano()
            midi_stream.append(new_note)
            offset += 0.5

    # Write stream to MIDI file
    try:
        midi_stream.write('midi', fp=str(output_filename))
        logger.info(f"Successfully saved MIDI to {output_filename}")
        return output_filename
    except Exception as e:
        logger.error(f"Failed to write MIDI file: {e}")
        raise e

def run_generation(output_filename=None, generate_len=None, temperature=None, bpm=None):
    """
    Orchestration function to load model, load vocabulary mapping, generate notes, and write MIDI.
    """
    # 1. Verify files exist
    vocab_file = config.PROCESSED_DIR / "vocab_mapping.pkl"
    if not vocab_file.exists():
        logger.error(f"Vocabulary mapping file not found at {vocab_file}. You must train the model first.")
        raise FileNotFoundError(f"Vocabulary file missing at {vocab_file}")
        
    if not config.MODEL_SAVE_FILE.exists():
        logger.error(f"Trained model not found at {config.MODEL_SAVE_FILE}. You must train the model first.")
        raise FileNotFoundError(f"Model file missing at {config.MODEL_SAVE_FILE}")

    # 2. Load vocab mapping
    with open(vocab_file, 'rb') as f:
        vocab_mapping = pickle.load(f)
        pitchnames = vocab_mapping['pitchnames']
        note_to_int = vocab_mapping['note_to_int']
        int_to_note = vocab_mapping['int_to_note']

    # 3. Load model
    lstm_model = model.load_trained_model(config.MODEL_SAVE_FILE)

    # 4. Generate note tokens
    notes = generate_notes_list(
        lstm_model=lstm_model,
        note_to_int=note_to_int,
        int_to_note=int_to_note,
        pitchnames=pitchnames,
        generate_len=generate_len,
        temperature=temperature
    )

    # 5. Export notes to MIDI file
    saved_path = notes_to_midi(notes, output_filename=output_filename, bpm=bpm)
    return saved_path, notes
