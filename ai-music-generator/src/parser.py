import os
from pathlib import Path
from music21 import converter, instrument, note, chord
from src.utils import logger
from src import config

def parse_single_midi(filepath):
    """
    Parses a single MIDI file into a list of note/chord/rest string tokens.
    
    Notes are stored as pitch names (e.g. 'A4').
    Chords are stored as dot-separated pitch names (e.g. 'C4.E4.G4').
    Rests are stored as 'REST'.
    """
    try:
        midi = converter.parse(str(filepath))
    except Exception as e:
        logger.error(f"Failed to parse MIDI file {filepath}: {e}")
        return []

    elements = []
    
    # Try to partition by instrument to focus on keyboard/piano parts
    try:
        parts = instrument.partitionByInstrument(midi)
        if parts:
            # Recursively find elements in the first part (or major keyboard part)
            notes_to_parse = parts.parts[0].recurse()
            logger.debug(f"Parsing instrument part 0: {parts.parts[0].partName or 'Unnamed Part'}")
        else:
            notes_to_parse = midi.flat
            logger.debug("No instrument parts found. Flattening MIDI stream.")
    except Exception as e:
        logger.warning(f"Error partitioning by instrument for {filepath.name}, falling back to flat parse: {e}")
        notes_to_parse = midi.flat

    for element in notes_to_parse:
        if isinstance(element, note.Note):
            elements.append(str(element.pitch))
        elif isinstance(element, chord.Chord):
            # Sort pitches by MIDI number to ensure consistent string representation (e.g. C4.E4.G4)
            sorted_notes = sorted(list(element.notes), key=lambda x: x.pitch.ps)
            elements.append(".".join(str(n.pitch) for n in sorted_notes))
        elif isinstance(element, note.Rest):
            # Rest token allows model to learn silence/rhythm
            elements.append("REST")

    logger.debug(f"Parsed {len(elements)} tokens from {filepath.name}")
    return elements

def parse_midi_dataset(midi_dir, force_reparse=False):
    """
    Parses all MIDI files in the given directory and caches the resulting tokens list.
    
    Returns:
        tuple: (list of all notes/chords across files, list of unique note/chord/rest tokens)
    """
    if config.NOTES_CACHE_FILE.exists() and not force_reparse:
        import pickle
        logger.info(f"Loading notes cache from {config.NOTES_CACHE_FILE}")
        with open(config.NOTES_CACHE_FILE, 'rb') as f:
            data = pickle.load(f)
            return data['notes'], data['pitchnames']
            
    midi_dir = Path(midi_dir)
    midi_files = list(midi_dir.glob("*.mid")) + list(midi_dir.glob("*.midi"))
    
    if not midi_files:
        logger.warning(f"No MIDI files found in {midi_dir}")
        return [], []
        
    logger.info(f"Parsing {len(midi_files)} MIDI files...")
    all_notes = []
    
    for i, f in enumerate(midi_files):
        notes = parse_single_midi(f)
        if notes:
            all_notes.extend(notes)
            # Log progress every 5 files
            if (i + 1) % 5 == 0 or (i + 1) == len(midi_files):
                logger.info(f"Parsed {i + 1}/{len(midi_files)} files...")
                
    if not all_notes:
        logger.error("No notes could be parsed from the MIDI dataset.")
        return [], []
        
    # Get all unique notes/chords/rests
    pitchnames = sorted(list(set(all_notes)))
    
    # Save cache
    cache_data = {
        'notes': all_notes,
        'pitchnames': pitchnames
    }
    
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.NOTES_CACHE_FILE, 'wb') as f:
        import pickle
        pickle.dump(cache_data, f)
        
    logger.info(f"Saved notes cache of {len(all_notes)} tokens and {len(pitchnames)} unique vocabulary to {config.NOTES_CACHE_FILE}")
    
    return all_notes, pitchnames
