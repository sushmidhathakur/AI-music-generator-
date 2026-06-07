import os
import sys
import unittest
import numpy as np
import pickle
from pathlib import Path
from music21 import stream, note, chord, instrument

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src import config
from src import utils
from src import parser
from src import preprocess
from src import model
from src import generate

class TestMusicGeneratorComponents(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """Set up testing environment directories."""
        config.setup_directories()
        
    def test_directories_creation(self):
        """Test if all standard project directories are created."""
        self.assertTrue(config.DATA_DIR.exists())
        self.assertTrue(config.RAW_DIR.exists())
        self.assertTrue(config.PROCESSED_DIR.exists())
        self.assertTrue(config.MODELS_DIR.exists())
        self.assertTrue(config.OUTPUTS_DIR.exists())

    def test_midi_parsing_and_token_extraction(self):
        """Test parsing a programmatically constructed MIDI file."""
        # Create a simple music21 stream
        test_stream = stream.Stream()
        test_stream.append(instrument.Piano())
        
        # Add single note, rest, and chord
        test_stream.append(note.Note('C4', quarterLength=1.0))
        test_stream.append(note.Rest(quarterLength=1.0))
        
        c_chord = chord.Chord(['E4', 'G4', 'C5'], quarterLength=1.0)
        test_stream.append(c_chord)
        
        # Save to temp MIDI file
        temp_midi_path = config.PROCESSED_DIR / "test_temp_score.mid"
        test_stream.write('midi', fp=str(temp_midi_path))
        
        self.assertTrue(temp_midi_path.exists())
        
        # Parse it using parser
        tokens = parser.parse_single_midi(temp_midi_path)
        
        # Clean up
        if temp_midi_path.exists():
            os.remove(temp_midi_path)
            
        # Verify tokens
        # Expecting at least ['C4', 'REST', 'E4.G4.C5'] or similar based on sorting
        self.assertGreater(len(tokens), 0)
        self.assertIn("REST", tokens)
        self.assertTrue(any("." in t for t in tokens if t != "REST"), "Chords should be represented with dot notation.")

    def test_sequence_preprocessing(self):
        """Test mapping note tokens to sliding window inputs and outputs."""
        sample_notes = ["C4", "E4", "G4", "C4", "E4", "G4", "REST", "C4", "C4.E4.G4", "REST"]
        vocab = sorted(list(set(sample_notes)))
        
        seq_len = 3
        # Preprocess
        x, y, note_to_int, int_to_note = preprocess.prepare_sequences(sample_notes, vocab, seq_len)
        
        # Check shapes
        # Total notes = 10, sequence length = 3 -> sliding window creates 7 patterns
        self.assertEqual(x.shape, (7, 3, 1))
        self.assertEqual(y.shape, (7, len(vocab)))
        
        # Assert normalization
        self.assertTrue(np.all(x >= 0.0) and np.all(x <= 1.0))
        
        # Check vocab mapping integrity
        self.assertEqual(len(note_to_int), len(vocab))
        for key, val in note_to_int.items():
            self.assertEqual(int_to_note[val], key)

    def test_model_compilation(self):
        """Test model compiling correctly (Keras LSTM or Markov Chain fallback)."""
        input_shape = (config.SEQUENCE_LENGTH, 1)
        n_vocab = 50
        
        lstm_model = model.create_lstm_model(input_shape, n_vocab)
        
        if model.HAS_TENSORFLOW:
            # Verify Keras layer shapes
            self.assertEqual(lstm_model.input_shape, (None, config.SEQUENCE_LENGTH, 1))
            self.assertEqual(lstm_model.output_shape, (None, n_vocab))
            self.assertEqual(lstm_model.loss, 'categorical_crossentropy')
        else:
            # Verify SimulatedModel characteristics
            self.assertTrue(isinstance(lstm_model, model.SimulatedModel))
            self.assertEqual(lstm_model.n_vocab, n_vocab)
            # Test simulated prediction dimensions
            dummy_input = np.zeros((1, config.SEQUENCE_LENGTH, 1))
            pred = lstm_model.predict(dummy_input)
            self.assertEqual(pred.shape, (1, n_vocab))
            self.assertAlmostEqual(np.sum(pred[0]), 1.0)

    def test_temperature_sampling(self):
        """Test temperature sampling logic boundaries."""
        # Setup mock probabilities (must sum to 1.0)
        mock_preds = np.array([0.1, 0.1, 0.6, 0.1, 0.1])
        
        # Test low temperature (should favor index 2 strongly)
        samples = [generate.sample_with_temperature(mock_preds, temperature=0.01) for _ in range(50)]
        # With temp 0.01, index 2 (prob 0.6) should be selected 100% of the time
        self.assertTrue(all(s == 2 for s in samples))
        
        # Test high temperature (should distribute choices more randomly)
        samples_high = [generate.sample_with_temperature(mock_preds, temperature=10.0) for _ in range(100)]
        unique_samples = set(samples_high)
        # With extremely high temperature, other choices should show up
        self.assertGreater(len(unique_samples), 1)

if __name__ == '__main__':
    unittest.main()
