import os
import sys
import argparse
import unittest
from pathlib import Path

# Add project root directory to python path
sys.path.append(str(Path(__file__).resolve().parent))

from src import config
from src import utils
from src import parser
from src import train
from src import generate
from src import visualize
from src.utils import logger

def run_tests():
    """Runs all automated unit tests in the tests directory."""
    logger.info("Discovering and running automated unit tests...")
    loader = unittest.TestLoader()
    start_dir = str(Path(__file__).resolve().parent / "tests")
    suite = loader.discover(start_dir, pattern="test_*.py")
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)

def run_webserver():
    """Launches the Flask web dashboard."""
    logger.info("Initializing Flask Web Server environment...")
    # Pre-populate sample dataset if raw folder is empty
    utils.populate_sample_dataset()
    
    # Import app and start
    from app.server import app
    app.run(host='127.0.0.1', port=5000, debug=True)

def main():
    config.setup_directories()
    
    parser_desc = "ANTIGRAVITY // Deep Learning LSTM Music Generator CLI"
    main_parser = argparse.ArgumentParser(description=parser_desc)
    subparsers = main_parser.add_subparsers(dest="command", help="Command to execute")

    # 'parse' command
    subparsers.add_parser("parse", help="Parse raw MIDI dataset files and cache tokens.")

    # 'train' command
    train_parser = subparsers.add_parser("train", help="Train the LSTM network model on preprocessed MIDI tokens.")
    train_parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs")
    train_parser.add_argument("--batch-size", type=int, default=None, help="Override training batch size")
    train_parser.add_argument("--reparse", action="store_true", help="Force reparsing of raw dataset")

    # 'generate' command
    gen_parser = subparsers.add_parser("generate", help="Generate a new MIDI file using the trained model.")
    gen_parser.add_argument("--temp", type=float, default=None, help="Creativity temperature (0.1 - 1.5)")
    gen_parser.add_argument("--length", type=int, default=None, help="Number of notes to generate")
    gen_parser.add_argument("--bpm", type=int, default=None, help="Tempo Beats Per Minute")
    gen_parser.add_argument("--out", type=str, default=None, help="Optional output midi file path")

    # 'visualize' command
    vis_parser = subparsers.add_parser("visualize", help="Plot and save a piano-roll image for a MIDI file.")
    vis_parser.add_argument("midi_file", type=str, help="Path to the MIDI file to visualize")
    vis_parser.add_argument("--out", type=str, default=None, help="Optional output image path")

    # 'serve' command
    subparsers.add_parser("serve", help="Launch the Flask web dashboard interface on localhost.")

    # 'test' command
    subparsers.add_parser("test", help="Run the automated test suite.")

    args = main_parser.parse_args()

    if not args.command:
        main_parser.print_help()
        sys.exit(0)

    try:
        if args.command == "parse":
            # Extract sample files if raw is empty
            utils.populate_sample_dataset()
            parser.parse_midi_dataset(config.RAW_DIR, force_reparse=True)
            
        elif args.command == "train":
            train.train_model(epochs=args.epochs, batch_size=args.batch_size, force_reparse=args.reparse)
            
        elif args.command == "generate":
            saved_path, _ = generate.run_generation(
                output_filename=args.out,
                generate_len=args.length,
                temperature=args.temp,
                bpm=args.bpm
            )
            # Plot piano roll for CLI generation
            visualize.plot_piano_roll(saved_path)
            
        elif args.command == "visualize":
            midi_path = Path(args.midi_file)
            if not midi_path.exists():
                logger.error(f"MIDI file not found: {midi_path}")
                sys.exit(1)
            visualize.plot_piano_roll(midi_path, save_path=args.out)
            
        elif args.command == "serve":
            run_webserver()
            
        elif args.command == "test":
            run_tests()
            
    except Exception as e:
        logger.error(f"Fatal error executing command '{args.command}': {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
