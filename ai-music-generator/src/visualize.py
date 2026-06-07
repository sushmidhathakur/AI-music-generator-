import os
import pandas as pd
import matplotlib
# Use Agg backend for running in background without desktop environment dependencies
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from music21 import converter, note, chord
from src.utils import logger
from src import config

def plot_piano_roll(midi_path, save_path=None, title="Generated Music Piano-Roll"):
    """
    Parses a MIDI file and renders a clean, professional piano-roll visualization.
    
    Args:
        midi_path (str/Path): Path to the MIDI file.
        save_path (str/Path, optional): Path to save the plot image. Defaults to next to midi_path.
        title (str): Title of the plot.
    """
    logger.info(f"Generating piano-roll visualization for {midi_path}...")
    try:
        midi = converter.parse(str(midi_path))
    except Exception as e:
        logger.error(f"Failed to parse MIDI file for piano roll: {e}")
        return None

    # Extract notes/chords and their durations
    notes_data = []
    
    # Flatten the stream to process all parts at their absolute offset
    flat_midi = midi.flat
    
    for element in flat_midi:
        if isinstance(element, note.Note):
            notes_data.append({
                'pitch_name': element.pitch.nameWithOctave,
                'pitch_midi': element.pitch.ps,
                'offset': float(element.offset),
                'duration': float(element.quarterLength)
            })
        elif isinstance(element, chord.Chord):
            for n in element.notes:
                notes_data.append({
                    'pitch_name': n.pitch.nameWithOctave,
                    'pitch_midi': n.pitch.ps,
                    'offset': float(element.offset),
                    'duration': float(element.quarterLength)
                })

    if not notes_data:
        logger.warning(f"No notes or chords found in MIDI file {midi_path} to plot.")
        return None

    df = pd.DataFrame(notes_data)
    
    # Determine save path if not specified
    midi_path = Path(midi_path)
    if save_path is None:
        save_path = midi_path.with_suffix('.png')
    else:
        save_path = Path(save_path)

    # Plot
    fig, ax = plt.subplots(figsize=(14, 6), facecolor='#0f172a') # Slate-900 background
    ax.set_facecolor('#1e293b') # Slate-800 grid background

    # Color palette based on pitches
    min_midi = df['pitch_midi'].min() - 2
    max_midi = df['pitch_midi'].max() + 2
    
    # Render note rectangles
    for _, row in df.iterrows():
        # Map pitch to color gradient
        normalized_pitch = (row['pitch_midi'] - min_midi) / max(1, (max_midi - min_midi))
        color = plt.cm.plasma(normalized_pitch)
        
        rect = patches.Rectangle(
            (row['offset'], row['pitch_midi'] - 0.4), # bottom-left corner
            row['duration'],                          # width
            0.8,                                      # height
            edgecolor='#0f172a',
            facecolor=color,
            alpha=0.9,
            lw=1
        )
        ax.add_patch(rect)

    # Custom styling
    ax.set_xlim(df['offset'].min() - 1, df['offset'].max() + df['duration'].max() + 1)
    ax.set_ylim(min_midi, max_midi)
    
    # Show MIDI pitches and label them as scientific pitch notation (e.g. C4)
    y_ticks = range(int(min_midi), int(max_midi) + 1)
    # Filter ticks to avoid clutter if range is wide
    if len(y_ticks) > 15:
        y_ticks = [y for y in y_ticks if y % 2 == 0]
        
    ax.set_yticks(y_ticks)
    # Convert midi numbers back to note names for labels
    note_labels = []
    for y in y_ticks:
        try:
            n = note.Note(y)
            note_labels.append(n.pitch.nameWithOctave)
        except Exception:
            note_labels.append(str(y))
    ax.set_yticklabels(note_labels, color='#94a3b8', fontsize=9)
    
    ax.set_title(title, fontsize=16, color='#f8fafc', fontweight='bold', pad=15)
    ax.set_xlabel('Time (Beats)', fontsize=12, color='#94a3b8', labelpad=10)
    ax.set_ylabel('Pitch', fontsize=12, color='#94a3b8', labelpad=10)
    
    ax.tick_params(colors='#64748b')
    ax.grid(color='#334155', linestyle=':', linewidth=0.5)
    
    # Remove top/right spines
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    ax.spines['left'].set_color('#334155')
    ax.spines['bottom'].set_color('#334155')
    
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=300)
    plt.close()
    logger.info(f"Saved piano-roll visualization to {save_path}")
    return save_path

def plot_training_csv(csv_path, save_path=None):
    """
    Plots training loss and accuracy from a CSV log file.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        logger.warning(f"Training log CSV not found at {csv_path}")
        return None
        
    df = pd.read_csv(csv_path)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5), facecolor='#0f172a')
    
    # Style helper
    def style_axis(ax, title, ylabel):
        ax.set_facecolor('#1e293b')
        ax.set_title(title, fontsize=14, color='#f8fafc', fontweight='bold', pad=12)
        ax.set_xlabel('Epoch', fontsize=11, color='#94a3b8')
        ax.set_ylabel(ylabel, fontsize=11, color='#94a3b8')
        ax.tick_params(colors='#64748b')
        ax.grid(color='#334155', linestyle=':', linewidth=0.5)
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        ax.spines['left'].set_color('#334155')
        ax.spines['bottom'].set_color('#334155')

    # Loss curves
    style_axis(ax1, "Training and Validation Loss", "Loss")
    if 'loss' in df.columns:
        ax1.plot(df['epoch'], df['loss'], label='Train Loss', color='#6366f1', lw=2)
    if 'val_loss' in df.columns:
        ax1.plot(df['epoch'], df['val_loss'], label='Val Loss', color='#ec4899', lw=2)
    ax1.legend(facecolor='#1e293b', edgecolor='#334155', labelcolor='#e2e8f0')

    # Accuracy curves
    style_axis(ax2, "Training and Validation Accuracy", "Accuracy")
    if 'accuracy' in df.columns:
        ax2.plot(df['epoch'], df['accuracy'], label='Train Acc', color='#10b981', lw=2)
    if 'val_accuracy' in df.columns:
        ax2.plot(df['epoch'], df['val_accuracy'], label='Val Acc', color='#f59e0b', lw=2)
    ax2.legend(facecolor='#1e293b', edgecolor='#334155', labelcolor='#e2e8f0')
    
    plt.tight_layout()
    
    if save_path is None:
        save_path = csv_path.parent / "training_history.png"
    else:
        save_path = Path(save_path)
        
    plt.savefig(str(save_path), dpi=300)
    plt.close()
    logger.info(f"Saved CSV training history plot to {save_path}")
    return save_path
