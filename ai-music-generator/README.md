# Antigravity // AI Music Generator using LSTM

A production-ready Deep Learning project that parses MIDI music files, preprocesses them into token sequences, trains a stacked LSTM neural network with TensorFlow/Keras to learn musical patterns, and generates novel piano melodies. It features a complete command-line interface (CLI) and a sleek glassmorphic Web Dashboard with built-in Tone.js synthesis and canvas playhead-synchronized visualizers.

---

## 📂 Project Architecture

```
ai-music-generator/
│
├── data/
│   ├── raw/                 # Raw classical MIDI dataset (.mid, .midi)
│   └── processed/           # Cached note datasets and vocabulary mappings
│
├── models/                  # Saved models (.keras) and training progress log CSVs
│
├── outputs/                 # Generated MIDI output tracks and matplotlib piano-roll plots
│
├── src/
│   ├── __init__.py
│   ├── config.py            # Global hyperparameters, path constants, and settings
│   ├── utils.py             # Logging, GPU diagnostics, and offline dataset populator
│   ├── parser.py            # MIDI file parser using music21 (extracts notes, chords, rests)
│   ├── preprocess.py        # Sequence mapping, scaling, and one-hot encoding
│   ├── model.py             # Keras stacked LSTM network architecture
│   ├── train.py             # Training orchestrator, callback setups, and loss plotting
│   ├── generate.py          # Seed-based inference with temperature probability sampling
│   └── visualize.py         # Matplotlib piano-roll and training history graphs
│
├── app/
│   ├── static/
│   │   ├── css/
│   │   │   └── style.css    # Modern Dark glassmorphism dashboard layout
│   │   └── js/
│   │       └── app.js       # Tone.js audio synth, canvas renderer, and live AJAX pollers
│   ├── templates/
│   │   └── index.html       # Web Dashboard HTML structure
│   └── server.py            # Flask API backend supporting background thread training
│
├── tests/
│   └── test_components.py   # Automated unit tests for all modules
│
├── requirements.txt         # Package dependencies
├── README.md                # Technical explanation and guide
├── main.py                  # CLI orchestrator
└── run_app.bat              # Double-click launcher for Windows
```

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 to 3.11 (Note: TensorFlow 2.15 is fully compatible with these versions).
- Standard audio drivers (for browser-based playback).

### Quickstart (Windows)
Double-click `run_app.bat` in the root folder. It will:
1. Automatically verify and install python requirements.
2. Launch the Flask server at `http://127.0.0.1:5000`.

### Manual Installation
1. Clone or open the project folder in your terminal.
2. Create and activate a Python virtual environment:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## 💻 Running the Application

The project uses `main.py` to route all commands. 

### 1. Web Dashboard (Recommended)
Launch the interactive dashboard:
```bash
python main.py serve
```
Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser. You can trigger training, monitor epochs in real-time, generate new music, visualize piano rolls, and synthesize audio with different instruments using Tone.js!

### 2. Command Line Interface (CLI)

* **Run Unit Tests**:
  Verify the environment is functioning properly:
  ```bash
  python main.py test
  ```
* **Parse MIDI Data**:
  Pre-extracts classical MIDI files from the `music21` corpus and saves the notes cache:
  ```bash
  python main.py parse
  ```
* **Train Model**:
  Train the LSTM model for a custom number of epochs:
  ```bash
  python main.py train --epochs 50 --batch-size 64
  ```
* **Generate Music**:
  Generate a new melody from the trained model:
  ```bash
  python main.py generate --temp 0.8 --length 200 --bpm 120 --out outputs/my_melody.mid
  ```
* **Visualize MIDI**:
  Manually generate a static piano-roll plot for any MIDI file:
  ```bash
  python main.py visualize outputs/generated_music_01.mid --out outputs/piano_roll.png
  ```

---

## 🧠 Technical Deep-Dive

### 1. How LSTM Learns Music Patterns
Standard Recurrent Neural Networks (RNNs) struggle with long sequence dependencies due to **vanishing and exploding gradients**. This happens because gradients must be backpropagated through time, multiplying numbers continuously across steps.

LSTMs (Long Short-Term Memory networks) solve this with a **Cell State** ($C_t$) regulated by three gates:
- **Forget Gate** ($f_t$): Decides what information to discard from the cell state using a sigmoid function:
  $$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$
- **Input Gate** ($i_t$): Decides which new values will be updated and creates a candidate vector $\tilde{C}_t$:
  $$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$
  $$\tilde{C}_t = \tanh(W_c \cdot [h_{t-1}, x_t] + b_c)$$
- **Output Gate** ($o_t$): Decides the next hidden state output $h_t$ based on the filtered cell state:
  $$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$
  $$h_t = o_t \cdot \tanh(C_t)$$

In music generation, the network processes a history of notes (e.g., a window of 64 tokens) and learns temporal structures (chords, scale runs, resolutions) by adjusting weights to minimize predictions errors on the next notes.

### 2. How MIDI Works
**MIDI** (Musical Instrument Digital Interface) is not a digital audio format (like `.mp3` or `.wav`). Instead, it is a lightweight control protocol. A MIDI file is a series of timed event commands:
- **Note On**: Initiates a note with a specific MIDI Number (0–127, where middle C is 60) and Velocity (volume, 0–127).
- **Note Off**: Terminates a note.
- **Tempo Changes / Metronome Marks**: Configures the rate of beats (BPM).
- **Control Change (CC)**: Adjusts parameters like pitch bends or expression pedals.

`music21` parses this protocol into object hierarchies. We translate notes and rests to string tokens:
- Single note: pitch name + octave (e.g. `'E4'`).
- Chord: dot-separated notes sorted ascendingly (e.g. `'C4.E4.G4'`).
- Rest: `'REST'`.

### 3. How Sequence Prediction Works
Music generation follows an autoregressive pipeline:
1. **Window Feed**: A seed sequence of length $L$ is passed to the network.
2. **Probability Output**: The network outputs a softmax probability vector $\hat{y}$ of size $V$ (vocabulary size), indicating the likelihood of each note/chord being next.
3. **Temperature Sampling**: We adjust probabilities before sampling. The adjusted probability $p_i$ for index $i$ is:
   $$p_i = \frac{e^{z_i / T}}{\sum_{j} e^{z_j / T}}$$
   Where $z_i$ is the raw model logit and $T$ is the temperature.
   - $T < 1.0$: Sharpens distribution; makes predictions highly predictable.
   - $T > 1.0$: Flattens distribution; allows lower probability, "creative" choices.
4. **Shifting Window**: The selected note is appended to the sequence, the oldest note is discarded, and the new sequence is fed back to the network.



## 🛠️ Performance Optimizations & Modern Best Practices

- **Memory-Optimized Inputs**: Instead of one-hot encoding the entire training input matrix (which creates a massive array of shape `(Patterns, SeqLen, VocabSize)` and exhausts RAM), we normalize indices as float values of shape `(Patterns, SeqLen, 1)` relative to vocabulary size.
- **GPU Acceleration**: Built-in compatibility with TensorFlow GPU runtimes, allocating resources dynamically through CUDA.
- **Async Web Worker Training**: Training runs in a background thread to prevent blocking Flask's event loop, reporting statistics via a custom callback directly back to the UI.
- **Web Audio Synthesis**: Browser audio synthesis uses the client-side Web Audio API (via Tone.js) and maps MIDI notes to synthesizer frequencies on the fly, keeping the backend lightweight and fast.
- **Clean Responsive Rendering**: The piano-roll is dynamically scaled to the client window sizes, utilizing `requestAnimationFrame` for stutter-free playhead animations.
user can add their own downloaded vesion to create a new one.
