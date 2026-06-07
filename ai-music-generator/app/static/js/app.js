// Global state
let modelTrained = false;
let isTraining = false;
let generatedNotes = [];
let noteToInt = {};
let int_to_note = {};
let pitchnames = [];

// Audio Synthesis State
let activeSynth = null;
let playbackEventIds = [];
let stepDuration = 0.25; // Duration of each step (0.5 beats = 8th note)
let isPlaying = false;
let playheadFrameId = null;
let totalPlaybackDuration = 0;

// UI Elements
const generateBtn = document.getElementById('generate-btn');
const trainBtn = document.getElementById('train-btn');
const playBtn = document.getElementById('play-btn');
const stopBtn = document.getElementById('stop-btn');
const playIcon = document.getElementById('play-icon');
const downloadMidiBtn = document.getElementById('download-midi-btn');
const playbackStatus = document.getElementById('playback-status');
const playbackProgress = document.getElementById('playback-progress');
const consoleOutput = document.getElementById('console-output');

// Canvas Elements
const canvas = document.getElementById('piano-roll-canvas');
const ctx = canvas.getContext('2d');
const placeholder = document.getElementById('canvas-placeholder');

// Dynamic Slider Displays
setupSlider('temp-slider', 'temp-val');
setupSlider('length-slider', 'length-val');
setupSlider('bpm-slider', 'bpm-val');

function setupSlider(sliderId, displayId) {
    const slider = document.getElementById(sliderId);
    const display = document.getElementById(displayId);
    slider.addEventListener('input', () => {
        display.textContent = slider.value;
    });
}

// Log utility
function appendLog(message, type = 'info') {
    const time = new Date().toLocaleTimeString();
    const line = document.createElement('div');
    line.className = `log-line log-${type}`;
    line.innerText = `[${time}] ${message}`;
    consoleOutput.appendChild(line);
    consoleOutput.scrollTop = consoleOutput.scrollHeight;
}

// Drag and Drop implementation
const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('file-input');
const filesList = document.getElementById('uploaded-files-list');

if (dropzone && fileInput) {
    dropzone.addEventListener('click', () => fileInput.click());
    
    dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropzone.classList.add('dragover');
    });
    
    dropzone.addEventListener('dragleave', () => {
        dropzone.classList.remove('dragover');
    });
    
    dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.classList.remove('dragover');
        const files = e.dataTransfer.files;
        handleFilesUpload(files);
    });
    
    fileInput.addEventListener('change', () => {
        const files = fileInput.files;
        handleFilesUpload(files);
    });
}

async function handleFilesUpload(files) {
    if (!files || files.length === 0) return;
    
    for (let i = 0; i < files.length; i++) {
        const file = files[i];
        if (!file.name.endsWith('.mid') && !file.name.endsWith('.midi')) {
            appendLog(`File ${file.name} is not a valid MIDI file.`, "error");
            continue;
        }
        
        // Add to UI list as uploading
        const item = document.createElement('div');
        item.className = 'uploaded-file-item';
        item.innerHTML = `
            <div class="uploaded-file-name">
                <i class="fa-solid fa-spinner fa-spin"></i>
                <span>${file.name}</span>
            </div>
            <span class="uploaded-file-status" style="color: var(--text-secondary);">Uploading...</span>
        `;
        filesList.appendChild(item);
        filesList.scrollTop = filesList.scrollHeight;
        
        const formData = new FormData();
        formData.append('file', file);
        
        try {
            const response = await fetch('/api/upload', {
                method: 'POST',
                body: formData
            });
            const data = await response.json();
            
            if (data.success) {
                item.innerHTML = `
                    <div class="uploaded-file-name">
                        <i class="fa-solid fa-file-audio"></i>
                        <span>${file.name}</span>
                    </div>
                    <span class="uploaded-file-status"><i class="fa-solid fa-check"></i> Ready</span>
                `;
                appendLog(`Successfully uploaded ${file.name} to training dataset.`, "info");
                // Update file count immediately
                document.getElementById('stat-files').textContent = data.raw_midi_files;
            } else {
                item.innerHTML = `
                    <div class="uploaded-file-name">
                        <i class="fa-solid fa-circle-exclamation" style="color: var(--danger);"></i>
                        <span>${file.name}</span>
                    </div>
                    <span class="uploaded-file-status" style="color: var(--danger);">Failed</span>
                `;
                appendLog(`Upload failed for ${file.name}: ${data.message}`, "error");
            }
        } catch (error) {
            item.innerHTML = `
                <div class="uploaded-file-name">
                    <i class="fa-solid fa-circle-exclamation" style="color: var(--danger);"></i>
                    <span>${file.name}</span>
                </div>
                <span class="uploaded-file-status" style="color: var(--danger);">Error</span>
            `;
            appendLog(`Network error uploading ${file.name}: ${error.message}`, "error");
        }
    }
}

// Scientific Pitch Notation to MIDI Number converter
function noteToMidi(noteName) {
    if (!noteName || noteName === 'REST') return null;
    const scale = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
    const match = noteName.match(/^([A-G]#?)(-?\d+)$/);
    if (!match) return null;
    const pitch = match[1];
    const octave = parseInt(match[2]);
    return 12 * (octave + 1) + scale.indexOf(pitch);
}

// Convert MIDI Number back to Note Name (e.g. 60 -> C4)
function midiToNoteName(midiNumber) {
    const scale = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
    const octave = Math.floor(midiNumber / 12) - 1;
    const noteIndex = midiNumber % 12;
    return scale[noteIndex] + octave;
}

// Poll server status on page load
async function checkServerStatus() {
    try {
        const response = await fetch('/api/status');
        const data = await response.json();
        
        modelTrained = data.model_trained;
        document.getElementById('stat-files').textContent = data.raw_midi_files;
        document.getElementById('stat-sequences').textContent = data.sequences_generated > 0 ? data.sequences_generated : '-';
        
        // Update GPU status display
        const statGpu = document.getElementById('stat-gpu');
        const statGpuSub = document.getElementById('stat-gpu-sub');
        if (data.gpu_enabled) {
            statGpu.innerHTML = '<i class="fa-solid fa-microchip"></i> GPU';
            statGpu.className = 'stat-value badge-gpu-active';
            statGpuSub.textContent = 'CUDA Accelerated';
            statGpuSub.style.color = '#10b981';
        } else {
            statGpu.innerHTML = '<i class="fa-solid fa-cpu"></i> CPU';
            statGpu.className = 'stat-value';
            statGpuSub.textContent = 'Fallback Mode';
            statGpuSub.style.color = 'var(--text-muted)';
        }
        
        // Update model training state in UI
        if (data.training_state.status === 'training') {
            if (!isTraining) {
                isTraining = true;
                appendLog("Live training pipeline running in background...", "info");
            }
            trainBtn.disabled = true;
            trainBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Model Fitting...';
            showTrainingProgress(data.training_state);
        } else {
            if (isTraining) {
                isTraining = false;
                if (data.training_state.status === 'completed') {
                    appendLog("Live training run finished successfully! Checkpoint saved.", "info");
                } else if (data.training_state.status === 'failed') {
                    appendLog(`Training failed: ${data.training_state.error_message}`, "error");
                }
            }
            
            trainBtn.disabled = false;
            trainBtn.innerHTML = '<i class="fa-solid fa-dumbbell"></i> Start Live Training';
            document.getElementById('training-progress-panel').style.display = 'none';
            
            if (modelTrained) {
                generateBtn.classList.remove('btn-disabled');
                
                // Show convergence curve chart
                const chartPanel = document.getElementById('history-chart-panel');
                const chartImg = document.getElementById('training-chart-img');
                if (chartPanel && chartImg) {
                    chartPanel.style.display = 'block';
                    // Force refresh to avoid caching
                    if (!chartImg.src.includes('/api/training_history_img')) {
                        chartImg.src = '/api/training_history_img?t=' + new Date().getTime();
                    }
                }
            } else {
                generateBtn.classList.add('btn-disabled');
            }
        }

    } catch (error) {
        console.error("Error checking server status:", error);
    }
}

// Periodic polling for status when training
let statusInterval = setInterval(checkServerStatus, 3000);

function showTrainingProgress(state) {
    const progressPanel = document.getElementById('training-progress-panel');
    const progressBar = document.getElementById('training-progress-bar');
    const stepText = document.getElementById('training-step-text');
    const pctText = document.getElementById('training-pct-text');
    
    progressPanel.style.display = 'block';
    stepText.textContent = `Epoch ${state.current_epoch}/${state.total_epochs}`;
    
    const pct = Math.round((state.current_epoch / state.total_epochs) * 100);
    pctText.textContent = `${pct}%`;
    progressBar.style.width = `${pct}%`;
    
    document.getElementById('train-metric-loss').textContent = state.loss ? state.loss.toFixed(4) : '-';
    document.getElementById('train-metric-acc').textContent = state.accuracy ? (state.accuracy * 100).toFixed(1) + '%' : '-';
    document.getElementById('train-metric-val-loss').textContent = state.val_loss ? state.val_loss.toFixed(4) : '-';
}

// Trigger background model training
async function startTraining() {
    const epochs = parseInt(document.getElementById('epochs-input').value) || 5;
    const batchSize = parseInt(document.getElementById('batch-input').value) || 64;
    const forceReparse = document.getElementById('reparse-checkbox').checked;
    
    appendLog(`Initializing training pipeline on backend: ${epochs} epochs...`, "info");
    
    try {
        const response = await fetch('/api/train', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ epochs, batch_size: batchSize, force_reparse: forceReparse })
        });
        const data = await response.json();
        
        if (data.success) {
            appendLog(data.message, "info");
            checkServerStatus();
        } else {
            appendLog(`Training failed to start: ${data.message}`, "error");
        }
    } catch (error) {
        appendLog(`Error starting training: ${error.message}`, "error");
    }
}

// Generate new music sequence
async function generateMusic() {
    if (!modelTrained) {
        appendLog("Model must be trained before generating music.", "warn");
        return;
    }
    
    const temp = parseFloat(document.getElementById('temp-slider').value);
    const length = parseInt(document.getElementById('length-slider').value);
    const bpm = parseInt(document.getElementById('bpm-slider').value);
    
    generateBtn.disabled = true;
    generateBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Generating Notes...';
    appendLog(`Starting generation sequence (Temp=${temp}, Length=${length}, Tempo=${bpm})...`, "info");
    
    try {
        const response = await fetch('/api/generate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ temperature: temp, length: length, bpm: bpm })
        });
        const data = await response.json();
        
        generateBtn.disabled = false;
        generateBtn.innerHTML = '<i class="fa-solid fa-wand-magic-sparkles"></i> Generate New MIDI Melody';
        
        if (data.success) {
            generatedNotes = data.notes;
            appendLog(`Successfully generated sequence of ${generatedNotes.length} notes!`, "info");
            
            // Enable download link
            downloadMidiBtn.href = data.midi_file_url;
            downloadMidiBtn.classList.remove('btn-disabled');
            
            // Enable playback controls
            playBtn.classList.remove('btn-disabled');
            stopBtn.classList.remove('btn-disabled');
            playBtn.disabled = false;
            stopBtn.disabled = false;
            
            // Calculate step duration based on BPM (0.5 beats per step)
            stepDuration = 30.0 / bpm; 
            
            // Plot piano roll on canvas
            renderPianoRollOnCanvas();
            placeholder.style.display = 'none';
        } else {
            appendLog(`Generation failed: ${data.message}`, "error");
        }
    } catch (error) {
        generateBtn.disabled = false;
        generateBtn.innerHTML = '<i class="fa-solid fa-wand-magic-sparkles"></i> Generate New MIDI Melody';
        appendLog(`Network error during generation: ${error.message}`, "error");
    }
}

// RENDER PIANO-ROLL ON CANVAS
let notesToDraw = [];
let minPitch = 0;
let maxPitch = 127;
let canvasPadding = { left: 60, right: 20, top: 20, bottom: 40 };

function renderPianoRollOnCanvas() {
    if (!generatedNotes || generatedNotes.length === 0) return;
    
    // Resize canvas to parent bounds
    canvas.width = canvas.offsetWidth;
    canvas.height = canvas.offsetHeight;
    
    notesToDraw = [];
    let allPitches = [];
    
    // Parse notes to MIDI pitch numbers and starting times
    for (let i = 0; i < generatedNotes.length; i++) {
        const token = generatedNotes[i];
        const startTime = i * stepDuration;
        
        if (token === 'REST') continue;
        
        if (token.includes('.')) {
            // Chord
            const chordNotes = token.split('.');
            chordNotes.forEach(n => {
                const pitch = noteToMidi(n);
                if (pitch !== null) {
                    notesToDraw.push({ pitch, startTime, duration: stepDuration, name: n });
                    allPitches.push(pitch);
                }
            });
        } else {
            // Single Note
            const pitch = noteToMidi(token);
            if (pitch !== null) {
                notesToDraw.push({ pitch, startTime, duration: stepDuration, name: token });
                allPitches.push(pitch);
            }
        }
    }
    
    if (allPitches.length === 0) return;
    
    minPitch = Math.min(...allPitches) - 2;
    maxPitch = Math.max(...allPitches) + 2;
    totalPlaybackDuration = generatedNotes.length * stepDuration;
    
    drawPianoRollFrame(0);
}

function drawPianoRollFrame(currentTime = 0) {
    const W = canvas.width;
    const H = canvas.height;
    
    // Clear canvas
    ctx.fillStyle = '#1e293b'; // Slate-800
    ctx.fillRect(0, 0, W, H);
    
    const plotW = W - canvasPadding.left - canvasPadding.right;
    const plotH = H - canvasPadding.top - canvasPadding.bottom;
    
    // Draw Grid Lines (Pitches)
    const pitchRange = maxPitch - minPitch;
    const pitchStep = plotH / pitchRange;
    
    ctx.strokeStyle = '#334155'; // Slate-700
    ctx.lineWidth = 0.5;
    
    for (let p = minPitch; p <= maxPitch; p++) {
        const y = canvasPadding.top + plotH - (p - minPitch) * pitchStep;
        
        // Draw horizontal line
        ctx.beginPath();
        ctx.moveTo(canvasPadding.left, y);
        ctx.lineTo(W - canvasPadding.right, y);
        ctx.stroke();
        
        // Label pitches on the left (e.g. C4)
        if (p % 4 === 0 || p === minPitch || p === maxPitch) {
            ctx.fillStyle = '#64748b';
            ctx.font = '10px Inter';
            ctx.textAlign = 'right';
            ctx.textBaseline = 'middle';
            ctx.fillText(midiToNoteName(p), canvasPadding.left - 8, y);
        }
    }
    
    // Draw Grid Lines (Time)
    const timeStep = plotW / totalPlaybackDuration;
    const beatInterval = 2.0; // Draw a vertical grid line every 2 seconds
    
    for (let t = 0; t <= totalPlaybackDuration; t += beatInterval) {
        const x = canvasPadding.left + t * timeStep;
        
        ctx.beginPath();
        ctx.moveTo(x, canvasPadding.top);
        ctx.lineTo(x, H - canvasPadding.bottom);
        ctx.stroke();
        
        // Draw axis text
        ctx.fillStyle = '#64748b';
        ctx.font = '10px Inter';
        ctx.textAlign = 'center';
        ctx.fillText(t.toFixed(0) + 's', x, H - canvasPadding.bottom + 15);
    }
    
    // Render Note Blocks
    notesToDraw.forEach(note => {
        const x = canvasPadding.left + note.startTime * timeStep;
        const w = note.duration * timeStep - 1; // Subtract 1px for separation
        
        const y = canvasPadding.top + plotH - (note.pitch - minPitch) * pitchStep - (pitchStep * 0.4);
        const h = pitchStep * 0.8;
        
        // Calculate beautiful linear gradient for each block based on pitch height
        const pct = (note.pitch - minPitch) / pitchRange;
        const grad = ctx.createLinearGradient(x, y, x + w, y);
        
        // Plasma colors: Purple to Pink to Orange
        const r = Math.floor(99 + pct * 140);
        const g = Math.floor(102 - pct * 50);
        const b = Math.floor(241 + pct * 14);
        
        grad.addColorStop(0, `rgb(${r}, ${g}, ${b})`);
        grad.addColorStop(1, `rgba(${r+30}, ${g+50}, ${b}, 0.85)`);
        
        ctx.fillStyle = grad;
        drawRoundedRect(ctx, x, y, w, h, 3);
        ctx.fill();
    });
    
    // Draw Playhead
    if (currentTime > 0) {
        const playheadX = canvasPadding.left + currentTime * timeStep;
        if (playheadX <= W - canvasPadding.right) {
            ctx.strokeStyle = '#06b6d4'; // Cyan-500
            ctx.lineWidth = 2;
            ctx.shadowBlur = 10;
            ctx.shadowColor = 'rgba(6, 182, 212, 0.8)';
            
            ctx.beginPath();
            ctx.moveTo(playheadX, canvasPadding.top);
            ctx.lineTo(playheadX, H - canvasPadding.bottom);
            ctx.stroke();
            
            ctx.shadowBlur = 0;
        }
    }
}

function drawRoundedRect(ctx, x, y, width, height, radius) {
    if (width < 2 * radius) radius = width / 2;
    if (height < 2 * radius) radius = height / 2;
    ctx.beginPath();
    ctx.moveTo(x + radius, y);
    ctx.arcTo(x + width, y, x + width, y + height, radius);
    ctx.arcTo(x + width, y + height, x, y + height, radius);
    ctx.arcTo(x, y + height, x, y, radius);
    ctx.arcTo(x, y, x + width, y, radius);
    ctx.closePath();
}

// SETUP SOUND SYNTHESIZER AND PLAYBACK
function createSynth(type) {
    if (activeSynth) {
        activeSynth.dispose();
    }
    
    if (type === 'piano') {
        activeSynth = new Tone.PolySynth(Tone.Synth, {
            oscillator: { type: 'triangle' },
            envelope: {
                attack: 0.01,
                decay: 0.6,
                sustain: 0.2,
                release: 1.2
            }
        }).toDestination();
        activeSynth.volume.value = -6;
    } else if (type === 'synth-pluck') {
        activeSynth = new Tone.PolySynth(Tone.Synth, {
            oscillator: { type: 'sine' },
            envelope: {
                attack: 0.002,
                decay: 0.15,
                sustain: 0.0,
                release: 0.2
            }
        }).toDestination();
        activeSynth.volume.value = -3;
    } else if (type === 'fm-chimes') {
        activeSynth = new Tone.PolySynth(Tone.FMSynth, {
            harmonicity: 3,
            modulationIndex: 10,
            oscillator: { type: 'sine' },
            envelope: {
                attack: 0.01,
                decay: 0.4,
                sustain: 0.0,
                release: 0.8
            },
            modulation: { type: 'square' }
        }).toDestination();
        activeSynth.volume.value = -12;
    }
}

// Schedule events in Tone.js Transport
function scheduleNotes() {
    Tone.Transport.cancel();
    
    const synthType = document.getElementById('synth-select').value;
    createSynth(synthType);
    
    playbackEventIds = [];
    
    for (let i = 0; i < generatedNotes.length; i++) {
        const token = generatedNotes[i];
        const startTime = i * stepDuration;
        
        if (token === 'REST') continue;
        
        if (token.includes('.')) {
            const notes = token.split('.');
            const eventId = Tone.Transport.schedule((time) => {
                activeSynth.triggerAttackRelease(notes, "8n", time);
                Tone.Draw.schedule(() => {
                    playbackStatus.textContent = `Playing Chord: ${notes.join(' + ')}`;
                }, time);
            }, startTime);
            playbackEventIds.push(eventId);
        } else {
            const eventId = Tone.Transport.schedule((time) => {
                activeSynth.triggerAttackRelease(token, "8n", time);
                Tone.Draw.schedule(() => {
                    playbackStatus.textContent = `Playing Note: ${token}`;
                }, time);
            }, startTime);
            playbackEventIds.push(eventId);
        }
    }
    
    const finalEventId = Tone.Transport.schedule((time) => {
        Tone.Draw.schedule(() => {
            stopPlayback();
            playbackStatus.textContent = "Finished Playing";
        }, time);
    }, totalPlaybackDuration);
    playbackEventIds.push(finalEventId);
}

// Playback Animation Loop
function animatePlayhead() {
    if (!isPlaying) return;
    
    const sec = Tone.Transport.seconds;
    const pct = Math.min(100, (sec / totalPlaybackDuration) * 100);
    
    playbackProgress.style.width = `${pct}%`;
    drawPianoRollFrame(sec);
    
    playheadFrameId = requestAnimationFrame(animatePlayhead);
}

async function togglePlayback() {
    await Tone.start();
    
    if (isPlaying) {
        Tone.Transport.pause();
        isPlaying = false;
        playIcon.className = 'fa-solid fa-play';
        playbackStatus.textContent = "Paused";
        if (playheadFrameId) {
            cancelAnimationFrame(playheadFrameId);
        }
    } else {
        if (Tone.Transport.state === 'stopped') {
            scheduleNotes();
        }
        
        Tone.Transport.start();
        isPlaying = true;
        playIcon.className = 'fa-solid fa-pause';
        playbackStatus.textContent = "Playing...";
        animatePlayhead();
    }
}

function stopPlayback() {
    Tone.Transport.stop();
    isPlaying = false;
    playIcon.className = 'fa-solid fa-play';
    playbackStatus.textContent = "Stopped";
    playbackProgress.style.width = "0%";
    
    if (playheadFrameId) {
        cancelAnimationFrame(playheadFrameId);
        playheadFrameId = null;
    }
    
    drawPianoRollFrame(0);
}

// Resize listener
window.addEventListener('resize', () => {
    if (generatedNotes.length > 0) {
        renderPianoRollOnCanvas();
    }
});

// Run check on load
checkServerStatus();
// Poll every 3 seconds
setInterval(checkServerStatus, 3000);
appendLog("Web UI loaded successfully.", "system");
