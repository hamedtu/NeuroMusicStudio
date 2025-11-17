"""
EEG Motor Imagery Music Composer - Clean Transition Version
=========================================================
This version implements a clear separation between the building phase (layering sounds) and the DJ phase (effect control),
with seamless playback of all layered sounds throughout both phases.
"""

# Set matplotlib backend to non-GUI for server/web use
import matplotlib
matplotlib.use('Agg')  # Set backend BEFORE importing pyplot
import matplotlib.pyplot as plt
import os
import gradio as gr
import numpy as np
from typing import Dict
from sound_manager import SoundManager
from data_processor import EEGDataProcessor
from classifier import MotorImageryClassifier
from config import DEMO_DATA_PATHS, CONFIDENCE_THRESHOLD

# --- Initialization ---
app_state = {
    'is_running': False,
    'demo_data': None,
    'demo_labels': None,
    'composition_active': False,
    'auto_mode': False
}

sound_manager = SoundManager()
data_processor = EEGDataProcessor()
classifier = MotorImageryClassifier()

# Load demo data
existing_files = [f for f in DEMO_DATA_PATHS if os.path.exists(f)]
if existing_files:
    app_state['demo_data'], app_state['demo_labels'], app_state['ch_names'] = data_processor.process_files(existing_files)
else:
    app_state['demo_data'], app_state['demo_labels'], app_state['ch_names'] = None, None, None

if app_state['demo_data'] is not None:
    classifier.load_model(n_chans=app_state['demo_data'].shape[1], n_times=app_state['demo_data'].shape[2])

# --- Helper Functions ---
def get_movement_sounds() -> Dict[str, str]:
    """Get the current sound files for each movement."""
    sounds = {}
    # Add a static cache for audio file paths per movement and effect state
    if not hasattr(get_movement_sounds, 'audio_cache'):
        get_movement_sounds.audio_cache = {m: {False: None, True: None} for m in ['left_hand', 'right_hand', 'left_leg', 'right_leg']}
        get_movement_sounds.last_effect_state = {m: None for m in ['left_hand', 'right_hand', 'left_leg', 'right_leg']}
    # Add a static counter to track how many times each movement's audio is played
    if not hasattr(get_movement_sounds, 'play_counter'):
        get_movement_sounds.play_counter = {m: 0 for m in ['left_hand', 'right_hand', 'left_leg', 'right_leg']}
        get_movement_sounds.total_calls = 0
    from sound_manager import AudioEffectsProcessor
    import tempfile
    import soundfile as sf
    # If in DJ mode, use effect-processed file if effect is ON
    dj_mode = getattr(sound_manager, 'current_phase', None) == 'dj_effects'
    for movement, sound_file in sound_manager.current_sound_mapping.items():
        if movement in ['left_hand', 'right_hand', 'left_leg', 'right_leg']:
            if sound_file is not None:
                sound_path = sound_manager.sound_dir / sound_file
                if sound_path.exists():
                    # Sticky effect for all movements: if effect was ON, keep returning processed audio until next ON
                    effect_on = dj_mode and sound_manager.active_effects.get(movement, False)
                    # If effect just turned ON, update sticky state
                    if effect_on:
                        get_movement_sounds.last_effect_state[movement] = True
                    # If effect is OFF, but sticky is set, keep using processed audio
                    elif get_movement_sounds.last_effect_state[movement]:
                        effect_on = True
                    else:
                        effect_on = False
                    # Check cache for this movement/effect state
                    cached_path = get_movement_sounds.audio_cache[movement][effect_on]
                    # Only regenerate if cache is empty or effect state just changed
                    if cached_path is not None and get_movement_sounds.last_effect_state[movement] == effect_on:
                        sounds[movement] = cached_path
                    else:
                        # Load audio
                        data, sr = sf.read(str(sound_path))
                        if len(data.shape) > 1:
                            data = np.mean(data, axis=1)
                        # Fade-in: apply to all audio on restart (0.5s fade for more gradual effect)
                        fade_duration = 10  # seconds
                        fade_samples = int(fade_duration * sr)
                        if fade_samples > 0 and fade_samples < len(data):
                            fade_curve = np.linspace(0, 1, fade_samples)
                            data[:fade_samples] = data[:fade_samples] * fade_curve
                        if effect_on:
                            # Apply effect
                            processed = AudioEffectsProcessor.process_layer_with_effects(
                                data, sr, movement, sound_manager.active_effects
                            )
                            # Save to temp file (persistent for this effect state)
                            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=f'_{movement}_effect.wav')
                            sf.write(tmp.name, processed, sr)
                            get_movement_sounds.audio_cache[movement][True] = tmp.name
                            sounds[movement] = tmp.name
                        else:
                            get_movement_sounds.audio_cache[movement][False] = str(sound_path.resolve())
                            sounds[movement] = str(sound_path.resolve())
                        get_movement_sounds.last_effect_state[movement] = effect_on
                    get_movement_sounds.play_counter[movement] += 1

    get_movement_sounds.total_calls += 1
    return sounds


def create_eeg_plot(eeg_data: np.ndarray, target_movement: str, predicted_name: str, confidence: float, sound_added: bool, ch_names=None) -> plt.Figure:
    '''Create a plot of EEG data with annotations. Plots C3 and C4 channels by name.'''
    if ch_names is None:
        ch_names = ['C3', 'C4']
    # Find indices for C3 and C4
    idx_c3 = ch_names.index('C3') if 'C3' in ch_names else 0
    idx_c4 = ch_names.index('C4') if 'C4' in ch_names else 1
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes = axes.flatten()
    time_points = np.arange(eeg_data.shape[1]) / 200
    for i, idx in enumerate([idx_c3, idx_c4]):
        color = 'green' if sound_added else 'blue'
        axes[i].plot(time_points, eeg_data[idx], color=color, linewidth=1)
        axes[i].set_title(f'{ch_names[idx] if idx < len(ch_names) else f"Channel {idx+1}"}')
        axes[i].set_xlabel('Time (s)')
        axes[i].set_ylabel('Amplitude (µV)')
        axes[i].grid(True, alpha=0.3)
    title = f"Target: {target_movement.replace('_', ' ').title()} | Predicted: {predicted_name.replace('_', ' ').title()} ({confidence:.2f})"
    fig.suptitle(title, fontsize=12, fontweight='bold')
    fig.tight_layout()
    plt.close(fig)
    return fig

def format_composition_summary(composition_info: Dict) -> str:
    '''Format the composition summary for display.
    '''
    if not composition_info.get('layers_by_cycle'):
        return "No composition layers yet"
    summary = []
    for cycle, layers in composition_info['layers_by_cycle'].items():
        summary.append(f"Cycle {cycle + 1}: {len(layers)} layers")
        for layer in layers:
            movement = layer.get('movement', 'unknown')
            confidence = layer.get('confidence', 0)
            summary.append(f"  • {movement.replace('_', ' ').title()} ({confidence:.2f})")
    # DJ Effects Status removed from status tab as requested
    return "\n".join(summary) if summary else "No composition layers"

# --- Main Logic ---
def start_composition():
    '''
    Start the composition process.
    '''
    global app_state
    if not app_state['composition_active']:
        app_state['composition_active'] = True
        sound_manager.start_new_cycle()
    if app_state['demo_data'] is None:
        return "❌ No data", "❌ No data", "❌ No data", None, None, None, None, None, None, "No EEG data available"
    # Force first trial to always be left_hand/instrumental
    if len(sound_manager.movements_completed) == 0:
        next_movement = 'left_hand'
        left_hand_label = [k for k, v in classifier.class_names.items() if v == 'left_hand'][0]
        import numpy as np
        matching_indices = np.where(app_state['demo_labels'] == left_hand_label)[0]
        chosen_idx = np.random.choice(matching_indices)
        epoch_data = app_state['demo_data'][chosen_idx]
        true_label = left_hand_label
        true_label_name = 'left_hand'
    else:
        epoch_data, true_label = data_processor.simulate_real_time_data(app_state['demo_data'], app_state['demo_labels'], mode="class_balanced")
        true_label_name = classifier.class_names[true_label]
        next_movement = sound_manager.get_current_target_movement()
    if next_movement == "cycle_complete":
        return continue_dj_phase()
    predicted_class, confidence, probabilities = classifier.predict(epoch_data)
    predicted_name = classifier.class_names[predicted_class]
    # Only add sound if confidence > threshold, predicted == true label, and true label matches the prompt
    if confidence > CONFIDENCE_THRESHOLD and predicted_name == true_label_name:
        result = sound_manager.process_classification(predicted_name, confidence, CONFIDENCE_THRESHOLD, force_add=True)
    else:
        result = {'sound_added': False}
    fig = create_eeg_plot(epoch_data, true_label_name, predicted_name, confidence, result['sound_added'], app_state.get('ch_names'))
    
    # Only play completed movement sounds (layered)
    sounds = get_movement_sounds()
    completed_movements = sound_manager.movements_completed

    # Assign audio paths only for completed movements
    left_hand_audio = sounds.get('left_hand') if 'left_hand' in completed_movements else None
    right_hand_audio = sounds.get('right_hand') if 'right_hand' in completed_movements else None
    left_leg_audio = sounds.get('left_leg') if 'left_leg' in completed_movements else None
    right_leg_audio = sounds.get('right_leg') if 'right_leg' in completed_movements else None

    # 2. Movement Commands: show mapping for all movements
    movement_emojis = {
        "left_hand": "🫲",
        "right_hand": "🫱",
        "left_leg": "🦵",
        "right_leg": "🦵",
    }

    movement_command_lines = []
    # Show 'Now Playing' for all completed movements (layers that are currently playing)
    completed_movements = sound_manager.movements_completed
    for movement in ["left_hand", "right_hand", "left_leg", "right_leg"]:
        sound_file = sound_manager.current_sound_mapping.get(movement, "")
        instrument_type = ""
        for key in ["bass", "drums", "instruments", "vocals"]:
            if key in sound_file.lower():
                instrument_type = key if key != "instruments" else "instrument"
                break
        pretty_movement = movement.replace("_", " ").title()
        # Always use 'Instruments' (plural) for the left hand stem
        if movement == "left_hand" and instrument_type.lower() == "instrument":
            pretty_instrument = "Instruments"
        else:
            pretty_instrument = instrument_type.capitalize() if instrument_type else "--"
        emoji = movement_emojis.get(movement, "")
        # Add 'Now Playing' indicator for all completed movements
        if movement in completed_movements:
            movement_command_lines.append(f"{emoji} {pretty_movement}: {pretty_instrument}  ▶️ Now Playing")
        else:
            movement_command_lines.append(f"{emoji} {pretty_movement}: {pretty_instrument}")
    movement_command_text = "🎼 Composition Mode - Movement to Stems Mapping\n" + "\n".join(movement_command_lines)

    # 3. Next Trial: always prompt user
    next_trial_text = "Imagine next movement"

    composition_info = sound_manager.get_composition_info()
    status_text = format_composition_summary(composition_info)
    return (
        movement_command_text,
        next_trial_text,
        fig,
        left_hand_audio,
        right_hand_audio,
        left_leg_audio,
        right_leg_audio,
        status_text
    )

def continue_dj_phase():
    ''' Continue in DJ phase, applying effects and always playing all layered sounds.
    '''
    global app_state
    if not app_state['composition_active']:
        return "❌ Not active", "❌ Not active", "❌ Not active", None, None, None, None, None, None, "Click 'Start Composing' first"
    if app_state['demo_data'] is None:
        return "❌ No data", "❌ No data", "❌ No data", None, None, None, None, None, None, "No EEG data available"
    # DJ phase: enforce strict DJ effect order
    epoch_data, true_label = data_processor.simulate_real_time_data(app_state['demo_data'], app_state['demo_labels'], mode="class_balanced")
    predicted_class, confidence, probabilities = classifier.predict(epoch_data)
    predicted_name = classifier.class_names[predicted_class]
    # Strict DJ order: right_hand, right_leg, left_leg, left_hand
    if not hasattr(continue_dj_phase, 'dj_order'):
        continue_dj_phase.dj_order = ["right_hand", "right_leg", "left_leg", "left_hand"]
        continue_dj_phase.dj_index = 0
    # Find the next movement in the DJ order that hasn't been toggled yet (using effect counters)
    while continue_dj_phase.dj_index < 4:
        next_movement = continue_dj_phase.dj_order[continue_dj_phase.dj_index]
        # Only proceed if the predicted movement matches the next in order
        if predicted_name == next_movement:
            break
        else:
            # Ignore this prediction, do not apply effect
            next_trial_text = "Imagine next movement"
            # UI update: show which movement is expected
            # Always play all completed movement sounds (layered)
            sounds = get_movement_sounds()
            completed_movements = sound_manager.movements_completed
            left_hand_audio = sounds.get('left_hand') if 'left_hand' in completed_movements else None
            right_hand_audio = sounds.get('right_hand') if 'right_hand' in completed_movements else None
            left_leg_audio = sounds.get('left_leg') if 'left_leg' in completed_movements else None
            right_leg_audio = sounds.get('right_leg') if 'right_leg' in completed_movements else None
            movement_map = {
                "left_hand": {"effect": "Fade In/Out", "instrument": "Instruments"},
                "right_hand": {"effect": "Low Pass", "instrument": "Bass"},
                "left_leg": {"effect": "Compressor", "instrument": "Drums"},
                "right_leg": {"effect": "Echo", "instrument": "Vocals"},
            }
            emoji_map = {"left_hand": "🫲", "right_hand": "🫱", "left_leg": "🦵", "right_leg": "🦵"}
            movement_command_lines = []
            for m in ["left_hand", "right_hand", "left_leg", "right_leg"]:
                status = "ON" if sound_manager.active_effects.get(m, False) else "off"
                movement_command_lines.append(f"{emoji_map[m]} {m.replace('_', ' ').title()}: {movement_map[m]['effect']} [{'ON' if status == 'ON' else 'off'}] → {movement_map[m]['instrument']}")
            target_text = "🎧 DJ Mode - Movement to Effect Mapping\n" + "\n".join(movement_command_lines)
            composition_info = sound_manager.get_composition_info()
            status_text = format_composition_summary(composition_info)
            fig = create_eeg_plot(epoch_data, classifier.class_names[true_label], predicted_name, confidence, False, app_state.get('ch_names'))
            return (
                target_text,            # Movement Commands (textbox)
                next_trial_text,         # Next Trial (textbox)
                fig,                    # EEG Plot (plot)
                left_hand_audio,        # Left Hand (audio)
                right_hand_audio,       # Right Hand (audio)
                left_leg_audio,         # Left Leg (audio)
                right_leg_audio,        # Right Leg (audio)
                status_text,            # Composition Status (textbox)
                gr.update(),            # Timer (update object)
                gr.update()             # Continue DJ Button (update object)
            )
    # If correct movement, apply effect and advance order
    effect_applied = False
    if confidence > CONFIDENCE_THRESHOLD and predicted_name == continue_dj_phase.dj_order[continue_dj_phase.dj_index]:
        result = sound_manager.toggle_dj_effect(predicted_name, brief=True, duration=1.0)
        effect_applied = result.get("effect_applied", False)
        continue_dj_phase.dj_index += 1
    else:
        result = None
    fig = create_eeg_plot(epoch_data, classifier.class_names[true_label], predicted_name, confidence, effect_applied, app_state.get('ch_names'))
    # Always play all completed movement sounds (layered)
    sounds = get_movement_sounds()
    completed_movements = sound_manager.movements_completed
    left_hand_audio = sounds.get('left_hand') if 'left_hand' in completed_movements else None
    right_hand_audio = sounds.get('right_hand') if 'right_hand' in completed_movements else None
    left_leg_audio = sounds.get('left_leg') if 'left_leg' in completed_movements else None
    right_leg_audio = sounds.get('right_leg') if 'right_leg' in completed_movements else None
    # Show DJ effect mapping for each movement with ON/OFF status and correct instrument mapping
    movement_map = {
        "left_hand": {"effect": "Fade In/Out", "instrument": "Instruments"},
        "right_hand": {"effect": "Low Pass", "instrument": "Bass"},
        "left_leg": {"effect": "Compressor", "instrument": "Drums"},
        "right_leg": {"effect": "Echo", "instrument": "Vocals"},
    }
    emoji_map = {"left_hand": "🫲", "right_hand": "🫱", "left_leg": "🦵", "right_leg": "🦵"}
    # Get effect ON/OFF status from sound_manager.active_effects
    movement_command_lines = []
    for m in ["left_hand", "right_hand", "left_leg", "right_leg"]:
        # Show [ON] only if effect is currently active (True), otherwise [off]
        status = "ON" if sound_manager.active_effects.get(m, False) else "off"
        movement_command_lines.append(f"{emoji_map[m]} {m.replace('_', ' ').title()}: {movement_map[m]['effect']} [{'ON' if status == 'ON' else 'off'}] → {movement_map[m]['instrument']}")
    target_text = "🎧 DJ Mode - Movement to Effect Mapping\n" + "\n".join(movement_command_lines)
    # In DJ mode, Next Trial should only show the prompt, not the predicted/target movement
    predicted_text = "Imagine next movement"
    composition_info = sound_manager.get_composition_info()
    status_text = format_composition_summary(composition_info)
    # Ensure exactly 10 outputs: [textbox, textbox, plot, audio, audio, audio, audio, textbox, timer, button]
    # Use fig for the plot, and fill all outputs with correct types
    return (
        target_text,            # Movement Commands (textbox)
        predicted_text,         # Next Trial (textbox)
        fig,                    # EEG Plot (plot)
        left_hand_audio,        # Left Hand (audio)
        right_hand_audio,       # Right Hand (audio)
        left_leg_audio,         # Left Leg (audio)
        right_leg_audio,        # Right Leg (audio)
        status_text,            # Composition Status (textbox)
        gr.update(),            # Timer (update object)
        gr.update()             # Continue DJ Button (update object)
    )

# --- Gradio UI ---
def create_interface():
    ''' Create the Gradio interface.
    '''
    with gr.Blocks(title="EEG Motor Imagery Music Composer", theme=gr.themes.Citrus()) as demo:
        with gr.Tabs():
            with gr.TabItem("🎵 Automatic Music Composer"):
                gr.Markdown("# 🧠 NeuroMusic Studio: An accessible, easy to use motor rehabilitation device.")
                gr.Markdown("""
                **How it works:**

                1. **Compose:** Imagine moving your left hand, right hand, left leg, or right leg to add musical layers. Each correct, high-confidence prediction adds a sound. Just follow the prompts.

                2. **DJ Mode:** After all four layers are added, you can apply effects and remix your composition using new brain commands.

                > **Tip:** In DJ mode, each effect is triggered only every 4th time you repeat a movement, to keep playback smooth.

                Commands and controls update as you progress. Just follow the on-screen instructions!
                """)
                
                with gr.Row():
                    with gr.Column(scale=2):
                        start_btn = gr.Button("🎵 Start Composing", variant="primary", size="lg")
                        stop_btn = gr.Button("🛑 Stop", variant="stop", size="md")
                        continue_btn = gr.Button("⏭️ Continue DJ Phase", variant="primary", size="lg", visible=False)
                        timer = gr.Timer(value=1.0, active=False)  # 4 second intervals
                        predicted_display = gr.Textbox(label="🧠 Movement Commands", interactive=False, value="--", lines=4)
                        timer_display = gr.Textbox(label="⏱️ Next Trial", interactive=False, value="--")
                        eeg_plot = gr.Plot(label="EEG Data Visualization")
                    with gr.Column(scale=1):
                        left_hand_sound = gr.Audio(label="🫲 Left Hand", interactive=False, autoplay=True, visible=True)
                        right_hand_sound = gr.Audio(label="🫱 Right Hand", interactive=False, autoplay=True, visible=True)
                        left_leg_sound = gr.Audio(label="🦵 Left Leg", interactive=False, autoplay=True, visible=True)
                        right_leg_sound = gr.Audio(label="🦵 Right Leg", interactive=False, autoplay=True, visible=True)
                        composition_status = gr.Textbox(label="Composition Status", interactive=False, lines=5)
                def start_and_activate_timer():
                    ''' Start composing and activate timer for trials.
                    '''
                    result = start_composition()
                    last_trial_result[:] = result  # Initialize with first trial result
                    if "DJ Mode" not in result[0]:
                        return (*result, gr.update(active=True), gr.update(visible=False))
                    else:
                        return (*result, gr.update(active=False), gr.update(visible=True))
                
                # ITI logic: 3s blank, 1s prompt, then trial
                timer_counter = {"count": 0}
                last_trial_result = [None] * 9  # Adjust length to match your outputs
                def timer_tick():
                    ''' Timer tick handler for ITI and trials.
                    '''
                    # 0,1,2: blank, 3: prompt, 4: trial
                    if timer_counter["count"] < 3:
                        timer_counter["count"] += 1
                        # Show blank prompt, keep last outputs
                        if len(last_trial_result) == 8:
                            return (*last_trial_result, gr.update(active=True), gr.update(visible=False))
                        elif len(last_trial_result) == 10:
                            # DJ mode: blank prompt
                            result = list(last_trial_result)
                            result[1] = ""
                            return tuple(result)
                        else:
                            raise ValueError(f"Unexpected last_trial_result length: {len(last_trial_result)}")
                    elif timer_counter["count"] == 3:
                        timer_counter["count"] += 1
                        # Show prompt
                        result = list(last_trial_result)
                        result[1] = "Imagine next movement"
                        if len(result) == 8:
                            return (*result, gr.update(active=True), gr.update(visible=False))
                        elif len(result) == 10:
                            return tuple(result)
                        else:
                            raise ValueError(f"Unexpected result length in prompt: {len(result)}")
                    else:
                        timer_counter["count"] = 0
                        # Run trial
                        result = list(start_composition())
                        last_trial_result[:] = result  # Save for next blanks/prompts
                        if len(result) == 8:
                            # Pre-DJ mode: add timer and button updates
                            if any(isinstance(x, str) and "DJ Mode" in x for x in result):
                                return (*result, gr.update(active=False), gr.update(visible=True))
                            else:
                                return (*result, gr.update(active=True), gr.update(visible=False))
                        elif len(result) == 10:
                            return tuple(result)
                        else:
                            raise ValueError(f"Unexpected result length in timer_tick: {len(result)}")
                
                def continue_dj():
                    ''' Continue DJ phase from button click.
                    '''
                    result = continue_dj_phase()
                    if len(result) == 8:
                        return (*result, gr.update(active=False), gr.update(visible=True))
                    elif len(result) == 10:
                        return result
                    else:
                        raise ValueError(f"Unexpected result length in continue_dj: {len(result)}")
                start_btn.click(
                    fn=start_and_activate_timer,
                    outputs=[predicted_display, timer_display, eeg_plot,
                            left_hand_sound, right_hand_sound, left_leg_sound, right_leg_sound, composition_status, timer, continue_btn]
                )
                timer_event = timer.tick(
                    fn=timer_tick,
                    outputs=[predicted_display, timer_display, eeg_plot,
                            left_hand_sound, right_hand_sound, left_leg_sound, right_leg_sound, composition_status, timer, continue_btn]
                )
                def stop_composing():
                    ''' Stop composing and reset state (works in both building and DJ mode). '''
                    timer_counter["count"] = 0
                    app_state['composition_active'] = False  # Ensure new cycle on next start
                    # Reset sound_manager state for new session
                    sound_manager.current_phase = "building"
                    sound_manager.composition_layers = {}
                    sound_manager.movements_completed = set()
                    sound_manager.active_effects = {m: False for m in ["left_hand", "right_hand", "left_leg", "right_leg"]}
                    # Clear static audio cache in get_movement_sounds
                    if hasattr(get_movement_sounds, 'audio_cache'):
                        for m in get_movement_sounds.audio_cache:
                            get_movement_sounds.audio_cache[m][True] = None
                            get_movement_sounds.audio_cache[m][False] = None
                    if hasattr(get_movement_sounds, 'last_effect_state'):
                        for m in get_movement_sounds.last_effect_state:
                            get_movement_sounds.last_effect_state[m] = None
                    if hasattr(get_movement_sounds, 'play_counter'):
                        for m in get_movement_sounds.play_counter:
                            get_movement_sounds.play_counter[m] = 0
                        get_movement_sounds.total_calls = 0
                    # Clear UI and deactivate timer, hide continue button, clear all audio
                    last_trial_result[:] = ["--", "Stopped", None, None, None, None, None, "Stopped"]
                    return ("--", "Stopped", None, None, None, None, None, "Stopped", gr.update(active=False), gr.update(visible=False))

                stop_btn.click(
                    fn=stop_composing,
                    outputs=[predicted_display, timer_display, eeg_plot,
                            left_hand_sound, right_hand_sound, left_leg_sound, right_leg_sound, composition_status, timer, continue_btn],
                    cancels=[timer_event]
                )
                continue_btn.click(
                    fn=continue_dj,
                    outputs=[predicted_display, timer_display, eeg_plot,
                            left_hand_sound, right_hand_sound, left_leg_sound, right_leg_sound, composition_status, timer, continue_btn]
                )

            with gr.TabItem("📝 Manual Classifier"):
                gr.Markdown("# Manual Classifier")
                gr.Markdown("Select a movement and run the classifier manually on a random epoch for that movement. Results will be accumulated below.")
                movement_dropdown = gr.Dropdown(choices=["left_hand", "right_hand", "left_leg", "right_leg"], label="Select Movement")
                manual_btn = gr.Button("Run Classifier", variant="primary")
                manual_predicted = gr.Textbox(label="Predicted Class", interactive=False)
                manual_confidence = gr.Textbox(label="Confidence", interactive=False)
                manual_plot = gr.Plot(label="EEG Data Visualization")
                manual_probs = gr.Plot(label="Class Probabilities")
                manual_confmat = gr.Plot(label="Confusion Matrix (Session)")

                # Session state for confusion matrix
                from collections import defaultdict
                session_confmat = defaultdict(lambda: defaultdict(int))

                def manual_classify(selected_movement):
                    ''' Manually classify a random epoch for the selected movement.
                    '''
                    import matplotlib.pyplot as plt
                    import numpy as np
                    if app_state['demo_data'] is None or app_state['demo_labels'] is None:
                        return "No data", "No data", None, None, None
                    label_idx = [k for k, v in classifier.class_names.items() if v == selected_movement][0]
                    matching_indices = np.where(app_state['demo_labels'] == label_idx)[0]
                    if len(matching_indices) == 0:
                        return "No data for this movement", "", None, None, None
                    chosen_idx = np.random.choice(matching_indices)
                    epoch_data = app_state['demo_data'][chosen_idx]
                    predicted_class, confidence, probs = classifier.predict(epoch_data)
                    predicted_name = classifier.class_names[predicted_class]
                    # Update confusion matrix
                    session_confmat[selected_movement][predicted_name] += 1
                    # Plot confusion matrix
                    classes = ["left_hand", "right_hand", "left_leg", "right_leg"]
                    confmat = np.zeros((4, 4), dtype=int)
                    for i, true_m in enumerate(classes):
                        for j, pred_m in enumerate(classes):
                            confmat[i, j] = session_confmat[true_m][pred_m]
                    fig_confmat, ax = plt.subplots(figsize=(4, 4))
                    ax.imshow(confmat, cmap="Blues")
                    ax.set_xticks(np.arange(4))
                    ax.set_yticks(np.arange(4))
                    ax.set_xticklabels(classes, rotation=45, ha="right")
                    ax.set_yticklabels(classes)
                    ax.set_xlabel("Predicted")
                    ax.set_ylabel("True")
                    for i in range(4):
                        for j in range(4):
                            ax.text(j, i, str(confmat[i, j]), ha="center", va="center", color="black")
                    fig_confmat.tight_layout()
                    # Plot class probabilities
                    if isinstance(probs, dict):
                        probs_list = [probs.get(cls, 0.0) for cls in classes]
                    else:
                        probs_list = list(probs)
                    fig_probs, ax_probs = plt.subplots(figsize=(4, 2))
                    ax_probs.bar(classes, probs_list)
                    ax_probs.set_ylabel("Probability")
                    ax_probs.set_ylim(0, 1)
                    fig_probs.tight_layout()
                    # EEG plot
                    fig = create_eeg_plot(epoch_data, selected_movement, predicted_name, confidence, False, app_state.get('ch_names'))
                    # Close all open figures to avoid warnings
                    plt.close(fig_confmat)
                    plt.close(fig_probs)
                    plt.close(fig)
                    return predicted_name, f"{confidence:.2f}", fig, fig_probs, fig_confmat

                manual_btn.click(
                    fn=manual_classify,
                    inputs=[movement_dropdown],
                    outputs=[manual_predicted, manual_confidence, manual_plot, manual_probs, manual_confmat]
                )
    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(server_name="0.0.0.0", server_port=7867)
