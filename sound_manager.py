"""
Sound Management System for EEG Motor Imagery Classification
-------------------------------------------------------------------------------
Handles sound mapping, layering, and music composition based on motor imagery predictions.
Supports seamless transition from building (layering) to DJ (effects) phase.
"""

import numpy as np
import soundfile as sf
from typing import Dict
from pathlib import Path

class AudioEffectsProcessor:
    @staticmethod
    def apply_fade_in_out(data: np.ndarray, samplerate: int, fade_duration: float = 0.5) -> np.ndarray:
        fade_samples = int(fade_duration * samplerate)
        data = np.copy(data)
        if fade_samples > 0 and fade_samples * 2 < len(data):
            fade_in_curve = np.linspace(0, 1, fade_samples)
            fade_out_curve = np.linspace(1, 0, fade_samples)
            data[:fade_samples] = data[:fade_samples] * fade_in_curve
            data[-fade_samples:] = data[-fade_samples:] * fade_out_curve
        return data
    @staticmethod
    def apply_high_pass_filter(data: np.ndarray, samplerate: int, cutoff: float = 800.0) -> np.ndarray:
        from scipy import signal
        nyquist = samplerate / 2
        normalized_cutoff = cutoff / nyquist
        b, a = signal.butter(4, normalized_cutoff, btype='high', analog=False)
        return signal.filtfilt(b, a, data)

    @staticmethod
    def apply_low_pass_filter(data: np.ndarray, samplerate: int, cutoff: float = 1200.0) -> np.ndarray:
        from scipy import signal
        nyquist = samplerate / 2
        normalized_cutoff = cutoff / nyquist
        b, a = signal.butter(4, normalized_cutoff, btype='low', analog=False)
        return signal.filtfilt(b, a, data)

    @staticmethod
    def apply_reverb(data: np.ndarray, samplerate: int, room_size: float = 0.5) -> np.ndarray:
        delay_samples = int(0.08 * samplerate)
        decay = 0.4 * room_size
        reverb_data = np.copy(data)
        for i in range(3):
            delay = delay_samples * (i + 1)
            if delay < len(data):
                gain = decay ** (i + 1)
                reverb_data[delay:] += data[:-delay] * gain
        return 0.7 * data + 0.3 * reverb_data

    @staticmethod
    def apply_echo(data: np.ndarray, samplerate: int, delay_time: float = 0.3, feedback: float = 0.4) -> np.ndarray:
        delay_samples = int(delay_time * samplerate)
        echo_data = np.copy(data)
        for i in range(delay_samples, len(data)):
            echo_data[i] += feedback * echo_data[i - delay_samples]
        return 0.7 * data + 0.3 * echo_data

    @staticmethod
    def apply_compressor(data: np.ndarray, samplerate: int, threshold: float = 0.2, ratio: float = 4.0) -> np.ndarray:
        # Simple compressor: reduce gain above threshold
        compressed = np.copy(data)
        over_threshold = np.abs(compressed) > threshold
        compressed[over_threshold] = np.sign(compressed[over_threshold]) * (threshold + (np.abs(compressed[over_threshold]) - threshold) / ratio)
        return compressed

    @staticmethod
    def process_layer_with_effects(audio_data: np.ndarray, samplerate: int, movement: str, active_effects: Dict[str, bool]) -> np.ndarray:
        processed_data = np.copy(audio_data)
        effect_map = {
            "left_hand": AudioEffectsProcessor.apply_fade_in_out,      # Fade in/out
            "right_hand": AudioEffectsProcessor.apply_low_pass_filter, # Low Pass
            "left_leg": AudioEffectsProcessor.apply_compressor,        # Compressor
            "right_leg": AudioEffectsProcessor.apply_echo,             # Echo (vocals)
        }
        effect_func = effect_map.get(movement)
        if active_effects.get(movement, False) and effect_func:
            if movement == "left_hand":
                processed_data = effect_func(processed_data, samplerate, fade_duration=0.5)
            else:
                processed_data = effect_func(processed_data, samplerate)
        return processed_data

class SoundManager:
    def __init__(self, sound_dir: str = "sounds"):
        self.available_sounds = [
            "SoundHelix-Song-6_bass.wav",
            "SoundHelix-Song-6_drums.wav",
            "SoundHelix-Song-6_instruments.wav",
            "SoundHelix-Song-6_vocals.wav"
        ]
        self.sound_dir = Path(sound_dir)
        self.current_cycle = 0
        self.current_step = 0
        self.cycle_complete = False
        self.completed_cycles = 0
        self.max_cycles = 2
        self.composition_layers = {}
        self.current_phase = "building"
        self.active_effects = {m: False for m in ["left_hand", "right_hand", "left_leg", "right_leg"]}
        self.active_movements = ["left_hand", "right_hand", "left_leg", "right_leg"]
        self.current_movement_sequence = []
        self.movements_completed = set()
        self.active_layers: Dict[str, str] = {}
        self.loaded_sounds = {}
        self._generate_new_sequence()
        self._load_sound_files()
        # Provide mapping from movement to sound file name for compatibility
        self.current_sound_mapping = {m: f for m, f in zip(self.active_movements, self.available_sounds)}
        # Track DJ effect trigger counts for each movement
        self.dj_effect_counters = {m: 0 for m in self.active_movements}
        self.cycle_stats = {'total_cycles': 0, 'successful_classifications': 0, 'total_attempts': 0}

    def _load_sound_files(self):
        self.loaded_sounds = {}
        for movement, filename in self.current_sound_mapping.items():
            file_path = self.sound_dir / filename
            if file_path.exists():
                data, sample_rate = sf.read(str(file_path))
                if len(data.shape) > 1:
                    data = np.mean(data, axis=1)
                self.loaded_sounds[movement] = {'data': data, 'sample_rate': sample_rate, 'sound_file': str(file_path)}

    def _generate_new_sequence(self):
        # Fixed movement order and mapping
        self.current_movement_sequence = ["left_hand", "right_hand", "left_leg", "right_leg"]
        self.current_sound_mapping = {
            "left_hand": "SoundHelix-Song-6_instruments.wav",
            "right_hand": "SoundHelix-Song-6_bass.wav",
            "left_leg": "SoundHelix-Song-6_drums.wav",
            "right_leg": "SoundHelix-Song-6_vocals.wav"
        }
        self.movements_completed = set()
        self.current_step = 0
        self._load_sound_files()

    def get_current_target_movement(self) -> str:
        # Always process left_hand last in DJ mode
        incomplete = [m for m in self.active_movements if m not in self.movements_completed]
        if not incomplete:
            return "cycle_complete"
        # If in DJ mode, left_hand should be last
        if getattr(self, 'current_phase', None) == 'dj_effects':
            # Remove left_hand from incomplete unless it's the only one left
            if 'left_hand' in incomplete and len(incomplete) > 1:
                incomplete = [m for m in incomplete if m != 'left_hand']
        import random
        movement = random.choice(incomplete)
        return movement


    def process_classification(self, predicted_class: str, confidence: float, threshold: float = 0.7, force_add: bool = False) -> Dict:
        result = {'sound_added': False, 'cycle_complete': False, 'audio_file': None}
        # If force_add is True, allow adding sound for any valid movement not already completed
        if force_add:
            if (
                confidence >= threshold and
                predicted_class in self.loaded_sounds and
                predicted_class not in self.composition_layers
            ):
                sound_info = dict(self.loaded_sounds[predicted_class])
                sound_info['confidence'] = confidence
                self.composition_layers[predicted_class] = sound_info
                self.movements_completed.add(predicted_class)
                result['sound_added'] = True
            else:
                pass
        else:
            current_target = self.get_current_target_movement()
            if (
                predicted_class == current_target and
                confidence >= threshold and
                predicted_class in self.loaded_sounds and
                predicted_class not in self.composition_layers
            ):
                sound_info = dict(self.loaded_sounds[predicted_class])
                sound_info['confidence'] = confidence
                self.composition_layers[predicted_class] = sound_info
                self.movements_completed.add(predicted_class)
                result['sound_added'] = True
            else:
                pass
        if len(self.movements_completed) >= len(self.active_movements):
            result['cycle_complete'] = True
            self.current_phase = "dj_effects"
        return result

    def start_new_cycle(self):
        self.current_cycle += 1
        self.current_step = 0
        self.cycle_complete = False
        self.cycle_stats['total_cycles'] += 1
        self._generate_new_sequence()
        self.composition_layers = {}  # Clear layers for new cycle
        self.movements_completed = set()
        self.current_phase = "building"
        self.active_layers = {}

    def transition_to_dj_phase(self):
        if len(self.composition_layers) >= len(self.active_movements):
            self.current_phase = "dj_effects"
            return True
        return False

    def toggle_dj_effect(self, movement: str, brief: bool = True, duration: float = 1.0) -> dict:
        import threading
        if self.current_phase != "dj_effects":
            return {"effect_applied": False, "message": "Not in DJ effects phase"}
        if movement not in self.active_effects:
            return {"effect_applied": False, "message": f"Unknown movement: {movement}"}
        # Only toggle effect at counts 1, 4, 8, ... (i.e., 1 and then every multiple of 4)
        self.dj_effect_counters[movement] += 1
        count = self.dj_effect_counters[movement]
        if count != 1 and (count - 1) % 4 != 0:
            return {"effect_applied": False, "message": f"Effect for {movement} only toggled at 1, 4, 8, ... (count={count})"}
        # Toggle effect ON
        self.active_effects[movement] = True
        effect_status = "ON"
        # Schedule effect OFF after duration if brief
        def turn_off_effect():
            self.active_effects[movement] = False
        if brief:
            timer = threading.Timer(duration, turn_off_effect)
            timer.daemon = True
            timer.start()
        return {"effect_applied": True, "effect_name": movement, "effect_status": effect_status, "count": count}

    def get_composition_info(self) -> Dict:
        layers_by_cycle = {0: []}
        for movement, layer_info in self.composition_layers.items():
            confidence = layer_info.get('confidence', 0) if isinstance(layer_info, dict) else 0
            layers_by_cycle[0].append({'movement': movement, 'confidence': confidence})
        # Add DJ effect status for each movement
        dj_effects_status = {m: self.active_effects.get(m, False) for m in self.active_movements}
        return {'layers_by_cycle': layers_by_cycle, 'dj_effects_status': dj_effects_status}

    def get_sound_mapping_options(self) -> Dict:
        return {
            'movements': self.active_movements,
            'available_sounds': self.available_sounds,
            'current_mapping': {m: self.loaded_sounds[m]['sound_file'] for m in self.loaded_sounds}
        }

    def get_all_layers(self):
        return {m: info['sound_file'] for m, info in self.composition_layers.items() if 'sound_file' in info}
