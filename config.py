"""
Configuration settings for the EEG Motor Imagery Music Composer
"""

import os
from pathlib import Path

# Data paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
SOUND_DIR = BASE_DIR / "sounds"
MODEL_DIR = BASE_DIR


# Classification settings
CONFIDENCE_THRESHOLD = 0.7  # Minimum confidence to add sound layer (lowered for testing)
MAX_COMPOSITION_LAYERS = 4  # Maximum layers in composition


# Demo data paths (optional) - updated with available files
DEMO_DATA_PATHS = [
    "data/HaLTSubjectA1602236StLRHandLegTongue.mat",
    "data/HaLTSubjectA1603086StLRHandLegTongue.mat",
    "data/HaLTSubjectA1603106StLRHandLegTongue.mat",
]
