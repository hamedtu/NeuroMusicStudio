"""
EEG Motor Imagery Classifier Module
----------------------------------
Handles model loading, inference, and real-time prediction for motor imagery classification.
Based on the ShallowFBCSPNet architecture from the original eeg_motor_imagery.py script.
"""

import torch
import torch.nn as nn
import numpy as np
from braindecode.models.shallow_fbcsp import ShallowFBCSPNet
from braindecode.modules.layers import Ensure4d  # necessary for loading
from typing import Dict, Tuple
import os
from data_processor import EEGDataProcessor
from config import DEMO_DATA_PATHS

class MotorImageryClassifier:
    """
    Motor imagery classifier using ShallowFBCSPNet model.
    """
    
    def __init__(self, model_path: str = "model.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.model_path = model_path
        self.class_names = {
            0: "left_hand",
            1: "right_hand", 
            2: "neutral",
            3: "left_leg",
            4: "tongue",
            5: "right_leg"
        }
        self.is_loaded = False
        
    def load_model(self, n_chans: int, n_times: int, n_outputs: int = 6):
        """Load the pre-trained ShallowFBCSPNet model.
        If model file not found or incompatible, fallback to LOSO training.
        """
        try:
            self.model = ShallowFBCSPNet(
                n_chans=n_chans,
                n_outputs=n_outputs,
                n_times=n_times,
                final_conv_length="auto"
            ).to(self.device)
            
            if os.path.exists(self.model_path):
                try:
                    # Load only the state_dict, using weights_only=True and allowlist ShallowFBCSPNet
                    with torch.serialization.safe_globals([Ensure4d, ShallowFBCSPNet]):
                        checkpoint = torch.load(
                        self.model_path,
                        map_location=self.device,
                        weights_only=False  # must be False to allow objects
                    )

                    # If checkpoint is a state_dict (dict of tensors)
                    if isinstance(checkpoint, dict) and all(isinstance(k, str) for k in checkpoint.keys()):
                        self.model.load_state_dict(checkpoint)

                    # If checkpoint is the full model object
                    elif isinstance(checkpoint, ShallowFBCSPNet):
                        self.model = checkpoint.to(self.device)

                    else:
                        raise ValueError("Unknown checkpoint format")


                    #self.model.load_state_dict(state_dict)
                    self.model.eval()
                    self.is_loaded = True
                except Exception:
                    self.is_loaded = False
            else:
                self.is_loaded = False
                
        except Exception:
            self.is_loaded = False
    
    def get_model_status(self) -> str:
        """Get current model status for user interface."""
        if self.is_loaded:
            return "✅ Pre-trained model loaded and ready"
        else:
            return "🔄 Using LOSO training (training new model from EEG data)"
            
    def predict(self, eeg_data: np.ndarray) -> Tuple[int, float, Dict[str, float]]:
        """
        Predict motor imagery class from EEG data.
        
        Args:
            eeg_data: EEG data array of shape (n_channels, n_times)
            
        Returns:
            predicted_class: Predicted class index
            confidence: Confidence score
            probabilities: Dictionary of class probabilities
        """
        if not self.is_loaded:
            return self._fallback_loso_classification(eeg_data)
            
        # Ensure input is the right shape: (batch, channels, time)
        if eeg_data.ndim == 2:
            eeg_data = eeg_data[np.newaxis, ...]
        
        # Convert to tensor
        x = torch.from_numpy(eeg_data.astype(np.float32)).to(self.device)
        
        with torch.no_grad():
            output = self.model(x)
            probabilities = torch.softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).cpu().numpy()[0]
            confidence = probabilities.max().cpu().numpy()
            
            # Convert to dictionary
            prob_dict = {
                self.class_names[i]: probabilities[0, i].cpu().numpy()
                for i in range(len(self.class_names))
            }
            
        return predicted_class, confidence, prob_dict

    def _fallback_loso_classification(self, eeg_data: np.ndarray) -> Tuple[int, float, Dict[str, float]]:
        """
        Fallback classification using LOSO (Leave-One-Session-Out) training.
        Trains a model on available data when pre-trained model isn't available.
        """
        try:
            
            # Initialize data processor
            processor = EEGDataProcessor()
            
            # Check if demo data files exist
            available_files = [f for f in DEMO_DATA_PATHS if os.path.exists(f)]
            if len(available_files) < 2:
                raise ValueError(f"Not enough data files for LOSO training. Need at least 2 files, found {len(available_files)}. "
                               f"Available files: {available_files}")
            
            # Perform LOSO split (using first session as test)
            X_train, y_train, X_test, y_test, session_info = processor.prepare_loso_split(
                available_files, test_session_idx=0
            )
            
            # Get data dimensions
            n_chans = X_train.shape[1]
            n_times = X_train.shape[2]
            
            # Create and train model
            self.model = ShallowFBCSPNet(
                n_chans=n_chans,
                n_outputs=6,
                n_times=n_times,
                final_conv_length="auto"
            ).to(self.device)
            
            # Simple training loop
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()
            
            # Convert training data to tensors
            X_train_tensor = torch.from_numpy(X_train).float().to(self.device)
            y_train_tensor = torch.from_numpy(y_train).long().to(self.device)
            
            # Quick training (just a few epochs for demo)
            self.model.train()
            for epoch in range(50):
                optimizer.zero_grad()
                outputs = self.model(X_train_tensor)
                loss = criterion(outputs, y_train_tensor)
                loss.backward()
                optimizer.step()
                
            
            # Switch to evaluation mode
            self.model.eval()
            self.is_loaded = True
            
            
            # Now make prediction with the trained model
            return self.predict(eeg_data)
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize classifier. Neither pre-trained model nor LOSO training succeeded: {e}")
    
    