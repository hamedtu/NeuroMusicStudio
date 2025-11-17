# NeuroMusic Studio

A user-friendly, accessible neuro-music studio for motor rehabilitation and creative exploration. Compose and remix music using EEG motor imagery signals—no musical experience required!

## Features

- **Automatic Composition:** Layer musical stems (bass, drums, instruments, vocals) by imagining left/right hand or leg movements. Each correct, high-confidence prediction adds a new sound.
- **DJ Mode:** After all four layers are added, apply real-time audio effects (Echo, Low Pass, Compressor, Fade In/Out) to remix your composition using new brain commands.
- **Seamless Playback:** All completed layers play continuously, with smooth transitions and effect toggling.
- **Manual Classifier:** Test the classifier on individual movements and visualize EEG data, class probabilities, and confusion matrix.
- **Accessible UI:** Built with Gradio for easy use in browser or on Hugging Face Spaces.

## How It Works

1. **Compose:**
   - Click "Start Composing" and follow the on-screen prompts.
   - Imagine the prompted movement (left hand, right hand, left leg, right leg) to add musical layers.
   - Each correct, confident prediction adds a new instrument to the mix.

2. **DJ Mode:**
   - After all four layers are added, enter DJ mode.
   - Imagine movements in a specific order to toggle effects on each stem.
   - Effects are sticky and only toggle every 4th repetition for smoothness.

3. **Manual Classifier:**
   - Switch to the Manual Classifier tab to test the model on random epochs for each movement.
   - Visualize predictions, probabilities, and confusion matrix.

## Technical Details

- **Model:** ShallowFBCSPNet architecture for EEG motor imagery classification
- **Data:** Pre-trained on EEG data with 4 movement classes (left/right hand, left/right leg)
- **Audio Processing:** Real-time audio effects using scipy and soundfile
- **Framework:** Gradio for web interface, PyTorch for deep learning

## Credits

- Developed and Deployed by Hamed Koochaki Kelardeh, Sofia Fregni, and Katarzyna Kuhlmann.
- Audio stems: [SoundHelix](https://www.soundhelix.com/)

## License


MIT License - see LICENSE file for details.

