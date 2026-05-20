import numpy as np
import soundfile as sf

# 1 second of random noise at 16000 Hz
sample_rate = 16000
duration = 1.0
t = np.linspace(0, duration, int(sample_rate * duration), False)
# Generate a simple sine wave instead of noise
audio = 0.5 * np.sin(2 * np.pi * 440 * t)

sf.write('test_audio.wav', audio, sample_rate)
print("Created test_audio.wav")
