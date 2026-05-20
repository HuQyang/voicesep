from funasr import AutoModel
import sys

try:
    model = AutoModel(
        model="SenseVoiceSmall",
        vad_model="fsmn-vad",
        vad_model_revision="v2.0.4",
        punc_model="ct-punc",
        punc_model_revision="v2.0.4",
        spk_model="cam++",
        spk_model_revision="v2.0.2",
    )
    print("Successfully loaded SenseVoiceSmall")
except Exception as e:
    print(f"Error: {e}")
