from funasr import AutoModel
model = AutoModel(
    model="paraformer-zh",
    vad_model="fsmn-vad",
    punc_model="ct-punc",
    spk_model="cam++",
    disable_update=True,
)
res = model.generate(input="test_audio.wav", hotword="阈值", batch_size_s=300)
print(res)
