from funasr import AutoModel
model = AutoModel(
    model="paraformer-zh",
    disable_update=True,
)
res = model.generate(input="test_audio.wav", hotword="阈值", batch_size_s=300)
print(res)
