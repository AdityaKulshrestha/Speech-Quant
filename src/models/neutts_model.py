from neutts import NeuTTS2E
import soundfile as sf

tts = NeuTTS2E()

wav = tts.infer(
    "I can't believe it's finally here!",
    speaker="emily",
    emotion="happy",
)
sf.write("test.wav", wav, 24000)
