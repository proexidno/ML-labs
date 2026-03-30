from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa
import torch

model_name = "openai/whisper-small.en"
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name)
model.eval()

model.config.forced_decoder_ids = None

import os
import librosa
import soundfile as sf

input_dir = "data"
output_dir = "data_wav"
os.makedirs(output_dir, exist_ok=True)

supported_ext = (".wav", ".flac", ".mp3")

for filename in os.listdir(input_dir):
    if not filename.lower().endswith(supported_ext):
        continue

    input_path = os.path.join(input_dir, filename)

    base_name = os.path.splitext(filename)[0] + ".wav"
    output_path = os.path.join(output_dir, base_name)

    y, sr = librosa.load(input_path, sr=16000, mono=True)

    sf.write(output_path, y, 16000)
    print(f"Converted: {filename} → {base_name}")

audio_dir = "data_wav"
transcriptions = {}

for filename in sorted(os.listdir(audio_dir)):
    if not filename.endswith(".wav"):
        continue

    audio_path = os.path.join(audio_dir, filename)

    audio, _ = librosa.load(audio_path, sr=16000)

    input_features = processor(
        audio, sampling_rate=16000, return_tensors="pt"
    ).input_features

    with torch.no_grad():
        predicted_ids = model.generate(input_features)

    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

    transcriptions[filename] = transcription
    print(f"{filename}: {transcription}")

with open("./transcriptions.txt", "w", encoding="utf-8") as f:
    for filename, transcription in transcriptions.items():
        f.write(f"{filename}: {transcription}\n")

ground_truth = {}

with open("./data/1673-143396.trans.txt", "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split(" ", 1)
        if len(parts) == 2:
            fname, text = parts
            ground_truth[fname + ".wav"] = text

example_key = None
for key in transcriptions:
    if key in ground_truth:
        example_key = key
        break

example_key = "1673-143396-0000.wav"
ref_text = ground_truth[example_key]
hyp_text = transcriptions[example_key]

print(f"file: {example_key}")
print("Given transcription: ", ref_text)
print("Our transcription:  ", hyp_text.upper())

import librosa.display
import matplotlib.pyplot as plt

audio_path = os.path.join("data_wav", example_key)
y, sr = librosa.load(audio_path, sr=16000)

n_fft = 400
hop_length = 160
n_mels = 80

mel_spec = librosa.feature.melspectrogram(
    y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
)
log_mel_spec = librosa.power_to_db(mel_spec, ref=1.0)

plt.figure(figsize=(14, 5))
librosa.display.specshow(
    log_mel_spec,
    sr=sr,
    hop_length=hop_length,
    x_axis="time",
    y_axis="mel",
    cmap="magma",
)
plt.colorbar(format="%+2.0f dB")
plt.title(f"{example_key}")
plt.tight_layout()
plt.show()

