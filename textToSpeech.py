# Install necessary libraries (if not yet installed)
!pip install transformers torch soundfile

# Import libraries
from transformers import AutoProcessor, BarkModel
import torch
import soundfile as sf

# Load the text-to-speech model
processor = AutoProcessor.from_pretrained("suno/bark")
model = BarkModel.from_pretrained("suno/bark")

# Move model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# Define your input text
text_prompt = "Hello! This is an AI-generated voice speaking through a text-to-speech model."

# Convert text to audio tensor
inputs = processor(text_prompt, return_tensors="pt").to(device)
audio_outputs = model.generate(**inputs)

# Convert the audio tensor to numpy and save it as a .wav file
audio_array = audio_outputs.cpu().numpy().squeeze()
sf.write("output_audio.wav", audio_array, 24000)

# Done — play or download output_audio.wav
print("✅ Audio file generated: output_audio.wav")
