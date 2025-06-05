# Libraries
from src.audio_extraction import extract_audio # <- importing the function from audio_extraction.py

### Main script to extract audio from an MP4 file and save it as a WAV file ###

# Define the input and output paths
input_mp4_path = 'data/raw/2025-06-04-2150.mp4'
output_wav_path = 'data/audio/2025-06-04-2150.wav'

# Call the function to extract audio
extract_audio(input_mp4_path, output_wav_path)
