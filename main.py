# Libraries
from src.audio_extraction import extract_audio # <- importing the function from audio_extraction.py
from src.diarization import run_diarization  # <- importing the function from diarization.py
from src.transcription import parse_rttm  # <- importing the function from transcription.py
from src.transcription import transcribe_with_diarization  # <- importing the function from transcription.py
import os
from dotenv import load_dotenv

### Main script to extract audio from an MP4 file and save it as a WAV file ###

# # Define the input and output paths
# input_mp4_path = 'data/raw/2025-06-04-2150.mp4'
output_wav_path = 'data/audio/2025-06-04-2150.wav'

# # Call the function to extract audio
# extract_audio(input_mp4_path, output_wav_path)


### Main script to run speaker diarization on the extracted audio ###

# Load the environment variables from the .env file
load_dotenv()

# # Take the Hugging Face authentication token from environment variables
# huggingface_token = os.getenv("HUGGINGFACE_TOKEN")

# # Call the function to run diarization
# run_diarization(output_wav_path,auth_token=huggingface_token)


### Main script to transcribe the audio with diarization ###

# Define the paths for the RTTM file and the output transcription text file
rttm_path = 'data/diarization/2025-06-04-2150.rttm'
output_txt_path = 'data/transcription'

# Take the OpenAI API key from environment variables
openai_token = os.getenv("OPENAI_API_KEY")

# Call the function to parse the RTTM file
parse_rttm(rttm_path)

# Call the function to transcribe the audio with diarization
transcribe_with_diarization(output_wav_path, rttm_path, output_txt_path, openai_token)