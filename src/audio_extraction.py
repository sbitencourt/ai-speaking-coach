# audio_extraction.py
import ffmpeg

def extract_audio(input_mp4_path: str, output_wav_path: str):
    stream = ffmpeg.input(input_mp4_path)
    stream = ffmpeg.output(stream, output_wav_path, format='wav', ac=1, ar='16000')
    ffmpeg.run(stream)
    print(f"Audio extracted to {output_wav_path}")