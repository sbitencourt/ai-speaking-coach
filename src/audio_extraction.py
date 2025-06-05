# audio_extraction.py
import ffmpeg

def extract_audio(input_mp4_path: str, output_wav_path: str):
    """
    Extracts audio from an MP4 file and saves it as a WAV file.

    Parameters:
    - input_mp4_path: path to the input MP4 file.
    - output_wav_path: path where the output WAV file will be saved.

    Returns:
    - None: The function saves the extracted audio to the specified output path.
    """

    stream = ffmpeg.input(input_mp4_path)
    stream = ffmpeg.output(stream, output_wav_path, format='wav', ac=1, ar='16000')
    ffmpeg.run(stream)
    print(f"Audio extracted to {output_wav_path}\n")