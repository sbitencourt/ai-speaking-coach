# transcription.py

# Libraries
from pydub import AudioSegment
import os
import openai
import subprocess

def parse_rttm(rttm_path: str):
    """
    Parse an RTTM file and return a list of tuples:
    (segment_start_time, segment_end_time, speaker_label)
    """
    segments = []
    with open(rttm_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) < 9:
                continue
            start_time = float(parts[3])
            duration = float(parts[4])
            speaker = parts[7]
            segments.append((start_time, start_time + duration, speaker))
    return segments

def stretch_audio(input_file, output_file, stretch_factor):
    """
    Stretch audio duration using ffmpeg's atempo filter.
    - input_file: path to input audio
    - output_file: path to save stretched audio
    - stretch_factor: factor < 1 speeds up, > 1 slows down
    """
    # atempo accepts range 0.5-2.0, so chain filters if necessary
    factors = []
    while stretch_factor > 2.0:
        factors.append(2.0)
        stretch_factor /= 2.0
    while stretch_factor < 0.5:
        factors.append(0.5)
        stretch_factor /= 0.5
    factors.append(stretch_factor)

    cmd = ['ffmpeg', '-y', '-i', input_file]
    for factor in factors:
        cmd += ['-filter:a', f'atempo={1/factor}']
    cmd += [output_file]

    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

def transcribe_with_diarization(original_audio_path: str, rttm_path: str, output_txt_path: str, auth_token: str, min_duration: float = 0.3):
    """
    Transcribe audio segments based on diarization RTTM file using OpenAI Whisper API.
    
    Parameters:
    - original_audio_path: Path to the original WAV audio file.
    - rttm_path: Path to the RTTM diarization file.
    - output_txt_path: Path to save the final transcription text file.
    - auth_token: OpenAI API authentication token.
    - min_duration: Minimum duration (in seconds) for a segment to be transcribed.
    """
    if not auth_token:
        raise ValueError("⚠️ auth_token is required for OpenAI API authentication.")

    # Set OpenAI API key
    openai.api_key = auth_token

    # Load the entire audio using pydub (milliseconds)
    audio = AudioSegment.from_file(original_audio_path)

    # Parse diarization RTTM file to get segments
    segments = parse_rttm(rttm_path)

    # Create a directory for temporary segment audio files
    segments_dir = 'data/audio/segments'
    os.makedirs(segments_dir, exist_ok=True)

    MIN_ABSOLUTE_DURATION = 0.1  # mínimo permitido pela API em segundos

    with open(output_txt_path, 'w') as output_file:
        for idx, (start, end, speaker) in enumerate(segments):
            segment_duration = end - start

            if segment_duration < MIN_ABSOLUTE_DURATION:
                print(f"⏩ Skipped segment {idx + 1}/{len(segments)} - Speaker: {speaker} - Duration {segment_duration:.3f}s (too short for transcription)")
                continue

            segment_audio = audio[start * 1000 : end * 1000]

            # Save the segment temporarily
            segment_file = os.path.join(segments_dir, f'segment_{idx}.wav')
            segment_audio.export(segment_file, format='wav', parameters=["-ac", "1", "-ar", "16000"])

            # Check if we need to stretch (between MIN_ABSOLUTE_DURATION and min_duration)
            if segment_duration < min_duration:
                stretch_factor = min_duration / segment_duration
                stretched_file = os.path.join(segments_dir, f'stretched_segment_{idx}.wav')
                try:
                    stretch_audio(segment_file, stretched_file, stretch_factor)
                    segment_to_use = stretched_file
                    print(f"🔊 Stretched segment {idx + 1}/{len(segments)} - Speaker: {speaker} - from {segment_duration:.2f}s to {min_duration:.2f}s")
                except subprocess.CalledProcessError as e:
                    print(f"❌ Failed to stretch audio: {e}. Using original segment.")
                    segment_to_use = segment_file
            else:
                segment_to_use = segment_file

            # Transcribe the audio segment using OpenAI Whisper API
            with open(segment_to_use, 'rb') as audio_file:
                response = openai.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="text",
                    language="en"
                )

            # Write the transcription with speaker label and timestamps
            output_file.write(f"[{start:.2f}s - {end:.2f}s] {speaker}: {response.strip()}\n")

            print(f"✅ Processed segment {idx + 1}/{len(segments)} - Speaker: {speaker}")

    print(f"✅ Transcription completed and saved to '{output_txt_path}'\n")
