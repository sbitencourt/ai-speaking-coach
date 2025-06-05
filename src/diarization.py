import os
from pyannote.audio import Pipeline

def run_diarization(input_wav_path: str, auth_token: str):
    """
    Run speaker diarization on a WAV audio file.

    Parameters:
    - input_wav_path: path to the input WAV audio file.
    - auth_token: Hugging Face authentication token for accessing the model.

    Returns:
    - diarization: the diarization result with speaker segments.
    """

    # Load the pre-trained speaker diarization pipeline
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=auth_token)

    # Apply the diarization pipeline to the input audio
    diarization = pipeline(input_wav_path)

    print("Diarization completed successfully.")

    # Prepare output directory and file name
    output_dir = os.path.join("data", "diarization")
    os.makedirs(output_dir, exist_ok=True)

    # Get base name of the input file (e.g., 'audio.wav' -> 'audio')
    base_name = os.path.splitext(os.path.basename(input_wav_path))[0]

    # Create full path for the RTTM file
    output_rttm_path = os.path.join(output_dir, f"{base_name}.rttm")

    # Save the diarization result in RTTM format
    with open(output_rttm_path, "w") as rttm_file:
        diarization.write_rttm(rttm_file)

    print(f"Diarization saved to: {output_rttm_path}")

    # Return the diarization object for further processing
    return diarization