import subprocess
import os
import torchaudio
from whisper import load_model

# Load the Whisper model (Tiny)
model = load_model('tiny')

def detect_language(video_path):
    # Check video duration and define segments
    duration = get_video_duration(video_path)
    segments = []
    if duration >= 240:
        segments = [(0, 30), (240, 270)]
    else:
        segments = [(0, 30)]
    
    language_counter = {}

    for start, end in segments:
        audio_segment = extract_audio_segment(video_path, start, end)
        language = transcribe_audio(audio_segment)
        if language:
            language_counter[language] = language_counter.get(language, 0) + 1

    # Final language decision logic
    if language_counter:
        detected_language = max(language_counter, key=language_counter.get)
    else:
        detected_language = 'unknown'

    return detected_language

def get_video_duration(video_path):
    command = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
               '-of', 'default=noprint_wrappers=1:nokey=1', video_path]
    duration = float(subprocess.check_output(command).strip())
    return duration

def extract_audio_segment(video_path, start, end):
    output_audio = 'temp_audio.wav'
    command = ['ffmpeg', '-i', video_path, '-ss', str(start), '-to', str(end),
               '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', output_audio]
    subprocess.run(command, check=True)
    return output_audio

def transcribe_audio(audio_file):
    audio_input, _ = torchaudio.load(audio_file)
    # Transcribing
    result = model.transcribe(audio_input)
    return result['language'] if 'language' in result else None

# Example usage
# detected_language = detect_language('path_to_video.mp4')
# print(f'Detected language: {detected_language}')
