"""
Add audio segments to video with correct timing
Speeds up audio if it's too long for the allocated time slot
"""

import json
import os
import subprocess
import math

# Configuration
VIDEO_FILE = "new.mp4"
JSON_FILE = "narration.json"
AUDIO_DIR = "audio_segments"
OUTPUT_VIDEO = "new2_with_audio.mp4"
TEMP_DIR = "temp_audio"

def get_audio_duration(file_path):
    """Get duration of audio file using ffprobe"""
    cmd = [
        'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1', file_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return float(result.stdout.strip())

def time_to_seconds(time_str):
    """Convert MM:SS format to seconds"""
    parts = time_str.split(':')
    if len(parts) == 2:
        return int(parts[0]) * 60 + int(parts[1])
    elif len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    return 0

def process_segments():
    """Process all audio segments without speed adjustment"""

    # Load narration data
    with open(JSON_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    segments = data['segments']

    print("="*60)
    print("PROCESSING AUDIO SEGMENTS")
    print("="*60)

    processed_segments = []

    for seg in segments:
        seg_id = seg['id']
        start_time = time_to_seconds(seg['start'])
        end_time = time_to_seconds(seg['end'])
        allocated_duration = end_time - start_time

        input_file = os.path.join(AUDIO_DIR, f"segment_{seg_id:02d}.mp3")

        if not os.path.exists(input_file):
            print(f"\n[SKIP] Segment {seg_id:02d}: File not found")
            continue

        actual_duration = get_audio_duration(input_file)

        print(f"\nSegment {seg_id:02d} ({seg['start']} -> {seg['end']})")
        print(f"  Allocated: {allocated_duration:.2f}s")
        print(f"  Actual: {actual_duration:.2f}s")

        if actual_duration > allocated_duration:
            print(f"  [NOTE] Audio is {actual_duration - allocated_duration:.2f}s longer (will NOT speed up)")
        else:
            print(f"  [OK] Duration fits")

        processed_segments.append({
            'file': input_file,
            'start': start_time,
            'id': seg_id
        })

    return processed_segments

def combine_with_video(segments):
    """Combine all audio segments with video"""

    print("\n" + "="*60)
    print("COMBINING AUDIO WITH VIDEO")
    print("="*60)

    # Create filter complex for mixing all audio segments
    filter_parts = []

    # Input video audio (if exists)
    input_files = ['-i', VIDEO_FILE]

    # Add all audio segment inputs
    for i, seg in enumerate(segments):
        input_files.extend(['-i', seg['file']])

    # Build filter complex
    # Delay each audio segment to its start time
    delayed_audios = []
    for i, seg in enumerate(segments):
        # i+1 because 0 is the video
        delay_ms = int(seg['start'] * 1000)
        filter_parts.append(f"[{i+1}:a]adelay={delay_ms}|{delay_ms}[a{i}]")
        delayed_audios.append(f"[a{i}]")

    # Mix all delayed audio tracks and increase volume
    mix_inputs = "".join(delayed_audios)
    filter_parts.append(f"{mix_inputs}amix=inputs={len(segments)}:duration=longest[amixed]")
    # Boost volume by 2x (double the loudness)
    filter_parts.append(f"[amixed]volume=2.0[aout]")

    filter_complex = ";".join(filter_parts)

    # Build ffmpeg command
    cmd = [
        'ffmpeg', '-y'
    ] + input_files + [
        '-filter_complex', filter_complex,
        '-map', '0:v',  # video from input 0
        '-map', '[aout]',  # mixed audio
        '-c:v', 'copy',  # copy video codec
        '-c:a', 'aac',  # encode audio as AAC
        '-b:a', '192k',  # audio bitrate
        OUTPUT_VIDEO
    ]

    print(f"\nGenerating: {OUTPUT_VIDEO}")
    print("This may take a moment...")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"\n[OK] Video created successfully!")
        print(f"[SAVED] {OUTPUT_VIDEO}")
    else:
        print(f"\n[ERROR] Failed to create video")
        print(result.stderr)

def main():
    print("\n" + "="*60)
    print("ADD AUDIO TO VIDEO")
    print("="*60)
    print(f"Video: {VIDEO_FILE}")
    print(f"Audio segments: {AUDIO_DIR}/")
    print(f"Output: {OUTPUT_VIDEO}")

    # Process segments (speed up if needed)
    segments = process_segments()

    print(f"\n[OK] Processed {len(segments)} segments")

    # Combine with video
    combine_with_video(segments)

    print("\n" + "="*60)
    print("COMPLETE")
    print("="*60)
    print(f"Output file: {OUTPUT_VIDEO}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[CANCELLED]")
    except Exception as e:
        print(f"\n\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
