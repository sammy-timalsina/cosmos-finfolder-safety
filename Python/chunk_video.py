"""
Video Chunker for Cosmos Reason 2 Testing
------------------------------------------
Splits a long video into small chunks for testing.

Requirements:
    pip install opencv-python
"""

import cv2
import os

# ── Config ─────────────────────────────────────────────────────────────────────

# Change this to your video path
VIDEO_PATH   = r"C:\Users\samy.timalsina.AUER\Pictures\Camera Roll\defect.mp4"

# Output folder for chunks
OUTPUT_FOLDER = r"C:\Users\samy.timalsina.AUER\Pictures\Camera Roll\chunk2"

# Chunk duration in seconds
CHUNK_SECONDS = 2

# ── Script ─────────────────────────────────────────────────────────────────────

def chunk_video(video_path, output_folder, chunk_seconds):
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"❌ Cannot open video: {video_path}")
        return

    fps        = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration   = total_frames / fps
    width      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames_per_chunk = int(fps * chunk_seconds)

    print(f"✅ Video loaded:")
    print(f"   Duration : {duration:.1f} seconds")
    print(f"   FPS      : {fps:.1f}")
    print(f"   Size     : {width}x{height}")
    print(f"   Total    : {total_frames} frames")
    print(f"   Chunks   : ~{int(duration / chunk_seconds)} x {chunk_seconds}s chunks")
    print()

    chunk_index  = 0
    frame_count  = 0
    writer       = None
    chunk_path   = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Start new chunk
        if frame_count % frames_per_chunk == 0:
            # Close previous writer
            if writer is not None:
                writer.release()
                print(f"✅ Saved: {chunk_path}")

            chunk_index += 1
            chunk_path = os.path.join(output_folder, f"chunk_{chunk_index:03d}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(chunk_path, fourcc, fps, (width, height))

        writer.write(frame)
        frame_count += 1

    # Close last writer
    if writer is not None:
        writer.release()
        print(f"✅ Saved: {chunk_path}")

    cap.release()
    print(f"\n🎉 Done! {chunk_index} chunks saved to: {output_folder}")
    print(f"   Now update VIDEO_PATH in test_cosmos_video.py to point to any chunk")
    print(f"   Example: chunks\\chunk_001.mp4")

if __name__ == "__main__":
    chunk_video(VIDEO_PATH, OUTPUT_FOLDER, CHUNK_SECONDS)