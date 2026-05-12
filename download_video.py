#!/usr/bin/env python3
# ============================================================
# download_video.py — Helper to download a public sports video
# ============================================================
# Usage:
#   python download_video.py "https://www.youtube.com/watch?v=XXXX"
#
# Requirements:
#   pip install yt-dlp
# ============================================================

import sys
import subprocess
import os

DEFAULT_URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # replace with your video

def download(url: str, output: str = "input_video.mp4"):
    print(f"[Download] URL  : {url}")
    print(f"[Download] Output: {output}")

    cmd = [
        "yt-dlp",
        "-f", "best[height<=720][ext=mp4]/best[height<=720]/best",
        "--merge-output-format", "mp4",
        "-o", output,
        url,
    ]

    result = subprocess.run(cmd)
    if result.returncode == 0 and os.path.exists(output):
        size_mb = os.path.getsize(output) / (1024 * 1024)
        print(f"\n✅ Downloaded → {output}  ({size_mb:.1f} MB)")
        print("   Now run:  python pipeline.py")
    else:
        print("\n❌ Download failed. Check the URL or install yt-dlp:")
        print("   pip install yt-dlp")


if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_URL
    download(url)
