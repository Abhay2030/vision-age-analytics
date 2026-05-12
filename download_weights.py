"""
Download pretrained Caffe models for face detection and age estimation.
Run once: python download_weights.py
"""
from __future__ import annotations

import os
import urllib.request

WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")

URLS = {
    "deploy.prototxt": "https://raw.githubusercontent.com/spmallick/learnopencv/master/FaceDetectionComparison/models/deploy.prototxt",
    "res10_300x300_ssd_iter_140000_fp16.caffemodel": "https://raw.githubusercontent.com/spmallick/learnopencv/master/FaceDetectionComparison/models/res10_300x300_ssd_iter_140000_fp16.caffemodel",
    "age_deploy.prototxt": "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/models/age_deploy.prototxt",
    "age_net.caffemodel": "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/models/age_net.caffemodel",
}


def main() -> None:
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    for name, url in URLS.items():
        dest = os.path.join(WEIGHTS_DIR, name)
        if os.path.isfile(dest) and os.path.getsize(dest) > 0:
            print(f"[skip] {name} already present")
            continue
        print(f"[get] {name} ...")
        urllib.request.urlretrieve(url, dest)
        print(f"[ok]  {dest}")
    print("[done] All weights ready.")


if __name__ == "__main__":
    main()
