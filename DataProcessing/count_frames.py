"""
FOr every video file, saves an additional file signerX_sampleY-nFrames
which contains a single integer (in text) with the number of frames.
"""
import argparse
import glob
import os

import torchvision

def main(args):
    for dataset in ["train", "val", "test"]:
        print(args.input_dir)
        print(os.path.join(args.input_dir, dataset))
        videos = glob.glob(os.path.join(args.input_dir, dataset, '*/*.avi'))
        print(videos)
        for video_file in videos:
            frames, _, _ = torchvision.io.read_video(video_file, pts_unit="sec")
            with open(video_file.replace(".avi", "_nframes"), "w") as of:
                of.write(f"{frames.size(0)}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str)
    args = parser.parse_args()
    main(args)