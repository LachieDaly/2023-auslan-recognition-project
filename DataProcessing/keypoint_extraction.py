import glob
import os

all_files = glob.glob('../Data/Demo/mp4/*.mp4')

CALL_STRING = 'bin\OpenPoseDemo.exe --render_pose 0 --number_people_max 1 --display 0 --video {} --write_json {} --model_pose BODY_135'

for sample in all_files:
    # Put each frame of keypoints from a video into its own folder
    out_dir = sample.replace("mp4", "kp").replace(".mp4", "").replace("videos", "kp")
    os.makedirs(out_dir, exist_ok=True)
    c = CALL_STRING.format(sample, out_dir)
    os.system(c)

