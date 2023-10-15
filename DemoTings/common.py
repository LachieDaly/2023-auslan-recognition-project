import csv
import os

def collect_samples(has_labels, root_path, job_path, sequence_length, 
                    temporal_stride, job, label_file_path, retrain_all=False):
    """
    Collects the required samples paths, labels, and frame indices

    :param has_labels: no longer fully necessary. should always be true
    :param root_path: path to the video directory
    :param job_path: train/val/test path - will be just train
    :param sequence_length: number of frames required
    :param temporal_stride: distance between selected frames
    :param job:  train or val 
    :param label_file_path: path to the label csv
    :return: sample list of dictionary of video_file, label, and frame indicies
    """
    if has_labels:  
        with open(label_file_path) as label_file:
            reader = csv.reader(label_file)
            samples = []
            for row in reader:
                if row[2] == job or (retrain_all and job == 'train'): 
                    video_file = os.path.join(root_path, job_path, row[0] + '.avi')
                    nframes_file = video_file.replace('.avi', '_nframes')
                    with open(nframes_file) as nff:
                        num_frames = int(nff.readline())
                    # Take frames around the middle of a video so long as its long enough
                    frame_start = (num_frames - sequence_length) // (2 * temporal_stride)
                    frame_end = frame_start + sequence_length * temporal_stride
                    if frame_start < 0:
                        frame_start = 0
                    if frame_end > num_frames:
                        frame_end = num_frames
                    frame_indices = list(range(frame_start, frame_end, temporal_stride))
                    # Add the last frame indice until we reach the desired sequence length
                    while len(frame_indices) < sequence_length:
                        frame_indices.append(frame_indices[-1])
                    samples.append({
                        'path': video_file,
                        'label': int(row[1]),
                        'frames': frame_indices
                    })
            return samples
