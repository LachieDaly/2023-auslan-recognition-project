import os
import cv2
from pathlib import Path

def accumulative_video_motion():
    # perform forward, backward, and concatenated fusion of the frames for ISA64 dataset where structure is [signer-->sign-->samples]
    signers = ['001', '002', '003', '004', '005', '006', '007', '008', '009', '010']
    for s in signers:
        homeSourceFolder = os.path.join('/home/eye/lsa64_raw/images/', s)
        print(homeSourceFolder)
        signs = Path(homeSourceFolder).rglob('*.jpg')
        targetPath_forward = os.path.join('/home/eye/lsa64_raw/fusion/forward/', s)
        targetPath_backward = os.path.join('/home/eye/lsa64_raw/fusion/backward/', s)
        targetPath_both = os.path.join('/home/eye/lsa64_raw/fusion/both/', s)

        for sign in signs:
            # extract frames
            fpath = str(sign)
            filename = os.path.basename(fpath)
            filePath = os.path.dirname(fpath)
            newPath_forward = filePath.replace(homeSourceFolder, targetPath_forward)
            newPath_backward = filePath.replace(homeSourceFolder, targetPath_backward)
            newPath_both = filePath.replace(homeSourceFolder, targetPath_both)
            sampleFolderFullPath = fpath

            if not os.path.exists(os.path.join(newPath_forward, filename)):
                os.makedirs(newPath_forward, exist_ok=True)
                os.makedirs(newPath_backward, exist_ok=True)
                os.makedirs(newPath_both, exist_ok=True)
                keyFrameImages = Path(sampleFolderFullPath).rglob('*.jpg')
                numFrames = len(list(keyFrameImages))

                # Backward
                f = numFrames
                if numFrames > 0:
                    while f > 0:
                        imageFramePath = str(Path(sampleFolderFullPath) / f"{f}.jpg")
                        frameRGB_1 = cv2.imread(imageFramePath)
                        if f == numFrames:
                            imgDiff = frameRGB_1
                        else:
                            imgDiff = cv2.addWeighted(imgDiff, 0.5, frameRGB_1, 0.5, 0)
                        f -= 1

                    # Forward
                    f = 1
                    imgDiff_backward = imgDiff
                    while f <= numFrames:
                        imageFramePath = str(Path(sampleFolderFullPath) / f"{f}.jpg")
                        frameRGB_1 = cv2.imread(imageFramePath)
                        if f == 1:
                            imgDiff = frameRGB_1
                        else:
                            imgDiff = cv2.addWeighted(imgDiff, 0.5, frameRGB_1, 0.5, 0)
                        f += 1

                    imgDiff_forward = imgDiff
                    imgDiff_both = cv2.addWeighted(imgDiff_backward, 0.5, imgDiff_forward, 0.5, 0)

                    cv2.imwrite(os.path.join(newPath_backward, f"{filename}.jpg"), imgDiff_backward)
                    cv2.imwrite(os.path.join(newPath_forward, f"{filename}.jpg"), imgDiff_forward)
                    cv2.imwrite(os.path.join(newPath_both, f"{filename}.jpg"), imgDiff_both)


accumulative_video_motion()