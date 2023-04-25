"""
This code has been used with some moficiations from
https://github.com/Hamzah-Luqman/SLR_AMN/blob/main/prepareData.py

Assume videos are stored in sVideoDir as:
... sVideoDir / train / class001 / gesture.mp4
... sVideoDir / val / class249 / gesture.avi

This pipeline
* extracts frames/images from videos (saved in sVideoDir path) and save frames in 
* Extract MobileNet features from frames and save them in diFeature folder
"""


