# Deep Cascade Model

This model is based on Rastgoo's Deep Cascade model. This model performed well on the researchers own dataset as well as AUTSL.

In the paper, they used an SSD to get the bounding boxes of the hands, but it was unclear whether they had cutout the hands to pass as images to a CNN, or simply removed the background. We were able to also pass the orientation, and hand keypoints to the LSTM model. This performed fairly averagely with our dataset, and extracting the features using MediaHandsPipe was limited by the CPU making it impractical to augment the data between epochs. This combined with the relatively small number of samples per label likely caused this model to suffer bad results

Attempts were made to use an SSD trained on the egohands dataset, however they performed terribly on our ELAR dataset. Instead we used Mediapipe-Hands to extract the features of the hands, this still sometimes performed less than what we wanted, but still much better.

Multiple hyperparameters for learning rate, number of nodes in the LSTM, as well as what degree of preprocessing was applied to images before being passed to the LSTM were tried, but none performed particularly well, either overfitting in most cases.

## Requirements

The requirements of the model are as follows

- We used Tensorflow
- Perhaps it could be modifed to include some more data augmentation
