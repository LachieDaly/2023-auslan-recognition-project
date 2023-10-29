import pickle

import numpy as np
from tqdm import tqdm

# Our data labels
label = open('Data/ELAR/sign/27/val_label.pkl', 'rb')
label = np.array(pickle.load(label))

r1 = open('./src/SAMSLR/SSTCN/results/T_Pose_model_test.pkl', 'rb')
r1 = list(pickle.load(r1).items())

# We can give give these results a bit more choice
alpha = [1]  # gcn, rgb, flow_color, sstcn valeus, 

right_num = total_num = right_num_5 = 0
names = []
preds = []
scores = []
mean = 0
with open('sstcn_predictions.csv', 'w') as f:

    for i in tqdm(range(len(label[0]))):
        name, l = label[:, i]
        names.append(name)
        name1, r11 = r1[i]
        assert name == name1
        mean += r11.mean()
        score = r11*alpha[0] / np.array(alpha).sum()
        score = score.squeeze()
        rank_5 = score.argsort()[-5:]
        right_num_5 += int(int(l) in rank_5)
        r = np.argmax(score)
        scores.append(score)
        preds.append(r)
        right_num += int(r == int(l))
        total_num += 1
        f.write('{}, {}\n'.format(name, r))
    acc = right_num / total_num
    acc5 = right_num_5 / total_num
    print(total_num)
    print('top1: ', acc)
    print('top5: ', acc5)

f.close()
print(mean/len(label[0]))

# src/SAMSLR/ensemble/gcn/ensemble_multimodal_rgb.py