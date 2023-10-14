import argparse
import pickle

import numpy as np
from tqdm import tqdm

"""
Ensembling of the SL-GCN components Bone, Joint, Bone-Motion, and Joint-Motions
"""
# Val labels
label = open('./Data/ELAR/sign/27/val_label.pkl', 'rb')
label = np.array(pickle.load(label))


# Joint results
r1 = open('./src/SAMSLR/SL-GCN/work_dir/sign_joint_final/eval_results/epoch_234_0.7847222222222222.pkl', 'rb')
r1 = list(pickle.load(r1).items())
print(len(r1[0][1]))


# Bone results
r2 = open('./src/SAMSLR/SL-GCN/work_dir/sign_bone_final/eval_results/epoch_247_0.75.pkl', 'rb')
r2 = list(pickle.load(r2).items())
print(len(r2[0][1]))


# Joint motion results
r3 = open('./src/SAMSLR/SL-GCN/work_dir/sign_joint_motion_final/eval_results/epoch_249_0.4722222222222222.pkl', 'rb')
r3 = list(pickle.load(r3).items())
print(len(r3[0][1]))


# Bone motion results
r4 = open('./src/SAMSLR/SL-GCN/work_dir/sign_bone_motion_final/eval_results/epoch_248_0.4288194444444444.pkl', 'rb')
r4 = list(pickle.load(r4).items())
print(len(r4[0][1]))


# We'll use these values for now
alpha = [1.0,0.9,0.5,0.5] # Joint, Bone, Joint-Motion, Bone-Motion

right_num = total_num = right_num_5 = 0
names = []
preds = []
scores = []
mean = 0

with open('Data/ELAR/sign/27/predictions.csv', 'w') as f:

    for i in tqdm(range(len(label[0]))):
        name, l = label[:, i]
        names.append(name)
        name1, r11 = r1[i]
        name2, r22 = r2[i]
        name3, r33 = r3[i]
        name4, r44 = r4[i]
        assert name == name1 == name2 == name3 == name4
        mean += r11.mean()
        score = (r11[:29]*alpha[0] + r22[:29]*alpha[1] + r33[:29]*alpha[2] + r44[:29]*alpha[3]) / np.array(alpha).sum()
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

with open('./src/SAMSLR/SL-GCN/work_dir/ensemble/gcn_ensembled.pkl', 'wb') as f:
    score_dict = dict(zip(names, scores))
    pickle.dump(score_dict, f)

    #src/SAMSLR/SL-GCN/ensemble/ensemble.py