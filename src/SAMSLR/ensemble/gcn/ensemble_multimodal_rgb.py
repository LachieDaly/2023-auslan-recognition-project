import pickle

import numpy as np
from tqdm import tqdm

# Our data labels
label = open('Data/ELAR/sign/27/val_label.pkl', 'rb')
label = np.array(pickle.load(label))

# Our GCN results
r1 = open('src/SAMSLR/SL-GCN/work_dir/ensemble/gcn_ensembled.pkl', 'rb')
r1 = list(pickle.load(r1).items())

# Our 3DCNN RGB results
# r2 = open('./src/SAMSLR/results/rgb_repeat_last_frame/results_epoch001_0.5989583333333334.pkl', 'rb')
# r2 = list(pickle.load(r2).items())
# Our VTN PF results instead?
r2 = open('./src/Transformer/results/vtnhcpf/results_epoch033_85.pkl', 'rb')
r2 = list(pickle.load(r2).items())
# Our Optical Flow results
# r3 = open('test_flow_color_w_val_finetune.pkl', 'rb')
# r3 = list(pickle.load(r3).items())

# Is this our SSTCN score
r4 = open('./src/SAMSLR/SSTCN/results/T_Pose_model_test.pkl', 'rb')
r4 = list(pickle.load(r4).items())
print(len(r4))


# We can give give these results a bit more choice
alpha = [1,1,1]  # gcn, rgb, flow_color, sstcn valeus, 

right_num = total_num = right_num_5 = 0
names = []
preds = []
scores = []
mean = 0
with open('predictions_poseflow_ensemble.csv', 'w') as f:

    for i in tqdm(range(len(label[0]))):
        name, l = label[:, i]
        names.append(name)
        name1, r11 = r1[i]
        name2, r22 = r2[i]
        # name3, r33 = r3[i]
        name4, r44 = r4[i]
        assert name == name1 == name2 #== name3
        mean += r11.mean()
        score = (r11*alpha[0] + r22*alpha[1] + r44*alpha[2]) / np.array(alpha).sum() # + r33*alpha[2] left out
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