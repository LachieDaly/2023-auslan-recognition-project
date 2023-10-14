import torch
import os
import csv
from T_Pose_model import *
from torch.autograd import Variable
import torch.nn.parallel
import argparse
import pickle
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="./Data/ELAR/skeleton_features/train", help="Path to input dataset")
    parser.add_argument("--checkpoint_model", type=str, default="./src/SAMSLR/SSTCN/save-models/T_Pose_model_244_85.41666666666667.pth", help="Optional path to checkpoint model")

    opt = parser.parse_args()
    print(opt)
    test_csv = './Data/ELAR/train_val_labels.csv'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = T_Pose_model(frames_number=60, joints_number=33, n_classes=29)
    #model = nn.DataParallel(model)    
    model = model.to(device)
    
    # Add weights from checkpoint model if specified
    if opt.checkpoint_model:
        model.load_state_dict(torch.load(opt.checkpoint_model, map_location='cuda:0'))#,strict=False)
    else:
        model.init_weights()
    model.eval()
    preds = []
    names = []
    index = 0
    with open(test_csv) as label_file:
        reader = csv.reader(label_file)
        for row in reader:
            if row[2] == "train":
                continue

            names.append(row[0])
            fea_name = row[0] + '.pt'
            fea_path = os.path.join(opt.dataset_path, fea_name)
            data = torch.load(fea_path)
            data = data.contiguous().view(1,-1,24,24)
            data_in = Variable(data.to(device), requires_grad=False)
            with torch.no_grad():
                pred = model(data_in)
            pred = pred.cpu().detach().numpy()
            preds.append(pred)
    with open('./src/SAMSLR/SSTCN/results/T_Pose_model_test.pkl', 'wb') as f:
         score_dict = dict(zip(names, preds))
         pickle.dump(score_dict, f)

# src/SAMSLR/SSTCN/test.py