if __name__ == '__main__':
    import os
    from datetime import datetime
    import logging
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from torch.utils.tensorboard import SummaryWriter
    import torchvision.transforms as transforms
    from models.Conv3D import r2plus1d_18
    from dataset_sign_clip import Sign_Isolated
    from train import train_epoch
    from validation_clip import val_epoch

    class LabelSmoothingCrossEntropy(nn.Module):
        """
        Label Smoothing supposedly improved the validation accuracy of 
        the SAM-SLR model
        """
        def __init__(self):
            super(LabelSmoothingCrossEntropy, self).__init__()
        def forward(self, x, target, smoothing=0.1):
            confidence = 1. - smoothing
            logprobs = F.log_softmax(x, dim=-1)
            nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
            nll_loss = nll_loss.squeeze(1)
            smooth_loss = -logprobs.mean(dim=-1)
            loss = confidence * nll_loss + smoothing * smooth_loss
            return loss.mean()

    # Path setting
    # Path setting
    exp_name = 'rgb_final_middle_finetune'
    data_path = "./Data/ELAR/sam_frames_crop/train"
    data_path2 = "./Data/ELAR/sam_frames_crop/train"
    label_train_path = "./Data/ELAR/avi/train_val_labels.csv"
    label_val_path = "./Data/ELAR/avi/train_val_labels.csv"
    model_path = "./src/SAMSLR/checkpoints/{}".format(exp_name)
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    if not os.path.exists(os.path.join('./src/SAMSLR/results', exp_name)):
        os.mkdir(os.path.join('./src/SAMSLR/results', exp_name))
    log_path = "./src/SAMSLR/results/log/sign_resnet2d+1_{}_{:%Y-%m-%d_%H-%M-%S}.log".format(exp_name, datetime.now())
    sum_path = "./src/SAMSLR/runs/sign_resnet2d+1_{}_{:%Y-%m-%d_%H-%M-%S}".format(exp_name, datetime.now())
    phase = 'Train'
    # Log to file & tensorboard writer
    logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=[logging.FileHandler(log_path), logging.StreamHandler()])
    logger = logging.getLogger('SLR')
    logger.info('Logging to file...')
    writer = SummaryWriter(sum_path)

    # Use specific gpus
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    # Device setting
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparams
    num_classes = 226 
    epochs = 100
    batch_size = 4
    learning_rate = 1e-3#1e-3 Train 1e-4 Finetune
    log_interval = 80
    sample_size = 128
    sample_duration = 16
    attention = False
    drop_p = 0.0
    hidden1, hidden2 = 512, 256

    def get_lr(optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

# Train with 3DCNN
    # Load data
    transform = transforms.Compose([transforms.Resize([sample_size, sample_size]),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5], std=[0.5])])
    
    train_set = Sign_Isolated(data_path=data_path, label_path=label_train_path, 
                              frames=sample_duration, num_classes=num_classes, 
                              train=True, transform=transform)
    
    val_set = Sign_Isolated(data_path=data_path2, label_path=label_val_path, 
                            frames=sample_duration, num_classes=num_classes, 
                            train=False, transform=transform)
    
    logger.info("Dataset samples: {}".format(len(train_set) + len(val_set)))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    # Create model
    model = r2plus1d_18(pretrained=False, num_classes=29)
    # load pretrained
    checkpoint = torch.load('./src/SAMSLR/checkpoints/rgb_repeat_last_frame/sign_resnet2d+1_epoch100.pth')
    for key in checkpoint:
        print(key)

    model.load_state_dict(checkpoint)
    print(model)

    model = model.to(device)
    # Run the model parallelly
    # Create loss criterion & optimizer
    criterion = LabelSmoothingCrossEntropy()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, threshold=0.0001)

    # Start training
    if phase == 'Train':
        logger.info("Training Started".center(60, '#'))
        for epoch in range(epochs):
            print('lr: ', get_lr(optimizer))
            # Train the model
            train_epoch(model, criterion, optimizer, train_loader, device, epoch, logger, log_interval, writer)

            # Validate the model
            val_loss = val_epoch(model, criterion, val_loader, device, epoch, logger, writer)
            scheduler.step(val_loss)
            
            # Save model
            torch.save(model.state_dict(), os.path.join(model_path, "sign_resnet2d+1_epoch{:03d}.pth".format(epoch+1)))
            logger.info("Epoch {} Model Saved".format(epoch+1).center(60, '#'))
    elif phase == 'Test':
        logger.info("Testing Started".center(60, '#'))
        val_loss = val_epoch(model, criterion, val_loader, device, 0, logger, writer, phase=phase, exp_name=exp_name)

    logger.info("Finished".center(60, '#'))