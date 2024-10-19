import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse
import os
import numpy as np
import clip
import wandb
import datetime
import torch.optim as optim

from data import CLIPDataset
from utils import augment_data
from contrastive import ContrastiveModule

def main(args):

    train_dataset = CLIPDataset(args)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)

    date = datetime.datetime.now().strftime("%d-%m-%y_%H:%M")
    wandb.init(
        project='UniMTS',
        name=f"{args.run_tag}_{args.stage}_" + f"{date}" 
    )

    model = ContrastiveModule(args).cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    save_path = './checkpoint/%s/' % args.run_tag
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for epoch in range(args.num_epochs):

        tol_loss = 0
        
        model.train()
        for i, batch in enumerate(train_loader):

            inputs_imu = batch['imu'].float().cuda()
            inputs_text = clip.tokenize(batch['text'], truncate=True).cuda()
            mask = batch['mask'].float().cuda()

            input = inputs_imu * mask

            # rotation invariant
            if args.aug:
                input = augment_data(input)

            if not args.gyro:
                b, t, c = input.shape
                indices = np.array([range(i, i+3) for i in range(0, c, 6)]).flatten()
                input = input[:,:,indices]
        
            b, t, c = input.shape
            if args.stft:
                input_stft = input.permute(0,2,1).reshape(b * c,t)
                input_stft = torch.abs(torch.stft(input_stft, n_fft = 25, hop_length = 28, onesided = False, center = True, return_complex = True))
                input_stft = input_stft.reshape(b, c, input_stft.shape[-2], input_stft.shape[-1]).reshape(b, c, t).permute(0,2,1)
                input = torch.cat((input, input_stft), dim=-1)

            input = input.reshape(b, t, 22, -1).permute(0, 3, 1, 2).unsqueeze(-1)
           
            # IMU and text representations
            logits_per_imu, logits_per_text = model(input, inputs_text)

            # positive keys are the entries on the diagonal
            labels = torch.arange(len(batch['imu'])).cuda()
        
            loss = F.cross_entropy(logits_per_imu / args.temperature, labels, reduction="mean")
           
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tol_loss += len(inputs_imu) * loss.item()
        
            # print(epoch, i, loss.item())
        
        print(f'Epoch [{epoch+1}/{args.num_epochs}], Loss: {tol_loss / len(train_dataset):.4f}')
        wandb.log({'loss': tol_loss / len(train_dataset)})

        if epoch > 0 and epoch % args.log == 0:
            torch.save(model.model.state_dict(), os.path.join(save_path, f'epoch_{epoch}.pth'))
                
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Unified Pre-trained Motion Time Series Model')

    # data
    parser.add_argument('--padding_size', type=int, default='200', help='padding size (default: 200)')
    parser.add_argument('--sample', type=float, default='1', help='pre-training down-sample ratio (default: 1)')
    parser.add_argument('--data_path', type=str, default='./data/', help='/path/to/data/')

    # training
    parser.add_argument('--run_tag', type=str, default='exp0', help='logging tag')
    parser.add_argument('--stage', type=str, default='pretrain', help='training stage')
    parser.add_argument('--num_epochs', type=int, default=100, help='number of pre-training epochs')
    parser.add_argument('--gyro', type=int, default=0, help='using gyro or not')
    parser.add_argument('--stft', type=int, default=0, help='using stft or not')
    parser.add_argument('--aug', type=int, default=1, help='using augmentation or not')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--temperature', type=float, default=0.1, help='temperature')
    parser.add_argument('--log', type=int, default=10, help='logging step')
    
    args = parser.parse_args()

    main(args)