import numpy as np
import torch
import torch.nn.functional as F

import argparse
import os
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import wandb
import datetime
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim

from data import load_multiple, load_custom_data
from utils import compute_metrics_np
from contrastive import ContrastiveModule

def main(args):
    
    train_inputs, train_masks, train_labels, _, _ = load_custom_data(
        args.X_train_path, args.y_train_path, args.config_path, args.joint_list, args.original_sampling_rate, padding_size=args.padding_size, split='train', k=args.k, few_shot_path=None
    )
    train_dataset = TensorDataset(train_inputs, train_masks, train_labels)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    test_inputs, test_masks, test_labels, _, _ = load_custom_data(
        args.X_test_path, args.y_test_path, args.config_path, args.joint_list, args.original_sampling_rate, padding_size=args.padding_size, split='test'
    )
    test_dataset = TensorDataset(test_inputs, test_masks, test_labels)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    date = datetime.datetime.now().strftime("%d-%m-%y_%H:%M")
    wandb.init(
        project='UniMTS',
        name=f"{args.run_tag}_{args.stage}_{args.mode}_k={args.k}_" + f"{date}" 
    )

    save_path = './checkpoint/%s/' % args.run_tag

    model = ContrastiveModule(args).cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    if args.mode == 'full' or args.mode == 'probe':
        model.model.load_state_dict(torch.load(f'{args.checkpoint}'))
    if args.mode == 'probe':
        for name, param in model.model.named_parameters():
            param.requires_grad = False
    
    best_loss = None
    for epoch in range(args.num_epochs):

        tol_loss = 0
        
        model.train()
        for i, (input, mask, label) in enumerate(train_dataloader):

            input = input.cuda()
            labels = label.cuda()

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
            
            output = model.classifier(input)
            
            loss = F.cross_entropy(output.float(), labels.long(), reduction="mean")
        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tol_loss += len(input) * loss.item()
        
            # print(epoch, i, loss.item())
        
        print(f'Epoch [{epoch+1}/{args.num_epochs}], Loss: {tol_loss / len(train_dataset):.4f}')
        wandb.log({' loss': tol_loss / len(train_dataset)})

        if best_loss is None or tol_loss < best_loss:
            best_loss = tol_loss
            torch.save(model.state_dict(), os.path.join(save_path, f'k={args.k}_best_loss.pth'))

    # evaluation
    model.load_state_dict(torch.load(os.path.join(save_path, f'k={args.k}_best_loss.pth')))
    model.eval()
    with torch.no_grad():

        pred_whole, logits_whole = [], []
        for input, mask, label in test_dataloader:
            
            input = input.cuda()
            label = label.cuda()

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

            logits_per_imu = model.classifier(input)
            logits_whole.append(logits_per_imu)
            
            pred = torch.argmax(logits_per_imu, dim=-1).detach().cpu().numpy()
            pred_whole.append(pred)

        pred = np.concatenate(pred_whole)
        acc = accuracy_score(test_labels, pred)
        prec = precision_score(test_labels, pred, average='macro')
        rec = recall_score(test_labels, pred, average='macro')
        f1 = f1_score(test_labels, pred, average='macro')

        print(f"acc: {acc}, prec: {prec}, rec: {rec}, f1: {f1}")
        wandb.log({f"acc": acc, f"prec": prec, f"rec": rec, f"f1": f1})

        logits_whole = torch.cat(logits_whole)
        r_at_1, r_at_2, r_at_3, r_at_4, r_at_5, mrr_score = compute_metrics_np(logits_whole.detach().cpu().numpy(), test_labels.numpy())
            
        print(f"R@1: {r_at_1}, R@2: {r_at_2}, R@3: {r_at_3}, R@4: {r_at_4}, R@5: {r_at_5}, MRR: {mrr_score}")
        wandb.log({f"R@1": r_at_1, f"R@2": r_at_2, f"R@3": r_at_3, f"R@4": r_at_4, f"R@5": r_at_5, f"MRR": mrr_score}) 


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Unified Pre-trained Motion Time Series Model')

    # model 
    parser.add_argument('--mode', type=str, default='full', choices=['random','probe','full'], help='full fine-tuning, linear probe, random init')

    # data
    parser.add_argument('--padding_size', type=int, default='200', help='padding size (default: 200)')
    parser.add_argument('--k', type=int, help='few shot samples per class (default: None)')
    parser.add_argument('--X_train_path', type=str, required=True, help='/path/to/train/data/')
    parser.add_argument('--X_test_path', type=str, required=True, help='/path/to/test/data/')
    parser.add_argument('--y_train_path', type=str, required=True, help='/path/to/train/label/')
    parser.add_argument('--y_test_path', type=str, required=True, help='/path/to/test/label/')
    parser.add_argument('--config_path', type=str, required=True, help='/path/to/config/')
    parser.add_argument('--few_shot_path', type=str, help='/path/to/few/shot/indices/')
    parser.add_argument('--joint_list', nargs='+', type=int, required=True, help='List of joint indices')
    parser.add_argument('--original_sampling_rate', type=int, required=True, help='original sampling rate')
    parser.add_argument('--num_class', type=int, required=True, help='number of classes')

    # training
    parser.add_argument('--stage', type=str, default='finetune', help='training stage')
    parser.add_argument('--num_epochs', type=int, default=200, help='number of fine-tuning epochs (default: 200)')
    parser.add_argument('--run_tag', type=str, default='exp0', help='logging tag')
    parser.add_argument('--gyro', type=int, default=0, help='using gyro or not')
    parser.add_argument('--stft', type=int, default=0, help='using stft or not')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')

    parser.add_argument('--checkpoint', type=str, default='./checkpoint/', help='/path/to/checkpoint/')
    
    args = parser.parse_args()

    main(args)
