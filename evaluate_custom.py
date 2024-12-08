import numpy as np
import torch
import argparse
import os
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import wandb
import datetime
from torch.utils.data import DataLoader, TensorDataset
from huggingface_hub import hf_hub_download
import zipfile

from data import load, load_multiple, load_custom_data
from utils import compute_metrics_np
from contrastive import ContrastiveModule

def main(args):

    repo_id = "xiyuanz/UniMTS"
    checkpoint_file = "checkpoint/UniMTS.pth"
    config_file = "config.json"
    data_file = "UniMTS_data.zip"

    if not os.path.exists("checkpoint"):
        hf_hub_download(repo_id=repo_id, filename=checkpoint_file, local_dir="./")
    hf_hub_download(repo_id=repo_id, filename=config_file, local_dir="./")
    if not os.path.exists("UniMTS_data"):
        hf_hub_download(repo_id=repo_id, filename=data_file, local_dir="./")
        with zipfile.ZipFile("UniMTS_data.zip", 'r') as zip_ref:
            zip_ref.extractall("./")

    # load real data
    
    real_inputs, real_masks, real_labels, label_list, all_text = load_custom_data(
        args.X_path, args.y_path, args.config_path, args.joint_list, args.original_sampling_rate, padding_size=args.padding_size, split='test'
    )
    real_dataset = TensorDataset(real_inputs, real_masks, real_labels)
    test_real_dataloader = DataLoader(real_dataset, batch_size=args.batch_size, shuffle=False)

    date = datetime.datetime.now().strftime("%d-%m-%y_%H:%M")
    wandb.init(
        project='UniMTS',
        name=f"{args.run_tag}_{args.stage}_" + f"{date}" 
    )

    model = ContrastiveModule(args).cuda()

    model.model.load_state_dict(torch.load(f'{args.checkpoint}'))
        
    model.eval()
    with torch.no_grad():
        pred_whole, logits_whole = [], []
        for input, mask, label in test_real_dataloader:
            
            input = input.cuda()
            mask = mask.cuda()
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
            
            logits_per_imu, logits_per_text = model(input, all_text)
            logits_whole.append(logits_per_imu)
            
            pred = torch.argmax(logits_per_imu, dim=-1).detach().cpu().numpy()
            pred_whole.append(pred)

        pred = np.concatenate(pred_whole)
        acc = accuracy_score(real_labels, pred)
        prec = precision_score(real_labels, pred, average='macro')
        rec = recall_score(real_labels, pred, average='macro')
        f1 = f1_score(real_labels, pred, average='macro')

        print(f"acc: {acc}, prec: {prec}, rec: {rec}, f1: {f1}")
        wandb.log({f"acc": acc, f"prec": prec, f"rec": rec, f"f1": f1})

        logits_whole = torch.cat(logits_whole)
        r_at_1, r_at_2, r_at_3, r_at_4, r_at_5, mrr_score = compute_metrics_np(logits_whole.detach().cpu().numpy(), real_labels.numpy())
        
        print(f"R@1: {r_at_1}, R@2: {r_at_2}, R@3: {r_at_3}, R@4: {r_at_4}, R@5: {r_at_5}, MRR: {mrr_score}")
        wandb.log({f"R@1": r_at_1, f"R@2": r_at_2, f"R@3": r_at_3, f"R@4": r_at_4, f"R@5": r_at_5, f"MRR": mrr_score})
            
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Unified Pre-trained Motion Time Series Model')

    # data
    parser.add_argument('--padding_size', type=int, default='200', help='padding size (default: 200)')
    parser.add_argument('--X_path', type=str, required=True, help='/path/to/data/')
    parser.add_argument('--y_path', type=str, required=True, help='/path/to/label/')
    parser.add_argument('--config_path', type=str, required=True, help='/path/to/config/')
    parser.add_argument('--joint_list', nargs='+', type=int, required=True, help='List of joint indices')
    parser.add_argument('--original_sampling_rate', type=int, required=True, help='original sampling rate')

    # training
    parser.add_argument('--run_tag', type=str, default='exp0', help='logging tag')
    parser.add_argument('--stage', type=str, default='evaluation', help='training or evaluation stage')
    parser.add_argument('--gyro', type=int, default=0, help='using gyro or not')
    parser.add_argument('--stft', type=int, default=0, help='using stft or not')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')

    parser.add_argument('--checkpoint', type=str, default='./checkpoint/UniMTS.pth', help='/path/to/checkpoint/')
    
    args = parser.parse_args()

    main(args)