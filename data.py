import torch
import numpy as np
import random
import os
import json
from scipy.signal import resample
import clip
from torch.utils.data import Dataset

class CLIPDataset(Dataset):

    def __init__(self, args):

        imu_dirs = [
            f'{args.data_path}/sim/',
        ]
        text_dirs = [
            f'{args.data_path}/aug_texts/',
        ]
        self.paths = []
        for imu_dir, text_dir in zip(imu_dirs, text_dirs):
            imu_files = [f.split('.')[0] for f in os.listdir(imu_dir) if os.path.isfile(os.path.join(imu_dir, f))]
            text_files = [f.split('.')[0] for f in os.listdir(text_dir) if os.path.isfile(os.path.join(text_dir, f))]
            common_files = [f for f in imu_files if f in text_files]
            for f in common_files:
                self.paths.append((os.path.join(imu_dir, f + '.npy'), os.path.join(text_dir, f + '.txt')))

        self.args = args
        if args.sample < 1:
            self.paths = random.sample(self.paths, int(len(self.paths) * args.sample))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):

        # load imu
        imu_path, text_path = self.paths[idx]
        imu = np.load(imu_path)
        imu[np.isnan(imu)] = 0

        # padding
        if len(imu) < self.args.padding_size:
            imu = np.pad(imu, ((0, self.args.padding_size - len(imu)), (0, 0), (0, 0)), mode='wrap')
        imu = imu[:self.args.padding_size] 

        # random masking
        mask = np.zeros_like(imu)
        k = np.random.randint(1, 6) # randomly select k joints
        selected_joints = np.random.choice(22, k, replace=False)
        mask[:,selected_joints] = 1
        imu = imu.reshape(len(imu), -1)
        mask = mask.reshape(len(mask), -1)
        
        # load text
        with open(text_path, 'r') as file:
            lines = file.readlines()

        text = random.choice(lines).split('#')[0].strip() # remove the comment starting from "#"

        batch = {}
        batch['imu'] = imu
        batch['text'] = text
        batch['mask'] = mask
        
        return batch
    
def select_samples(data, masks, labels, k, name, data_path):
    unique_labels = torch.unique(labels)
    selected_data = []
    selected_masks = []
    selected_labels = []
    all_indices = torch.load(f'{data_path}/few_shot_data_2/{name}_k={k}.pth')

    for i, label in enumerate(unique_labels):
        selected_indices = all_indices[i]
        selected_data.append(data[selected_indices])
        selected_masks.append(masks[selected_indices])
        selected_labels.append(labels[selected_indices])

    selected_data = torch.cat(selected_data, dim=0)
    selected_masks = torch.cat(selected_masks, dim=0)
    selected_labels = torch.cat(selected_labels, dim=0)

    return selected_data, selected_masks, selected_labels

def load(dataset, padding_size, data_path, split='test', k=None):

    print(dataset)

    X = np.load(f'{data_path}/{dataset}/X_{split}.npy')
    real_labels = torch.from_numpy(np.load(f'{data_path}/{dataset}/y_{split}.npy'))
    with open(f'{data_path}/{dataset}/{dataset}.json', 'r') as file:
        data = json.load(file)
    all_X = np.zeros((X.shape[0], X.shape[1], 22, 6))

    if dataset == 'PAMAP':
        all_X[:,:,21] = np.concatenate((X[:,:,0:3], X[:,:,3:6]), axis=-1) 
        all_X[:,:,11] = np.concatenate((X[:,:,18:21], X[:,:,21:24]), axis=-1)
        all_X[:,:,7] = np.concatenate((X[:,:,9:12], X[:,:,12:15]), axis=-1)
        original_sampling_rate = 100
        num_classes = 12

    elif dataset == 'USCHAD':
        all_X[:,:,5] = np.concatenate((X[:,:,0:3] * 9.80665, X[:,:,3:6] / 180 * np.pi), axis=-1)
        original_sampling_rate = 100
        num_classes = 12

    elif dataset == 'UCIHAR':
        all_X[:,:,9] = np.concatenate((X[:,:,6:9] * 9.80665, X[:,:,3:6]), axis=-1) # linear accel, gyro, total accel
        original_sampling_rate = 50
        num_classes = 6

    elif dataset == 'Opp_g':
        all_X[:,:,10] = np.concatenate((X[:,:,0:3] / 1000 * 9.8, X[:,:,3:6] / 1000), axis=-1) # convert unit from milli g to m/s^2
        all_X[:,:,19] = np.concatenate((X[:,:,9:12] / 1000 * 9.8, X[:,:,12:15] / 1000), axis=-1) 
        all_X[:,:,20] = np.concatenate((X[:,:,18:21] / 1000 * 9.8, X[:,:,21:24] / 1000), axis=-1)
        all_X[:,:,15] = np.concatenate((X[:,:,27:30] / 1000 * 9.8, X[:,:,30:33] / 1000), axis=-1)
        all_X[:,:,16] = np.concatenate((X[:,:,36:39] / 1000 * 9.8, X[:,:,39:42] / 1000), axis=-1)
        original_sampling_rate = 30
        num_classes = 4 # locomotion

    elif dataset == 'WISDM':
        all_X[:,:,21] = np.concatenate((X[:,:,0:3], X[:,:,3:6]), axis=-1) 
        original_sampling_rate = 20
        num_classes = 18

    elif dataset == 'DSADS':
        all_X[:,:,11] = np.concatenate((X[:,:,0:3], X[:,:,3:6]), axis=-1) 
        all_X[:,:,21] = np.concatenate((X[:,:,9:12], X[:,:,12:15]), axis=-1) 
        all_X[:,:,17] = np.concatenate((X[:,:,18:21], X[:,:,21:24]), axis=-1)
        all_X[:,:,6] = np.concatenate((X[:,:,27:30], X[:,:,30:33]), axis=-1)
        all_X[:,:,2] = np.concatenate((X[:,:,36:39], X[:,:,39:42]), axis=-1)
        original_sampling_rate = 25
        num_classes = 19

    elif dataset == 'Harth':
        all_X[:,:,9,:3] = X[:,:,:3] * 9.80665
        all_X[:,:,6,:3] = X[:,:,3:6] * 9.80665
        original_sampling_rate = 50
        num_classes = 12

    elif dataset == 'Wharf':
        X = -14.709 + X / 63 * (2 * 14.709)
        all_X[:,:,21,:3] = X
        original_sampling_rate = 32
        num_classes = 14
        
    elif dataset == 'Mhealth':
        all_X[:,:,11,:3] = X[:,:,0:3]
        all_X[:,:,3] = np.concatenate((X[:,:,6:9], X[:,:,9:12] / 180 * np.pi), axis=-1)
        all_X[:,:,21] = np.concatenate((X[:,:,15:18], X[:,:,18:21] / 180 * np.pi), axis=-1)
        original_sampling_rate = 50
        num_classes = 12

    elif dataset == 'UTD-MHAD':
        all_X[real_labels < 21,:,21,:] = np.concatenate((X[real_labels < 21,:,0:3] * 9.80665, X[real_labels < 21,:,3:6] / 180 * np.pi), axis=-1)
        all_X[real_labels >= 21,:,5,:] = np.concatenate((X[real_labels >= 21,:,0:3] * 9.80665, X[real_labels >= 21,:,3:6] / 180 * np.pi), axis=-1)
        original_sampling_rate = 50
        num_classes = 27

    elif dataset == 'MotionSense':
        all_X[:,:,5] = np.concatenate((X[:,:,:3] * 9.80665, X[:,:,3:6]), axis=-1)
        all_X[:,:,1] = np.concatenate((X[:,:,:3] * 9.80665, X[:,:,3:6]), axis=-1)
        original_sampling_rate = 50
        num_classes = 6

    elif dataset == 'w-HAR':
        all_X[:,:,7] = np.concatenate((X[:,:,:3] * 9.80665, X[:,:,3:6] / 180 * np.pi), axis=-1)
        original_sampling_rate = 250
        num_classes = 7

    elif dataset == 'Shoaib':
        all_X[:,:,1] = X[:,:,:6]
        all_X[:,:,5] = X[:,:,6:12]
        all_X[:,:,21] = X[:,:,12:18]
        all_X[:,:,20] = X[:,:,18:24]
        all_X[:,:,0] = X[:,:,24:30]
        original_sampling_rate = 50
        num_classes = 7
    
    elif dataset == 'har70plus':
        all_X[:,:,0,:3] = X[:,:,:3] * 9.80665
        all_X[:,:,5,:3] = X[:,:,3:6] * 9.80665
        original_sampling_rate = 50
        num_classes = 7

    elif dataset == 'MMAct':
        all_X[:,:,5] = np.concatenate((X[:,:,:3], X[:,:,3:6]), axis=-1)
        all_X[:,:,21,:3] = X[:,:,6:9]
        original_sampling_rate = 50
        num_classes = 35
    
    elif dataset == 'realworld':
        all_X[:,:,14] = np.concatenate((X[:,:,:3], X[:,:,3:6]), axis=-1)
        all_X[:,:,16] = np.concatenate((X[:,:,6:9], X[:,:,9:12]), axis=-1)
        all_X[:,:,13] = np.concatenate((X[:,:,12:15], X[:,:,15:18]), axis=-1)
        all_X[:,:,3] = np.concatenate((X[:,:,18:21], X[:,:,21:24]), axis=-1)
        all_X[:,:,1] = np.concatenate((X[:,:,24:27], X[:,:,27:30]), axis=-1)
        all_X[:,:,15] = np.concatenate((X[:,:,30:33], X[:,:,33:36]), axis=-1)
        all_X[:,:,9] = np.concatenate((X[:,:,36:39], X[:,:,39:42]), axis=-1)
        original_sampling_rate = 50
        num_classes = 8
    
    elif dataset == 'TNDA-HAR':
        all_X[:,:,20] = np.concatenate((X[:,:,:3], X[:,:,3:6]), axis=-1)
        all_X[:,:,2] = np.concatenate((X[:,:,6:9], X[:,:,9:12]), axis=-1)
        all_X[:,:,21] = np.concatenate((X[:,:,12:15], X[:,:,15:18]), axis=-1)
        all_X[:,:,3] = np.concatenate((X[:,:,18:21], X[:,:,21:24]), axis=-1)
        all_X[:,:,11] = np.concatenate((X[:,:,24:27], X[:,:,27:30]), axis=-1)
        original_sampling_rate = 50
        num_classes = 8
    
    elif dataset == 'ut-complex':
        all_X[:,:,5] = np.concatenate((X[:,:,:3], X[:,:,3:6]), axis=-1)
        all_X[:,:,21] = np.concatenate((X[:,:,6:9], X[:,:,9:12]), axis=-1)
        original_sampling_rate = 50
        num_classes = 13
    
    all_X = all_X.reshape(all_X.shape[0], all_X.shape[1], 22 * 6)

    # resample real data to 20 Hz
    new_sampling_rate = 20
    new_length = int((all_X.shape[1] / original_sampling_rate) * new_sampling_rate)
    resampled_data = np.array([resample(sequence, new_length) for sequence in all_X])

    # pad real data to args.padding_size
    masks = np.ones_like(resampled_data)
    if resampled_data.shape[1] < padding_size:
        resampled_data = np.pad(resampled_data, ((0, 0), (0, padding_size - resampled_data.shape[1]), (0, 0)), 'wrap') # N, 200, 6
        masks = np.pad(masks, ((0, 0), (0, padding_size - masks.shape[1]), (0, 0)), 'constant') # N, 200, 6
    real_inputs = torch.from_numpy(resampled_data[:,:padding_size,:]).float() 
    real_masks = torch.from_numpy(masks[:,:padding_size,:]).float() 

    if split == 'train' and k and k < len(real_inputs):
        real_inputs, real_masks, real_labels = select_samples(real_inputs, real_masks, real_labels, k, dataset, data_path)
    print(real_inputs.shape, real_labels.shape)
 
    # load text
    label_dictionary = data['label_dictionary']
    label_list = [' '.join(labels) for labels in label_dictionary.values()]
    all_text = clip.tokenize(label_list).cuda()
    
    return real_inputs, real_masks, real_labels, label_list, all_text, num_classes

def load_multiple(dataset_list, padding_size, data_path, split='test', k=None):

    real_inputs_list, real_masks_list, real_labels_list, label_list_list, all_text_list, num_classes_list = [], [], [], [], [], []
    for dataset in dataset_list:
        real_inputs, real_masks, real_labels, label_list, all_text, num_classes = load(dataset, padding_size, data_path, split, k)
        real_inputs_list.append(real_inputs)
        real_masks_list.append(real_masks)
        real_labels_list.append(real_labels)
        label_list_list.append(label_list)
        all_text_list.append(all_text)
        num_classes_list.append(num_classes)

    return real_inputs_list, real_masks_list, real_labels_list, label_list_list, all_text_list, num_classes_list

def load_custom_data(X_path, y_path, config_path, joint_list, original_sampling_rate, padding_size=200, split='test', k=None, few_shot_path=None):

    X = np.load(X_path)
    real_labels = torch.from_numpy(np.load(y_path))
    with open(config_path, 'r') as file:
        data = json.load(file)
    all_X = np.zeros((X.shape[0], X.shape[1], 22, 6))

    for i, joint in enumerate(joint_list):
        all_X[:,:,joint] = np.concatenate((X[:,:,6*i:6*i+3], X[:,:,6*i+3:6*i+6]), axis=-1)

    all_X = all_X.reshape(all_X.shape[0], all_X.shape[1], 22 * 6)

    # resample real data to 20 Hz
    new_sampling_rate = 20
    new_length = int((all_X.shape[1] / original_sampling_rate) * new_sampling_rate)
    resampled_data = np.array([resample(sequence, new_length) for sequence in all_X])

    # pad real data to args.padding_size
    masks = np.ones_like(resampled_data)
    if resampled_data.shape[1] < padding_size:
        resampled_data = np.pad(resampled_data, ((0, 0), (0, padding_size - resampled_data.shape[1]), (0, 0)), 'wrap') # N, 200, 6
        masks = np.pad(masks, ((0, 0), (0, padding_size - masks.shape[1]), (0, 0)), 'constant') # N, 200, 6
    real_inputs = torch.from_numpy(resampled_data[:,:padding_size,:]).float() 
    real_masks = torch.from_numpy(masks[:,:padding_size,:]).float() 

    if split == 'train' and k and k < len(real_inputs):

        unique_labels = torch.unique(real_labels)

        if few_shot_path is None:
            print('Generating few shot indices ...')
            all_indices = []
            for i, label in enumerate(unique_labels):
                indices = torch.where(real_labels == label)[0]
                selected_indices = indices[torch.randperm(len(indices))[:k]]
                all_indices.append(selected_indices)
        else:
            print('Loading existing few shot indices ...')
            all_indices = torch.load(few_shot_path)
    
        selected_data = []
        selected_masks = []
        selected_labels = []
        for i, label in enumerate(unique_labels):
            selected_indices = all_indices[i]
            selected_data.append(real_inputs[selected_indices])
            selected_masks.append(real_masks[selected_indices])
            selected_labels.append(real_labels[selected_indices])
        selected_data = torch.cat(selected_data, dim=0)
        selected_masks = torch.cat(selected_masks, dim=0)
        selected_labels = torch.cat(selected_labels, dim=0)
        real_inputs, real_masks, real_labels = selected_data, selected_masks, selected_labels

    print(real_inputs.shape, real_labels.shape)
 
    # load text
    label_dictionary = data['label_dictionary']
    label_list = [' '.join(labels) for labels in label_dictionary.values()]
    all_text = clip.tokenize(label_list).cuda()
    
    return real_inputs, real_masks, real_labels, label_list, all_text