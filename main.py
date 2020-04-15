# Main entry point for clasiication 

import torch
import torchvision
import models.CNN as CNN
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import os

if __name__ == '__main__':
    
    train_flag = False
    
    # Parameters
    train_test_split_ratio = 0.5
    number_of_iterations = 10
    train_batch_size = 20
    number_of_epochs = 2 # 6
    
    # Load our data
    print("[Status] Loading Data")
    data_filepath = './data/png'
    datasets = torchvision.datasets.ImageFolder(root=data_filepath, transform=torchvision.transforms.ToTensor())
    
    # Split into Train and Test
    print("[Status] Splitting Train/Test")
    num_samples = len(datasets)
    indices = list(range(num_samples))
    split_loc = int(np.floor(num_samples*train_test_split_ratio))
    print("[Info] Testing on ", split_loc, " images")
    
    np.random.shuffle(indices)
    
    train_idx, test_idx = indices[split_loc:], indices[:split_loc]
    
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    
    train_loader = torch.utils.data.DataLoader(datasets, batch_size=train_batch_size, num_workers=0, sampler=train_sampler)
    test_loader = torch.utils.data.DataLoader(datasets, batch_size=num_samples, num_workers=0, sampler=test_sampler)
    

    if train_flag:
        # Train the CNN -- Edwin
        print("[Status] Train CNN")
        net = CNN.fit(train_loader, number_of_iterations, number_of_epochs)
    else:
        repo_dir = os.path.abspath(os.getcwd())
        model_dir = os.path.join(repo_dir, 'checkpoints\\net1.model')
        print("[Status] Load CNN @ ", model_dir)
        
        # Load the CNN -- Edwin 
        net = torch.load(model_dir)
        net.eval()
    
    print("[Status] Evaluate CNN")
    # Evaluate on Test Set
    for data, label in test_loader:
        out = net(data)
        prediction = torch.argmax(out, axis=1)
        
        accuracy = torch.sum(prediction==label).float()/len(data)
        
        print("Accuracy: ", float(accuracy)*100, "%")