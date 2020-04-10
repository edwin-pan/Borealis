# Main entry point for clasiication 

import torch
import torchvision
import models.CNN as CNN
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler

if __name__ == '__main__':
    # Parameters
    train_test_split_ratio = 0.1
    number_of_iterations = 10
    train_batch_size = 20
    number_of_epochs = 6
    
    # Load our data
    data_filepath = './data/png'
    datasets = torchvision.datasets.ImageFolder(root=data_filepath, transform=torchvision.transforms.ToTensor())
    
    # Split into Train and Test
    num_samples = len(datasets)
    indices = list(range(num_samples))
    split_loc = int(np.floor(num_samples*train_test_split_ratio))
    
    np.random.shuffle(indices)
    
    train_idx, test_idx = indices[split_loc:], indices[:split_loc]
    
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    
    train_loader = torch.utils.data.DataLoader(datasets, batch_size=train_batch_size, num_workers=0, sampler=train_sampler)
    test_loader = torch.utils.data.DataLoader(datasets, batch_size=num_samples, num_workers=0, sampler=test_sampler)

       
    # Train the CNN -- Edwin
    net = CNN.fit(train_loader, number_of_iterations, number_of_epochs)
    
    # Evaluate on Test Set
    for data, label in test_loader:
        out = net(data)
        prediction = torch.argmax(out, axis=1)
        
        accuracy = torch.sum(prediction==label).float()/len(data)
        
        print("Accuracy: ", float(accuracy)*100, "%")