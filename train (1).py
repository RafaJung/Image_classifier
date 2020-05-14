import argparse
import sys
import os
import numpy as np 
import pandas as pd
import torch
from torchvision import datasets, transforms
from torch import nn
from torch import optim
import torch.nn.functional as F
import torchvision.models as models
from collections import OrderedDict
#Main Referrence - https://realpython.com/command-line-interfaces-python-argparse/

#1 - o Primeiro passo já foi feito, que é Importar o argparse
#2 - Create the parser
parser = argparse.ArgumentParser(prog = "train", description = "Training a Neural network")


#Start step every datatype since this error: TypeError: unsupported operand type(s) for +: 'NoneType' and 'str'
parser.add_argument('--data', default = 'flowers',type = str, help = 'Training, testing and Validation directory' )
parser.add_argument('--save_dir', default = 'checkpoint.pth' )
parser.add_argument('--arch',
                    default = 'vgg16',
                    choices=['densenet121','vgg16'],
                    help = 'Select the Pre-trained data Architeture')
parser.add_argument('--learning_rate',default = 0.001, type =int,  help = 'setting learning for the neural network')
parser.add_argument('--hu', default = 512, type = int, help = 'Define de hidden Units, must be lower than 2048')
parser.add_argument('--epochs',default = 3, type = int,  help = 'setting epochs of training')
parser.add_argument('--gpu', default = 'cuda', help = "To turn on GPU" )

args = parser.parse_args()

#Data directory
data_dir = args.data
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

if args.data:
    #Trainning Data
    train_transforms = transforms.Compose([transforms.RandomRotation (30),
                                           transforms.RandomResizedCrop (224),
                                           transforms.RandomHorizontalFlip (),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)

    trainloader = torch.utils.data.DataLoader(train_dataset,
                                              batch_size=64,
                                              shuffle=True)
    
    #Valid Data 
    val_transforms = transforms.Compose([transforms.RandomRotation (30),
                                         transforms.RandomResizedCrop (224),
                                         transforms.RandomHorizontalFlip (),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    val_dataset = datasets.ImageFolder(valid_dir, transform=val_transforms)
    
    valloader = torch.utils.data.DataLoader(val_dataset,
                                            batch_size=64,
                                            shuffle=True)
    
    
    #Test Data 
    test_transforms = transforms.Compose([transforms.RandomRotation (30),
                                          transforms.RandomResizedCrop (224),
                                          transforms.RandomHorizontalFlip (),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transforms)
    
    testloader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=64,
                                             shuffle=True)


#import data Label
import json

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

#Select Gpu device    
if args.gpu: 
    if torch.cuda.is_available():
        
        device = torch.device(args.gpu)
    
    else:
        device = torch.device('cpu')

#The Pre-Trained Models will be Resnet18 and vgg16
#vgg16 structure is in=25088
# optional argumenets = --arch --learning_rate --epochs --hu

#Define the classifier structures
if args.arch:
    
    if args.arch== 'vgg16':
    
        model = models.vgg16(pretrained = True)
    
        #Freeze the parameters of the model
        for param in model.parameters():
            param.requires_grad = False
    
        if args.hu:
            classifier = nn.Sequential(OrderedDict([
                                 ('fc1',nn.Linear(25088, args.hu)),
                                 ('relu',nn.ReLU()),
                                 ('dropout',nn.Dropout(0.2)),
                                 ('fc2',nn.Linear(args.hu, 102)),
                                 ('output',nn.LogSoftmax(dim=1))]))
   
    else:
        
        #inception_v3 structure is in=2048
        args.arch == 'densenet121'
        model = models.densenet121(pretrained = True)
    
        #Freeze the parameters of the model
        for param in model.parameters():
            param.requires_grad = False
    
        if args.hu:
            classifier = nn.Sequential(OrderedDict([
                                    ('fc1',nn.Linear(1024, args.hu)),
                                    ('relu',nn.ReLU()),
                                    ('dropout',nn.Dropout(0.2)),
                                    ('fc2',nn.Linear(args.hu, 102)),
                                    ('output',nn.LogSoftmax(dim=1))]))
    
     
model.classifier = classifier

#Define how to calculate Loss
criterion = nn.NLLLoss()

#Define the optimizer
if args.learning_rate:
    optimizer = optim.Adam(model.classifier.parameters(), lr= args.learning_rate)


model.to(device)


#Training model

if args.epochs:
    
    epochs = args.epochs
    steps = 0
    running_loss = 0
    print_every = 5
    
    for epoch in range(epochs):
        model.train()
        for inputs, labels in trainloader:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)
        
            #Zero parmeters when iterate over
            optimizer.zero_grad()
        
            #Forward pass and calculate loss
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
        
            #Do the backpropagation to calculate the new gradients and update them
            loss.backward()
            optimizer.step()
        
            #Fill train_loss values
            running_loss += loss.item()
        
            if steps % print_every == 0:
                val_loss = 0
                accuracy = 0
                model.eval()
        
                with torch.no_grad():
                    for inputs, labels in valloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                    
                        #Forward pass and calculate loss of validation
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                    
                        #Fill Validation loss
                        val_loss += batch_loss.item()
                    
                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
                        print(f"Epoch {epoch+1}/{epochs}.. "
                              f"Train loss: {running_loss/print_every:.3f}.. "
                              f"Val loss: {val_loss/len(valloader):.3f}.. "
                              f"val accuracy: {accuracy/len(valloader):.3f}")
                        running_loss = 0
                        #turn back to train
                        model.train() 
        
#Testing the network
#I opt to didn't include that on this run since it isn't on rubric : The training loss, validation loss, and validation accuracy are printed out as a network trains

#Save the Model Parameters
model.class_to_idx = train_dataset.class_to_idx

if args.save_dir:
    checkpoint = {"arch": args.arch,
               'classifier': model.classifier,
               'class_to_idx': model.class_to_idx,
               'state_dict': model.state_dict()}
    
    torch.save(checkpoint,args.save_dir)
               
               
   




    