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
import json
from PIL import Image

parser = argparse.ArgumentParser(prog = "predict", description = "Predict Flowers labels")

parser.add_argument('--input_checkpoint', default = 'checkpoint.pth',type = str, help = 'Set the trained model directory' )
parser.add_argument('--top_k', default = 5 ,type = int, help= 'Return K most likely classes')
parser.add_argument('--image',default = './flowers/test/100/image_07939.jpg', type = str,  help = 'select the image to predict')
parser.add_argument('--gpu', default = 'cuda', help = "To turn on GPU" )
parser.add_argument('--cat_name', default = 'cat_to_name.json',type = str, help = "select the cattegory features dictionary names" )
args = parser.parse_args()


with open(args.cat_name, 'r') as f:
    cat_to_name = json.load(f)
    
#Select Gpu device    
if args.gpu: 
    if torch.cuda.is_available():
        
        device = torch.device(args.gpu)
    
    else:
        device = torch.device('cpu')
        
        
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    
    if checkpoint['arch'] == 'vgg16':
        model = models.vgg16(pretrained=True)
    else:
        model = models.densenet121(pretrained=True)
    
    #Network Structure
    model.classifier = checkpoint['classifier']
    
    
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    
    return model    

model = load_checkpoint(args.input_checkpoint)


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
        
    '''
    image = Image.open(image)
    
    # TODO: Process a PIL image for use in a PyTorch model
    preprocessing_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop (224),
                                      transforms.RandomHorizontalFlip (),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    
    image = preprocessing_transforms(image)
    
    return image



def predict(image_path, model, k):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # TODO: Implement the code to predict the class from an image file
    model.to(device)
    
    #Processing the image
    image = process_image(image_path)
    
    #correct the format
    image = torch.unsqueeze(image, dim=0)
    #CORRECT That Error RuntimeError: Expected object of type torch.FloatTensor but found type torch.cuda.FloatTensor for argument #2 'weight'
    image = image.float().cuda()
    
    model.eval()
    # Turn off gradients to speed up this part
    with torch.no_grad():
        logits = model.forward(image)
    
    # Use softmax to get the probabilities 
    softmax = F.softmax(logits.data, dim=1)
    
    #Get the top k probabilities
    ps_k, labels_k = softmax.topk(k)
    
    #Solving this problem: TypeError: unhashable type: 'list' 
    #since they are a tensor, i've change to list
    ps_k = ps_k.detach().tolist()
    labels_k = labels_k.detach().tolist()
    
    #convert the indices using class_to_idx and cat_to_nome since it's the true name
    df = pd.DataFrame({'class':pd.Series(model.class_to_idx),'flower':pd.Series(cat_to_name)})
    
        
    #Error: Length of values does not match length of index
    df = df.set_index('class')
    
    df = df.iloc[labels_k[0]]
    
    #Combine Probabilities to class
    df['probabilities'] = ps_k[0]
    
    return df


#Testing function
df = predict(args.image,model, k= args.top_k)
print(df.head(5))  
  