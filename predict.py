import argparse
import json

import numpy as np

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

from collections import OrderedDict

from PIL import Image

torch.manual_seed(47)

#-----------------------------------------------------------------------------------------------------------------------------------------------
def arg_parse():
    parser = argparse.ArgumentParser(description="parser for predict.py")
    
    parser.add_argument("img_path", type=str, help="Testing image path")
    parser.add_argument("checkpoint", type=str, help="Saved trained model checkpoint")
    parser.add_argument("--top_k", type=int, default=5, help="Top K most likely classes")
    parser.add_argument("--category_names", type=str, default="cat_to_name.json", help="JSPN object to map category label to name")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to train on")
    
    return parser.parse_args()

#-----------------------------------------------------------------------------------------------------------------------------------------------
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    
    model = checkpoint['pretrained_model']   
    
    for param in model.parameters():
        param.requires_grad = False
            
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    
    model.optimizer = checkpoint['optimizer']
    model.learning_rate = checkpoint['learning_rate']
    model.epochs = checkpoint['epochs']
    model.class_to_idx = checkpoint['class_to_idx']    
    
    return model

#-----------------------------------------------------------------------------------------------------------------------------------------------
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # Process a PIL image for use in a PyTorch model
    # Get image original sizes
    w, h = image.size
    
    # Resize image
    if w > h:
        ratio = w/h
        image.thumbnail((w*ratio, 256))
    else:
        ratio = h/w
        image.thumbnail((256, h*ratio))
    
    # Crop image
    w, h = image.size
    
    left = (w-224)/2
    upper = (h-224)/2
    right = left + 224
    lower = upper + 224
    
    image = image.crop((left, upper, right, lower))
    
    # Change color channels to be between 0 and 1
    image = np.array(image)
    image = image/225.
    
    # Normalize image
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    image = (image - mean)/std
    
    # Reorder NumPy and PIL
    image = np.transpose(image, (2, 0, 1))
    
    return image

#-----------------------------------------------------------------------------------------------------------------------------------------------
def predict(device, image_path, model, topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # Implement the code to predict the class from an image file
    model.to(device)
    model.eval()

    image = Image.open(image_path)
    image = process_image(image)
    image = torch.from_numpy(image)
    image = image.float()
    image.unsqueeze_(0)
    
    image = image.to(device)
    
    # Turn off autograd
    with torch.no_grad():
        
        # Feedworwad
        logps = model(image)
                
        # Calculate probability
        ps = torch.exp(logps)
        top_ps, top_class = ps.topk(topk, dim=1)
        
        top_ps = np.array(top_ps[0])
        top_class = np.array(top_class[0])
    
    model.train()
    
    # Convert indices to actual category names
    index_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_class = [index_to_class[idx] for idx in top_class]
    
    return top_ps, top_class

#-----------------------------------------------------------------------------------------------------------------------------------------------
def main():
    args = arg_parse()
    
    # Label mapping
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
    
    # Load the checkpoint
    model = load_checkpoint(args.checkpoint)
    
    # Class prediction
    device = args.device
    image_path = args.img_path
    top_k = args.top_k
    
    probs, classes = predict(device, image_path, model, top_k)
    names = [cat_to_name[str(idx)] for idx in classes]
    
    # Print
    print("\nTop {} most likely classes are...".format(top_k))
    print("Names: ", names)
    print("Probabilities: ", probs)
    print()
    
#-----------------------------------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()