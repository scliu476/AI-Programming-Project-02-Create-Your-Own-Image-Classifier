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
    parser = argparse.ArgumentParser(description="parser for train.py")
    
    parser.add_argument("data_dir", type=str, help="Directory with train, validation, and test data")
    parser.add_argument("--save_dir", type=str, default="checkpoint_part_2.pth", help="Directory to save/load trained model")
    parser.add_argument("--arch", type=str, default="vgg16", choices=["vgg16", "densenet121"], help="Pretrained model for transfer learning")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--hidden_units", type=int, default=512, help="Number of hidden units")
    parser.add_argument("--epochs", type=int, default=3, help="Numer of epochs")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to train on")
    
    return parser.parse_args()
    
#-----------------------------------------------------------------------------------------------------------------------------------------------
def main():
    args = arg_parse()
    
    train_dir = args.data_dir + '/train'
    valid_dir = args.data_dir + '/valid'
    
    # Define your transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(255),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    # Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)

    # Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)

    # Label mapping
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    
    cat_num = len(cat_to_name)
    
    # Device to train on
    device = args.device
    
    # Pretrained model
    model = getattr(models, args.arch)(pretrained=True)
    
    # Freeze model parameters when training; turn off gradients for our model
    for param in model.parameters():
        param.requires_grad = False
        
    # Define new classifier
    if args.arch == "vgg16":
        input_num = 25088
    elif args.arch == "densenet121":
        input_num = 1024    
    
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(input_num, args.hidden_units)),
        ('act', nn.ReLU()),
        ('drop', nn.Dropout(p=0.2)),
        ('fc2', nn.Linear(args.hidden_units, cat_num)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

    model.classifier = classifier
    
    criterion = nn.NLLLoss()

    learning_rate = args.learning_rate
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    model.to(device)
    
    # Train model
    epochs = args.epochs
    steps = 0
    running_loss = 0
    print_every = 5

    for epoch in range(epochs):
            
        # Training loop
        for images, labels in trainloader:
            steps += 1
        
            # Move images & labels to GPU if available
            images, labels = images.to(device), labels.to(device)
        
            # Set gradients to zero
            optimizer.zero_grad()
        
            # Feedforward
            logps = model(images)
            loss = criterion(logps, labels)
        
            # Backpropagation
            loss.backward()
        
            # Gradient descent
            optimizer.step()
        
            running_loss += loss.item()
        
            if steps % print_every == 0:
            
                # Turn on evaluation, inference mode; turn off dropout
                model.eval()
                valid_loss = 0
                accuracy = 0
            
                # Turn off autograd
                with torch.no_grad():

                    # Validation loop
                    for images, labels in validloader:
                
                        # Move images & labels to GPU if available
                        images, labels = images.to(device), labels.to(device)
        
                        logps = model(images)
                        batch_loss = criterion(logps, labels)
                        valid_loss += batch_loss.item()
                
                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_ps, top_class = ps.topk(1, dim=1)
                        equality = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equality.type(torch.FloatTensor)).item()
            
                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Validation loss: {valid_loss/len(validloader):.3f}.. "
                      f"Validation accuracy: {accuracy/len(validloader):.3f}.. ") 
            
                running_loss = 0
            
                # Set model back to training mode
                model.train()   

    # Save the checkpoint
    model.class_to_idx = train_data.class_to_idx

    checkpoint = {'input_size': input_num,
                  'output_size': cat_num,
                  'pretrained_model': getattr(models, args.arch)(pretrained=True),
                  'classifier': model.classifier,
                  'state_dict': model.state_dict(),
                  'optimizer': optimizer,
                  'learning_rate': learning_rate,
                  'epochs': epochs,
                  'class_to_idx': model.class_to_idx
                  }

    torch.save(checkpoint, args.save_dir)
    
    print("\nTraining process is completed!")
    print("Checkpoint is saved at: {}".format(args.save_dir))
    print()

#-----------------------------------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()