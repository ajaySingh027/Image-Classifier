# #!/usr/bin/python

import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import torch
import sys
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models, utils

# model = None

def args_manage():
    """
    Controliing all the arguments passed to script
    """

    parser = argparse.ArgumentParser(description='Train & Process some Data')

    parser.add_argument("data_dir", help="The path of dir containing the image datasets")
    parser.add_argument("--save_dir",type=str, help="Directory path to save checkpoints")
    parser.add_argument("--arch", default='vgg19', help="Choosing the architecture")
    parser.add_argument("--learning_rate", default=0.003, type=float, help="rate at which Models are trained")
    parser.add_argument("--hidden_units", default=4096, type=int, help="hidden layers: 4096 for vgg19 and 1024 for densenet121")
    parser.add_argument("--epochs", default=14, type=int, help="looping cycles")
    parser.add_argument("--gpu", action='store_const', const='gpu', help="device for training")
                        
    args = parser.parse_args()
    print(args.data_dir)
    if args.save_dir:
        print("save dir option is used")
    return args



def data_load():
    """
        Loading the data from the Image folder and transforming the data for training , testing & validation
    """
    # data_dir = 'flowers'
    data_dir = args.data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

    # TODO: Load the datasets with ImageFolder
    train_image_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_image_datasets = datasets.ImageFolder(test_dir, transform=test_transforms)
    valid_image_datasets = datasets.ImageFolder(valid_dir, transform=valid_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloaders = torch.utils.data.DataLoader(train_image_datasets, batch_size=64, shuffle=True)
    testloaders = torch.utils.data.DataLoader(test_image_datasets, batch_size=32)
    validloaders = torch.utils.data.DataLoader(valid_image_datasets, batch_size=32)
    print(trainloaders)

    return trainloaders, testloaders, validloaders, train_image_datasets



def build_classifier():
    """
        Building and training the classifier
    """
    # Build and train your network
    learn_rate = args.learning_rate
    hidden_layer = args.hidden_units
    

    from collections import OrderedDict
    if(args.arch == 'vgg19'):
        model = models.vgg19(pretrained=True)
        print(model)

        # Freeze parameters so we don't backprop through them
        for param in model.parameters():
            param.requires_grad = False
        classifier = nn.Sequential(OrderedDict([
                                ('fc1', nn.Linear(25088, hidden_layer)),
                                ('relu', nn.ReLU()),
                                ('drpt', nn.Dropout(p=0.5)),
                                ('fc2', nn.Linear(hidden_layer, 102)),
                                ('output', nn.LogSoftmax(dim=1))
                                ]))

    elif(args.arch == 'densenet121'):
        model = models.densenet121(pretrained=True)
        print(model)

        # Freeze parameters so we don't backprop through them
        for param in model.parameters():
            param.requires_grad = False
        classifier = nn.Sequential(OrderedDict([
                                ('fc1', nn.Linear(1024, hidden_layer)),
                                ('relu', nn.ReLU()),
                                ('drpt', nn.Dropout(p=0.5)),
                                ('fc2', nn.Linear(hidden_layer, 102)),
                                ('output', nn.LogSoftmax(dim=1))
                                ]))
    else:
        print("Please select from the provided option values")
        sys.exit(1)

    model.classifier = classifier

    # Train the model
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learn_rate)
    
    return model, optimizer, criterion



def do_deep_learning(model, validloader, epochs, print_every, criterion, optimizer, device):
    """
    #Functions for calculating running_loss
    """
    epochs = epochs
    print_every = print_every
    steps = 0
    
    #Change to cuda
    model.to(device)
    
    for e in range(epochs):
        model.train()
        running_loss = 0
        for ii , (inputs, labels) in enumerate(validloader):
            # model.train()
            steps += 1

            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            #Forward and backward passess
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if steps % print_every == 0:
                print("Epoch: {}/{}... ".format(e+1, epochs),
                    "Loss: {:.4f}".format(running_loss/print_every))

                running_loss = 0
    

                
def check_accuracy_on_test(validloader):
    """
    #Functions for calculating accuracy
    """
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data in validloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 validation data images: %d %%' % (100 * correct / total))


# TODO: Do validation on the test set
#Function for validation pass
def validation(model, testloader, criterion):
    valid_loss = 0
    valid_accuracy = 0
    
    for images2, labels2 in testloader:
        images2, labels2 = images2.to(device), labels2.to(device)
        model.to(device)
        output = model.forward(images2)
        valid_loss += criterion(output, labels2).item()
        
        ps = torch.exp(output)
        equality = (labels2.data == ps.max(dim=1)[1])
        valid_accuracy += equality.type(torch.FloatTensor).mean()
        
    return valid_loss, valid_accuracy


def test_network(model, trainloaders, testloaders, optimizer, criterion, epochs, device):
    # epochs = 10
    steps = 0
    running_loss = 0
    print_every = 40
    #Change to cuda
    model.to(device)
        
    for e in range(epochs):
        model.train()
        for images, labels in trainloaders:
            steps += 1
            
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if steps % print_every == 0:
                # Make sure network is in eval mode for inference
                model.eval()
                
                # Turn off gradients for validation, saves memory and computations
                with torch.no_grad():
                    valid_loss, valid_accuracy = validation(model, testloaders, criterion)
                    
                print("Epoch: {}/{}.. ".format(e+1, epochs),
                    "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                    "Validation Loss: {:.3f}.. ".format(valid_loss/len(testloaders)),
                    "validation Accuracy: {:.3f}.. ".format(valid_accuracy/len(testloaders)))
                
                    
                running_loss = 0
                #Make sure training is back on
                model.train()


def save_chcekpoint(model, train_image_datasets):
    """
        Saving and loading the checkpoint of the trained Model for dynamic use
    """
    # Save the checkpoint 
    model.class_to_idx = train_image_datasets.class_to_idx

    checkpoint = {'classifier': model.classifier,
                'class_to_idx': model.class_to_idx,
                'arch': 'vgg19',
                'state_dict': model.state_dict()}

    model.cpu()
    torch.save(checkpoint, args.save_dir)



def load_checkpoint(filepath):
    """
        function that loads a checkpoint and rebuilds the model
    """
    checkpoint = torch.load(filepath)
    
    # model = models.checkpoint['arch']
    if checkpoint['arch'] == 'vgg19':
        model = models.vgg19(pretrained=True)
    model.classifier = checkpoint['classifier']
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    
    return model



if __name__ == '__main__':
    args = args_manage()

    hidden_layer = args.hidden_units
    epochs = args.epochs
    print_every = 40
    
    trainloaders, testloaders, validloaders, train_image_datasets = data_load()
    
    if(args.gpu):
        device = 'cuda'
    else: 
        device = 'cpu'
    print("Test data used : ", testloaders)

    model, optimizer, criterion = build_classifier()

    ## ----- Claculating Loss on model
    #do_deep_learning(model, trainloaders, epochs, print_every, criterion, optimizer, device)
    #check_accuracy_on_test(validloaders)

    test_network(model, trainloaders, testloaders, optimizer, criterion, epochs, device)

    # Saving & Loading checkpoint
    save_chcekpoint(model, train_image_datasets)
    load_checkpoint(args.save_dir)


