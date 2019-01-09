# #!/usr/bin/python

import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

import train

def control_args():

    """
    Handling all arguments passed
    """
    parser = argparse.ArgumentParser(description="Taking arguments for predict.py script")

    parser.add_argument("image_path", help="the path to single flower image")
    parser.add_argument("checkpoint", help="folder path where trained Model's dictionary and values are saved")
    parser.add_argument("--top_k", type=int, default=5, help="Return top KK most likely classes")
    parser.add_argument("--category_names", help="a mapping of categories to real names")
    parser.add_argument("--gpu", action='store_const', const='gpu', help="device for training")

    args = parser.parse_args()
    return args


def label_mapping(category_path):
    import json

    with open(category_path, 'r') as f:
        cat_to_name = json.load(f)
    
    return cat_to_name




def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    # from PIL import Image
    img = Image.open(image)
    
    # Resize
    if img.size[0] > img.size[1]:
        img.thumbnail((10000, 256))
    else:
        img.thumbnail((256, 10000))
    # Crop
    left_margin = (img.width - 224) / 2
    bottom_margin = (img.height - 224) / 2
    right_margin = left_margin + 224
    top_margin = bottom_margin + 224
    img = img.crop((left_margin, bottom_margin, right_margin,
                   top_margin))
    # Normalize
    img = np.array(img) / 255
    mean = np.array([0.485, 0.456, 0.406])  # provided mean
    std = np.array([0.229, 0.224, 0.225])  # provided std
    img = (img - mean) / std

    # Move color channels to first dimension as expected by PyTorch
    img = img.transpose((2, 0, 1))
    
    return img



def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax


def predict(image_path, model, topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model.to(device)
    
    # model.eval()
    # Process image
    img = process_image(image_path)
    
    # Numpy -> Tensor
    image_tensor = torch.from_numpy(img).type(torch.FloatTensor)
    # Add batch of size 1 to image
    model_input = image_tensor.unsqueeze(0)
    
    # Probs
    model_input.to(device)
    probs = torch.exp(model.forward(model_input))
    
    # Top probs
    top_probs, top_labs = probs.topk(args.top_k)
    top_probs = top_probs.detach().numpy().tolist()[0] 
    top_labs = top_labs.detach().numpy().tolist()[0]
    

    # Convert indices to classes
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_labels = [idx_to_class[lab] for lab in top_labs]
    top_flowers = [cat_to_name[idx_to_class[lab]] for lab in top_labs]
    return top_probs, top_labels, top_flowers



def plot_solution(image_path, model, topk):
    # Set up plot
    print("Plotting the image")
    # plt.figure(figsize = (6,10))
    # ax = plt.subplot(2,1,1)
    # Set up title
    flower_num = image_path.split('/')[-2]
    print(image_path)
    print(flower_num)
    title_ = cat_to_name[flower_num]
    print(title_)
    # Plot flower
    img = process_image(image_path)
    # imshow(img, ax, title = title_);   //-- Commenting for showing image
    # Make prediction
    probs, labs, flowers = predict(image_path, model, topk) 
    # Plot bar chart
    # plt.subplot(2,1,2)
    # sns.barplot(x=probs, y=flowers, color=sns.color_palette()[0]);
    # plt.show()
    print(probs, labs, flowers)



if __name__ == '__main__':
    args = control_args()
    image_path = args.image_path
    checkpoint = args.checkpoint
    top_k = args.top_k
    category_path = args.category_names
    device = args.gpu

    #Loading the trained model
    print("Model load")
    model = train.load_checkpoint(checkpoint)

    cat_to_name = label_mapping(category_path)
    # process_image(image_path)
    # imshow(image_path)
    # predict(image_path, model, top_k)
    plot_solution(image_path, model, top_k)