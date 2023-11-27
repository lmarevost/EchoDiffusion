# Logger
import logging
# Data science tools
import numpy as np
# paths
import os
from pathlib import Path 
# Parser
import argparse
# t-sne embedding
from sklearn.manifold import TSNE
# PyTorch
from torchvision import transforms, datasets, models
import torch
from torch import optim, cuda
from torch.utils.data import DataLoader, sampler
import torch.nn as nn
torch.manual_seed(42)
# Timing utility
from datetime import datetime
import time
# Visualizations
import matplotlib.pyplot as plt
# warnings
import warnings
warnings.filterwarnings("ignore")

# Set device to GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# set current time to know when script ran
time_now = datetime.now().strftime("%Y_%m_%d_%H%M%S")


# --------------------------------------------
# ARG PARSER
# --------------------------------------------
parser = argparse.ArgumentParser()
# dirs
parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset to be visualized')
parser.add_argument('--save_dir', type=str, default=None, help='Path to save PNGs')
parser.add_argument('--log_dir', type=str, default=None, help='Path to log process')
# model flags
parser.add_argument('--model_checkpoint', type=str, default=None, help='Path to model checkpoint to be used')
parser.add_argument('--model_name', type=str, default=None, choices=('VGG16', 'ResNet18', 'DenseNet121', 'InceptionV3'), help='name of model to fetch from torchvision.models')
parser.add_argument('--pretrained', type=str, required=True, choices=('True', 'False'), help='if True, use pretrained weights, else train from scratch')
parser.add_argument('--freeze', type=str, required=True, choices=('True', 'False'), help='if True, freeze all layers except the last classifier layer')
# hyperparameters
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
args = parser.parse_args()

# path to data to be visualized
data_dir = args.data_dir
if 'synthetic_samples' in data_dir:
    data_dir = os.path.join(data_dir,'PNG')
    data_type = 'synthetic'
    dataset_name = data_dir.split('/')[-2]
else:
    data_type = 'real'
    dataset_name = os.path.basename(data_dir)

data_dir = Path(data_dir)

# --------------------------------------------
# FOR PLOTS
# --------------------------------------------
perplexity=20
n_iter=1000
markers = {'sample':'v'} if 'synthetic_samples' in str(data_dir) else {'TMED2': 'x', 'LVH': '*', 'Dynamic': 'o'} 
colors = {0:'b', # 'A2C' = blue
          1:'c', # 'A4C' = cyan
          2:'m', # 'PLAX' = magenta
          3:'y'} # 'PSAX' = yellow

# path to the directory where the visualizations are to be saved
save_dir = '/mimer/NOBACKUP/priv/chair/echo_anouka_luismi/streamlined/saved_cluster_viz/PNGs' if args.save_dir is None else args.save_dir
save_dir = Path(save_dir)
save_dir.mkdir(parents=True, exist_ok=True)

# path to the directory where process is logged
log_dir = '/mimer/NOBACKUP/priv/chair/echo_anouka_luismi/streamlined/saved_cluster_viz/logs' if args.log_dir is None else args.log_dir
log_dir = Path(log_dir)
log_dir.mkdir(parents=True, exist_ok=True)

# --------------------------------------------
# MODEL FLAGS
# --------------------------------------------
model_checkpoint = Path(args.model_checkpoint)
assert os.path.isfile(model_checkpoint), f'No model checkpoint found at {model_checkpoint}.'

if model_checkpoint:        
    for m in ('VGG16', 'ResNet18', 'DenseNet121', 'InceptionV3'):
        print(m, args.model_checkpoint)
        if m in args.model_checkpoint:
            model_name = m
            break
    else:
        raise ValueError('Unrecognized model from provided checkpoint.')
else:
    model_name = args.model_name                               

pretrained = (args.pretrained == 'True')
freeze = (args.freeze == 'True')
input_size = 299 if model_name == 'InceptionV3' else 112

# --------------------------------------------
# LOGGING
# --------------------------------------------
save_basename = log_dir/f"{time_now}_{data_type}_{dataset_name}_{model_name}_{'T' if pretrained else 'F'}{'T' if freeze else 'F'}"  # e.g. VGG16_FF_2023_04_23_115719
logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)
logger.addHandler(logging.FileHandler(f'{save_basename}.log', 'w'))
logger.info(f"Logging to:\n'{save_basename}.log'")

# --------------------------------------------
# SETTING HYPERPARAMETERS
# --------------------------------------------
n_classes = len(os.listdir(data_dir))
class_idx_dict = {i:c for i,c in enumerate(sorted(os.listdir(data_dir)))} # e.g. {0: 'A2C', 1: 'A4C', 2: 'PLAX', 3: 'PSAX'}
batch_size = args.batch_size



def get_elapsed_time(start_time):
    # Elapsed_time in seconds
    elapsed_time = time.time() - start_time
    # Convert time in seconds to days, hours, minutes, and seconds
    days, remainder = divmod(elapsed_time, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    # Format the time as a string in d:hh:mm:ss format
    time_string = f"{days:,.0f}d {hours:02.0f}h {minutes:02.0f}m {seconds:02.0f}s"
    return time_string


def get_dataset_info():
    '''
    Returns the number of images in subdirectories and total in dataset
    '''
    logger.info(f'\nCurrent time: {datetime.now().strftime("%Y_%m_%d_%H%M%S")}')
    logger.info('*'*100)
    logger.info(f'DATASET')
    count_sub_dir = 0
    for class_dir in os.listdir(data_dir):
        class_dir_path = os.path.join(data_dir, class_dir)
        count_class_dir = len(os.listdir(class_dir_path))
        count_sub_dir += count_class_dir
    logger.info(f'  - Data: {count_sub_dir:,} images found in {data_dir}')
    

def print_settings_info():
    ''' Prints the settings used for training '''
    logger.info(
        f'\nSETTINGS: \
        \n  - Batch size:       {batch_size} \
        \n  - Num of Classes:   {n_classes} \
        \n  - Device used:      {device} \
        \n\nPNGs SAVED TO: {save_dir}'
        )
    logger.info('*'*100)
    
    
# --------------------------------------------
# LOAD DATA
# --------------------------------------------
def get_transform():
    ''' 
    Define data transforms and load data into dataloaders
    Special case for InceptionV3, which requires 299 input size and 3 color channels
    '''
    # Define transforms
    if pretrained or model_name == 'InceptionV3':
        #print(f'\nTransform data to 3 color channels with size {input_size}x{input_size}')
        transform = transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, ), (0.5, )) ])
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) ])
    else:
        #print(f'\nTransform data to 1 color channels with size {input_size}x{input_size}')
        transform = transforms.Compose([
            transforms.Resize(input_size),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.5, ), (0.5, )) ])
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) ])
    return transform


def get_data_loader():
    ''' 
    Returns dataloaders for train and val sets in dictionary format
    '''
    transform = get_transform()
    data = datasets.ImageFolder(root=data_dir, transform=transform)
    print(data.classes)
    print(data.class_to_idx)
    
    dataloader =  DataLoader(data, batch_size=batch_size, shuffle=False)
    
    # return dataloader
    return dataloader


def get_imgs_labels_filenames(dataloader):
    images, labels = [], []
    # imgs_paths = [x[0] for x in dataloader.dataset.samples] # image path
    #file_names = [os.path.basename(x[0]) for x in dataloader.dataset.samples] # file names
    dataset_names = [os.path.basename(x[0]).split('_')[0] for x in dataloader.dataset.samples] # dataset name per image 
    # Iterate over all batches in the dataloader 
    for batch_images, batch_labels in dataloader:
        # append flattened images and labels to the lists
        images.append(batch_images.view(batch_images.shape[0], -1))
        labels.append(batch_labels)
    # Concatenate the images and labels from all batches into single tensors
    images = torch.cat(images)
    labels = torch.cat(labels)
    return images.numpy(), labels, dataset_names

# --------------------------------------------
# t-SNE EMBEDDINGS
# --------------------------------------------
def tsne(X_flat):
    '''Perform t-sne embedding'''
    return TSNE(n_components=2, init='random', random_state=42, perplexity=perplexity, n_iter=n_iter).fit_transform(X_flat)

def x_y_coordinates(X_embedded):
    '''get x- and y-coordinates from 2-dimensional t-sne embedding for scatter plot'''
    return X_embedded[:,0], X_embedded[:,1]

def scale_0_1(X_embedded_x, X_embedded_y):
    '''scale x- and y-coordinates to [0,1] for better plotting'''
    X_embedded_x_scaled = (X_embedded_x - X_embedded_x.min()) / (X_embedded_x.max() - X_embedded_x.min())
    X_embedded_y_scaled = (X_embedded_y - X_embedded_y.min()) / (X_embedded_y.max() - X_embedded_y.min())
    return X_embedded_x_scaled, X_embedded_y_scaled


# --------------------------------------------
# SET AND EVALUATE MODEL
# --------------------------------------------
def set_model_input_layer(model):
    '''
    Set the first layer of the model to fit the input color dimension of the dataset.
    '''
    if not pretrained:
        #logger.info('Do not use pretrained weights... Train model from scratch') 
        #logger.info('Replacing model input layer to account for greyscale images (1 color channel instead of 3)...')
        if model_name == 'VGG16':
            model.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) # Change first layer input channels to 1
            return model
        elif model_name == 'ResNet18':
            model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) # Change first layer input channels to 1
            return model
        elif model_name == 'DenseNet121':
            model.features[0] = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) # Change first layer input channels to 1
            return model
        elif model_name == 'InceptionV3':
            model.Conv2d_1a_3x3.conv = nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False) # Change first layer input channels to 3
            return model
    elif pretrained and model_name == 'InceptionV3': 
        model.Conv2d_1a_3x3.conv = nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False) # Change first layer input channels to 3
        return model
    
    
def set_parameter_requires_grad(model):
    ''' 
    Freeze all layers except the last classifier layer if freeze is True 
    '''
    if freeze:
        #logger.info('Freezing all layers except the last classifier layer')
        for param in model.parameters():
            param.requires_grad = False
            
            
def initialize_model():
    ''' 
    Choose a model from torchvision.models and modify first layer to fit input color dimension 
    and modify last layer to fit the number of classes in the dataset.
    
    Input Parameters:
    model_name: string, name of model to fetch from torchvision.models
    pretrained: bool, if True, use pretrained weights, else train from scratch
    freeze: bool, if True, freeze all layers except the last classifier layer
    
    Return: model, name of model save file, input size for model
    ''' 
    #logger.info('\n------------------------------------------------------------------------------------')
    logger.info(f'MODEL: {model_name}, with pretrained weights: {pretrained}, and frozen layers: {freeze}')
    logger.info(f'model checkpoint: {args.model_checkpoint}')
    #logger.info('------------------------------------------------------------------------------------')

    if model_name == 'VGG16':
        # VGG16
        model = models.vgg16(pretrained=pretrained)   
        set_model_input_layer(model)
        set_parameter_requires_grad(model)
        # Update final layer
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, n_classes)
           
    elif model_name == 'ResNet18':
        model = models.resnet18(pretrained=pretrained)
        set_model_input_layer(model)
        set_parameter_requires_grad(model)
        # Update final layer
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, n_classes) # reset final layer (requireds_grad=True by default)
        
    elif model_name == 'DenseNet121':
        model = models.densenet121(pretrained=pretrained)
        set_model_input_layer(model)
        set_parameter_requires_grad(model)
        # Update final layer
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, n_classes)  
    
    elif model_name == 'InceptionV3':
        model = models.inception_v3(pretrained=pretrained)
        # Change first layer input channels to 3
        set_model_input_layer(model)
        set_parameter_requires_grad(model)
        # Update final layer for Auxilary net
        num_ftrs1 = model.AuxLogits.fc.in_features
        model.AuxLogits.fc = nn.Linear(num_ftrs1, n_classes)
        # Update final layer for main net
        num_ftrs2 = model.fc.in_features 
        model.fc = nn.Linear(num_ftrs2, n_classes)
    return model


def evaluate_model(model, dataloader):
    features = []
    l = []
    # Iterate over data batches
    with torch.no_grad():
        for batch_images, batch_labels in dataloader:
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)
            outputs = model(batch_images)
            features.append(outputs.clone().cpu().numpy())
            l.append(batch_labels.clone().cpu().numpy())
    features = np.concatenate(features)
    l = np.concatenate(l)
    return features, l



# --------------------------------------------
# PLOT FUNCTIONS
# --------------------------------------------
def plot_tsne(X_embedded_x_scaled, X_embedded_y_scaled, Y, dataset_names, title, save_name):
    for x, y, label, name in zip(X_embedded_x_scaled, X_embedded_y_scaled, Y.numpy() if torch.is_tensor(Y) else Y, np.array(dataset_names)):
        plt.scatter(x, y, color=colors[label], marker=markers[name], s=8)
    # Add the legend for the dataset names
    legend2 = plt.legend(handles=[plt.scatter([],[],c='k',marker=markers[name]) for name in markers.keys()], 
                        labels=list(markers.keys()), loc='upper right',  title='Dataset')
    plt.gca().add_artist(legend2)
    # Add the legend for the class names
    class_legend_handles = []
    for idx, name in class_idx_dict.items():
        class_legend_handles.append(plt.scatter([], [], marker='o', label=name, color=colors[idx]))
    plt.legend(handles=class_legend_handles, title='Views', loc='upper left')
    # Set the plot title and display the plot
    plt.title(title)
    save_name = f'{save_dir}/{time_now}_{data_type}_{dataset_name}_{os.path.basename(model_checkpoint)}_{save_name}.png'
    plt.savefig(save_name, bbox_inches='tight')
    logger.info(f'\nplot saved to {save_name}')
    plt.show()
    plt.close()



def plot_tsne_subplots(X_embedded_x_scaled_1, X_embedded_y_scaled_1, Y_1, X_embedded_x_scaled_2, X_embedded_y_scaled_2, Y_2, dataset_names, title1, title2):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    #fig.suptitle(f"2D t-SNE: {data_type.upper()} '{dataset_name.upper()}' DATASET\n(perplexity:{perplexity}, iterations: {n_iter})", fontsize=16, fontweight='bold')

    # PLOT 1 - PIXELS
    for x, y, label, name in zip(X_embedded_x_scaled_1, X_embedded_y_scaled_1, Y_1.numpy() if torch.is_tensor(Y_1) else Y_1, np.array(dataset_names)):
        ax1.scatter(x, y, color=colors[label], marker=markers[name], s=8)
    ax1.set_title(title1)
    legend1 = ax1.legend(handles=[ax1.scatter([],[],c='k',marker=markers[name]) for name in markers.keys()], 
                        labels=list(markers.keys()), loc='upper right',  title='Dataset')
    ax1.add_artist(legend1)
    # Add the legend for the class names
    class_legend_handles = []
    for idx, name in class_idx_dict.items():
        class_legend_handles.append(ax1.scatter([], [], marker='s', label=name, color=colors[idx]))
    ax1.legend(handles=class_legend_handles, title='Views', loc='upper left')

    # PLOT 2 - FEATURES
    for x, y, label, name in zip(X_embedded_x_scaled_2, X_embedded_y_scaled_2, Y_2.numpy() if torch.is_tensor(Y_2) else Y_2, np.array(dataset_names)):
        ax2.scatter(x, y, color=colors[label], marker=markers[name], s=8)
    ax2.set_title(title2)
    legend2 = ax2.legend(handles=[ax2.scatter([],[],c='k',marker=markers[name]) for name in markers.keys()], 
                        labels=list(markers.keys()), loc='upper right',  title='Dataset')
    ax2.add_artist(legend2)
    # Add the legend for the class names
    class_legend_handles = []
    for idx, name in class_idx_dict.items():
        class_legend_handles.append(ax2.scatter([], [], marker='s', label=name, color=colors[idx]))
    ax2.legend(handles=class_legend_handles, title='Views', loc='upper left')
    
    fig.text(0.5, 0.93, f"2D t-SNE: {data_type.upper()} '{dataset_name.upper()}' DATASET", ha='center', fontsize=16, fontweight='bold')
    fig.text(0.5, 0.90, f"(perplexity:{perplexity}, iterations: {n_iter})", ha='center', fontsize=10)
     
    # SAVE
    save_name = f'{save_dir}/{time_now}_{data_type}_{dataset_name}_{os.path.basename(model_checkpoint)}_tsne_subplots.png'
    plt.savefig(save_name, bbox_inches='tight')
    logger.info(f'\nplot saved to {save_name}')
    plt.show()
    plt.close()

def print_data_shape(class_dirs_idx,X_flat,X_embedded,Y):
    logger.info(f'\nclass index:',class_dirs_idx)
    logger.info(f'X before t-SNE:',X_flat.shape, '= (images, dimensions)') # for pixel data w*h = 112*112=12544
    logger.info(f'X after t-SNE:',X_embedded.shape, '= (images, dimensions)')
    logger.info(f'Y shape:',Y.shape, '= (labels,)')




# --------------------------------------------
# MAIN
# --------------------------------------------
def main():  
    '''
    Initialize model, get data loaders, train model and visualize performance
    Return model and history and path to where model is saved
    '''
    
    start_time = time.time()
    get_dataset_info()
    print_settings_info()
    
    # Get dataloader
    dataloader = get_data_loader()

    # FOR PLOTTING PIXELS
    X_flat, Y, dataset_names = get_imgs_labels_filenames(dataloader) # get flat images, labels, dataset name per image
    X_embedded = tsne(X_flat) # get embedded features using t-SNE
    X_embedded_x, X_embedded_y = x_y_coordinates(X_embedded)  # get x and y coordinates for plotting
    X_embedded_x_scaled, X_embedded_y_scaled = scale_0_1(X_embedded_x, X_embedded_y) # scale x and y coordinates to 0-1 range

    # FOR PLOTTING FEATURES
    model = initialize_model() # initialize model architecture
    model = model.to(device)  # move model to GPU if available else CPU
    if model_checkpoint: # if model_checkpoint is not None, load model checkpoint
        model.load_state_dict(torch.load(model_checkpoint, map_location=device))   
    model.classifier = nn.Sequential(*[model.classifier[i] for i in range(5)]) # remove final output layey to get features from next to last layer, including ReLu
    model.eval() # set model to evaluation mode
    features, labels = evaluate_model(model, dataloader) # get features and labels
    features_embedded = tsne(features) # get embedded features using t-SNE
    f_embedded_x, f_embedded_y = x_y_coordinates(features_embedded) # get x and y coordinates for plotting
    f_embedded_x_scaled, f_embedded_y_scaled = scale_0_1(f_embedded_x, f_embedded_y) # scale x and y coordinates to 0-1 range


    #print_data_shape(class_idx_dict,X_flat,X_embedded,Y)
    #print_data_shape(class_idx_dict, features, features_embedded, labels)

    # PLOT
    logger.info(f'\nPlotting...')
    logger.info(f'colors:\n{colors}')
    title_pix = 'Image Pixels'
    title_features = f'Image Features from own trained {model_name}'
    
    #plot_tsne(X_embedded_x_scaled, X_embedded_y_scaled, Y, 
              #dataset_names, title_pix, 'tsne_pix')
    
    #plot_tsne(f_embedded_x_scaled, f_embedded_y_scaled, labels, 
              #dataset_names, title_features, 'tsne_features')
    
    plot_tsne_subplots(X_embedded_x_scaled, X_embedded_y_scaled, Y, 
                       f_embedded_x_scaled, f_embedded_y_scaled, labels, 
                       dataset_names, title_pix, title_features)

    logger.info(f'\nFinished! Total run time: {get_elapsed_time(start_time)}')



if __name__ == '__main__':
    main()
    