# Logger
import logging
from pathlib import Path 
# Parser
import argparse
# PyTorch
from torchvision import transforms, datasets, models
import torch
from torch import optim, cuda
from torch.utils.data import DataLoader, sampler
import torch.nn as nn
from torch.optim import lr_scheduler
# Torch metrics
from torchmetrics.classification import MulticlassF1Score, MulticlassAUROC
from torchmetrics.classification import MulticlassConfusionMatrix, MulticlassPrecisionRecallCurve, MulticlassROC
# Data science tools
import numpy as np
import pandas as pd
import os
import shutil
import copy
# Image manipulations
import PIL
# Timing utility
import time
from datetime import datetime
from timeit import default_timer as timer
from tqdm import tqdm
# Visualizations
import matplotlib.pyplot as plt
# warnings
import warnings
warnings.filterwarnings("ignore")


# --------------------------------------------
# ARG PARSER
# --------------------------------------------
parser = argparse.ArgumentParser()
# dirs
parser.add_argument('--real_dir', type=str, required=True, help='Path to real training set')
parser.add_argument('--fake_dir', type=str, required=True, help='Path to synthetic training set')
parser.add_argument('--val_dir', type=str, required=True, help='Path to validation set')
parser.add_argument('--save_dir', type=str, required=True, help='Path to save classifier checkpoint')
# model flags
parser.add_argument('--model_name', type=str, required=True, choices=('VGG16', 'ResNet18', 'DenseNet121', 'InceptionV3'), help='name of model to fetch from torchvision.models')
parser.add_argument('--pretrained', type=str, required=True, choices=('True', 'False'), help='if True, use pretrained weights, else train from scratch')
parser.add_argument('--freeze', type=str, required=True, choices=('True', 'False'), help='if True, freeze all layers except the last classifier layer')
parser.add_argument('--frac', type=float, required=True, help='Number of synthetic data to use for augmentation, as a fraction of real dataset')
# hyperparameters
parser.add_argument('--epochs', type=int, default=25, help='Training epochs')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
args = parser.parse_args()

# Set device to GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# set current time to know when script ran
time_now = datetime.now().strftime("%Y_%m_%d_%H%M%S")

# --------------------------------------------
# TRAIN, VAL, SAVE PATH
# --------------------------------------------
# real train, fake train & val datasets
real_dir = args.real_dir
fake_dir = args.fake_dir
val_dir = args.val_dir
sub_dirs = [real_dir, fake_dir, val_dir]

# save directory for model checkpoints, logs, and plots
save_dir = args.save_dir
save_dir = Path(save_dir)
save_dir.mkdir(parents=True, exist_ok=True)

# --------------------------------------------
# MODEL FLAGS
# --------------------------------------------
model_name = args.model_name
pretrained = (args.pretrained == 'True')
freeze = (args.freeze == 'True')
frac = args.frac

# --------------------------------------------
# LOGGING
# --------------------------------------------
save_basename = save_dir/f"{model_name}_{'T' if pretrained else 'F'}{'T' if freeze else 'F'}_{frac}_{time_now}"  # e.g. VGG16_FF_0.1_2023_04_23_115719
logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)
logger.addHandler(logging.FileHandler(f'{save_basename}.log', 'w'))
logger.info(f"Logging to:\n'{save_basename}.log'")

# --------------------------------------------
# SETTING HYPERPARAMETERS
# --------------------------------------------
n_classes = len(os.listdir(real_dir))
learning_rate = args.lr
batch_size = args.batch_size
epochs = args.epochs

# --------------------------------------------
# METRICS
# --------------------------------------------
# training
f1 = MulticlassF1Score(num_classes=n_classes).to(device)
auroc = MulticlassAUROC(num_classes=n_classes).to(device)


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
    logger.info(f'\nCurrent time: {time_now}')
    logger.info('*'*100)
    logger.info(f'DATASETS')
    total_img_count = 0
    for sub_dir,sd in zip(sub_dirs,['Real', 'Synth', 'Val']):
        count_sub_dir = 0
        for class_dir in os.listdir(sub_dir):
            class_dir_path = os.path.join(sub_dir, class_dir)
            count_class_dir = len(os.listdir(class_dir_path))
            count_sub_dir += count_class_dir
        logger.info(f'  - {sd}: {count_sub_dir:,} images found in {sub_dir}')
        for dirpath, dirnames, filenames in os.walk(sub_dir):
            total_img_count += len(filenames)
    logger.info(f'TOTAL DATASET SIZE: {total_img_count:,}')


def print_settings_info():
    ''' Prints the settings used for training '''
    logger.info(
        f'\nSETTINGS: \
        \n  - Augmentation fraction:    {frac} \
        \n  - Learning rate:    {learning_rate} \
        \n  - Batch size:       {batch_size} \
        \n  - Num of Epochs:    {epochs} \
        \n  - Num of Classes:   {n_classes} \
        \n  - Device used:      {device} \
        \n\nRESULTS SAVED TO: {save_dir}'
        )
    logger.info('*'*100)


def set_parameter_requires_grad(model, freeze):
    ''' 
    Freeze all layers except the last classifier layer if freeze is True 
    '''
    if freeze:
        logger.info('Freezing all layers except the last classifier layer')
        for param in model.parameters():
            param.requires_grad = False


def print_params_to_learn(model):
    '''
    Prints the parameters that will be updated during training of the given model.
    '''
    params_to_update = model.parameters()
    logger.info("\nParams to learn:")
    if freeze:
        params_to_update = []
        for m_n,param in model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                logger.info(f"\t {m_n}")
    else:
        for m_n,param in model.named_parameters():
            if param.requires_grad == True:
                logger.info(f"\t {m_n}")


def initialize_model():
    ''' 
    Choose a model from torchvision.models and modify first layer to fit input color dimension 
    and modify last layer to fit the number of classes in the dataset.
    
    Return: model, name of model save file, input size for model
    ''' 
    
    logger.info('\n------------------------------------------------------------------------------------')
    logger.info(f'MODEL: {model_name}, with pretrained weights: {pretrained}, and frozen layers: {freeze}')
    logger.info('------------------------------------------------------------------------------------\n')

    input_size=112
    
    if model_name == 'VGG16':
        # VGG16
        model = models.vgg16(pretrained=pretrained)
        model.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) # Change first layer input channels to 1
        set_parameter_requires_grad(model, freeze)
        # Update final layer
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, n_classes)
           
    elif model_name == 'ResNet18':
        model = models.resnet18(pretrained=pretrained)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) # Change first layer input channels to 1
        set_parameter_requires_grad(model, freeze)
        # Update final layer
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, n_classes) # reset final layer (requireds_grad=True by default)
        
    elif model_name == 'DenseNet121':
        model = models.densenet121(pretrained=pretrained)
        model.features[0] = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) # Change first layer input channels to 1
        set_parameter_requires_grad(model, freeze)
        # Update final layer
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, n_classes)  
    
    elif model_name == 'InceptionV3':
        model = models.inception_v3(pretrained=pretrained)
        # Change first layer input channels to 3
        model.Conv2d_1a_3x3.conv = nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        set_parameter_requires_grad(model, freeze)
        # Update final layer for Auxilary net
        num_ftrs1 = model.AuxLogits.fc.in_features
        model.AuxLogits.fc = nn.Linear(num_ftrs1, n_classes)
        # Update final layer for main net
        num_ftrs2 = model.fc.in_features 
        model.fc = nn.Linear(num_ftrs2, n_classes)
        input_size = 299
 
    # Print count of non-trainable and trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'{total_trainable_params:,} Trainable Parameters / {total_params:,} Total Parameters')

    return model, input_size


def get_transforms(input_size):
    ''' 
    Define data transforms and load data into dataloaders
    Special case for InceptionV3, which requires 299 input size and 3 color channels
    '''
    # Define transforms
    if model_name == 'InceptionV3':
        transform = transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, ), (0.5, )) ])
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) ])
    else:
        transform = transforms.Compose([
            transforms.Resize(input_size),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.5, ), (0.5, )) ])
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) ])
    
    return transform


def get_data_loaders(input_size, batch_size):
    ''' 
    Returns dataloaders for train and val sets in dictionary format
    '''
    # get transformer
    transform = get_transforms(input_size)
    
    # Datasets from each folder
    data = {
        'real': datasets.ImageFolder(root=real_dir, transform=transform),
        'fake': datasets.ImageFolder(root=fake_dir, transform=transform),
        'val': datasets.ImageFolder(root=val_dir, transform=transform),
    }

    # Build real dataset augmented with synthetic data
    n_fake = int(frac * len(data['real']))  # number of synth examples to take
    subset_indices = torch.randperm(len(data['fake']))[:n_fake]  # select a random subset of examples from dataset B
    data['train'] = torch.utils.data.ConcatDataset([data['real'], torch.utils.data.Subset(data['fake'], subset_indices)])  # concatenate real dataset with subset of fake dataset
    logger.info(f"Augmenting real dataset with {n_fake} synthetic examples ({frac*100}%).")

    # Dataloader iterators
    dataloaders = {
        'train': DataLoader(data['train'], batch_size=batch_size, shuffle=True),
        'val': DataLoader(data['val'], batch_size=batch_size, shuffle=False),
    }

    # Check shape of data and labels
    logger.info('\nShape of data and labels:')
    for d in dataloaders.keys():
        for (data, label) in (dataloaders[d]):
            logger.info(f'Dataset: {d} \t Datashape: {data.shape}\t Labelshape: {label.shape}')
            break
    
    return dataloaders


# https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html#vgg
def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs, is_inception=False):
    ''' 
    Train and validate model for num_epochs.
    Save best model weights. 
    Track training and validation loss and accuracy.
    Time total run time, epoch and store best epoch.
    
    Return trained model and histiry
    '''
    
    start_time = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_auroc = 0.0  # imbalanced data -> go for AUROC (maybe AUPRC?) instead of Accuracy
    history = {'train_acc': [], 'val_acc': [], 'train_loss': [], 'val_loss': [], 'train_f1': [], 'val_f1': [], 'train_auroc': [], 'val_auroc': [], 'time_epoch': []}

    for epoch in range(num_epochs):
        logger.info(f'\nEpoch {epoch+1}/{num_epochs}')

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            # reset metrics
            running_loss = 0.0
            running_corrects = 0
            f1.reset()
            auroc.reset()

            # Iterate over data batches
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #  mode we calculate the loss by summing the final output and the auxiliary output
                    #  but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # metrics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                f1.update(outputs, labels)
                auroc.update(outputs, labels)

            if phase == 'train':
                scheduler.step()

            # Print per epoch metrics
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            epoch_f1 = f1.compute()
            epoch_auroc = auroc.compute()
            
            logger.info(f'\t{phase.capitalize()} - Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} F1: {epoch_f1:.4f} AUROC: {epoch_auroc:.4f}')
            
            # Saving metrics
            if phase == 'train':
                history['train_acc'].append(epoch_acc.cpu())
                history['train_loss'].append(epoch_loss)
                history['train_f1'].append(epoch_f1.cpu())
                history['train_auroc'].append(epoch_auroc.cpu())
            else:
                history['val_acc'].append(epoch_acc.cpu())
                history['val_loss'].append(epoch_loss)
                history['val_f1'].append(epoch_f1.cpu())
                history['val_auroc'].append(epoch_auroc.cpu())
                history['time_epoch'].append(time.time() - start_time)

            # deep copy the model
            if phase == 'val' and epoch_auroc > best_auroc:
                logger.info(f'\tValidation AUROC increased ({best_auroc:.6f} --> {epoch_auroc:.6f}).  Saving model ...')
                best_acc = epoch_acc
                best_loss = epoch_loss
                best_f1 = epoch_f1
                best_auroc = epoch_auroc
                best_epoch = epoch+1
                best_model_wts = copy.deepcopy(model.state_dict())
                # torch.save(model.state_dict(), f'{save_basename}.pt')


    # Total time and best ROC AUC
    logger.info(f'\nTraining completed in {get_elapsed_time(start_time)}.')
    logger.info(f'Best Val AUROC: {best_auroc:4f}, corresponding Val Loss: {best_loss:4f}, Best Epoch: {best_epoch}')
    logger.info('------------------------------------------------------------------------------------')

    # Load to return best model weights
    model.load_state_dict(best_model_wts)
    
    return model, history


def visualize_performance(history):
    '''
    Visualize training and validation performance along with training time per epoch and save plot
    '''
    # training evolution
    fig, ax = plt.subplots(2, 3, figsize=(15, 6))
    fig.suptitle(f"{model_name} Model Performance", fontsize=16, fontweight='bold')
    
    ax = ax.flatten()
    metrics = 'acc', 'loss', 'f1', 'auroc'
    metric_names = 'Accuracy', 'Loss', 'F1-Score', 'ROC AUC'
    for i, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        ax[i].plot(history[f'train_{metric}'], label='Train', marker='o')
        ax[i].plot(history[f'val_{metric}'], label='Validation', marker='o')
        ax[i].set_title(metric_name)
        ax[i].set_xlabel('Epoch')
        ax[i].set_ylabel(metric_name)
        ax[i].set_xticks(np.arange(len(history[f'train_{metric}'])), np.arange(1, len(history[f'train_{metric}'])+1))
        ax[i].legend()
    
    # ax[0].plot(history['train_acc'], label='Train', marker='o')
    # ax[0].plot(history['val_acc'], label='Validation', marker='o')
    # ax[0].set_title('Accuracy')
    # ax[0].set_xlabel('Epoch')
    # ax[0].set_ylabel('Accuracy')
    # ax[0].set_xticks(np.arange(len(history['train_acc'])), np.arange(1, len(history['train_acc'])+1))
    # ax[0].legend()
    
    # ax[1].plot(history['train_loss'], label='Train', marker='o')
    # ax[1].plot(history['val_loss'], label='Validation', marker='o')
    # ax[1].set_title('Loss')
    # ax[1].set_xlabel('Epoch')
    # ax[1].set_ylabel('Loss')
    # ax[1].set_xticks(np.arange(len(history['train_loss'])), np.arange(1, len(history['train_loss'])+1))
    # ax[1].legend()
    
    # ax[2].plot(history['train_f1'], label='Train', marker='o')
    # ax[2].plot(history['val_f1'], label='Validation', marker='o')
    # ax[2].set_title('F1-Score')
    # ax[2].set_xlabel('Epoch')
    # ax[2].set_ylabel('F1-Score')
    # ax[2].set_xticks(np.arange(len(history['train_f1'])), np.arange(1, len(history['train_f1'])+1))
    # ax[2].legend()
    
    # ax[3].plot(history['train_auroc'], label='Train', marker='o')
    # ax[3].plot(history['val_auroc'], label='Validation', marker='o')
    # ax[3].set_title('ROC AUC')
    # ax[3].set_xlabel('Epoch')
    # ax[3].set_ylabel('ROC AUC')
    # ax[3].set_xticks(np.arange(len(history['train_auroc'])), np.arange(1, len(history['train_auroc'])+1))
    # ax[3].legend()
    
    ax[4].plot(history['time_epoch'], label='Time', marker='o')
    ax[4].set_title('Time')
    ax[4].set_xlabel('Epoch')
    ax[4].set_ylabel('Time (sec)')
    ax[4].set_xticks(np.arange(len(history['time_epoch'])), np.arange(1, len(history['time_epoch'])+1))

    ax[5].axis('off')

    plt.tight_layout()
    plt.savefig(f'{save_basename}.png')
    plt.close()

    
def main():  
    '''
    Initialize model, get data loaders, train model and visualize performance
    Return model and history and path to where model is saved
    '''
    
    start_time = time.time()
    get_dataset_info()
    print_settings_info()

    # Initialize model
    model, input_size = initialize_model()
    model = model.to(device)  # move model to GPU if available else CPU
    
    # Print params to learn (this looks a bit messy, should we just remove it?)
    # print_params_to_learn(model)
    
    # Get data loaders
    dataloaders = get_data_loaders(input_size, batch_size)
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    
    # Decay LR by a factor of 0.1 every 5 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1) 
    
    # Train model and get history
    is_inception = (model_name == 'InceptionV3')
    trained_model, history = train_model(model, dataloaders, criterion, optimizer, exp_lr_scheduler, epochs, is_inception)

    # Plot training performance
    # visualize_performance(history)

    logger.info(f'\nFinished! Total run time: {get_elapsed_time(start_time)}')


if __name__ == '__main__':
    main()


# list of changes
# ================
# changed from printing to logging
# added tracking f1 and auroc during training
# updated visualization function to include f1 and auroc
# added a timing function
# added a model evaluation function (confusion matrix, ROC, PRC)
# eventually moving classifier evaluation to a separate script
# removed all references to test split, this should probably be a script on its own at the very end of the project
# renamed function get_transformer to get_transforms, as transformer could cause confusion since it is a type of model
# moved printing of model params to its own function, to make the main loop look cleaner
# shortened savename, made it consistent across model checkpoints, logs, and plots
# modified some functions to take arguments from the global scope (model_name, pretrained, freeze), simplifying a bit
# cleaned up a bit the datasets printout

# NOTE: InceptionV3 takes about 1h/epoch only for synth data. why?