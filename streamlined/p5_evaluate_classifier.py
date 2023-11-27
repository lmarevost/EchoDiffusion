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
from torchmetrics.classification import MulticlassF1Score, MulticlassAUROC, MulticlassAveragePrecision
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
from sklearn.metrics import ConfusionMatrixDisplay
# warnings
import warnings
warnings.filterwarnings("ignore")


# --------------------------------------------
# ARG PARSER
# --------------------------------------------
parser = argparse.ArgumentParser()
# dirs
parser.add_argument('--val_dir', type=str, required=True, help='Path to validation set')
parser.add_argument('--log_dir', type=str, required=True, help='Path to log evaluation results')
# model flags
parser.add_argument('--model_checkpoint', type=str, required=True, help='Path to model checkpoint to evaluate')
# hyperparameters
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
args = parser.parse_args()

# Set device to GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --------------------------------------------
# TRAIN, VAL, SAVE PATH
# --------------------------------------------
# train & val datasets
valdir = args.val_dir
# save path for model checkpoints and logs
log_dir = Path(args.log_dir)
log_dir.mkdir(parents=True, exist_ok=True)

# --------------------------------------------
# MODEL FLAGS
# --------------------------------------------
model_checkpoint = Path(args.model_checkpoint)
assert os.path.isfile(model_checkpoint), f'No model checkpoint found at {model_checkpoint}.'
for m in ('VGG16', 'ResNet18', 'DenseNet121', 'InceptionV3'):
    print(m, args.model_checkpoint)
    if m in args.model_checkpoint:
        model_name = m
        break
else:
    raise ValueError('Unrecognized model from provided checkpoint.')

# --------------------------------------------
# LOGGING
# --------------------------------------------
filename = Path(model_checkpoint).stem  # removing path and extension, we can then match train logs to eval logs.
save_basename = os.path.join(log_dir, f'{filename}_eval')
logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)
logger.addHandler(logging.FileHandler(f'{save_basename}.log', 'w'))
logger.info(f"Logging to:\n'{save_basename}.log'")
# log file is built from model checkpoint path. e.g.:
# model_ckpt:   /mimer/NOBACKUP/priv/chair/echo_anouka_luismi/streamlined/saved_classifiers/synth_data/VGG16_FF_2023_04_23_122735.pt
# log_file:     /mimer/NOBACKUP/priv/chair/echo_anouka_luismi/streamlined/saved_classifiers/synth_data/eval/VGG16_FF_2023_04_23_122735_eval.log

# --------------------------------------------
# SETTING HYPERPARAMETERS
# --------------------------------------------
val_classes = len(os.listdir(valdir))  # if 3 then we need to ignore PSAX for plots
ignore_psax = (val_classes == 3)
n_classes = 2 if val_classes == 2 else 4  # hacky fix, needs correction if time allows
class_labels = {0:'A4C', 1:'PLAX'} if n_classes == 2 else {0:'A2C', 1:'A4C', 2:'PLAX', 3:'PSAX'}
batch_size = args.batch_size

# --------------------------------------------
# METRICS
# --------------------------------------------
average = None  # None for per-class metrics, 'macro' for macro average
thresholds = None  # number of thresholds for computing pairs of (fpr, tpr) in ROC or (precision, recall) in PRC. thresholds evenly space in [0, 1] range. None for automatic thresholds
f1 = MulticlassF1Score(num_classes=n_classes, average=average).to(device)
auroc = MulticlassAUROC(num_classes=n_classes, average=average).to(device)
auprc = MulticlassAveragePrecision(num_classes=n_classes, average=average).to(device)
cm = MulticlassConfusionMatrix(num_classes=n_classes, normalize='none').to(device)
cm_norm = MulticlassConfusionMatrix(num_classes=n_classes, normalize='true').to(device)
prc = MulticlassPrecisionRecallCurve(num_classes=n_classes, thresholds=thresholds).to(device)
roc = MulticlassROC(num_classes=n_classes, thresholds=thresholds).to(device)


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
    for class_dir in os.listdir(valdir):
        class_dir_path = os.path.join(valdir, class_dir)
        count_class_dir = len(os.listdir(class_dir_path))
        count_sub_dir += count_class_dir
    logger.info(f'  - Val: {count_sub_dir:,} images found in {valdir}')


def print_settings_info():
    ''' Prints the settings used for training '''
    logger.info(
        f'\nSETTINGS: \
        \n  - Batch size:       {batch_size} \
        \n  - Num of Classes:   {n_classes} \
        \n  - Device used:      {device} \
        \n\nRESULTS SAVED TO: {log_dir}'
        )
    logger.info('*'*100)


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
    (pretrained, freeze) = (True, True) if 'TT' in args.model_checkpoint else (False, False)  # hacky shortcut, might fix later
    logger.info('\n------------------------------------------------------------------------------------')
    logger.info(f'MODEL: {model_name}, with pretrained weights: {pretrained}, and frozen layers: {freeze}')
    logger.info(f'model checkpoint: {args.model_checkpoint}')
    logger.info('------------------------------------------------------------------------------------')

    input_size=112
    
    if model_name == 'VGG16':
        # VGG16
        model = models.vgg16()
        model.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) # Change first layer input channels to 1
        # Update final layer
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, n_classes)
           
    elif model_name == 'ResNet18':
        model = models.resnet18()
        model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) # Change first layer input channels to 1
        # Update final layer
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, n_classes) # reset final layer (requireds_grad=True by default)
        
    elif model_name == 'DenseNet121':
        model = models.densenet121()
        model.features[0] = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) # Change first layer input channels to 1
        # Update final layer
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, n_classes)  
    
    elif model_name == 'InceptionV3':
        model = models.inception_v3()
        # Change first layer input channels to 3
        model.Conv2d_1a_3x3.conv = nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        # Update final layer for Auxilary net
        num_ftrs1 = model.AuxLogits.fc.in_features
        model.AuxLogits.fc = nn.Linear(num_ftrs1, n_classes)
        # Update final layer for main net
        num_ftrs2 = model.fc.in_features 
        model.fc = nn.Linear(num_ftrs2, n_classes)
        input_size = 299
 
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


def get_data_loaders(input_size):
    ''' 
    Returns dataloaders for train and val sets in dictionary format
    '''
    # get transformer
    transform = get_transforms(input_size)
    
    # Datasets from each folder
    data = {
        'val': datasets.ImageFolder(root=valdir, transform=transform),
    }

    logger.info('\nClass mapping:')
    logger.info(data['val'].class_to_idx)


    # Dataloader iterators
    dataloaders = {
        'val': DataLoader(data['val'], batch_size=batch_size, shuffle=False),
    }

    return dataloaders


def main():  
    '''
    Initialize model, get data loaders, train model and visualize performance
    Return model and history and path to where model is saved
    '''
    
    start_time = time.time()
    get_dataset_info()
    print_settings_info()

    # initialize model
    model, input_size = initialize_model()
    model = model.to(device)  # move model to GPU if available else CPU

    # load model checkpoint
    model.load_state_dict(torch.load(model_checkpoint))
    model.eval()
    
    # get data loaders
    dataloaders = get_data_loaders(input_size)
    
    # evaluate
    evaluate_model(model, dataloaders)

    logger.info(f'\nFinished! Total run time: {get_elapsed_time(start_time)}')


def evaluate_model(model, dataloaders):

    # Reset metrics
    f1.reset()
    auroc.reset()
    auprc.reset()
    cm.reset()
    cm_norm.reset()
    prc.reset()
    roc.reset()

    # Iterate over data batches
    for inputs, labels in dataloaders['val']:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        # Update metrics
        f1.update(outputs, labels)
        auroc.update(outputs, labels)
        auprc.update(outputs, labels)
        cm.update(outputs, labels)
        cm_norm.update(outputs, labels)
        prc.update(outputs, labels)
        roc.update(outputs, labels)

    # Compute metrics
    eval_f1 = f1.compute().cpu().numpy()
    eval_auroc = auroc.compute().cpu().numpy()
    eval_auprc = auprc.compute().cpu().numpy()
    eval_cm = cm.compute().cpu().numpy()  # rows: ground truth, columns: predictions
    eval_cm_norm = cm_norm.compute().cpu().numpy()  # rows: ground truth, columns: predictions
    precision, recall, _ = prc.compute()
    fpr, tpr, _ = roc.compute()

    if ignore_psax:
        # removing zeros corresponding to PSAX, if model was trained on 4 classes but evaluated on 3
        precision = precision[:-1]
        recall = recall[:-1]
        fpr = fpr[:-1]
        tpr = tpr[:-1]
        eval_f1 = eval_f1[:-1]
        eval_auroc = eval_auroc[:-1]
        eval_auprc = eval_auprc[:-1]
    
    logger.info(f'\nNumber of classes (train): {n_classes}')
    logger.info(f'Number of classes (val): {val_classes}')
    logger.info(f'F1-Score per class: { {class_labels[i]: v for i, v in enumerate(eval_f1)} }')
    logger.info(f'ROC AUC per class: { {class_labels[i]: v for i, v in enumerate(eval_auroc)} }')
    logger.info(f'PRC AUC per class: { {class_labels[i]: v for i, v in enumerate(eval_auprc)} }')
    logger.info(f'Macro averages:')
    logger.info(f'F1-Score: {eval_f1.mean()}')
    logger.info(f'ROC AUC: {eval_auroc.mean()}')
    logger.info(f'PRC AUC: {eval_auprc.mean()}')

    # Plotting Confusion Matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=eval_cm, display_labels=[class_labels[i] for i in range(n_classes)])
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.savefig(f"{save_basename}_CM.png")
    plt.close()

    # Plotting Confusion Matrix Normalized
    disp = ConfusionMatrixDisplay(confusion_matrix=eval_cm_norm, display_labels=[class_labels[i] for i in range(n_classes)])
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix (Normalized)')
    plt.savefig(f"{save_basename}_CM_norm.png")
    logger.info(f'\nNormalized Confusion Matrix plot saved to {save_basename}_CM_norm.png')
    plt.close()
        
    # Plotting PRC
    for i, (c_precision, c_recall) in enumerate(zip(precision, recall)):
        plt.plot(c_recall.cpu(), c_precision.cpu(), label=class_labels[i])
    plt.title('Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.savefig(f"{save_basename}_PRC.png")
    logger.info(f'\nPrecision-Recall Curve plot saved to {save_basename}_PRC.png')
    plt.close()

    # Plotting ROC
    for i, (c_fpr, c_tpr) in enumerate(zip(fpr, tpr)):
        plt.plot(c_fpr.cpu(), c_tpr.cpu(), label=class_labels[i])
    plt.title('Receiver Operating Characteristic')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend()
    plt.savefig(f"{save_basename}_ROC.png")
    logger.info(f'\nReceiver Operating Characteristic plot saved to {save_basename}_ROC.png')
    plt.close()


if __name__ == '__main__':
    main()
    # runs in about 1 min