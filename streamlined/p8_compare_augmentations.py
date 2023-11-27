import os
import math
import matplotlib.pyplot as plt


def main():

    root = '/mimer/NOBACKUP/priv/chair/echo_anouka_luismi/streamlined/synth_augmentation'
    datasets = [f for f in os.listdir(root) if '4class' in f]  # e.g. ['4class_balanced_2720', ...]
    datasets = sorted(datasets, key=lambda x: int(x.split('_')[-1]))  # sorting low to high size
    
    for dataset in datasets:
        size = int(dataset.split('_')[-1])
        log_files = [f for f in os.listdir(os.path.join(root, dataset)) if '.log' in f]  # e.g. ['VGG16_FF_0.1_2023_05_01_151250.log', ...]
        log_files = sorted(log_files, key=lambda x: float(x.split('_')[2]))  # sorting low to high by augmentation fraction
        print(f'\n{dataset}')
        fracs = []
        aurocs = []
        for log in log_files:
            with open(os.path.join(root, dataset, log), 'r') as f:
                for line in f:
                    if '- Augmentation fraction:' in line:
                        frac = float(line.strip().split(':')[-1])
                    if 'Best Val AUROC' in line:
                        auroc = float(line.split(',')[0].split(' ')[-1])
            fracs.append(frac)
            aurocs.append(auroc)
            print(f'  Frac: {frac} - ROC AUC: {auroc:.6f}')
        plt.plot(fracs, aurocs, label=f'{size:,}')
    plt.title('Augmenting real with synthetic data')
    plt.ylabel('Val ROC AUC')
    plt.xlabel('Augmentation fraction')
    plt.legend(title='Real dataset size')
    plt.tight_layout()
    plt.savefig('/mimer/NOBACKUP/priv/chair/echo_anouka_luismi/streamlined/augmentation_comparison.png')


if __name__ == '__main__':
    main()