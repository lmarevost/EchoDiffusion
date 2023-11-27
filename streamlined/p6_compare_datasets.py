import os
import math
import matplotlib.pyplot as plt


def main():

    root = '/mimer/NOBACKUP/priv/chair/echo_anouka_luismi/streamlined/saved_classifiers'
    datasets = [f for f in os.listdir(root) if '4class' in f]
    datasets = sorted(datasets, key=lambda x: int(x.split('_')[-1]))  # sorting low to high size
    xlabels = [int(x.split('_')[-1]) for x in datasets]
    xlogs = [math.log(x) for x in xlabels]
    
    reals = []
    fakes = []
    for dataset in datasets:
        log_files = [f for f in os.listdir(os.path.join(root, dataset)) if '.log' in f]
        print(f'\n{dataset}')
        for log in log_files:
            with open(os.path.join(root, dataset, log), 'r') as f:
                for line in f:
                    if '- Train:' in line:
                        data = 'Fake' if 'synthetic_samples' in line else 'Real'
                    if 'Best Val AUROC' in line:
                        auroc = float(line.split(',')[0].split(' ')[-1])
            if data == 'Real':
                reals.append(auroc)
            else:
                fakes.append(auroc)
            print(f'  {data} ROC AUC: {auroc:.6f}')
            del data, auroc
            
    plt.plot(xlogs, reals, label='Real', c='b')
    plt.plot(xlogs, fakes, label='Fake', c='r')
    plt.xticks(xlogs, xlabels)
    plt.title('Val ROC AUC')
    plt.ylabel('Val ROC AUC')
    plt.xlabel('Size of original dataset (Log scale)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('/mimer/NOBACKUP/priv/chair/echo_anouka_luismi/streamlined/dataset_comparison.png')


if __name__ == '__main__':
    main()