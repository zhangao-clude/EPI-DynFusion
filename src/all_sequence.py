import os
import numpy as np

# names = ['GM12878', 'HUVEC', 'HeLa-S3', 'IMR90', 'K562', 'NHEK']
names = ['GM12878', 'HUVEC', 'HeLa-S3', 'IMR90', 'K562']

all_enhancers = []
all_promoters = []
all_labels = []


for name in names:
    train_dir = f'data/{name}/train/'

    with open(f'{train_dir}{name}_enhancer.fasta', 'r') as f:
        enhancers_tra = f.read().splitlines()[1::2]

    with open(f'{train_dir}{name}_promoter.fasta', 'r') as f:
        promoters_tra = f.read().splitlines()[1::2]

    y_tra = np.loadtxt(f'{train_dir}{name}_label.txt').tolist()  


    assert len(enhancers_tra) == len(promoters_tra) == len(y_tra)

   
    all_enhancers.extend(enhancers_tra)
    all_promoters.extend(promoters_tra)
    all_labels.extend(y_tra)


assert len(all_enhancers) == len(all_promoters) == len(all_labels)


combined_data = list(zip(all_enhancers, all_promoters, all_labels))
np.random.shuffle(combined_data)

all_enhancers, all_promoters, all_labels = zip(*combined_data)


combined_dir = './data/all/'
os.makedirs(combined_dir, exist_ok=True)


with open(f'{combined_dir}all_enhancer_noNHEK.fasta', 'w') as f:
    for line in all_enhancers:
        f.write(line + '\n')


with open(f'{combined_dir}all_promoter_noNHEK.fasta', 'w') as f:
    for line in all_promoters:
        f.write(line + '\n')


np.savetxt(f'{combined_dir}all_label_noHEK.txt', all_labels, fmt='%d')


