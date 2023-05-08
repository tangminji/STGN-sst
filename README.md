# STGN: an Implicit Regularization Method for Learning with Noisy Labels in Natural Language Processing

Main experiment of ["STGN: an Implicit Regularization Method for Learning with Noisy Labels in Natural Language Processing"](https://aclanthology.org/2022.emnlp-main.515/) (EMNLP 2022) by Tingting Wu, Xiao Ding, Minji Tang, Hao Zhang, Bing Qin, Ting Liu.

Note:
+ To reproduce the paper results, you can run the stable version ['v5.0'](https://github.com/tangminji/STGN-sst/tree/v5.0) on `tesla_v100-sxm2-16gb`. However, the noise on SST is not in strictly uniform distribution.
+ We will fix data with uniform distribution and adjust code in [`dev`](https://github.com/tangminji/STGN-sst/tree/dev) branch.

Experiment on NoisyNER and wikiHow:
-  NoisyNER: 
   - [our code](https://github.com/tangminji/STGN-NoisyNER)
   - [original work](https://github.com/uds-lsv/noise-estimation)

- wikiHow: 
  - [our code](https://github.com/tangminji/STGN-wikiHow) 
  - [original work](https://github.com/zharry29/wikihow-goal-step)

## Models
+ BERT  bert-base-uncased, batch_size=32, epochs=10, Adam(lr=1e-5)
  + tesla_v100-sxm2-16gb    0.2h/run;   for GMMP:  0.8h/run.

## Files
+ cmd_args_sst.py            The command arguments.
+ tm_train_hy_nruns.py       The entry and main code for BERT. Read paramters from json file, run experiments and log results.
+ data/
    + sst_dataset.py         Dataloader and Models for BERT.
    + sst/                   SST Dataset for 5-class classification.
+ common/                    Useful codes.
+ choose_params/             Best params for BERT.

## Before run
You need to make sure the output folder exists. 
+ with sbatch:
```shell
mkdir -p ../sst-bert-output/output
sbatch base.sh
```
+ without sbatch:
```shell
mkdir -p ../sst-bert-output/output
bash base.sh >../sst-bert-output/output/base.txt 2>../sst-bert-output/output/base.err
```
You can also use run it as a nohupping backgrounded job.

## Shell

For BERT:
```shell
#!/bin/bash
            
#SBATCH -J SST_base
#SBATCH -p compute
#SBATCH -N 1
#SBATCH --gres gpu:tesla_v100-sxm2-16gb:1
#SBATCH -t 10:00:00
#SBATCH --mem 20240
#SBATCH -e ../sst-bert-output/output/SST_base.err
#SBATCH -o ../sst-bert-output/output/SST_base.txt

source ~/.bashrc
conda activate base

dataset=SST
noise_rate=0.0
method=base
i=0

for noise_rate in 0.0 0.2 0.4 0.6
do
  for i in 0 
  do
    echo "${i}"
    python tm_train_hy_nruns.py \
    --dataset $dataset \
    --noise_rate $noise_rate \
    --seed $i \
    --exp_name ../sst-bert-output/nrun/$dataset-$method/nr$noise_rate/seed$i \
    --params_path choose_params/$dataset/$method/best_params$noise_rate.json
  done
done

```

You can change arguments for different experiments.

+ dataset
    + We provide `['SST', 'QQP', 'MNLI']`
    + For `['QQP','MNLI']`, We provide experimental parameters for `['SLN', 'STGN']` under 40% uniform label noise.
+ method 
    + You can choose `['base', 'GCE', 'GNMO', 'GNMP', 'SLN', 'STGN', 'STGN_GCE']`
+ noise_rate
    + For `'base'`, you can choose `[0.0, 0.1, 0.2, 0.4, 0.6]`.
    + For other methods, you can choose `[0.1, 0.2, 0.4, 0.6]`.
+ seed (i)
    + You may try many different seeds to analyse the method performance, since the seeds make a difference on the results(peak test acc). For example, you can choose `[0, 1, 2, 3, 4]`.