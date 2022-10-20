# STGN

source code and data for
+ [EMNLP2022] STGN: an Implicit Regularization Method for Learning with Noisy
Labels in Natural Language Processing

Note:
+ To reproduce the paper results, you can choose the stable version ['master/v5.0'](https://github.com/tangminji/STGN-sst/tree/v5.0). However, the noise on SST is not in strictly uniform distribution.
+ We will fix data with uniform distribution and adjust code, params in later versions.

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
            
#SBATCH -J base
#SBATCH -p compute
#SBATCH -N 1
#SBATCH --gres gpu:tesla_v100-sxm2-16gb:1
#SBATCH -t 10:00:00
#SBATCH --mem 20240
#SBATCH -e ../sst-bert-output/output/base.err
#SBATCH -o ../sst-bert-output/output/base.txt

source ~/.bashrc
conda activate base

noise_rate=0.0
method=base
i=0

for noise_rate in 0.0 0.2 0.4 0.6
do
python tm_train_hy_nruns.py \
  --dataset SST \
  --noise_rate $noise_rate \
  --seed $i \
  --exp_name ../sst-bert-output/nrun/SST_$method/nr$noise_rate/seed$i \
  --params_path choose_params/$method/best_params$noise_rate.json
done

```

You can change arguments for different experiments.

+ method 
    + You can choose `['base', 'GCE', 'GNMO', 'GNMP', 'SLN', 'STGN', 'STGN_GCE']`
+ noise_rate
    + For `'base'`, you can choose `[0.0, 0.1, 0.2, 0.4, 0.6]`.
    + For other methods, you can choose `[0.1, 0.2, 0.4, 0.6]`.
+ seed (i)
    + You may try many different seeds to analyse the method performance, since the seeds make a difference on the results(peak test acc). For example, you can choose `[0, 1, 2, 3, 4]`.