# STGN

## Files
+ tm_train_hy_nruns.py  The entry and main code. Read paramters from json file, run experiments and log results.
+ cmd_args_sst.py       The command arguments.
+ data/
    + sst_dataset.py    Dataloader and Models for experiments.

## Shell

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
  --params_path choose_params/$method/best_params$noise_rate.json \
  --out_tmp sst_out_tmp.json \
  --sub_script sbatch_sst_hy_sub.sh
done

```

You can change arguments for different experiments.

+ method 
    + You can choose `['base', 'GCE', 'GNMO', 'GNMP', 'SLN', 'STGN', 'STGN_GCE']`
+ noise_rate
    + For `'base'`, you can choose `[0.0, 0.1, 0.2, 0.4, 0.6]`.
    + For other methods, you can choose `[ 0.1, 0.2, 0.4, 0.6]`.