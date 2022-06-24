# STGN

## Models
+ LSTM  LSTM+fc, batch_size=32, epoch=30, hidden_size=168, embed_size=300, RMSProp(lr=0.001,alpha=0.9,weight_decay=0.0001)
+ BERT  bert-base-uncased, batch_size=32, epochs=10, Adam(lr=1e-5)

## Files
+ cmd_args_sst.py            The command arguments.
+ tm_train_hy_nruns.py       The entry and main code for BERT. Read paramters from json file, run experiments and log results.
+ tm_train_hy_nruns_lstm.py  The entry and main code for LSTM.
+ data/
    + sst_dataset.py         Dataloader and Models for BERT.
    + sst_dataset_lstm.py    Dataloader and Models for LSTM.
    + sst_preprocess.py      Preprocess data and glove embedding for LSTM. Note you should download glove embeddings.
    + sst/                   SST Dataset for 5-class classification.
    + sst-2/                 SST Dataset for 2-class classification.
+ models/
    + sst_lstm.py            LSTM model.
+ common/                    Useful codes.
+ choose_params/             Best params for BERT.
+ choose_params_lstm/        Best params for LSTM.

## Shell

For LSTM:
```shell
#!/bin/bash
            
#SBATCH -J STGN
#SBATCH -p compute
#SBATCH -N 1
#SBATCH --gres gpu:tesla_v100-sxm2-16gb:1
#SBATCH -t 10:00:00
#SBATCH --mem 20240
#SBATCH -e ../sst-lstm-output/output/STGN.err
#SBATCH -o ../sst-lstm-output/output/STGN.txt

source ~/.bashrc
conda activate base

noise_rate=0.0
method=STGN
i=0

for noise_rate in 0.0 0.2 0.4 0.6
do
python tm_train_hy_nruns_lstm.py \
  --dataset SST \
  --noise_rate $noise_rate \
  --seed $i \
  --exp_name ../sst-lstm-output/nrun/SST_$method/nr$noise_rate/seed$i \
  --params_path choose_params_lstm/$method/best_params$noise_rate.json
done
```

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