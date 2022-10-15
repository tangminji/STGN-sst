import argparse


# Generate shells for sst-2
nrun_shell='''#!/bin/bash
            
#SBATCH -J nrun{noise_rate}
#SBATCH -p compute
#SBATCH -N 1
#SBATCH --gres gpu:tesla_v100-sxm2-16gb:1
#SBATCH -t 2:00:00
#SBATCH --mem 20240
#SBATCH -e output2/nrun{noise_rate}.err
#SBATCH -o output2/nrun{noise_rate}.txt

source ~/.bashrc
conda activate base

noise_rate={noise_rate}
method={method}

for i in 0 1 2
do
python tm_train_hy_nruns.py \\
  --dataset SST \\
  --noise_rate $noise_rate \\
  --seed $i \\
  --num_class 2 \\
  --exp_name nrun2/SST_$method/nr$noise_rate/seed$i \\
  --params_path best_params$noise_rate.json \\
  --out_tmp sst_out_tmp.json \\
  --sub_script sbatch_sst_hy_sub.sh
done
'''

hy_shell='''#!/bin/bash
            
#SBATCH -J hy_{noise_rate}
#SBATCH -p compute
#SBATCH -N 1
#SBATCH --gres gpu:tesla_v100-sxm2-16gb:1
#SBATCH -t 6:00:00
#SBATCH --mem 20240
#SBATCH -o output2/{method}_hy{noise_rate}.out
#SBATCH -e output2/{method}_hy{noise_rate}.err


source ~/.bashrc
conda activate base

i=0
noise_rate={noise_rate}
method={method}

python tm_train_hy_params.py \\
--dataset SST \\
--noise_rate $noise_rate \\
--seed $i \\
--num_class 2 \\
--exp_name hy2/SST_$method/nr$noise_rate/ \\
--params_path sst_params$noise_rate.json \\
--out_tmp sst_out_tmp$noise_rate.json \\
--sub_script sbatch_sst_hy_sub$noise_rate.sh
'''

sub_shell='''#!/bin/bash

i=0
noise_rate={noise_rate}
method={method}

python tm_train_hy_sub.py \\
--dataset SST \\
--noise_rate $noise_rate \\
--seed $i \\
--num_class 2 \\
--exp_name hy2/SST_$method/nr$noise_rate/ \\
--params_path sst_params$noise_rate.json \\
--out_tmp sst_out_tmp$noise_rate.json \\
--sub_script sbatch_sst_hy_sub$noise_rate.sh
'''

method='STGN'

mode='nrun' #hy, mode

def main():
    if mode=='hy':
        for noise_rate in [0.2,0.4,0.6]:
            with open(f"sbatch_sst_hy_params{noise_rate}.sh","w") as f:
                f.write(hy_shell.format(noise_rate=noise_rate,method=method))
            with open(f"sbatch_sst_hy_sub{noise_rate}.sh","w") as f:
                f.write(sub_shell.format(noise_rate=noise_rate,method=method))
    else:
        for noise_rate in [0.2,0.4,0.6]:
            with open(f"sbatch_sst_hy_nrun{noise_rate}.sh","w") as f:
                f.write(nrun_shell.format(noise_rate=noise_rate,method=method))

if __name__ == '__main__':
    main()