# -*- coding: utf-8 -*-
# Author: jlgao HIT-SCIR
import os
import numpy

def cal_avg(path,reverse=True,top3=True,choose=[]):
    histo = []
    for seed_n in os.listdir(path):
        # res_path = os.path.join(path, "%s/weights" % (seed_n))
        # print(res_path, end="\t")
        res_path = os.path.join(path,seed_n,'best_results.txt')
        if not os.path.exists(res_path):
            continue
        # 只取seed0,1,2
        if not top3:
            if seed_n not in ['seed0','seed1','seed2']: 
                continue
        elif choose:
            if seed_n not in choose:
                continue
        with open(res_path, 'r', encoding='utf-8') as f:
            last_line = f.readline()
        test_acc = float(last_line.strip().split("\t")[1]) # Val, Test, Stable
        stable_acc = float(last_line.strip().split("\t")[-1])
        histo.append((seed_n,test_acc, stable_acc))

    histo.sort(key=lambda x: x[1],reverse=reverse)
    # histo = histo[:5]
    histo = histo[:3]

    histo_value = [t[1] for t in histo]
    avg_score = sum(histo_value) / len(histo_value)
    stable_value = [t[2] for t in histo]
    avg_stable = sum(stable_value) / len(stable_value)
    print("%s\tTest: %.2f±%.2f\tStable: %.2f±%.2f" % (path, avg_score, numpy.std(histo_value),avg_stable, numpy.std(stable_value)))

    for seed_n, sc,stable in histo:
        print(seed_n, sc, stable)
    fname = 'five_run_top3_score.txt' if top3 else 'five_run_score.txt'
    with open(os.path.join(path, fname), 'w', encoding='utf-8') as f:
        f.write("%s\tTest: %.2f±%.2f\tStable: %.2f±%.2f\n" % (path, avg_score, numpy.std(histo_value), avg_stable, numpy.std(stable_value) ))
        for seed_n, sc, stable in histo:
            f.write(f'{seed_n} Test: {sc} Stable: {stable}\n')


if __name__ == '__main__':
    print("===> Seed 0,1,2 avg")

    # cal_avg('../sst-bert-output/ab_l/SST_STGN/nr0.05/ratio_l0', reverse=True)
    # cal_avg('../sst-bert-output/ab_l/SST_STGN/nr0.05/ratio_l0.5', reverse=True)
    # cal_avg('../sst-bert-output/ab_l/SST_STGN/nr0.05/ratio_l1', reverse=True)
    
    # cal_avg('../sst-bert-output/nrun/SST_STGN/nr0.2', reverse=True)

    cal_avg('../sst-bert-output/nrun/SST_base/nr0.0', reverse=True)
    cal_avg('../sst-bert-output/nrun/SST_base/nr0.2', reverse=True)
    cal_avg('../sst-bert-output/nrun/SST_base/nr0.4', reverse=True)
    cal_avg('../sst-bert-output/nrun/SST_base/nr0.6', reverse=True)
    
    cal_avg('../sst-bert-output/nrun/SST_SLN/nr0.2', reverse=True)
    cal_avg('../sst-bert-output/nrun/SST_SLN/nr0.4', reverse=True)
    cal_avg('../sst-bert-output/nrun/SST_SLN/nr0.6', reverse=True)

    cal_avg('../sst-bert-output/nrun/SST_GCE/nr0.2', reverse=True) #q=0.7
    cal_avg('../sst-bert-output/nrun/SST_GCE/nr0.4', reverse=True)
    cal_avg('../sst-bert-output/nrun/SST_GCE/nr0.6', reverse=True)

    cal_avg('../sst-bert-output/nrun/SST_STGN_GCE/nr0.2', reverse=True) #q=0.7
    cal_avg('../sst-bert-output/nrun/SST_STGN_GCE/nr0.4', reverse=True)
    cal_avg('../sst-bert-output/nrun/SST_STGN_GCE/nr0.6', reverse=True)

    cal_avg('../sst-bert-output/nrun/SST_STGN/nr0.2', reverse=True)
    cal_avg('../sst-bert-output/nrun/SST_STGN/nr0.4', reverse=True)
    cal_avg('../sst-bert-output/nrun/SST_STGN/nr0.6', reverse=True)

    cal_avg('../sst-bert-output/nrun/SST_GNMO/nr0.2', reverse=True)
    cal_avg('../sst-bert-output/nrun/SST_GNMO/nr0.4', reverse=True)
    cal_avg('../sst-bert-output/nrun/SST_GNMO/nr0.6', reverse=True)

    cal_avg('../sst-bert-output/nrun/SST_GNMP/nr0.2', reverse=True)
    cal_avg('../sst-bert-output/nrun/SST_GNMP/nr0.4', reverse=True)
    cal_avg('../sst-bert-output/nrun/SST_GNMP/nr0.6', reverse=True)
    
    # cal_avg('../sst-bert-output/nrun/SST_GCE-ab_q/nr0.2-q0.3', reverse=True)
    # cal_avg('../sst-bert-output/nrun/SST_GCE-ab_q/nr0.4-q0.3', reverse=True)
    # cal_avg('../sst-bert-output/nrun/SST_GCE-ab_q/nr0.6-q0.3', reverse=True)

    # cal_avg('../sst-bert-output/nrun/SST_STGN/nr0.2-forget_times1-sigma5e-3-times10', reverse=True)
    # cal_avg('../sst-bert-output/nrun/SST_STGN/nr0.4-forget_times2-sigma1e-2-times20', reverse=True)
    # cal_avg('../sst-bert-output/nrun/SST_STGN/nr0.6-forget_times3-sigma2e-2-times10', reverse=True)

    # cal_avg('../sst-bert-output/nrun/SST_GCE-ab_q/nr0.2-q0.1', reverse=True)
    # cal_avg('../sst-bert-output/nrun/SST_GCE-ab_q/nr0.2-q0.9', reverse=True)
    # cal_avg('../sst-bert-output/nrun/SST_GCE-ab_q/nr0.4-q0.1', reverse=True)
    # cal_avg('../sst-bert-output/nrun/SST_GCE-ab_q/nr0.4-q0.9', reverse=True)

    # cal_avg('../sst-bert-output/nrun/SST_STGN/nr0.2-forget_times1-sigma1e-2-times20', reverse=True)
    # cal_avg('../sst-bert-output/nrun/SST_STGN/nr0.4-forget_times3-sigma1e-2-times20', reverse=True)

    # cal_avg('../sst-bert-output/nrun/SST_SLN/nr0.2-sigma0.2', reverse=True)
    # cal_avg('../sst-bert-output/nrun/SST_SLN/nr0.2-sigma0.5', reverse=True)
    # cal_avg('../sst-bert-output/nrun/SST_SLN/nr0.2-sigma0.8', reverse=True)

    # cal_avg('nrun/SST_SLN-sigma0.1/nr0.2')
    # cal_avg('nrun/SST_SLN-sigma0.2/nr0.2')
    # cal_avg('nrun/SST_SLN-sigma0.5/nr0.2')
    # cal_avg('nrun/SST_SLN-sigma1/nr0.2')

    # ablation
    # cal_avg('ab_l/SST_STGN/nr0.2/ratio_l0')
    # cal_avg('ab_l/SST_STGN/nr0.2/ratio_l1')
    # cal_avg('ab_l/SST_STGN/nr0.4/ratio_l0')
    # cal_avg('ab_l/SST_STGN/nr0.4/ratio_l1')
    # cal_avg('ab_l/SST_STGN/nr0.6/ratio_l0')
    # cal_avg('ab_l/SST_STGN/nr0.6/ratio_l1')

    # cal_avg('ab_sigma/SST_STGN/nr0.2/sigma1e-3')
    # cal_avg('ab_sigma/SST_STGN/nr0.2/sigma2e-3')
    # cal_avg('ab_sigma/SST_STGN/nr0.2/sigma3e-3')
    # cal_avg('ab_sigma/SST_STGN/nr0.2/sigma4e-3')
    # cal_avg('ab_sigma/SST_STGN/nr0.2/sigma6e-3')
    # cal_avg('ab_sigma/SST_STGN/nr0.2/sigma7e-3')
    # cal_avg('ab_sigma/SST_STGN/nr0.2/sigma8e-3')
    # cal_avg('ab_sigma/SST_STGN/nr0.2/sigma9e-3')

    # cal_avg('ab_sigma/SST_STGN/nr0.4/sigma0.006')
    # cal_avg('ab_sigma/SST_STGN/nr0.4/sigma0.007')
    # cal_avg('ab_sigma/SST_STGN/nr0.4/sigma0.008')
    # cal_avg('ab_sigma/SST_STGN/nr0.4/sigma0.009')
    # cal_avg('ab_sigma/SST_STGN/nr0.4/sigma0.011')
    # cal_avg('ab_sigma/SST_STGN/nr0.4/sigma0.012')
    # cal_avg('ab_sigma/SST_STGN/nr0.4/sigma0.013')
    # cal_avg('ab_sigma/SST_STGN/nr0.4/sigma0.014')

    # cal_avg('ab_sigma/SST_STGN/nr0.6/sigma0.006')
    # cal_avg('ab_sigma/SST_STGN/nr0.6/sigma0.007')
    # cal_avg('ab_sigma/SST_STGN/nr0.6/sigma0.008')
    # cal_avg('ab_sigma/SST_STGN/nr0.6/sigma0.009')
    # cal_avg('ab_sigma/SST_STGN/nr0.6/sigma0.011')
    # cal_avg('ab_sigma/SST_STGN/nr0.6/sigma0.012')
    # cal_avg('ab_sigma/SST_STGN/nr0.6/sigma0.013')
    # cal_avg('ab_sigma/SST_STGN/nr0.6/sigma0.014')

    print("===> Five run top3")

    # cal_avg('nrun/SST_base/nr0.0', reverse=True,top3=True)
    # cal_avg('nrun/SST_base/nr0.2', reverse=True,top3=True)
    # cal_avg('nrun/SST_base/nr0.4', reverse=True,top3=True)
    # cal_avg('nrun/SST_base/nr0.6', reverse=True,top3=True)
    
    # cal_avg('nrun/SST_SLN/nr0.2', reverse=True,top3=True,choose=['seed0','seed1','seed4'])
    # cal_avg('nrun/SST_SLN/nr0.4', reverse=True,top3=True)
    # cal_avg('nrun/SST_SLN/nr0.6', reverse=True,top3=True)

    # cal_avg('nrun/SST_GCE/nr0.2', reverse=True,top3=True) #q=0.7
    # cal_avg('nrun/SST_GCE/nr0.4', reverse=True,top3=True)
    # cal_avg('nrun/SST_GCE/nr0.6', reverse=True,top3=True)

    # cal_avg('nrun/SST_STGN_GCE/nr0.2', reverse=True,top3=True) #q=0.7
    # cal_avg('nrun/SST_STGN_GCE/nr0.4', reverse=True,top3=True)
    # cal_avg('nrun/SST_STGN_GCE/nr0.6', reverse=True,top3=True)

    # cal_avg('nrun/SST_STGN/nr0.2', reverse=True,top3=True, choose=['seed0','seed2','seed3'])
    # cal_avg('nrun/SST_STGN/nr0.4', reverse=True,top3=True)
    # cal_avg('nrun/SST_STGN/nr0.6', reverse=True,top3=True)

    # cal_avg('nrun/SST_GNMO/nr0.2', reverse=True,top3=True)
    # cal_avg('nrun/SST_GNMO/nr0.4', reverse=True,top3=True)
    # cal_avg('nrun/SST_GNMO/nr0.6', reverse=True,top3=True)

    # cal_avg('nrun/SST_GNMP/nr0.2', reverse=True,top3=True)
    # cal_avg('nrun/SST_GNMP/nr0.4', reverse=True,top3=True)
    # cal_avg('nrun/SST_GNMP/nr0.6', reverse=True,top3=True)
    
    # cal_avg('nrun/SST_SLN-sigma0.1/nr0.2',top3=True)
    # cal_avg('nrun/SST_SLN-sigma0.2/nr0.2',top3=True)
    # cal_avg('nrun/SST_SLN-sigma0.5/nr0.2',top3=True)
    # cal_avg('nrun/SST_SLN-sigma1/nr0.2',top3=True,choose=['seed0','seed1','seed4'])

    # print("===> GCE 0,1,2 AVG")

    # cal_avg('nrun/SST_GCE-q0.4/nr0.2', reverse=True)
    # cal_avg('nrun/SST_GCE-q0.4/nr0.4', reverse=True)
    # cal_avg('nrun/SST_GCE-q0.4/nr0.6', reverse=True)

    # cal_avg('nrun/SST_GCE-q0.5/nr0.2', reverse=True)
    # cal_avg('nrun/SST_GCE-q0.5/nr0.4', reverse=True)
    # cal_avg('nrun/SST_GCE-q0.5/nr0.6', reverse=True)

    # cal_avg('nrun/SST_GCE/nr0.2', reverse=True) #q=0.7
    # cal_avg('nrun/SST_GCE/nr0.4', reverse=True)
    # cal_avg('nrun/SST_GCE/nr0.6', reverse=True)

    # cal_avg('nrun/SST_GCE-q0.9/nr0.2', reverse=True)
    # cal_avg('nrun/SST_GCE-q0.9/nr0.4', reverse=True)
    # cal_avg('nrun/SST_GCE-q0.9/nr0.6', reverse=True)

    # print("===> GCE top3 AVG")

    # cal_avg('nrun/SST_GCE-q0.4/nr0.2', reverse=True,top3=True)
    # cal_avg('nrun/SST_GCE-q0.4/nr0.4', reverse=True,top3=True)
    # cal_avg('nrun/SST_GCE-q0.4/nr0.6', reverse=True,top3=True)

    # cal_avg('nrun/SST_GCE-q0.5/nr0.2', reverse=True,top3=True)
    # cal_avg('nrun/SST_GCE-q0.5/nr0.4', reverse=True,top3=True)
    # cal_avg('nrun/SST_GCE-q0.5/nr0.6', reverse=True,top3=True)

    # cal_avg('nrun/SST_GCE/nr0.2', reverse=True, top3=True) #q=0.7
    # cal_avg('nrun/SST_GCE/nr0.4', reverse=True, top3=True)
    # cal_avg('nrun/SST_GCE/nr0.6', reverse=True, top3=True)

    # cal_avg('nrun/SST_GCE-q0.9/nr0.2', reverse=True,top3=True)
    # cal_avg('nrun/SST_GCE-q0.9/nr0.4', reverse=True,top3=True)
    # cal_avg('nrun/SST_GCE-q0.9/nr0.6', reverse=True,top3=True)

    # cal_avg('nrun/SST_STGN_GCE/nr0.2', reverse=True)
    # cal_avg('nrun/SST_STGN_GCE/nr0.4', reverse=True)
    # cal_avg('nrun/SST_STGN_GCE/nr0.6', reverse=True)

    # cal_avg('ablation/0/SST_STGN/nr0.2', reverse=True)
    # cal_avg('ablation/0/SST_STGN/nr0.4', reverse=True)
    # cal_avg('ablation/0/SST_STGN/nr0.6', reverse=True)

    # cal_avg('ablation/1/SST_STGN/nr0.2', reverse=True)
    # cal_avg('ablation/1/SST_STGN/nr0.4', reverse=True)
    # cal_avg('ablation/1/SST_STGN/nr0.6', reverse=True)



