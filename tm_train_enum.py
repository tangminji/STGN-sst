# our method main.py
# tailor-made regularization
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2019 Apple Inc. All Rights Reserved.
#
import json
import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np

from cmd_args_sst import args

best_acc = 0
ITERATION = 0

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

def main(params):
    global ITERATION
    ITERATION += 1
    params['ITERATION'] = ITERATION
    json.dump(params, open(args.params_path, 'w+', encoding="utf-8"), ensure_ascii=False)
    sig = os.system("sh %s" % args.sub_script)
    assert sig == 0
    res = json.load(open(args.out_tmp, 'r', encoding="utf-8"))
    return res

def get_trials(fixed, space, MAX_EVALS):
    for k in space:
        times = len(space[k])
        break
    if times > MAX_EVALS:
        times = MAX_EVALS
    for t in range(times):
        params = {k: space[k][t] for k in space}
        params.update(fixed)
        yield params

# enum settings which run once

if __name__ == '__main__':

    assert args.dataset == "SST"

    MAX_EVALS = 10

    fixed = {

    }
    space = {

    }
    # TODO: times, sigma (key hyperparameters)
    # space中所有参数都需要是hp对象，否则best会缺失相应超参数值
    # --noise_rate 0.4 \ #generally known
    # --forget_times 10 \ discrete value
    # --ratio_l 1.0 \
    # --times 50.0 \
    # --avg_steps 20 \
    # --sigma 1e-3 \
    # --sig_max  2e-3 \ #can be obtained by sigma
    # --lr_sig 1e-4 \ #can be obtained by sigma

    if 'STGN' in args.exp_name:
        # e.g:
        # {"q":0.2, "avg_steps": 20, "forget_times": 2.0, "ratio_l": 0.5, "sigma": 0.005, "times": 10}
        # 调sigma best sigma 1e-2(数据上选择了1e-3)
        fixed = {
            "ratio_l": 0.5,
            "forget_times": 2,
            "avg_steps": 20,
        }
        space = {
            'sigma': [],
            'times': []
        }
        for sigma in [1e-3,5e-3,1e-2]:
            for times in [10, 20]:
                space['sigma'].append(sigma)
                space['times'].append(times)
        # times=20比times>20好
        # fixed = {
        #     'forget_times': 3,
        #     'ratio_l': 0.5,
        #     'avg_steps': 20,
        #     'times': 30,
        # }
        # space = {
        #     'sigma': [5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1]
        # }

        # 调 sigma, times
        # 最优 sigma=0.01, times=20
        # fixed = {
        #     'forget_times': 3,
        #     'ratio_l': 0.5,
        #     'avg_steps': 20,
        # }
        # space = {
        #     'sigma': [],
        #     'times': []
        # }
        # for sigma in [1e-3,5e-3,1e-2]:
        #     for times in [10, 20]:
        #         space['sigma'].append(sigma)
        #         space['times'].append(times)

        # 调 forget_times
        # fixed = {
        #     'ratio_l': 0.5,
        #     'avg_steps': 20,
        #     'sigma': 0.01,
        #     'times': 20,
        # }
        # space = {
        #     'forget_times': [1, 2, 3, 4],
        # }

        #  "avg_steps": 20, "forget_times": 2.0, "ratio_l": 0.5, "sigma": 0.005, "times": 10
        # "avg_steps": 20, "forget_times": 2.0, "ratio_l": 0.5, "sigma": 0.005, "times": 10
        

    if 'GCE' in args.exp_name:
        # space = {
        #     'q': [0.1, 0.4, 0.7, 1.0],
        # }
        space = {
            'q': [0.1, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
        }
    if 'SLN' in args.exp_name:
        space = {
            'sigma': [0.1,0.2,0.5, 1]
        }

    if 'STGN_GCE' in args.exp_name:
        space = {
            'sigma': [],
            'times': [],
            'q': []
        }
        fixed = {
            'ratio_l': 0.5,
            'avg_steps': 20,
            # 'sigma': 0.01,
            # 'times': 20,
            'forget_times': 3
        }
        for q in [0.2,0.4]:
            for sigma in [0.005, 0.01]:
                for times in [10, 20]:
                    space['sigma'].append(sigma)
                    space['times'].append(times)
                    space['q'].append(q)
    # if 'STGN_GCE' in args.exp_name:
    #     space = {
    #         'forget_times': hp.quniform('forget_times', 3, 8, 1), # 只有30epoch,不用设太大
    #         'ratio_l': hp.uniform('ratio_l', 0, 1.0), #loss vs forget的权重,0~1
    #         'avg_steps': hp.choice('avg_steps', [20]),
    #         'times': hp.choice('times', [ 20, 30, 40, 50, 60]),
    #         'sigma': hp.choice('sigma', [5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]),
    #         'q': hp.choice('q', [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 0.1, 0.3, 0.5, 0.7]),
    #     }
    if 'GNMO' in args.exp_name:
        space = {
            'sigma': [1e-3,5e-3,1e-2,5e-2,1e-1]
        }
    if 'GNMP' in args.exp_name:
        space = {
            'sigma': [1e-3,5e-3,1e-2,5e-2,1e-1]
        }
    trials = get_trials(fixed, space, MAX_EVALS)
    all_trials = []
    best_loss = None
    best = None
    for params in trials:
        res = main(params)
        all_trials.append(res)
        loss = res['loss']
        if best_loss is None or loss<best_loss:
            best_loss = loss
            best = params

    print(best, best_loss)
    print(all_trials)
    #TODO: use only using hp.choice
    #https://github.com/hyperopt/hyperopt/issues/284
    #https://github.com/hyperopt/hyperopt/issues/492
    # print(space_eval(space, best))
    # best = space_eval(space, best)
    args.log_dir = args.exp_name
    json.dump({"best": best, "trials": all_trials},
              open(os.path.join(args.log_dir, "hy_best_params.json"), "w+", encoding="utf-8"),
              ensure_ascii=False, cls=NpEncoder)
    #os.remove(args.params_path)
    os.remove(args.out_tmp)
