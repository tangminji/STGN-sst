import json
import os

def best_result(path,mode='test'):
    file = os.path.join(path, 'hy_best_params.json')
    with open(file,'r') as f:
        results = json.load(f)
    best, trials = results['best'],results['trials']

    scores = []
    for trial in trials:
        flag = True    
        # for key in best:
        #     if best[key]!=trial['params'][key]:
        #         flag = False
        #         break
        if flag:
            scores.append((trial['best_acc'], trial['test_at_best'], trial['stable_acc'], trial['params']))
    if mode=='test':
        # 按测试集
        scores.sort(key=lambda x: x[1], reverse=True)
    else:
        # 按稳定acc
        scores.sort(key=lambda x: x[2], reverse=True)
    
    print(path)
    print("Val: %f\tTest: %f\tStable: %f" % (scores[0][0],scores[0][1],scores[0][2])) # Val acc, Test acc
    print(scores[0][-1])
    #print(scores)

if __name__ == '__main__':

    print("===>Sort by Test_acc")
    # best_result('hy/SST_GCE/nr0.2')
    # best_result('hy/SST_GCE/nr0.4')
    # best_result('hy/SST_GCE/nr0.6')

    # best_result('hy/SST_SLN/nr0.2')
    # best_result('hy/SST_SLN/nr0.4')
    # best_result('hy/SST_SLN/nr0.6')

    # best_result('hy/SST_STGN/nr0.2')
    # best_result('hy/SST_STGN/nr0.4')
    # best_result('hy/SST_STGN/nr0.6')

    best_result('hy/SST_STGN_GCE/nr0.2')
    best_result('hy/SST_STGN_GCE/nr0.4')
    best_result('hy/SST_STGN_GCE/nr0.6')

    print("===>Sort by Stable_test_acc")
    # best_result('hy/SST_STGN/nr0.2',mode='stable')
    # best_result('hy/SST_STGN/nr0.4',mode='stable')
    # best_result('hy/SST_STGN/nr0.6',mode='stable')