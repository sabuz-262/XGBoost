import sys
from pathlib import Path
import json

def main(dataDir='../data/'):
    results = dict()
    dataPath = Path(dataDir)
    for i, dataFile in enumerate(dataPath.iterdir()):
        if dataFile.is_file():
            print(dataFile)
            with dataFile.open() as f:
                jsonDict = json.loads(f.read())
                for fn, fnResults in jsonDict.items():
                    if fn not in results:
                        results[fn] = fnResults
                    else:
                        results[fn].update(fnResults)
    # plot(results, skipAlgs=['RAND', 'GP_UCB', 'LOGO'])
    plot(results)

'''
runs.append({'Costs': costs,
             'Values': values,
             'QueryPoints': queryPoints,
             'BestQuery': alg.bestQuery()
             })

{functionName: {algorithm: {'TrueOptima': trueOptima,
                            'Runs': runs}}}
'''
def plot(results, skipAlgs=[]):
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    import numpy as np
    import math
    rcParams.update({'font.size': 18})
    binSize = 5
    for i, (fn, fnResults) in enumerate(results.items()):
        plt.figure(i)
        for alg, algResults in fnResults.items():
            if alg in skipAlgs:
                continue
            if 'TrueOptima' in algResults:
                trueOptima = algResults['TrueOptima']
            errorBins = dict()  # Find a list of the means of each run
            for run in algResults['Runs']:
                runErrorBins = dict()
                costs = np.array(run['Costs']).flatten()
                if 'Errors' in run:
                    errors = np.array(run['Errors']).flatten()
                elif 'Values' in run:
                    errors = trueOptima - np.array(run['Values'])
                for c, e in zip(costs, errors):
                    _bin = math.floor(c / binSize) * binSize
                    if _bin not in runErrorBins:
                        runErrorBins[_bin] = [e]
                    else:
                        runErrorBins[_bin].append(e)
                # Get the mean errors of the run
                for _bin, es in runErrorBins.items():
                    if _bin not in errorBins:
                        errorBins[_bin] = [np.mean(es)]
                    else:
                        errorBins[_bin].append(np.mean(es))

            costValues = list(errorBins.keys())
            errorValues = list(errorBins.values())
            perm = np.argsort(costValues)
            costValues = np.array(costValues)[perm]
            errorValues = np.array(errorValues)[perm]
            means = np.array([np.mean(es) for es in errorValues])
            np.savetxt("currin"+"_values",means)
            np.savetxt("currin"+"_cost", costValues)
            lows = np.array([np.min(es) for es in errorValues])
            highs = np.array([np.max(es) for es in errorValues])
            plt.plot(costValues, means, label=alg)
            if alg == 'MF-BaMLOGO':
                alpha = 0.5
            else:
                alpha = 0.1
            plt.fill_between(costValues,
                             lows,
                             highs,
                             alpha=alpha)
        plt.legend()
        plt.title(fn)
        plt.xlabel('Cumulative Cost')
        plt.ylabel('Simple Regret (log scale)')
        plt.yscale('log')
    plt.show()

if __name__ == '__main__':
    if len(sys.argv) == 2:
        main(sys.argv[1])
    else:
        main()
