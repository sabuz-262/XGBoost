import sys
import numpy as np
from news_classifier import NGMFOptFunction


def main(argv):
    optionExample = (argv[0] + ' -h'
                               ' -f <test_function>'
                               ' -a <algorithm>'
                               ' -r <budget>'
                               ' -n <num_runs>'
                               ' -s <num_initial_samples>'
                               ' -o <output>'
                               ' -v <verbose_level>')
    testFunction = 'real'
    algorithm = 'MF-BaMLOGO'
    budget = 360
    outputDir = None
    numRuns = 5
    numInitSamples = 10
    import getopt
    try:
        opts, args = getopt.getopt(argv[1:], 'hf:a:r:n:s:o:v:')
    except getopt.GetoptError:
        print(optionExample)
        exit(1)
    for opt, arg in opts:
        if opt == '-h':
            print(optionExample)
            exit(0)
        elif opt == '-v':
            import logging
            if arg == '0':
                logging.basicConfig(level=logging.ERROR)
            elif arg == '1':
                logging.basicConfig(level=logging.INFO)
            else:
                logging.basicConfig(level=logging.DEBUG)
        elif opt == '-f':
            testFunction = arg
        elif opt == '-a':
            algorithm = arg
        elif opt == '-r':
            budget = float(arg)
        elif opt == '-n':
            numRuns = int(arg)
        elif opt == '-s':
            numInitSamples = int(arg)
        elif opt == '-o':
            outputDir = arg

    if testFunction == 'real':
        from mf_test_functions import mfReal
        def fn(x, f, object):
            value, cost = mfReal(x, f, object)
            return value, cost
        costEstimations = [7.35, 22.40, 34.86]
        lows = 5 * [0]
        highs = 5 * [1]
        trueOptima = 1


    elif testFunction == 'Hartmann-3D':
        from mf_test_functions import mfHartmann3
        def fn(x, f):
            value, cost = mfHartmann3(x, f)
            return -value, cost
        costEstimations = [0.0052, 0.05, 0.955]
        lows = 3 * [0.]
        highs = 3 * [1.]
        trueOptima = 3.86278
    elif testFunction == 'Hartmann-6D':
        from mf_test_functions import mfHartmann6
        def fn(x, f):
            value, cost = mfHartmann6(x, f)
            return -value, cost
        costEstimations = [1, 10, 100, 1000.]
        lows = 6 * [0.]
        highs = 6 * [1.]
        trueOptima = 3.32237
    elif testFunction == 'Park1-4D':
        from mf_test_functions import mfPark1
        def fn(x, f):
            value, cost = mfPark1(x, f)
            return -value, cost
        costEstimations = [0.1, 1.]
        lows = 4 * [0.]
        highs = 4 * [1.]
        trueOptima = 25.5893
    elif testFunction == 'Park2-4D':
        from mf_test_functions import mfPark2
        def fn(x, f):
            value, cost = mfPark2(x, f)
            return -value, cost
        costEstimations = [0.1, 1.]
        lows = 4 * [0.]
        highs = 4 * [1.]
        trueOptima = 5.92604
    elif testFunction == 'CurrinExponential-2D':
        from mf_test_functions import mfCurrinExp
        fn = mfCurrinExp
        costEstimations = [0.21, .55, 1.1]
        lows = [0., 0.]
        highs = [1., 1.]
        trueOptima = 13.7987
    elif testFunction == 'BadCurrinExponential-2D':
        from mf_test_functions import mfBadCurrinExp
        fn = mfBadCurrinExp
        costEstimations = [0.21, 0.55 ,1.1]
        lows = [0., 0.]
        highs = [1., 1.]
        trueOptima = 13.7987
    elif testFunction == 'Borehole-8D':
        from mf_test_functions import mfBorehole
        fn = mfBorehole
        costEstimations = [0.29,.644, 1.1]
        lows = [.05, 100., 63070., 990., 63.1, 700., 1120., 9855.]
        highs = [.15, 50000., 115600., 1110., 116., 820., 1680., 12045.]
        trueOptima = 309.523221
    elif testFunction == 'SCALE-8D':
        true_dim = 38
        dim = 8
        from eval_scale import evalWeightsSCALE
        def fn(x, f):
            u = np.append(x, (true_dim - dim) * [0.5])
            if f == 0:
                value, cost = evalWeightsSCALE(u, 30)
            elif f == 1:
                value, cost = evalWeightsSCALE(u, 150)
            elif f == 2:
                value, cost = evalWeightsSCALE(u, 569)
            return 1 - value, cost
        costEstimations = [0.1, 0.25, 1.0]
        lows = dim * [0.]
        highs = dim * [1.]
        trueOptima = 1.
    elif testFunction == 'Rosenbrock-2D':
        from mf_test_functions import mfRosenbrock
        def fn(x, f):
            value, cost = mfRosenbrock(x, f)
            return -value, cost
        costEstimations = [0.01, 0.1, 1.]
        lows = [-2., -2.]
        highs = [2., 2.]
        trueOptima = 0.
    elif testFunction == 'Hosaki-2D':
        from mf_test_functions import mfHosaki
        def fn(x, f):
            value, cost = mfHosaki(x, f)
            return -value, cost
        costEstimations = [0.01, 0.1, 1.]
        lows = [0., 0.]
        highs = [10., 10.]
        trueOptima = 2.345811576101292
    elif testFunction == 'Univariate-1D':
        from mf_test_functions import mfUnivariate
        fn = mfUnivariate
        costEstimations = [0.1, 1.]
        lows = [0.]
        highs = [1.2]
        trueOptima = 1.48907
    elif testFunction == 'Shekel-4D':
        from mf_test_functions import mfShekel
        dim = 4
        numFidelities = 3
        fn = mfShekel
        costEstimations = [0.25, 0.5, 1.]
        lows = dim * [0.]
        highs = dim * [10.]
        trueOptima = mfShekel([4, 4, 4, 4], numFidelities - 1)[0]
    else:
        print('Unknown test function.')
        exit(1)

    results = runAlgorithm(testFunction, fn, costEstimations, lows, highs,
                           trueOptima, budget, numInitSamples,
                           algorithm, numRuns)

    if outputDir:
        with open(outputDir, 'w') as outFile:
            import json
            json.dump(results, outFile)
    else:
        best = results[testFunction][algorithm]['Runs'][0]['BestQuery']
        print('Found f{0} = {1}'.format(*best))
        from plot import plot
        plot(results)

def runAlgorithm(functionName, fn,
                 costEstimations, lows, highs,
                 trueOptima, budget, numInitSamples, algorithm, numRuns):
    import os, sys
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.dirname(CURRENT_DIR))
    from mfbamlogo import MFBaMLOGO
    FIDEL_BOUNDS = [[100, 5000]]
    fidel_bounds = np.array(FIDEL_BOUNDS)
    mfobject = NGMFOptFunction(fidel_bounds)
    runs = []
    for i in range(numRuns):
        filename = str(i) + "real_fidelity" + ".txt"
        alg = MFBaMLOGO(mfobject, fn, filename, costEstimations, lows, highs,
                         numInitSamples=numInitSamples, algorithm=algorithm)
        costs, values, queryPoints, fidelity = alg.maximize(budget=budget,
                                                  ret_data=True)
        #np.savetxt(str(i) + "currin" + "_values", values)
        #np.savetxt(str(i) + "currin"+ "_cost", costs)
        print(costs)
        print(values)
        print(fidelity)
        runs.append({'Costs': costs,
                     'Values': values,
                     'QueryPoints': queryPoints,
                     'BestQuery': alg.bestQuery()
                     })

    results = {functionName: {algorithm: {'TrueOptima': trueOptima,
                                          'Runs': runs}}}
    return results

if __name__ == '__main__':
    main(sys.argv)
