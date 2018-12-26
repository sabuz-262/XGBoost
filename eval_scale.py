import subprocess
import re
import time

policyScorePattern = re.compile("(?<=test policy score = )[0-9.e-]+")

ROOT_PATH = '/data/doppa/users/ehoag/SCALE/'
JAR_FILE = ROOT_PATH + 'executable/BatchTestPolicyTimeMultipleBeamNew.jar'
GRAPH_DATA = ROOT_PATH + 'data/syago.txt'
TEST_QUERY_DATA = ROOT_PATH + 'data/fixedSolvedQueryNew.txttestingFold.txt'
WORKING_TEST_QUERY_DATA = (ROOT_PATH +
            'data/workingFixedSolvedQueryNew.txttestingFold.txt')
QUERY_DATA = ROOT_PATH + 'data/idealQuery.txt'
MODEL_DATA = (ROOT_PATH +
            'output/policy/SaveYagoNewWORandB_10-tau_25.txt60pc.1.model')
WORKING_MODEL_DATA = (ROOT_PATH +
            'output/policy/workingSaveYagoNewWORandB_10-tau_25.txt60pc.1.model')

CMD = ['java',
       '-Xms81920m',
       '-Xmx81920m',
       '-jar', JAR_FILE,
       '-linearPolicy',
       '-oraclePath', GRAPH_DATA,
       '-testQueryPath', WORKING_TEST_QUERY_DATA,
       '-idealQueryPath', QUERY_DATA,
       '-tauTest', '25',
       '-hop', '10',
       '-dghop', '5',
       '-k', '1',
       '-randomFeatures', '1',
       '-logLevel', '2',
       '-modelSaveFileName', WORKING_MODEL_DATA]

'''
To execute SCALE (Yago):
    ~30GB of memory
    2-4 cpus
    ~3 minutes for querySize=30 (walltime=~3)
    ~8 minutes for querySize=50 (walltime=~4)
    ~11 minutes for querySize=100 (walltime=~6)
    ~19 minutes for querySize=200 (walltime=~10)
    ~36 minutes for querySize=569 (walltime=~30)
'''
def evalWeightsSCALE(weights, testQuerySize):

    startTime = time.time()

    # Write weights to WORKING_MODEL_DATA
    with open(MODEL_DATA, 'r') as modelFile:
        data = ''
        for i, line in enumerate(modelFile.readlines()):
            if i == 11:
                line = ('1 1:{0} 2:{1} 3:{2} 4:{3} 5:{4} '
                          '6:{5} 7:{6} 8:{7} 9:{8} 10:{9} '
                          '11:{10} 12:{11} 13:{12} 14:{13} 15:{14} '
                          '16:{15} 17:{16} 18:{17} 19:{18} 20:{19} '
                          '21:{20} 22:{21} 23:{22} 24:{23} 25:{24} '
                          '26:{25} 27:{26} 28:{27} 29:{28} 30:{29} '
                          '31:{30} 32:{31} 33:{32} 34:{33} 35:{34} '
                          '36:{35} 37:{36} 38:{37} #\n'.format(*weights))
            data += line
        with open(WORKING_MODEL_DATA, 'w') as workingModelFile:
            workingModelFile.seek(0)
            workingModelFile.write(data)
            workingModelFile.truncate()

    # Write test query size to WORKING_TEST_QUERY_DATA
    with open(TEST_QUERY_DATA, 'r', errors='replace') as testQueryFile:
        data = ''
        for i, line in enumerate(testQueryFile.readlines()):
            if i == 0:
                line = '{0}\n'.format(testQuerySize)
            data += line
        with open(WORKING_TEST_QUERY_DATA, 'w', errors='replace') as wQueryFile:
            wQueryFile.seek(0)
            wQueryFile.write(data)
            wQueryFile.truncate()

    # Execute SCALE
    result = subprocess.run(CMD, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode:
        print('Could not execute {0}!'.format(CMD))
        print('Weights={0}, Test query size={1}'.format(weights, testQuerySize))
        error = result.stderr.decode('utf-8')
        print(error)
        return float('nan'), startTime - time.time()
    output = result.stdout.decode('utf-8')

    # Read results from stdout
    score = policyScorePattern.search(output)
    if score:
        return float(score[0]), startTime - time.time()
    else:
        print('Could not find policy score!')
        print(output)
        print('Weights={0}, Test query size={1}'.format(weights, testQuerySize))
        return float('nan'), startTime - time.time()

def makeRandomEmbedding(fn, fn_bounds, fn_dim, result_dim):
    A = np.random.rand(fn_dim, result_dim)
    def embeddedFn(u, fidelity):
        return fn(A * u, fidelity)
    return embeddedFn

if __name__ == '__main__':
    weights = (0.1, 0.1, 0.1, 0.1, 0.1,
               0.1, 0.0, 0.1, 0.1, 0.1,
               0.1, 0.1, 0.1, 0.1, 0.1,
               0.1, 0.1, 0.1, 0.1, 0.1,
               0.1, 0.1, 0.1, 0.1, 0.1,
               0.1, 0.1, 0.1, 0.1, 0.1,
               0.0, 0.1, 0.1, 0.1, 0.1,
               0.1, 0.1, 0.1)
    testQuerySize = 569
    y, cost = evalWeightsSCALE(weights, testQuerySize)
    print('y = {0} with with cost = {1} and test query size = {2}'
            .format(y, cost, testQuerySize))
