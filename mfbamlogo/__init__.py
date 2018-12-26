import numpy as np
import math
import logging


X_DIM = 2
Z_DIM = 1
Threshold_Constant = 1
CONSTANT_KERNEL = 1

class MFBaMLOGO:

    def __init__(self, object, fn, filename, costEstimations, lows, highs,
                        numInitSamples=10, algorithm='MF-BaMLOGO'):
        assert algorithm in ['MF-BaMLOGO', 'BaMLOGO', 'LOGO']
        assert len(lows) == len(highs)
        self.algorithm = algorithm
        self.wSchedule = [3, 4, 5, 6, 8, 30]
        self.fn = fn
        self.lows = np.array(lows)
        self.highs = np.array(highs)
        self.dim = len(self.lows)
        self.costs = costEstimations
        self.totalCost = 0.
        self.numFidelities = 1
        self.maxFidelity = 1
        self.numExpansions = 0
        self.wIndex = 0
        self.stepBestValue = -float('inf')
        self.fidelity = []
        self.lastBestValue = -float('inf')
        self.bestNode = None
        from .model import GaussianProcess
        self.model = GaussianProcess(self.numFidelities, self.dim)
        self.file = open(filename, 'w')
        self.mfobject = object

        self.epsilon = 0
        if self.algorithm == 'MF-BaMLOGO':
            samples = []
            for i in range(numInitSamples):
                x = np.random.uniform([0.] * self.dim, [1.] * self.dim)
                y = self.evaluate(x, .333)
                samples.append(y)
                self.model.addSample(x, y, .333)
                if i % 3 == 0:
                    y1 = self.evaluate(x, 1)
                    self.epsilon = max(self.epsilon, abs(y - y1))
                    samples.append(y1)
                    self.model.addSample(x, y1, fidelity=1)

            r = 1e-6 * (max(samples) - min(samples))
            self.thresholds = (self.numFidelities - 1) * [r]
            self.timeSinceEval = np.zeros((self.numFidelities),)
            logging.debug('Thresholds initialized to {0}'.format(r))
            logging.debug('Epsilon initialzed to {0}'.format(self.epsilon))

        from .partitiontree import PartitionTree
        self.space = PartitionTree(self.dim)
        self.observeNode(self.space.nodes[0])

    def maximize(self, budget=100., ret_data=False, plot=True):
        costs, bestValues, queryPoints = [], [], []
        while self.totalCost < budget:
            print(self.totalCost)
            self.stepBestValue = -float('inf')
            self.expandStep()
            self.adjustW()

            if self.bestNode:
                cost = self.totalCost
                x = self.transformToDomain(self.bestNode.center)
                y = self.bestNode.value
                costs.append(cost)
                queryPoints.append(x)
                bestValues.append(y)
                logging.info('Best value is {0} with cost {1}'.format(y, cost))
            if plot and self.dim == 1:
                self.plotInfo()

        if ret_data:
            return costs, bestValues, queryPoints, self.fidelity

    def maxLevel(self):
        depthWidth = self.wSchedule[self.wIndex]
        hMax = math.sqrt(self.numExpansions + 1)
        return math.floor(min(hMax, self.space.maxDepth()) / depthWidth)

    def expandStep(self):
        logging.debug('Starting expand step')
        vMax = -float('inf')
        depthWidth = self.wSchedule[self.wIndex]
        level = 0
        while level <= self.maxLevel():

            logging.debug('Expanding level {0}'.format(level))
            idx, bestNode = self.space.bestNodeInRange(level, depthWidth)
            if idx is not None and bestNode.value > vMax:
                vMax = bestNode.value
                logging.debug('vMax is now {0}'.format(vMax))
                self.space.expandAt(idx)
                self.observeNode(self.space.nodes[-3])  # Left node
                self.observeNode(self.space.nodes[-2])  # Center node
                self.observeNode(self.space.nodes[-1])  # Right node
                self.numExpansions = self.numExpansions + 1
            level = level + 1

    def observeNode(self, node):

        x = node.center
        if node.value is not None and not node.isFakeValue:
            if node.fidelity == self.maxFidelity:
                logging.debug('Already had node at x={0}'
                                .format(self.transformToDomain(x)))

                return
        lcb, ucb = self.computeLCBUCB(x)

        if ucb is None or self.bestNode is None or ucb >= self.bestNode.value:
            fidelity = self.chooseFidelity(node)
            self.evaluateNode(node, fidelity, offset=self.error(fidelity),
                                updateGP=True, adjustThresholds=True)

        elif self.algorithm == 'MF-BaMLOGO' and 2. * self.error(0) < ucb - lcb:
            logging.debug('Unfavorable region at x={0}; Using lowest fidelity'
                            .format(self.transformToDomain(x)))
            self.evaluateNode(node, fidelity=0, offset=-self.error(0))

        else:
            logging.debug('Unfavorable region at x={0}. Using LCB = {1}'
                            .format(self.transformToDomain(x), lcb))

            node.setFakeValue(lcb)

    def error(self, z):
        return 0

    def evaluateNode(self, node, fidelity,
                    offset=0., updateGP=False, adjustThresholds=False):

        x = node.center
        if node.value is not None and not node.isFakeValue:
            if fidelity <= node.fidelity:
                logging.debug('Already had node at x={0}'
                                .format(self.transformToDomain(x)))
                return

        y = self.evaluate(x, fidelity)
        node.setFidelity(y + offset, fidelity)

        self.stepBestValue = max(self.stepBestValue, y)
        logging.debug('Step best is now {0}'.format(self.stepBestValue))
        if fidelity == self.maxFidelity:
            if not self.bestNode or self.bestNode.value < y:
                self.bestNode = node

        if self.algorithm == 'MF-BaMLOGO' or self.algorithm == 'BaMLOGO':
            if updateGP:
                self.model.addSample(x, y, fidelity)




    def evaluate(self, x, f):
        args = self.transformToDomain(x)
        logging.debug('Evaluating f{0} at fidelity {1}'.format(args, f))
        y, cost = self.fn(args, f, self.mfobject)
        logging.debug('Got y = {0} with cost {1}'.format(y, cost))
        self.totalCost += cost
        self.file.write(str(y) + ' ' + str(self.totalCost) + ' ' + str(f) + ' ' + str(cost) + '\n')
        self.fidelity.append(f)
        return y

    def cost_currin(self, z):
        import time
        t1 = time.time()
        value = self.mfobject.eval_at_fidel_single_point_normalised([z, 1.0], [.5,.5])
        t2 = time.time()
        t0 = t2 - t1
        return t0
    def fineSampling(self):
        fidelity = []
        fidelity.append(.333)
        fidelity.append(.667)
        return fidelity

    def inforamtionGap(self, z):
        K = np.exp(-.5 * (1 - z))
        K = K * K
        K = 1 - K
        return pow(K, .5)

    def threshold(self, z):
        q = (1 / (X_DIM + Z_DIM + 2))
        ratio = (self.cost_currin(z) / self.cost_currin(1))
        ratio = pow(ratio, q)
        return Threshold_Constant * pow(CONSTANT_KERNEL, .5) * self.inforamtionGap(z) * ratio

    def secondConditions(self,z, B_t):
        temp = pow(B_t, -.5) * self.inforamtionGap(pow(1, .5))
        if self.inforamtionGap(z) > temp:
            return True
        else:
            return False

    def conditions(self, fidelities, gp, x_t, B_t):
        f = []
        for fidelity in fidelities:
            mean, std = gp.getPrediction(x_t, fidelity)
            variance = std * std
            if (variance > self.threshold(fidelity)) and self.secondConditions(fidelity, B_t):
                f.append(fidelity)
        f.append(1)
        return f

    def finalFidelity(self, fidelities):
        if (fidelities is None) or (len(fidelities) == 0):
            return 1
        minCost = 100000
        minFid = 2
        for fidelity in fidelities:
            if self.cost_currin(fidelity) < minCost:
                minCost = self.cost_currin(fidelity)
                minFid = fidelity
        return minFid

    def chooseFidelity(self, node):
        if self.algorithm == 'MF-BaMLOGO':
            x = node.center
            beta = self.beta()
            x_t = node.center
            fidelities = self.fineSampling()
            fidelities = self.conditions(fidelities, self.model, x_t, beta)
            fidelity = self.finalFidelity(fidelities)
            return fidelity
        else:
            return self.maxFidelity

    def beta(self):
        n = 0.5
        return math.sqrt(2. * math.log(
                math.pi ** 2. * (self.numExpansions + 1) ** 2. / (6. * n)))

    def computeLCBUCB(self, x):
        if self.algorithm == 'MF-BaMLOGO' or self.algorithm == 'BaMLOGO':
            beta = self.beta()

            def uncertainty(args):
                f, (_, std) = args
                return beta * std

            predictions = []
            fidelities = self.fineSampling()
            for fidelity in fidelities:
                predictions.append(self.model.getPrediction(x, fidelity))
            if not predictions:
                return None, None

            f, (mean, std) = min(enumerate(predictions), key=uncertainty)
            lcb = float(mean - beta * std )
            ucb = float(mean + beta * std )

            logging.debug('LCB/UCB for f{0} (fidelity {1})'
                            .format(self.transformToDomain(x), f))
            logging.debug('Mean={0}, std={1}, beta={2}'.format(mean, std, beta))
            logging.debug('LCB={0}, UCB={1}'.format(lcb, ucb))

            return lcb, ucb
        else:
            return None, None

    def adjustW(self):
        if self.stepBestValue > self.lastBestValue:
            self.wIndex = min(self.wIndex + 1, len(self.wSchedule) - 1)
        else:
            self.wIndex = max(self.wIndex - 1, 0)
        self.lastBestValue = self.stepBestValue
        logging.debug('Width is now {0}'.format(self.wSchedule[self.wIndex]))

    def transformToDomain(self, x):
        return tuple(x * (self.highs - self.lows) + self.lows)

    def bestQuery(self):
        return self.transformToDomain(self.bestNode.center), self.bestNode.value


