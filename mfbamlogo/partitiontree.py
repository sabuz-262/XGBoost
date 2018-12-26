import numpy as np
from copy import deepcopy
import logging

class PartitionTree:
    def __init__(self, dim):
        self.nodes = [Node([0.] * dim, [1.] * dim, depth=0)]

    def maxDepth(self):
        return max([n.depth for n in self.nodes])

    def bestNodeInRange(self, level, width):
        depthRange = range(width * level, width * (level + 1))

        def inRange(n):
            return n[1].depth in depthRange

        nodesInLevel = list(filter(inRange, enumerate(self.nodes)))
        if not nodesInLevel:
            return None, None
        return max(nodesInLevel, key=lambda n: n[1].value)

    def expandAt(self, index):
        node = self.nodes.pop(index)
        newNodes = node.split()
        self.nodes.extend(newNodes)

    def plotTree(self, ax, numFidelities):
        ax.set_title('Partition Tree')
        fake = list(filter(lambda n: n.isFakeValue, self.nodes))
        fidel = [list(filter(lambda n: n.fidelity == i, self.nodes))
                        for i in range(numFidelities)]
        xs = [n.center for n in fake]
        depths = [n.depth for n in fake]
        ax.scatter(xs, depths, label='Fake Nodes', color='#1B9E77')
        for i in range(numFidelities):
            xs = [n.center for n in fidel[i]]
            depths = [n.depth for n in fidel[i]]
            ax.scatter(xs, depths, label=['Low', 'High'][i] + ' Fidelity',
                                   color=['#7570B3', 'blue'][i])
        ax.set_ylabel('Depth')
        ax.legend()
        ax.set_xlim([0., 1.])

class Node:
    def __init__(self, lows, highs, depth):
        self.lows = np.array(lows)
        self.highs = np.array(highs)
        self.center = (self.lows + self.highs) / 2.
        self.value = None
        self.fidelity = None
        self.isFakeValue = False
        self.depth = depth

    def setFidelity(self, value, fidelity):
        self.value = value
        self.fidelity = fidelity
        self.isFakeValue = False

    def setFakeValue(self, fakeValue):
        self.value = fakeValue
        self.fidelity = None
        self.isFakeValue = True

    def split(self):
        lengths = self.highs - self.lows
        longestDimension = np.argmax(lengths)
        logging.debug('Splitting node {0} along axis {1}'
                        .format(tuple(self.center), longestDimension))
        t = lengths[longestDimension] / 3.
        lowerThird = self.lows[longestDimension] + t
        upperThird = self.highs[longestDimension] - t
        listOfLows = [deepcopy(self.lows) for _ in range(3)]
        listOfHighs = [deepcopy(self.highs) for _ in range(3)]
        listOfHighs[0][longestDimension] = lowerThird   # Left node
        listOfLows[1][longestDimension] = lowerThird    # Center node
        listOfHighs[1][longestDimension] = upperThird   # Center node
        listOfLows[2][longestDimension] = upperThird    # Right node
        newNodes = [Node(listOfLows[i], listOfHighs[i], self.depth + 1)
                        for i in range(3)]
        newNodes[1].value = self.value
        newNodes[1].fidelity = self.fidelity
        newNodes[1].isFakeValue = self.isFakeValue
        return newNodes
