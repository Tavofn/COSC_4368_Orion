"""" PEPLOIT: If pickup and dropoff is applicable, 
    choose this operator; 
    otherwise, apply the applicable operator with the highest q-value 
    (break ties by rolling a dice for operators with the same q-value) 
    with probability 0.8 and choose a different applicable operator randomly with probability 0.2. """
import random
from enum import Enum

from world.agent import Action, ActionType
from world.cell import CellType


class PolicyType(Enum):
    PRANDOM = 1
    PEXPLOIT = 2
    PGREEDY = 3


class Policy:
    applicableOperators = set()
    canPickup = True
    canDropoff = True
    world = None
    agent = None
    type = None

    def __init__(self, type, agent):
        self.type = type
        self.world = agent.world
        self.agent = agent
        self.agent.updateApplicableOperators()
        self.applicableOperators = self.agent.applicableOperators

    # returns action in accordance with the random action policy
    def pRandom(self):
        self.type = PolicyType.PRANDOM
        if Action.getApplicability(Action(ActionType.DROPOFF), self.agent):
            return Action(ActionType.DROPOFF)
        elif Action.getApplicability(Action(ActionType.PICKUP), self.agent):
            return Action(ActionType.PICKUP)
        else:
            randomOperator = random.randint(0, self.applicableOperators.__len__() - 1)
            if self.applicableOperators is not None:
                ap = list(self.applicableOperators)
                return Action(ap[randomOperator].type)

    # returns action in accordance with the exploit action policy
    def pExploit(self, operator):
        self.type = PolicyType.PEXPLOIT
        if self.canDropoff and self.canPickup:
            return operator
        else:
            highestQValue = 0
            op = None
            randomOperator = random.randint(0, self.applicableOperators.__len__())
            for o in self.applicableOperators:
                if o.qValue > highestQValue:
                    highestQValue = o.qValue
                    probability = [highestQValue] * 80 * self.applicableOperators[randomOperator] * 20
                    random.choice(probability)
                    op = o
                elif o.qValue == highestQValue:
                    highestQValue = self.break_tie(o.qValue, highestQValue)
                    op = o
            return op

    # returns action in accordance with the greedy action policy
    def pGreedy(self):
        self.type = PolicyType.PGREEDY
        if Action.getApplicability(Action(ActionType.DROPOFF), self.agent):
            return Action(ActionType.DROPOFF)
        elif Action.getApplicability(Action(ActionType.PICKUP), self.agent):
            return Action(ActionType.PICKUP)
        else:
            highestQValue = 0
            op = None
            for o in self.applicableOperators:
                if o.qValue > highestQValue:
                    highestQValue = o.qValue
                    op = o
                elif o.qValue == highestQValue:
                    highestQValue = self.break_tie(o.qValue, highestQValue)
                    op = o
            return op

    def break_tie(self, o1, o2):
        randomOp = random.randint(0, 2)
        if randomOp == 1:
            return o1
        return o2
