# reinforcement learning
import random

from numpy.core._dtype import __repr__

from algo.policies import PolicyType
from world.agent import Action, ActionType


class RL:
    alpha = None
    discount_factor = None
    steps = 0
    steps2 = 0  # for experiments that switch policies
    policy = None
    policy2 = None
    agent = None
    #customActionsForDebugging = [Action(ActionType.WEST), Action(ActionType.WEST), Action(ActionType.SOUTH), Action(ActionType.SOUTH), Action(ActionType.PICKUP), Action(ActionType.SOUTH), Action(ActionType.SOUTH), Action(ActionType.DROPOFF)]
    customActionsForDebugging = [Action(ActionType.WEST), Action(ActionType.WEST), Action(ActionType.SOUTH), Action(ActionType.SOUTH), Action(ActionType.PICKUP), Action(ActionType.SOUTH)]

    def __init__(self, alpha, dF, steps, policy, steps2, policy2):
        self.alpha = alpha # learning rate
        self.discount_factor = dF
        self.steps = steps
        self.policy = policy
        self.steps2 = steps2
        self.policy2 = policy2

    def setAgent(self, agent):
        self.agent = agent

    def qLearn(self):
        self.position = self.agent.agentPosition #initial state
        action = None
        self.reward = None
        #while(not self.agent.world.isInTerminalState):

        oldAgentPos = self.agent.agentPosition
        print("Agent initial position: ", end="")
        print(oldAgentPos.__dict__)
        self.newAgentPos = None
        random.seed(1)
        a = 0
        for x in range(0, self.steps): #self.steps
            oldAgentPos = self.agent.agentPosition
            self.action = self.getNextAction(self.policy)
            self.logInfoBeforeAction()
            self.agent.move(self.action) # perform action

            self.agent.interface.pd_world_window.update_idletasks()
            self.agent.interface.updateAgentPosition(self.agent,self.agent.agentPosition)

            self.logInfoAfterAction()
            newAgentPos = self.agent.agentPosition
            self.reward = self.action.reward # measure reward
            # Q(a,s)  (1-alpha)*Q(a,s) + alpha*[R(s’,a,s)+ γ*maxa’Q(a’,s’)]
            oldAgentPos.qValue = (1-self.alpha) * oldAgentPos.qValue + self.alpha * (self.action.reward + self.discount_factor * self.maxFutureReward(newAgentPos))# update q
            self.agent.interface.updateQTable(oldAgentPos.position[0], oldAgentPos.position[1], self.action.type.value, round(oldAgentPos.qValue,3))

            a = a + 1

    def getNextAction(self, policy):
        if self.policy.type == PolicyType.PRANDOM:
            return policy.pRandom()
        if self.policy.type == PolicyType.PGREEDY:
            return policy.pGreedy()
        if self.policy.type == PolicyType.PEXPLOIT:
            return policy.pGreedy()

    def sarsa(self):
        self.position = self.agent.agentPosition  # initial state
        action = None
        self.reward = None

    def maxFutureReward(self, currentState):
        maxReward = -1
        for a in self.policy.applicableOperators:
            #newPosition = self.agent.getProjectedPos(currentState, a)
            if a.reward >= maxReward:
                maxReward = a.reward
        return maxReward

    def logInfoBeforeAction(self):
        print("Action Chosen at Random: ", end="")
        print(self.action.type.__dict__)

    def logInfoAfterAction(self):
        print("Agent's position: ", end="")
        print(self.agent.agentPosition.__dict__)


    #def sarsa(self):

