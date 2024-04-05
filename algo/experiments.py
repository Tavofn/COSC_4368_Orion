from algo.policies import PolicyType, Policy
from algo.rl import RL


class Experiment:

    def __init__(self, agent):
        self.agent = agent

    def experiment1(self):
        rl = RL(0.3, 0.5, 4000, PolicyType.PRANDOM)
        rl2 = RL(0.3, 0.5, 4000, PolicyType.PGREEDY)
        rl.setAgent(self.agent)
        rl2.setAgent(self.agent)
        qlearn = rl.qLearn()
        qlearn2 = rl2.qLearn()

    def experiment2(self):
        rl = RL(0.3, 0.5, 200, Policy(PolicyType.PRANDOM, self.agent), 7800, PolicyType.PEXPLOIT)
        rl.setAgent(self.agent)
        qlearn = rl.qLearn()

    def experiment3(self):
        rl = RL(0.3, 0.5, 200, PolicyType.PRANDOM, 7800, PolicyType.PEXPLOIT)
        rl.setAgent(self.agent)
        sarsa = rl.sarsa()

    def experiment4(self):
        rl = RL(0.3, 1.0, 200, PolicyType.PRANDOM, 7800, PolicyType.PEXPLOIT)
        sarsa = rl.sarsa()

    def experiment5(self):
        rl = RL(0.3, 0.5, 200, PolicyType.PRANDOM, 7800, PolicyType.PEXPLOIT)
        rl.setAgent(self.agent)
        qlearn = rl.qLearn()
        # after the agent reaches a terminal state the second time, you will swap pickup and drop-off locations
