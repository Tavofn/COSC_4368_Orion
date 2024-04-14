import numpy as np
import random

class Agent:
    def __init__(self, start_position, name):
        self.position = start_position
        self.name = name
        self.has_block = False

    def move(self, direction, world):
        moves = {'north': (-1, 0), 'south': (1, 0), 'east': (0, 1), 'west': (0, -1)}
        new_position = tuple(np.add(self.position, moves[direction]))
        if world.within_bounds(new_position) and new_position not in [agent.position for agent in world.agents.values() if agent.name != self.name]:
            self.position = new_position

    def pickup(self, world):
        if world.pickup_cells.get(self.position, 0) > 0 and not self.has_block:
            self.has_block = True
            world.pickup_cells[self.position] -= 1

    def dropoff(self, world):
        if world.dropoff_cells.get(self.position, 0) < 5 and self.has_block:
            self.has_block = False
            world.dropoff_cells[self.position] += 1

class PDWorld:
    def __init__(self):
        self.grid_size = (5, 5)
        self.agents = {'red': Agent((2, 2), 'Red'), 'blue': Agent((4, 2), 'Blue'), 'black': Agent((0, 2), 'Black')}
        self.pickup_cells = {(1, 4): 5, (2, 3): 5, (4, 1): 5}
        self.dropoff_cells = {(0, 0): 0, (2, 0): 0, (4, 4): 0}

    def within_bounds(self, position):
        x, y = position
        return 0 <= x < self.grid_size[0] and 0 <= y < self.grid_size[1]

    def display_world(self):
        grid = [['.' for _ in range(self.grid_size[1])] for _ in range(self.grid_size[0])]
        for pos, blocks in self.pickup_cells.items():
            grid[pos[0]][pos[1]] = 'P' if blocks > 0 else '.'
        for pos, blocks in self.dropoff_cells.items():
            grid[pos[0]][pos[1]] = 'D' if blocks < 5 else '.'
        for agent in self.agents.values():
            grid[agent.position[0]][agent.position[1]] = agent.name[0]
        print('\n'.join([' '.join(row) for row in grid]))

class RLAlgorithm:
    def __init__(self, learning_rate=0.1, discount_factor=0.9, actions=['north', 'south', 'east', 'west', 'pickup', 'dropoff']):
        self.q_table = {}
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.actions = actions

    def select_action(self, state, policy):
        if policy == 'PRandom' or (policy == 'PExploit' and random.random() < 0.2):
            return random.choice(self.actions)
        best_action = max(self.actions, key=lambda action: self.q_table.get((state, action), 0))
        return best_action if best_action else random.choice(self.actions)

    def update_q_table(self, current_state, action, reward, next_state, policy):
        if (current_state, action) not in self.q_table:
            self.q_table[(current_state, action)] = 0
        next_max = max(self.q_table.get((next_state, a), 0) for a in self.actions)
        self.q_table[(current_state, action)] += self.learning_rate * (reward + self.discount_factor * next_max - self.q_table[(current_state, action)])

def simulate(world, algorithm, policy, steps):
    for step in range(steps):
        for name, agent in world.agents.items():
            state = (agent.position, agent.has_block)
            action = algorithm.select_action(state, policy)
            if action in ['north', 'south', 'east', 'west']:
                agent.move(action, world)
            elif action == 'pickup':
                agent.pickup(world)
            elif action == 'dropoff':
                agent.dropoff(world)
            next_state = (agent.position, agent.has_block)
            reward = -1 if action in ['north', 'south', 'east', 'west'] else 13
            algorithm.update_q_table(state, action, reward, next_state, policy)
        world.display_world()

# Initialize the world and run the simulation
world = PDWorld()
algorithm = RLAlgorithm(learning_rate=0.1, discount_factor=0.9)
simulate(world, algorithm, 'PExploit', 100)
