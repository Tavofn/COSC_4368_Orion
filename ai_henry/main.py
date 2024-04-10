import random
import numpy as np

# Constants for the world size and parameters
WORLD_SIZE = (5, 5)  # Assuming a 5x5 grid for simplicity
PICKUP_LOCATIONS = [(0, 4), (1, 3), (4, 1)]  # Example pickup locations
DROPOFF_LOCATIONS = [(0, 0), (2, 0), (3, 4)]  # Example dropoff locations
BLOCKS_AT_PICKUP = 5  # Initial blocks at pickup locations
CAPACITY_DROPOFF = 5  # Capacity at dropoff locations
N_AGENTS = 3  # Number of agents
ACTIONS = ['north', 'south', 'east', 'west', 'pickup', 'dropoff']  # Possible actions
EPSILON = 0.2  # For exploration in PEXPLOIT

# Parameters for Q-learning
LEARNING_RATE = 0.3
GAMMA = 0.5
STEPS = 9000
PRANDOM_STEPS = 500  # Initial steps with PRANDOM

# Setting up the initial state of the world
class PDWorld:
    def __init__(self):
        self.pickup_locations = {loc: BLOCKS_AT_PICKUP for loc in PICKUP_LOCATIONS}
        self.dropoff_locations = {loc: 0 for loc in DROPOFF_LOCATIONS}
        self.agents = {'red': (4, 2), 'blue': (2, 2), 'black': (0, 2)}
        self.blocks_carried = {agent: 0 for agent in self.agents}
        # Separate Q-table for each agent
        self.q_tables = {
            'red': np.zeros((WORLD_SIZE[0] * WORLD_SIZE[1] * 2, len(ACTIONS))),
            'blue': np.zeros((WORLD_SIZE[0] * WORLD_SIZE[1] * 2, len(ACTIONS))),
            'black': np.zeros((WORLD_SIZE[0] * WORLD_SIZE[1] * 2, len(ACTIONS)))
        }
        
    def reset(self):
        self.pickup_locations = {loc: BLOCKS_AT_PICKUP for loc in PICKUP_LOCATIONS}
        self.dropoff_locations = {loc: 0 for loc in DROPOFF_LOCATIONS}
        self.agents = {'red': (4, 2), 'blue': (2, 2), 'black': (0, 2)}
        self.blocks_carried = {agent: 0 for agent in self.agents}
        # Note: Q-table is not reset here to continue learning across simulations
        
    def is_move_valid(self, new_pos):
        if not (0 <= new_pos[0] < WORLD_SIZE[0] and 0 <= new_pos[1] < WORLD_SIZE[1]):
            return False
        if new_pos in self.agents.values():
            return False
        return True

    def move_agent(self, agent, direction):
        current_pos = self.agents[agent]
        if direction == 'north' and self.is_move_valid((current_pos[0] - 1, current_pos[1])):
            self.agents[agent] = (current_pos[0] - 1, current_pos[1])
        elif direction == 'south' and self.is_move_valid((current_pos[0] + 1, current_pos[1])):
            self.agents[agent] = (current_pos[0] + 1, current_pos[1])
        elif direction == 'east' and self.is_move_valid((current_pos[0], current_pos[1] + 1)):
            self.agents[agent] = (current_pos[0], current_pos[1] + 1)
        elif direction == 'west' and self.is_move_valid((current_pos[0], current_pos[1] - 1)):
            self.agents[agent] = (current_pos[0], current_pos[1] - 1)

    def perform_pickup(self, agent):
        if self.agents[agent] in self.pickup_locations and self.blocks_carried[agent] < self.max_blocks_carried:
            self.blocks_carried[agent] += 1
            self.pickup_locations[self.agents[agent]] -= 1

    def perform_dropoff(self, agent):
        if self.agents[agent] in self.dropoff_locations and self.blocks_carried[agent] > 0:
            self.blocks_carried[agent] -= 1
            self.dropoff_locations[self.agents[agent]] += 1

    def step(self, agent_actions):
        for agent, action in agent_actions.items():
            if action in ['north', 'south', 'east', 'west']:
                self.move_agent(agent, action)
            elif action == 'pickup':
                self.perform_pickup(agent)
            elif action == 'dropoff':
                self.perform_dropoff(agent)

    def print_world(self):
        world_map = [['.' for _ in range(WORLD_SIZE[0])] for _ in range(WORLD_SIZE[1])]
        for loc in self.pickup_locations:
            if self.pickup_locations[loc] > 0:
                world_map[loc[0]][loc[1]] = 'P'
        for loc in self.dropoff_locations:
            if self.dropoff_locations[loc] < CAPACITY_DROPOFF:
                world_map[loc[0]][loc[1]] = 'D'
        for agent, pos in self.agents.items():
            symbol = agent[0].upper()
            world_map[pos[0]][pos[1]] = symbol
        for row in world_map:
            print(' '.join(row))
            
    def get_state(self, agent):
        pos = self.agents[agent]
        carrying_block = 1 if self.blocks_carried[agent] > 0 else 0
        state_index = (pos[0] * WORLD_SIZE[1] + pos[1]) * 2 + carrying_block
        return state_index

    def select_action(self, agent, policy):
        state = self.get_state(agent)
        if policy == "PRANDOM":
            return random.choice(ACTIONS)
        elif policy == "PEXPLOIT":
            if random.random() < EPSILON:
                return random.choice(ACTIONS)
            else:
                return ACTIONS[np.argmax(self.q_tables[agent][state])]
        else:  # PGREEDY
            return ACTIONS[np.argmax(self.q_tables[agent][state])]

    def simulate_action(self, agent, action):
        reward, _ = -1, self.get_state(agent)  # Default penalty for movement
        self.step({agent: action})
        new_state = self.get_state(agent)
        if action == 'pickup' and self.agents[agent] in PICKUP_LOCATIONS and self.blocks_carried[agent] == 0:
            reward = 10  # Successful pickup
        elif action == 'dropoff' and self.agents[agent] in DROPOFF_LOCATIONS and self.blocks_carried[agent] == 1:
            reward = 10  # Successful dropoff
        return reward, new_state

    def update_q_value(self, agent, action, reward, next_state):
        state = self.get_state(agent)
        action_index = ACTIONS.index(action)
        q_table = self.q_tables[agent]
        future_rewards = np.max(q_table[next_state])
        q_table[state, action_index] = (
            (1 - LEARNING_RATE) * q_table[state, action_index] +
            LEARNING_RATE * (reward + GAMMA * future_rewards)
        )

    def run(self):
        policy = "PRANDOM"
        for step in range(STEPS):
            if step == PRANDOM_STEPS:
                policy = "PGREEDY"
            for agent in self.agents:
                current_state = self.get_state(agent)
                action = self.select_action(current_state, policy)
                reward, next_state = self.simulate_action(agent, action)
                self.update_q_value(agent, action, reward, next_state)

    
world = PDWorld()
world.print_world()
world.run()  # Run the Q-learning simulation
print("\n")
world.print_world()
