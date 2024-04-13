import numpy as np
import random
class Agent:
    def __init__(self, start_position, name):
        self.position = start_position
        self.name = name
        self.has_block = False  # Indicates whether the agent is carrying a block

    def move(self, direction, world):
        moves = {
            'north': (-1, 0),
            'south': (1, 0),
            'east': (0, 1),
            'west': (0, -1)
        }
        if direction in moves:
            new_x = self.position[0] + moves[direction][0]
            new_y = self.position[1] + moves[direction][1]
            new_position = (new_x, new_y)
            if world.within_bounds(new_position):  # Move only if within bounds
                self.position = new_position

    def pickup(self, world):
        if world.is_pickup_cell(self.position) and world.pickup_cells[self.position] > 0 and not self.has_block:
            self.has_block = True
            world.pickup_cells[self.position] -= 1

    def dropoff(self, world):
        if world.is_dropoff_cell(self.position) and world.dropoff_cells[self.position] < 5 and self.has_block:
            self.has_block = False
            world.dropoff_cells[self.position] += 1

    def __repr__(self):
        return f"{self.name}(Position: {self.position}, Carries Block: {self.has_block})"


class PDWorld:
    def __init__(self):
        self.agents = {
            'red': Agent((4, 3), 'Red'),
            'blue': Agent((3, 5), 'Blue'),
            'black': Agent((2, 3), 'Black')
        }
        self.pickup_cells = {
            (1, 5): 5,  # (15)
            (2, 4): 5,  # (24)
            (5, 2): 5   # (52)
        }
        self.dropoff_cells = {
            (1, 1): 0,  # (11)
            (3, 1): 0,  # (31)
            (4, 5): 0   # (45)hhgg
        }
        self.grid_size = (5, 5)  # Assuming a 5x5 grid

    def within_bounds(self, position):
        x, y = position
        return 1 <= x <= self.grid_size[0] and 1 <= y <= self.grid_size[1]

    def is_pickup_cell(self, position):
        return position in self.pickup_cells

    def is_dropoff_cell(self, position):
        return position in self.dropoff_cells

    def display_world(self):
        grid = [['.' for _ in range(self.grid_size[1])] for _ in range(self.grid_size[0])]
        for pos, blocks in self.pickup_cells.items():
            x, y = pos
            grid[x-1][y-1] = f'P{blocks}'
        for pos, blocks in self.dropoff_cells.items():
            x, y = pos
            grid[x-1][y-1] = f'D{blocks}'
        for agent in self.agents.values():
            x, y = agent.position
            if agent.name == "Black":
                grid[x-1][y-1] = "Ba"
            elif agent.name == "Blue":
                grid[x-1][y-1] = "Bu"
            else:
                grid[x-1][y-1] = agent.name[0]
        for row in grid:
            print(' '.join(row))

# Initial setup for reinforcement learning algorithms
class RLAlgorithm:
    def __init__(self, learning_rate=0.1, discount_factor=0.9):
        self.q_table = {}
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

    def update_q_table(self, current_state, action, reward, next_state):
        pass  # Placeholder for Q-learning and SARSA update rules


class QLearning(RLAlgorithm):
    def update_q_table(self, current_state, action, reward, next_state, next_action=None):
        # Q-Learning update rule
        if (current_state, action) not in self.q_table:
            self.q_table[(current_state, action)] = 0  # Initialize if not present

        max_q_next = max(self.q_table.get((next_state, a), 0) for a in ['north', 'south', 'east', 'west', 'pickup', 'dropoff'])
        self.q_table[(current_state, action)] += self.learning_rate * (
            reward + self.discount_factor * max_q_next - self.q_table[(current_state, action)]
        )

class SARSA(RLAlgorithm):
    def update_q_table(self, current_state, action, reward, next_state, next_action):
        # SARSA update rule
        if (current_state, action) not in self.q_table:
            self.q_table[(current_state, action)] = 0  # Initialize if not present

        next_q = self.q_table.get((next_state, next_action), 0)  # Next action is part of the update
        self.q_table[(current_state, action)] += self.learning_rate * (
            reward + self.discount_factor * next_q - self.q_table[(current_state, action)]
        )

# Define Agent Policies
def choose_action(world, agent, q_learning, policy):
    valid_actions = ['north', 'south', 'east', 'west', 'pickup', 'dropoff']
    if policy == 'random':
        return random.choice(valid_actions)
    elif policy in ['exploitative', 'greedy']:
        q_values = {action: q_learning.q_table.get((agent.position, action), 0) for action in valid_actions}
        if policy == 'exploitative' and random.random() < 0.8:
            return max(q_values, key=q_values.get)
        elif policy == 'greedy':
            return max(q_values, key=q_values.get)
        else:
            return random.choice(valid_actions)

# Simulation Loop
def simulate(world, num_steps, policy):
    q_learning = QLearning()
    for step in range(num_steps):
        for name, agent in world.agents.items():
            action = choose_action(world, agent, q_learning, policy)
            current_state = agent.position
            if action in ['north', 'south', 'east', 'west']:
                agent.move(action, world)
            elif action == 'pickup':
                agent.pickup(world)
            elif action == 'dropoff':
                agent.dropoff(world)
            next_state = agent.position
            reward = -1  # Assume a default reward for movement
            if action == 'pickup' or action == 'dropoff':
                reward = 13  # Adjusted reward for successful interaction
            # Q-learning update (could switch to SARSA)
            q_learning.update_q_table(current_state, action, reward, next_state)
        world.display_world()
        print(f"Step {step + 1} complete")
        
        # Define applicable actions based on current state to make decisions more intelligent
def applicable_actions(agent, world):
    actions = []
    x, y = agent.position
    # Movement actions
    if world.within_bounds((x - 1, y)):  # North
        actions.append('north')
    if world.within_bounds((x + 1, y)):  # South
        actions.append('south')
    if world.within_bounds((x, y + 1)):  # East
        actions.append('east')
    if world.within_bounds((x, y - 1)):  # West
        actions.append('west')
    # Pickup and dropoff actions
    if world.is_pickup_cell(agent.position) and world.pickup_cells[agent.position] > 0 and not agent.has_block:
        actions.append('pickup')
    if world.is_dropoff_cell(agent.position) and world.dropoff_cells[agent.position] < 5 and agent.has_block:
        actions.append('dropoff')
    return actions

def choose_action(world, agent, q_learning, policy):
    actions = applicable_actions(agent, world)
    if not actions:  # Fallback if no actions are applicable
        return None
    if policy == 'random':
        return random.choice(actions)
    else:
        q_values = {action: q_learning.q_table.get((agent.position, action), 0) for action in actions}
        if policy == 'exploitative':
            if random.random() < 0.8:
                return max(q_values, key=q_values.get)  # Exploit
            else:
                return random.choice(actions)  # Explore
        elif policy == 'greedy':
            return max(q_values, key=q_values.get)  # Always exploit

# Enhanced simulation loop to handle different policies and smarter actions
def simulate(world, num_steps, policy):
    q_learning = QLearning()
    for step in range(num_steps):
        print(f"\nStep {step + 1} starting:")
        for name, agent in world.agents.items():
            action = choose_action(world, agent, q_learning, policy)
            if not action:
                continue  # Skip if no valid action is possible
            current_state = agent.position
            if action in ['north', 'south', 'east', 'west']:
                agent.move(action, world)
            elif action == 'pickup':
                agent.pickup(world)
            elif action == 'dropoff':
                agent.dropoff(world)
            next_state = agent.position
            reward = -1  # Assume a default reward for movement
            if action in ['pickup', 'dropoff']:
                reward = 13  # Reward for interacting with blocks
            # Update Q-table using Q-learning
            q_learning.update_q_table(current_state, action, reward, next_state)
        world.display_world()

# Run simulations with different policies
world = PDWorld()  # Reset world
print("Running simulation with Exploitative Policy:")
simulate(world, 10, 'exploitative')
world = PDWorld()  # Reset world for a clean start
print("\nRunning simulation with Greedy Policy:")
simulate(world, 10, 'greedy')


# Run the simulation
# simulate(world, 10, 'random')


# # Example use of the classes
# world = PDWorld()
# world.display_world()
# print(world.agents['red'])
# world.agents['red'].move('north', world)
# print(world.agents['red'])
# world.agents['red'].pickup(world)
# world.display_world()
