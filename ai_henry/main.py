import random

class Agent:
    def __init__(self, start_position, name):
        self.position = start_position
        self.name = name
        self.has_block = False

    def move(self, direction, world):
        moves = {'north': (-1, 0), 'south': (1, 0), 'east': (0, 1), 'west': (0, -1)}
        # Calculate new position
        new_x = self.position[0] + moves[direction][0]
        new_y = self.position[1] + moves[direction][1]
        new_position = (new_x, new_y)
        
        # Check if the new position is within bounds and not occupied by another agent
        if world.within_bounds(new_position) and not world.is_occupied(new_position, self):
            self.position = new_position
        

    def pickup(self, world):
        if world.is_pickup_cell(self.position) and world.pickup_cells[self.position] > 0 and not self.has_block:
            self.has_block = True
            world.pickup_cells[self.position] -= 1
            print(f"{self.name} picked up a block at {self.position}.")

    def dropoff(self, world):
        if world.is_dropoff_cell(self.position) and world.dropoff_cells[self.position] < 5 and self.has_block:
            self.has_block = False
            world.dropoff_cells[self.position] += 1
            print(f"{self.name} dropped off a block at {self.position}.")

class PDWorld:
    def __init__(self):
        self.grid_size = (5, 5)
        # Assuming the grid positions are 0-indexed.
        # Adjust the coordinates if your grid is 1-indexed or follows a different system.
        self.agents = {
            'red': Agent((2, 2), 'Red'),  # Corrected to start in the middle of a 5x5 grid
            'blue': Agent((4, 2), 'Blue'),  # Starts on the bottom row, middle column
            'black': Agent((0, 2), 'Black')  # Starts on the top row, middle column
        }
        self.pickup_cells = {(0, 4): 5, (1, 3): 5, (4, 1): 5}
        self.dropoff_cells = {(0, 0): 0, (2, 0): 0, (3, 4): 0}
    
    def is_occupied(self, position, current_agent):
        # Check all agents to see if any occupy the given position
        for name, agent in self.agents.items():
            if agent != current_agent and agent.position == position:
                return True
        return False
    
    def check_terminal_state(self):
        # Check if all pickup locations are empty
        all_picked = all(blocks == 0 for blocks in self.pickup_cells.values())
        # Check if all dropoff locations are at capacity
        all_dropped = all(blocks == 5 for blocks in self.dropoff_cells.values())
        return all_picked and all_dropped
    
    
    def is_dropoff_cell(self, position):
        return position in self.dropoff_cells
    
    def is_pickup_cell(self, position):
        return position in self.pickup_cells
    
    def within_bounds(self, position):
        x, y = position
        return 0 <= x < self.grid_size[0] and 0 <= y < self.grid_size[1]

    def display_world(self):
        grid = [['.' for _ in range(self.grid_size[1])] for _ in range(self.grid_size[0])]
        for pos, blocks in self.pickup_cells.items():
            grid[pos[0]][pos[1]] = 'P' + str(blocks)
        for pos, blocks in self.dropoff_cells.items():
            grid[pos[0]][pos[1]] = 'D' + str(blocks)
        for agent in self.agents.values():
            grid[agent.position[0]][agent.position[1]] = agent.name[0] + ('(B)' if agent.has_block else '')
        print('\n'.join(' '.join(row) for row in grid))
        print()
        

class RLAlgorithm:
    def __init__(self, learning_rate=0.1, discount_factor=0.9, actions=['north', 'south', 'east', 'west', 'pickup', 'dropoff'],epsilon=0.1, randomseed=42):
        self.q_table = {}
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.actions = actions
        self.epsilon = epsilon  # Exploration rate
        self.randomseed = randomseed
        
    def select_action(self, state, policy):
        random.seed(self.randomseed)
        if policy == 'PRandom' or (policy == 'PExploit' and random.random() < 0.2) or (policy == 'PGreedy' and random.random() < self.epsilon):
            return random.choice(self.actions)
        best_action = max(self.actions, key=lambda action: self.q_table.get((state, action), 0))
        return best_action if best_action else random.choice(self.actions)
    
    def update_q_table(self, current_state, action, reward, next_state, policy):
        if (current_state, action) not in self.q_table:
            self.q_table[(current_state, action)] = 0
        next_max = max(self.q_table.get((next_state, a), 0) for a in self.actions)
        self.q_table[(current_state, action)] += self.learning_rate * (reward + self.discount_factor * next_max - self.q_table[(current_state, action)])
        
    def print_q_table(self):
        # Print the Q-table in a formatted manner
        print("Q-Table:")
        for key, value in sorted(self.q_table.items()):
            state, action = key
            print(f"State {state}, Action {action}: {value:.2f}")

class Sarsa(RLAlgorithm):
    def __init__(self, learning_rate=0.1, discount_factor=0.9, actions=['north', 'south', 'east', 'west', 'pickup', 'dropoff'],epsilon=0.1, randomseed=42):
        self.q_table = {}
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.actions = actions
        self.epsilon = epsilon  # Exploration rate
        self.randomseed = randomseed
        
    def select_action(self, state, policy):
        random.seed(self.randomseed)

        if policy == 'PRandom' or (policy == 'PExploit' and random.random() < 0.2) or (policy == 'PGreedy' and random.random() < self.epsilon):
            return random.choice(self.actions)
        best_action = max(self.actions, key=lambda action: self.q_table.get((state, action), 0))
        return best_action if best_action else random.choice(self.actions)
    
    def update_q_table(self, current_state, action, reward, next_state, policy):
        if (current_state, action) not in self.q_table:
            self.q_table[(current_state, action)] = 0
        next_action = self.select_action(next_state, policy)
        target = reward + self.discount_factor * self.q_table.get((next_state, next_action), 0)
        self.q_table[(current_state, action)] += self.learning_rate * (target - self.q_table[(current_state, action)])
        
    
    def print_q_table(self):
        # Print the Q-table in a formatted manner
        print("Q-Table:")
        for key, value in sorted(self.q_table.items()):
            state, action = key
            print(f"State {state}, Action {action}: {value:.2f}")

def simulate(world, algorithm, policy, steps):
    # for step in range(steps):
    #     if world.check_terminal_state():
    #         print(f"Terminal state reached after {step} steps.")
    #         world.__init__()
        
    #     for name, agent in world.agents.items():
    #         state = (agent.position, agent.has_block)
    #         action = algorithm.select_action(state, policy)
    #         if action in ['north', 'south', 'east', 'west']:
    #             agent.move(action, world)
    #         elif action == 'pickup':
    #             agent.pickup(world)
    #         elif action == 'dropoff':
    #             agent.dropoff(world)
    #         next_state = (agent.position, agent.has_block)
    #         reward = -1 if action in ['north', 'south', 'east', 'west'] else 13
    #         algorithm.update_q_table(state, action, reward, next_state, policy)
    # world.display_world()
    for step in range(steps):
        print(f"Step {step+1}:")
        for name, agent in world.agents.items():
            state = (agent.position, agent.has_block)
            print(f"Current State of {name}: {state}")
            action = algorithm.select_action(state, policy)
            print(f"{name} takes action: {action}")
            if action in ['north', 'south', 'east', 'west']:
                agent.move(action, world)
            elif action == 'pickup':
                agent.pickup(world)
            elif action == 'dropoff':
                agent.dropoff(world)
            next_state = (agent.position, agent.has_block)
            reward = -1 if action in ['north', 'south', 'east', 'west'] else 13
            algorithm.update_q_table(state, action, reward, next_state, policy)
            print(f"New State of {name}: {next_state}, Reward: {reward}")
        world.display_world()
        if world.check_terminal_state():
            print("Terminal state reached.")
            break
    
def reset_simulation(world, algorithm):
    world.__init__()  # Reinitialize world to reset agent positions and blocks
    algorithm.q_table.clear()  # Clear Q-table for a fresh start in learning

        

# Initialize the world and run the simulation
world = PDWorld()
algorithm = RLAlgorithm(learning_rate=0.3, discount_factor=0.5)
print("initial world: ")
world.display_world()
print("simulation a 500: ")
simulate(world, algorithm, 'PRandom', 500)
print("simulation a 8500: ")
simulate(world, algorithm, 'PRandom', 8500)
print()


# reset_simulation(world, algorithm)  # Reset for next experiment

# print("simulation b 500: ")
# simulate(world, algorithm, 'PRandom', 500)
# print("simulation b 8500: ")
# simulate(world, algorithm, 'PGreedy', 8500)
# print()


# reset_simulation(world, algorithm)  # Reset for next experiment

# print("simulation c 500: ")
# simulate(world, algorithm, 'PRandom', 500)
# print("simulation c 8500: ")
# simulate(world, algorithm, 'PExploit', 8500)
# print()

# world = PDWorld()
# algorithm = Sarsa(learning_rate=0.3, discount_factor=0.5)
# reset_simulation(world, algorithm)  # Reset for next experiment

# print("SARSA simulation:")
# simulate(world, algorithm, 'PExploit', 9000)

