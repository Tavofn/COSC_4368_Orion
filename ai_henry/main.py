import random


#The agent class is the basis of our 3 agents in out PD world, the move, pickup, and dropoff functions are used in our simulate function 
#When we are moving around agents around the pdworld based on our algorithm in simulate.
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
        #the agent will pick up a block if its in the pickup cell, the number of blocks is greater than 0, and the agent does not have a block
        #This will be called by our simulate function and will be used in the process of determining the action our agents will take
        if world.is_pickup_cell(self.position) and world.pickup_cells[self.position] > 0 and not self.has_block:
            self.has_block = True
            world.pickup_cells[self.position] -= 1
            print(f"{self.name} picked up a block at {self.position}.")
    
    def dropoff(self, world):
        #the agent will drop off a block if its in the dropoff cell, the number of blocks is less than 5, and the agent has a block
        #This will be called by our simulate function and will be used in the process of determining the action our agents will take

        if world.is_dropoff_cell(self.position) and world.dropoff_cells[self.position] < 5 and self.has_block:
            self.has_block = False
            world.dropoff_cells[self.position] += 1
            print(f"{self.name} dropped off a block at {self.position}.")
     
#PD world creates our vizualization of our grid and has functions that prevent oddities in the creation of the visuals such
#as boundary checking, terminal state checks, and determining whether a position is a dropoff cell or pickup cell.
# THis is where the instantiation of our agents locations and pickup and dropoff locations are in.
class PDWorld:
    def __init__(self, randomseed, experiment4 = False):
        self.grid_size = (5, 5)
        # Assuming the grid positions are 0-indexed.
        # Adjust the coordinates if your grid is 1-indexed or follows a different system.
        self.agents = {
            'red': Agent((2, 2), 'Red'),  # Corrected to start in the middle of a 5x5 grid
            'blue': Agent((4, 2), 'Blue'),  # Starts on the bottom row, middle column
            'black': Agent((0, 2), 'Black')  # Starts on the top row, middle column
        }
        
        #Will change the pickup locations if we using experiment 4
        self.experiment4 = experiment4
        if self.experiment4 == False:

            self.pickup_cells = {(0, 4): 5, (1, 3): 5, (4, 1): 5}
        else:
            print("changed pickup positions")
            self.pickup_cells = {(2, 4): 5, (3, 3): 5, (4, 2): 5}
            
        self.dropoff_cells = {(0, 0): 0, (2, 0): 0, (3, 4): 0}
        
        self.randomseed = randomseed
        random.seed(self.randomseed)
    
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
        
#This is our regular Q-learning algorithm with a q table function and policies instantiated within. This algorithm function
#will be passed into our simulate function and will be a major factor in determining the moves that the simulate function will take
#This class specifies the policies, the best action based on the policy and actions that the program is allowed to take under
#the get_applicable_actions function. This also has our qtable function which has our q-learning equation and is a big factor on
#what our pvalues for our pd world will be.
class RLAlgorithm:
    def __init__(self, learning_rate=0.1, discount_factor=0.9, actions=['north', 'south', 'east', 'west', 'pickup', 'dropoff']):
        self.q_table = {}
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.actions = actions
    
    def select_action(self, state, policy, world):
        position, has_block = state
        applicable_actions = self.get_applicable_actions(position, has_block, world)
        
        if policy == 'PRandom':
            return random.choice(applicable_actions)
        
        if policy == 'PExploit':
            # With probability 0.8, exploit; with probability 0.2, explore
            if random.random() < 0.8:
                return self.get_best_action(state, applicable_actions)
            else:
                return random.choice(applicable_actions)
        
        if policy == 'PGreedy':
            return self.get_best_action(state, applicable_actions)

    def get_best_action(self, state, applicable_actions):
        # Fetch the best action based on Q-values from applicable actions
        best_q_value = max(self.q_table.get((state, action), 0) for action in applicable_actions)
        best_actions = [action for action in applicable_actions if self.q_table.get((state, action), 0) == best_q_value]
        return random.choice(best_actions)  # Break ties randomly

    def get_applicable_actions(self, position, has_block, world):
        # Determine actions that are actually possible in the current state
        applicable_actions = []
        if world.is_dropoff_cell(position) and has_block and world.dropoff_cells[position] < 5:
            applicable_actions.append('dropoff')
        if world.is_pickup_cell(position) and not has_block and world.pickup_cells[position] > 0:
            applicable_actions.append('pickup')
        if world.within_bounds((position[0] - 1, position[1])):  # North
            applicable_actions.append('north')
        if world.within_bounds((position[0] + 1, position[1])):  # South
            applicable_actions.append('south')
        if world.within_bounds((position[0], position[1] + 1)):  # East
            applicable_actions.append('east')
        if world.within_bounds((position[0], position[1] - 1)):  # West
            applicable_actions.append('west')
        return applicable_actions
    
    
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


#This is our SARSA algorithm, it is a extension of our q-learning algorithm class however the biggest difference is the determining of our q values 
#which is in our update_q_table function. The equation in this function is vastly different from our update_q_table in the regular
#q-learning algorithm. Most of the methology is the same where we have our policy specification, our best action under each policy determining function, a
#and our applicable actions function.

class Sarsa(RLAlgorithm):
    def __init__(self, learning_rate=0.1, discount_factor=0.9, actions=['north', 'south', 'east', 'west', 'pickup', 'dropoff']):
        self.q_table = {}
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.actions = actions
     
        
    def select_action(self, state, policy, world):
        position, has_block = state
        applicable_actions = self.get_applicable_actions(position, has_block, world)
        
        if policy == 'PRandom':
            return random.choice(applicable_actions)
        
        if policy == 'PExploit':
            # With probability 0.8, exploit; with probability 0.2, explore
            if random.random() < 0.8:
                return self.get_best_action(state, applicable_actions)
            else:
                return random.choice(applicable_actions)
        
        if policy == 'PGreedy':
            return self.get_best_action(state, applicable_actions)

    def get_best_action(self, state, applicable_actions):
        # Fetch the best action based on Q-values from applicable actions
        best_q_value = max(self.q_table.get((state, action), 0) for action in applicable_actions)
        best_actions = [action for action in applicable_actions if self.q_table.get((state, action), 0) == best_q_value]
        return random.choice(best_actions)  # Break ties randomly

    def get_applicable_actions(self, position, has_block, world):
        # Determine actions that are actually possible in the current state
        applicable_actions = []
        if world.is_dropoff_cell(position) and has_block and world.dropoff_cells[position] < 5:
            applicable_actions.append('dropoff')
        if world.is_pickup_cell(position) and not has_block and world.pickup_cells[position] > 0:
            applicable_actions.append('pickup')
        if world.within_bounds((position[0] - 1, position[1])):  # North
            applicable_actions.append('north')
        if world.within_bounds((position[0] + 1, position[1])):  # South
            applicable_actions.append('south')
        if world.within_bounds((position[0], position[1] + 1)):  # East
            applicable_actions.append('east')
        if world.within_bounds((position[0], position[1] - 1)):  # West
            applicable_actions.append('west')
        return applicable_actions
    
    
    def update_q_table(self, current_state, action, reward, next_state, next_action, policy):
        if (current_state, action) not in self.q_table:
            self.q_table[(current_state, action)] = 0
        target = reward + self.discount_factor * self.q_table.get((next_state, next_action), 0)
        self.q_table[(current_state, action)] += self.learning_rate * (target - self.q_table[(current_state, action)])
    
    def print_q_table(self):
        # Print the Q-table in a formatted manner
        print("Q-Table:")
        for key, value in sorted(self.q_table.items()):
            state, action = key
            print(f"State {state}, Action {action}: {value:.2f}")



#Experiment 1
#This simulate function is our runtime function for our regular q-learning algorithm class.
#This function is what outputs our pdworld and moves the agents at the same time. The q-learning 
#algorithm is passed through here alongside its policy and the q-learning class determines the best action
#based on the policy and returns the best action, this action moves the agents in all sorts of directions. After the 
#agent is moved, the q value is updated alonside with it our q table is outputted when we finish all the steps values specified.
#the movement of the agent is resulted in our visualization which is called under world.displayworld()
def simulate(world, algorithm, policy, steps,randomseed):
    for step in range(steps):
        if world.check_terminal_state():
            print(f"Terminal state reached after {step} steps.")
            world.__init__(randomseed=randomseed)  
        for name, agent in world.agents.items():
            state = (agent.position, agent.has_block)
            action = algorithm.select_action(state, policy,world)
            # print(f"Current State of {name}: {state}")
            # print(f"{name} takes action: {action}")
            if action in ['north', 'south', 'east', 'west']:
                agent.move(action, world)
            elif action == 'pickup':
                agent.pickup(world)
            elif action == 'dropoff':
                agent.dropoff(world)
            next_state = (agent.position, agent.has_block)
            reward = -1 if action in ['north', 'south', 'east', 'west'] else 13
            algorithm.update_q_table(state, action, reward, next_state, policy)
    algorithm.print_q_table()
    world.display_world()
    
#Experiment 2/3    
#This simulate function is very similar to the first simulate function however it is built for SARSA/
#What differs is the implementation of a queue helps remember the action that has already been determined. This 
#helps determine the guarenteed next action for our agent.
def simulate2(world, algorithm, policy, steps, randomseed):
    Actions = ['','','']
    terminal_counter = 0
    for step in range(steps):                                                                                                                               
        if world.check_terminal_state():
            print(f"Terminal state reached after {step} steps.")
            terminal_counter+=1
            print(terminal_counter)
            world.__init__(randomseed=randomseed)                                                                                                                                                                                                
            Actions = ['','','']
        for name, agent in world.agents.items():
            state = (agent.position, agent.has_block)
            if Actions[0] == '':
                action = algorithm.select_action(state, policy,world)
            else:
                action = Actions[0]
            if action in ['north', 'south', 'east', 'west']:
                agent.move(action, world)
            elif action == 'pickup':
                agent.pickup(world)
            elif action == 'dropoff':
                agent.dropoff(world)
            next_state = (agent.position, agent.has_block)
            next_action = algorithm.select_action(next_state, policy, world)
            Actions.pop(0)
            Actions.append(next_action)
            reward = -1 if next_action in ['north', 'south', 'east', 'west'] else 13
            algorithm.update_q_table(state, action, reward, next_state, next_action, policy)
    algorithm.print_q_table()      
    world.display_world()

    
#experiment 4 works very similar to experiment 2/3 however with the implementation of the terminal state conditions
#where if its less than 3, reset the pd world like normal and if it's greater than 3 or less than 6, reset the pdworld but
#with the experiment4 variable enabled, this changes the pickup locations to the new locations in our PD world class specified in our requriements
#Once it reaches 6, the program terminates completely.
def simulate4(world, algorithm, policy, steps, randomseed, TerminalStates):
    terminalStateCount = TerminalStates
    for step in range(steps):
        if world.check_terminal_state():
            print(f"Terminal state reached after {step} steps.")
            terminalStateCount += 1
            print("TERMINAL STATE COUNT ADDED")
            print(terminalStateCount)
            if terminalStateCount < 3:
                world.__init__(randomseed=randomseed)
            elif terminalStateCount < 6:
                world.__init__(randomseed=randomseed, experiment4=True)
            else:
                world.display_world()
                return
        for name, agent in world.agents.items():
            state = (agent.position, agent.has_block)
            action = algorithm.select_action(state, policy, world)
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
    if not world.check_terminal_state():
        print(f"Simulation ended without reaching the terminal state after {steps} steps.")
    # algorithm.print_q_table()  # Print the Q-table at the end of the simulation
    return terminalStateCount

def reset_simulation(world, algorithm):
    world.__init__()  # Reinitialize world to reset agent positions and blocks
    algorithm.q_table.clear()  # Clear Q-table for a fresh start in learning

        

# Initialize the world and run the simulation

#Experiment 1.c
# world = PDWorld(42)
# algorithm = RLAlgorithm(learning_rate=0.3, discount_factor=0.5)
# print("initial world: ")
# world.display_world()
# print("simulation a 500: ")
# simulate(world, algorithm, 'PRandom', 500)
# print("simulation a 8500: ")
# simulate(world, algorithm, 'PExploit', 8500)
# print()

#Experiment 2
# world = PDWorld(randomseed=42)
# algorithm = Sarsa(learning_rate=0.3, discount_factor=0.5)
# print("initial world: ")
# world.display_world()
# print("simulation a 500: ")
# simulate2(world, algorithm, 'PRandom', 500, randomseed=42)
# print("simulation a 8500: ")
# simulate2(world, algorithm, 'PExploit', 8500, randomseed=42)
# print()

#Experiment 3 using sarsa
#randomseed = 42
#world = PDWorld(randomseed)
#algorithm = Sarsa(learning_rate=0.45, discount_factor=0.5)
#print("initial world: ")
#world.display_world()
#print("simulation a 500: ")
#simulate2(world, algorithm, 'PRandom', 500,randomseed=randomseed)
#print("simulation a 8500: ")
#simulate2(world, algorithm, 'PExploit', 8500,randomseed=randomseed)
#print()

#Experiment 4
#randomseed = 42
#world = PDWorld(randomseed=randomseed)
#algorithm = RLAlgorithm(learning_rate=0.3, discount_factor=0.5)
#print("initial world: ")
#world.display_world()
#print("simulation a 500: ")
#terminal_states = simulate4(world, algorithm, 'PRandom', 500,randomseed, 0)
#print("simulation a 8500: ")
#simulate4(world, algorithm, 'PExploit', 8500,randomseed, terminal_states)
#print()


