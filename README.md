# COSC 4368 Group Project

Path Discovery in a 3-Agent Transportation World Using Reinforcement Learning 


## Running

```bash
python main.py
```

## Usage
```bash
#Uncomment the experiment you wish to run

#Experiment 1.a

world = PDWorld(randomseed=42)
algorithm = RLAlgorithm(learning_rate=0.3, discount_factor=0.5)
print("initial world: ")
world.display_world()
print("simulation a 500: ")
simulate(world, algorithm, 'PRandom', 500,randomseed=42)
print("simulation a 8500: ")
simulate(world, algorithm, 'PRandom', 8500,randomseed=42)
print()

#Experiment 1.b

# world = PDWorld(randomseed=42)
# algorithm = RLAlgorithm(learning_rate=0.3, discount_factor=0.5)
# print("initial world: ")
# world.display_world()
# print("simulation a 500: ")
# simulate(world, algorithm, 'PRandom', 500,randomseed=42)
# print("simulation a 8500: ")
# simulate(world, algorithm, 'PGreedy', 8500,randomseed=42)
# print()

#etc, etc
```
