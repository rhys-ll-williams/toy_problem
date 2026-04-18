# Example Q-learn reinforcement learning
#
# Important - the penalties and rewards need to be carefully considered
# the alpha, gamma, epsilon and episodes control the algorithm as hyperparameters
#
import numpy as np
import random
import copy

SEED=62
NUM_TESTS=100

GRID_SIZE = 5
ACTIONS = ['up','down','left','right','maintain']
ACTION_IDX = {a: i for i, a in enumerate(ACTIONS)}
goal = (GRID_SIZE-1,GRID_SIZE-1)
start = (0,0)

# other vehicles in the simulation
initial_obstacles = [(2,2),(3,1)]

# parameters for reinforcement learning with Q-learn
alpha=0.1
gamma = 0.9
epsilon = 0.1
episodes = 500 # for quick test that the code runs
episodes = 5000 # fast but looking for >85% success learned
episodes = 50000 # should be getting 95% or more success
episodes = 500000 # takes some time, but looking for 100% learned
episodes = 5000000 # takes some time, but looking for 100% learned

# Initialize the Q-table with state including the obstacle positions
# use a dictionary so we don't upfront decide what the total table looks like
# The RL model is simple for Q-Learn...
Q = {}


# TASK 1 => create a simulator class that includes
# is_valid()
# get_next_state()
# get_reward()
# move_obstacles() - rename as step() so that it is generic rather than specific to this system
#
def is_valid(pos):
    """
    The proposed move is valid if it stays insid
    e the grid
    """
    global GRID_SIZE
    x,y = pos
    return 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE

def get_next_state(pos, action):
    """
    Apply the proposed action and update the position only
    if the action is valid
    """
    global ACTIONS
    x, y = pos
    if action == ACTIONS[0]: x-=1 # 'up'
    if action == ACTIONS[1]: x+=1 # 'down'
    if action == ACTIONS[2]: y-=1 # 'left'
    if action == ACTIONS[3]: y+=1 # 'right'
    # ACTIONS[4] is 'maintain' so do nothing...
    next_pos = (x,y)
    return next_pos if is_valid(next_pos) else pos 
        
def get_reward(pos, obstacles):
    """
    Reward lot for reaching the finish
    Opposite for hitting the obstacles
    Slight penalty for any other move
    """
    global goal
    if pos == goal:
        return 100  # Finish line
    elif pos in obstacles:
        return -1000000000000 # crash!
    else:
        return -1 # everything else

def move_obstacles(obstacles):
    """
    Obstacles move randomly - this is the simulator
    ... an obstacle can run into the agent, but the agent will try to move
    ... away on the next turn
    """
    global ACTIONS
    next_pos = []
    for obs in obstacles:
        action = random.choice(ACTIONS)
        next_obs_pos = get_next_state(obs, action)
        next_pos.append(next_obs_pos)
    return next_pos
    #return [get_next_state(obs, random.choice(ACTIONS)) for obs in obstacles]


# 
# TRAINING LOOP
random.seed(1)
for ep in range(episodes):
    agent_pos = copy.deepcopy(start)
    obstacles = copy.deepcopy(initial_obstacles) # same initial starting point every time (easier to learn than if the obstacles were in randomly different places each time.

    while agent_pos != goal: # in each simulation play all the way to the end
        state_key = (*agent_pos, *obstacles[0], *obstacles[1])
        # if this is the first time for this situation
        if state_key not in Q:
            Q[state_key] = np.zeros(len(ACTIONS)) # no actions yet

        # choose either a random action or the best action we know for this
        # epsilon is the fraction of random steps
        if random.uniform(0, 1) < epsilon:
            action = random.choice(ACTIONS)
        else: # pick the highest scoring action
            action = ACTIONS[np.argmax(Q[state_key])]

        # try to move
        next_agent_pos = get_next_state(agent_pos, action) 

        # calculate the reward for this move
        reward = get_reward(next_agent_pos, obstacles)

        # UPDATE SIMULATION BY MOVING 1 TIMESTEP
        next_obstacles = move_obstacles(obstacles)
        next_state_key = (*next_agent_pos, *next_obstacles[0], *next_obstacles[1])
        # again if this is the first time in this state add it to Q
        if next_state_key not in Q:
            Q[next_state_key] = np.zeros(len(ACTIONS))

        # Do a Q-learn update
        old_value = Q[state_key][ACTION_IDX[action]] # current value of the move at this step
        next_max = np.max(Q[next_state_key]) # current best move at next step

        # Bellman equation update
        #   adjust old_value by adding on an updated score
        #       reward    = reward for this move (in this example 100, -100, or -1)
        #  gamma*next_max = best move after this move times gamma
        #                   ...(could play 2 or more forward moves if we wanted by having gamma_1, gamma_2 etc.
        #                   ...this is like a chess computer looking multiple moves ahead.
        #   old_value     = the orignal value, this is removed from the update
        #
        # the whole update is multiplied by alpha (so default is 0.1, which means only 10% of the reward value is used
        #     to update the action.
        Q[state_key][ACTION_IDX[action]] = (1-alpha)* old_value + alpha*(reward + gamma*next_max)
        

        # finish by making the move
        agent_pos = next_agent_pos
        obstacles = next_obstacles


# RUN a check on how well the problem has been learned
def test_agent(Q, test_episodes=100, seed=1, verbose=True):
    random.seed(seed)
    success_count = 0
    total_steps = 0

    # loop over the number of tests requested
    for _ in range(test_episodes):
        agent_pos = copy.deepcopy(start)
        obstacles = copy.deepcopy(initial_obstacles)
        steps = 0 # count how many steps to get to the finish
        max_steps = 100 # in case the car gets losts and is driving aimlessly aroun

        while agent_pos != goal and steps < max_steps:
            state_key = (*agent_pos, *obstacles[0], *obstacles[1])
            if state_key not in Q:
                break # Never was in this state before, so assume we have failed this test

            # Choose the action which worked best in this situation during training...
            action = ACTIONS[np.argmax(Q[state_key])]
            next_agent_pos = get_next_state(agent_pos, action)

            if next_agent_pos in obstacles:
                break # had a crash so failed this test

            if test_episodes<2 and verbose==True: # run an individual case explicitly
                print(steps,agent_pos,obstacles)


            # update the agent's position
            agent_pos = next_agent_pos

            # simulate the next timestep
            obstacles = move_obstacles(obstacles)
            steps+=1

        # is it successful?
        if agent_pos == goal:
            success_count += 1
            total_steps += steps


    success_rate = success_count/test_episodes
    avg_steps = total_steps / success_count if success_count > 0 else float('inf')

    if verbose==True:
        print("Q-LEARNING BASED")
        print(f"Success rate: {success_rate * 100:.2f}%")
        print(f"Average steps to goal (successful episodes): {avg_steps:.2f}")
    return avg_steps


# run the test
test_agent(Q,test_episodes=NUM_TESTS,seed=SEED)

# look at a small part of the learned system
for a in range(10):
    key, value = random.choice(list(Q.items()))
    print(f"{key}: {value}")

print(f"Q has {len(list(Q.keys()))} states.")


######## TEST The rule based version
# Rule-based model
# test in order. First accepted move wins
# 1. move down
# 2. move right
# 3. move up
# 4. move left
# 5. do nothing (maybe the obstacles will move out of the way!
# ...problem is that we could end up in a corner or deadend.
# ...is the performance better or worse than Q-Learn?
    

def test_rules_based(test_episodes=NUM_TESTS,seed=1,verbose=True):
    random.seed(seed) # use a random seed so that the same test problems are applied to RL and rules-based
    success_count = 0
    total_steps = 0

    # loop over the number of tests requested
    for _ in range(test_episodes):
        agent_pos = copy.deepcopy(start)
        obstacles = copy.deepcopy(initial_obstacles)
        steps = 0 # count how many steps to get to the finish
        max_steps = 100 # in case the car gets losts and is driving aimlessly aroun

        while agent_pos != goal and steps < max_steps:
            # Q-learn code for comparison
            # ...Rules never fail just because we didn't see the case before
            #state_key = (*agent_pos, *obstacles[0], *obstacles[1])
            #if state_key not in Q:
            #    break # Never was in this state before, so assume we have failed this test

            # Choose the action which worked best in this situation during training...
            #action = ACTIONS[np.argmax(Q[state_key])]
            #next_agent_pos = get_next_state(agent_pos, action)
            # RULES BASED ACTION
            for action in [ACTIONS[1],ACTIONS[3],ACTIONS[0],ACTIONS[2]]:
                next_agent_pos = get_next_state(agent_pos, action)
                if next_agent_pos == agent_pos:
                    continue # move was outside the grid
                # if there is no obstacle, we can make the move
                if not (next_agent_pos in obstacles):
                    break
                # if no valid move is found, stay still
                next_agent_pos=agent_pos

            if test_episodes<2 and verbose==True: # run an individualcase explicitly
                print(steps,agent_pos,obstacles)

            

            # ...Rules never hit an obstacle, but can fail to move
            # ...off an obstacle in the next turn
            if next_agent_pos in obstacles:
                break # had a crash so failed this test

            # update the agent's position
            agent_pos = next_agent_pos

            # simulate the next timestep
            obstacles = move_obstacles(obstacles)
            steps+=1

        #...Rules fail by meandering around (getting stuck next to
        #...and obstacle an going back and forth in a corner...
        # is it successful?
        if agent_pos == goal:
            success_count += 1
            total_steps += steps


    success_rate = success_count/test_episodes
    avg_steps = total_steps / success_count if success_count > 0 else float('inf')

    if verbose==True:
        print("RULES-BASED")
        print(f"Success rate: {success_rate * 100:.2f}%")
        print(f"Average steps to goal (successful episodes): {avg_steps:.2f}")
    return avg_steps

            
# run the test
test_rules_based(test_episodes=NUM_TESTS,seed=SEED)
    


# find an example of RL doing something clever...
for a in range(1000):
    avg1 = test_agent(Q,test_episodes=1,seed=a,verbose=False)
    avg2 = test_rules_based(test_episodes=1,seed=a,verbose=False)
    if avg1<avg2:
        break
print(a,avg1,avg2)
        




