
import numpy as np 
import random
import matplotlib.pyplot as plt # Graphical library
#from sklearn.metrics import mean_squared_error # Mean-squared error function

def get_CID():
  return "02461091" # Return your CID (add 0 at the beginning to ensure it is 8 digits long)

def get_login():
  return "jcv23" # Return your short imperial login



class GraphicsMaze(object):

  def __init__(self, shape, locations, default_reward, obstacle_locs, absorbing_locs, absorbing_rewards, absorbing):

    self.shape = shape
    self.locations = locations
    self.absorbing = absorbing

    # Walls
    self.walls = np.zeros(self.shape)
    for ob in obstacle_locs:
      self.walls[ob] = 20

    # Rewards
    self.rewarders = np.ones(self.shape) * default_reward
    for i, rew in enumerate(absorbing_locs):
      self.rewarders[rew] = 10 if absorbing_rewards[i] > 0 else -10

    # Print the map to show it
    self.paint_maps()

  def paint_maps(self):
    """
    Print the Maze topology (obstacles, absorbing states and rewards)
    input: /
    output: /
    """
    plt.figure(figsize=(15,10))
    plt.imshow(self.walls + self.rewarders)
    plt.show()

  def paint_state(self, state):
    """
    Print one state on the Maze topology (obstacles, absorbing states and rewards)
    input: /
    output: /
    """
    states = np.zeros(self.shape)
    states[state] = 30
    plt.figure(figsize=(15,10))
    plt.imshow(self.walls + self.rewarders + states)
    plt.show()

  def draw_deterministic_policy(self, Policy):
    """
    Draw a deterministic policy
    input: Policy {np.array} -- policy to draw (should be an array of values between 0 and 3 (actions))
    output: /
    """
    plt.figure(figsize=(15,10))
    plt.imshow(self.walls + self.rewarders) # Create the graph of the Maze
    for state, action in enumerate(Policy):
      if(self.absorbing[0,state]): # If it is an absorbing state, don't plot any action
        continue
      arrows = [r"$\uparrow$",r"$\rightarrow$", r"$\downarrow$", r"$\leftarrow$"] # List of arrows corresponding to each possible action
      action_arrow = arrows[action] # Take the corresponding action
      location = self.locations[state] # Compute its location on graph
      plt.text(location[1], location[0], action_arrow, ha='center', va='center') # Place it on graph
    plt.show()

  def draw_policy(self, Policy):
    """
    Draw a policy (draw an arrow in the most probable direction)
    input: Policy {np.array} -- policy to draw as probability
    output: /
    """
    deterministic_policy = np.array([np.argmax(Policy[row,:]) for row in range(Policy.shape[0])])
    self.draw_deterministic_policy(deterministic_policy)

  def draw_value(self, Value):
    """
    Draw a policy value
    input: Value {np.array} -- policy values to draw
    output: /
    """
    plt.figure(figsize=(15,10))
    plt.imshow(self.walls + self.rewarders) # Create the graph of the Maze
    for state, value in enumerate(Value):
      if(self.absorbing[0, state]): # If it is an absorbing state, don't plot any value
        continue
      location = self.locations[state] # Compute the value location on graph
      plt.text(location[1], location[0], round(value,2), ha='center', va='center') # Place it on graph
    plt.show()

  def draw_deterministic_policy_grid(self, Policies, title, n_columns, n_lines):
    """
    Draw a grid representing multiple deterministic policies
    input: Policies {np.array of np.array} -- array of policies to draw (each should be an array of values between 0 and 3 (actions))
    output: /
    """
    plt.figure(figsize=(20,8))
    for subplot in range (len(Policies)): # Go through all policies
      ax = plt.subplot(n_columns, n_lines, subplot+1) # Create a subplot for each policy
      ax.imshow(self.walls+self.rewarders) # Create the graph of the Maze
      for state, action in enumerate(Policies[subplot]):
        if(self.absorbing[0,state]): # If it is an absorbing state, don't plot any action
          continue
        arrows = [r"$\uparrow$",r"$\rightarrow$", r"$\downarrow$", r"$\leftarrow$"] # List of arrows corresponding to each possible action
        action_arrow = arrows[action] # Take the corresponding action
        location = self.locations[state] # Compute its location on graph
        plt.text(location[1], location[0], action_arrow, ha='center', va='center') # Place it on graph
      ax.title.set_text(title[subplot]) # Set the title for the graph given as argument
    plt.show()

  def draw_policy_grid(self, Policies, title, n_columns, n_lines):
    """
    Draw a grid representing multiple policies (draw an arrow in the most probable direction)
    input: Policy {np.array} -- array of policies to draw as probability
    output: /
    """
    deterministic_policies = np.array([[np.argmax(Policy[row,:]) for row in range(Policy.shape[0])] for Policy in Policies])
    self.draw_deterministic_policy_grid(deterministic_policies, title, n_columns, n_lines)

  def draw_value_grid(self, Values, title, n_columns, n_lines):
    """
    Draw a grid representing multiple policy values
    input: Values {np.array of np.array} -- array of policy values to draw
    output: /
    """
    plt.figure(figsize=(20,8))
    for subplot in range (len(Values)): # Go through all values
      ax = plt.subplot(n_columns, n_lines, subplot+1) # Create a subplot for each value
      ax.imshow(self.walls+self.rewarders) # Create the graph of the Maze
      for state, value in enumerate(Values[subplot]):
        if(self.absorbing[0,state]): # If it is an absorbing state, don't plot any value
          continue
        location = self.locations[state] # Compute the value location on graph
        plt.text(location[1], location[0], round(value,1), ha='center', va='center') # Place it on graph
      ax.title.set_text(title[subplot]) # Set the title for the graoh given as argument
    plt.show()



# ## Maze class


# This class define the Maze environment

class Maze(object):

  # [Action required]
  def __init__(self):
    """
    Maze initialisation.
    input: /
    output: /
    """
    
    # [Action required]
    # Properties set from the CID
    self._prob_success = 0.8 + 0.02 * (9-int(get_CID()[-2])) # float
    self._gamma = 0.8 + 0.02 * int(get_CID()[-2]) # float
    self._goal = 1 # integer (0 for R0, 1 for R1, 2 for R2, 3 for R3)

    # Build the maze
    self._build_maze()
                              

  # Functions used to build the Maze environment 
  # You DO NOT NEED to modify them
  def _build_maze(self):
    """
    Maze initialisation.
    input: /
    output: /
    """

    # Properties of the maze
    self._shape = (13, 10)
    self._obstacle_locs = [
                          (1,0), (1,1), (1,2), (1,3), (1,4), (1,7), (1,8), (1,9), \
                          (2,1), (2,2), (2,3), (2,7), \
                          (3,1), (3,2), (3,3), (3,7), \
                          (4,1), (4,7), \
                          (5,1), (5,7), \
                          (6,5), (6,6), (6,7), \
                          (8,0), \
                          (9,0), (9,1), (9,2), (9,6), (9,7), (9,8), (9,9), \
                          (10,0)
                         ] # Location of obstacles
    self._absorbing_locs = [(2,0), (2,9), (10,1), (12,9)] # Location of absorbing states
    self._absorbing_rewards = [ (500 if (i == self._goal) else -50) for i in range (4) ]
    self._starting_locs = [(0,0), (0,1), (0,2), (0,3), (0,4), (0,5), (0,6), (0,7), (0,8), (0,9)] #Reward of absorbing states
    self._default_reward = -1 # Reward for each action performs in the environment
    self._max_t = 500 # Max number of steps in the environment

    # Actions
    self._action_size = 4
    self._direction_names = ['N','E','S','W'] # Direction 0 is 'N', 1 is 'E' and so on
        
    # States
    self._locations = []
    for i in range (self._shape[0]):
      for j in range (self._shape[1]):
        loc = (i,j) 
        # Adding the state to locations if it is no obstacle
        if self._is_location(loc):
          self._locations.append(loc)
    self._state_size = len(self._locations)

    # Neighbours - each line is a state, ranked by state-number, each column is a direction (N, E, S, W)
    self._neighbours = np.zeros((self._state_size, 4)) 
    
    for state in range(self._state_size):
      loc = self._get_loc_from_state(state)

      # North
      neighbour = (loc[0]-1, loc[1]) # North neighbours location
      if self._is_location(neighbour):
        self._neighbours[state][self._direction_names.index('N')] = self._get_state_from_loc(neighbour)
      else: # If there is no neighbour in this direction, coming back to current state
        self._neighbours[state][self._direction_names.index('N')] = state

      # East
      neighbour = (loc[0], loc[1]+1) # East neighbours location
      if self._is_location(neighbour):
        self._neighbours[state][self._direction_names.index('E')] = self._get_state_from_loc(neighbour)
      else: # If there is no neighbour in this direction, coming back to current state
        self._neighbours[state][self._direction_names.index('E')] = state

      # South
      neighbour = (loc[0]+1, loc[1]) # South neighbours location
      if self._is_location(neighbour):
        self._neighbours[state][self._direction_names.index('S')] = self._get_state_from_loc(neighbour)
      else: # If there is no neighbour in this direction, coming back to current state
        self._neighbours[state][self._direction_names.index('S')] = state

      # West
      neighbour = (loc[0], loc[1]-1) # West neighbours location
      if self._is_location(neighbour):
        self._neighbours[state][self._direction_names.index('W')] = self._get_state_from_loc(neighbour)
      else: # If there is no neighbour in this direction, coming back to current state
        self._neighbours[state][self._direction_names.index('W')] = state

    # Absorbing
    self._absorbing = np.zeros((1, self._state_size))
    for a in self._absorbing_locs:
      absorbing_state = self._get_state_from_loc(a)
      self._absorbing[0, absorbing_state] = 1

    # Transition matrix
    self._T = np.zeros((self._state_size, self._state_size, self._action_size)) # Empty matrix of domension S*S*A
    for action in range(self._action_size):
      for outcome in range(4): # For each direction (N, E, S, W)
        # The agent has prob_success probability to go in the correct direction
        if action == outcome:
          prob = 1 - 3.0 * ((1.0 - self._prob_success) / 3.0) # (theoritically equal to self.prob_success but avoid rounding error and garanty a sum of 1)
        # Equal probability to go into one of the other directions
        else:
          prob = (1.0 - self._prob_success) / 3.0
          
        # Write this probability in the transition matrix
        for prior_state in range(self._state_size):
          # If absorbing state, probability of 0 to go to any other states
          if not self._absorbing[0, prior_state]:
            post_state = self._neighbours[prior_state, outcome] # Post state number
            post_state = int(post_state) # Transform in integer to avoid error
            self._T[prior_state, post_state, action] += prob

    # Reward matrix
    self._R = np.ones((self._state_size, self._state_size, self._action_size)) # Matrix filled with 1
    self._R = self._default_reward * self._R # Set default_reward everywhere
    for i in range(len(self._absorbing_rewards)): # Set absorbing states rewards
      post_state = self._get_state_from_loc(self._absorbing_locs[i])
      self._R[:,post_state,:] = self._absorbing_rewards[i]

    # Creating the graphical Maze world
    self._graphics = GraphicsMaze(self._shape, self._locations, self._default_reward, self._obstacle_locs, self._absorbing_locs, self._absorbing_rewards, self._absorbing)
    
    # Reset the environment
    self.reset()


  def _is_location(self, loc):
    """
    Is the location a valid state (not out of Maze and not an obstacle)
    input: loc {tuple} -- location of the state
    output: _ {bool} -- is the location a valid state
    """
    if (loc[0] < 0 or loc[1] < 0 or loc[0] > self._shape[0]-1 or loc[1] > self._shape[1]-1):
      return False
    elif (loc in self._obstacle_locs):
      return False
    else:
      return True


  def _get_state_from_loc(self, loc):
    """
    Get the state number corresponding to a given location
    input: loc {tuple} -- location of the state
    output: index {int} -- corresponding state number
    """
    return self._locations.index(tuple(loc))


  def _get_loc_from_state(self, state):
    """
    Get the state number corresponding to a given location
    input: index {int} -- state number
    output: loc {tuple} -- corresponding location
    """
    return self._locations[state]

  # Getter functions used only for DP agents
  # You DO NOT NEED to modify them
  def get_T(self):
    return self._T

  def get_R(self):
    return self._R

  def get_absorbing(self):
    return self._absorbing

  # Getter functions used for DP, MC and TD agents
  # You DO NOT NEED to modify them
  def get_graphics(self):
    return self._graphics

  def get_action_size(self):
    return self._action_size

  def get_state_size(self):
    return self._state_size

  def get_gamma(self):
    return self._gamma

  # Functions used to perform episodes in the Maze environment
  def reset(self):
    """
    Reset the environment state to one of the possible starting states
    input: /
    output: 
      - t {int} -- current timestep
      - state {int} -- current state of the envionment
      - reward {int} -- current reward
      - done {bool} -- True if reach a terminal state / 0 otherwise
    """
    self._t = 0
    self._state = self._get_state_from_loc(self._starting_locs[random.randrange(len(self._starting_locs))])
    self._reward = 0
    self._done = False
    return self._t, self._state, self._reward, self._done

  def step(self, action):
    """
    Perform an action in the environment
    input: action {int} -- action to perform
    output: 
      - t {int} -- current timestep
      - state {int} -- current state of the envionment
      - reward {int} -- current reward
      - done {bool} -- True if reach a terminal state / 0 otherwise
    """

    # If environment already finished, print an error
    if self._done or self._absorbing[0, self._state]:
      print("Please reset the environment")
      return self._t, self._state, self._reward, self._done

    # Drawing a random number used for probaility of next state
    probability_success = random.uniform(0,1)

    # Look for the first possible next states (so get a reachable state even if probability_success = 0)
    new_state = 0
    while self._T[self._state, new_state, action] == 0: 
      new_state += 1
    assert self._T[self._state, new_state, action] != 0, "Selected initial state should be probability 0, something might be wrong in the environment."

    # Find the first state for which probability of occurence matches the random value
    total_probability = self._T[self._state, new_state, action]
    while (total_probability < probability_success) and (new_state < self._state_size-1):
     new_state += 1
     total_probability += self._T[self._state, new_state, action]
    assert self._T[self._state, new_state, action] != 0, "Selected state should be probability 0, something might be wrong in the environment."
    
    # Setting new t, state, reward and done
    self._t += 1
    self._reward = self._R[self._state, new_state, action]
    self._done = self._absorbing[0, new_state] or self._t > self._max_t
    self._state = new_state
    return self._t, self._state, self._reward, self._done


# ## DP Agent


# This class define the Dynamic Programing agent 

class DP_agent(object):

  # [Action required]
  # WARNING: make sure this function can be called by the auto-marking script
  def solve(self, env):
    """
    Solve a given Maze environment using Dynamic Programming
    input: env {Maze object} -- Maze to solve
    output: 
      - policy {np.array} -- Optimal policy found to solve the given Maze environment 
      - V {np.array} -- Corresponding value function 
    """
    
    # Initialisation (can be edited)
    policy = np.zeros((env.get_state_size(), env.get_action_size())) 
    V = np.zeros(env.get_state_size())
    gamma = env.get_gamma() #Discounted factor
    threshold = 0.0001  #threshold for policy evaluation

    policy[:, 0] = 1 # Initialise policy to choose action 1 systematically

    #### 
    # Add your code here
    # WARNING: for this agent only, you are allowed to access env.get_T(), env.get_R() and env.get_absorbing()
    ####

    policy, V, epochs = self.value_iteration(env, gamma, V, policy, threshold)
  
    return policy, V
  

  def policy_evaluation(self, env, policy, gamma, threshold = 0.0001):
    """
    Policy evaluation on Maze
    input: 
      - env {Maze object} -- Maze to solve
      - policy {np.array} -- policy to evaluate
      - threshold {float} -- threshold value used to stop the policy evaluation algorithm
      - gamma {float} -- discount factor
    output: 
      - V {np.array} -- value function corresponding to the policy 
      - epochs {int} -- number of epochs to find this value function
    """
    
    # Ensure inputs are valid
    assert (policy.shape[0] == env.get_state_size()) and (policy.shape[1] == env.get_action_size()), "The dimensions of the policy are not valid."
    assert (gamma <=1) and (gamma >= 0), "Discount factor should be in [0, 1]."

    # Initialisation
    delta = 2*threshold # Ensure delta is bigger than the threshold to start the loop
    T = env.get_T()
    R = env.get_R()
    state_size = env.get_state_size()
    action_size = env.get_action_size()
    V = np.zeros(state_size) # Initialise value function to 0  
    epoch = 0
    abs_states = env.get_absorbing()  #Get the absorbing states from Maze


    while delta > threshold:
      delta = 0 # Reset delta at the start of each iteration
      V_save = V.copy() # copy of the current value function for the formula 

      for s in range(state_size):
        #Init phase for a given state s
        v = V[s] # Store the current value of state s
        new_v = 0 # update value for a given state s

        #Check if the current state is absorbing
        if abs_states[0, s]:
          V[s] = 0
          continue

        for a in range(action_size):
          for s_prime in range(state_size):
            new_v += policy[s, a] * T[s, s_prime, a] * (R[s, s_prime, a] + gamma * V_save[s_prime])

        V[s] = new_v
        delta = max(delta, abs(v - new_v))

      epoch += 1

    return V, epoch
  

  def policy_iteration(self, env, gamma, V, policy, threshold = 0.0001):
    """
    Policy iteration on Maze
    input: 
      - env {Maze object} -- Maze to solve
      - threshold {float} -- threshold value used to stop the policy iteration algorithm
      - gamma {float} -- discount factor
    output:
      - policy {np.array} -- policy found using the policy iteration algorithm
      - V {np.array} -- value function corresponding to the policy 
      - epochs {int} -- number of epochs to find this policy
    """

    # Ensure gamma value is valid
    assert (gamma <=1) and (gamma >= 0), "Discount factor should be in [0, 1]."

    T = env.get_T()
    R = env.get_R()
    state_size = env.get_state_size()
    action_size = env.get_action_size()

    # Initialisation
    epochs = 0
    policy_stable = False # Condition to stop the main loop


    while policy_stable == False:
      V, epochs_eval = self.policy_evaluation(env, policy, gamma, threshold)
      epochs = epochs + epochs_eval

      policy_save = policy.copy() #copy of the current policy

      for s in range(state_size):

        action_to_evaluate = []
        
        for a in range(action_size):
          val_action = 0
          for s_prime in range(state_size):
              val_action += T[s, s_prime, a] * (R[s, s_prime, a] + gamma * V[s_prime])

          action_to_evaluate.append(val_action)

        best_action = np.argmax(np.array(action_to_evaluate))
        policy[s,:] = 0
        policy[s,best_action] = 1

      if np.array_equal(policy, policy_save) == False:
        policy_stable = False

      else:
        policy_stable = True

    return policy, V, epochs
  

  def value_iteration(self, env, gamma, V, policy, threshold = 0.0001):
    """
    Value iteration on GridWorld
    input: 
      - env {Maze object} -- Maze to solve
      - threshold {float} -- threshold value used to stop the value iteration algorithm
      - gamma {float} -- discount factor
    output: 
      - policy {np.array} -- optimal policy found using the value iteration algorithm
      - V {np.array} -- value function corresponding to the policy
      - epochs {int} -- number of epochs to find this policy
    """

    # Ensure gamma value is valid
    assert (gamma <=1) and (gamma >= 0), "Discount factor should be in [0, 1]."

    T = env.get_T()
    R = env.get_R()
    state_size = env.get_state_size()
    action_size = env.get_action_size()

    # Initialisation
    epochs = 0
    delta = threshold # Setting value of delta to go through the first breaking condition
    abs_states = env.get_absorbing()  #Get the absorbing states from Maze

    
    while delta >= threshold:
      delta = 0 # Reset delta at the start of each iteration
      V_save = V.copy() # copy of the current value function for the formula
      epochs = epochs+1

      for s in range(state_size):
        #Init phase for a given state s
        v = V[s] # Store the current value of state s
        action_to_evaluate = []

        #Check for absorbing state
        if abs_states[0,s]:
          V[s] = 0
          continue

        for a in range(action_size):
          new_v = 0 # update value for a given state s

          for s_prime in range(state_size):
            new_v += T[s, s_prime, a] * (R[s, s_prime, a] + gamma * V_save[s_prime])

          action_to_evaluate.append(new_v)

        V[s] = max(action_to_evaluate)
        delta = max(delta, abs(v - V[s]))

      #Output the deterministic policy:
      for s in range(state_size):

        action_to_evaluate = []
          
        for a in range(action_size):
          val_action = 0
          for s_prime in range(state_size):
            val_action += T[s, s_prime, a] * (R[s, s_prime, a] + gamma * V[s_prime])

          action_to_evaluate.append(val_action)

        best_action = np.argmax(np.array(action_to_evaluate))
        policy[s,:] = 0   #reset the action for a given state
        policy[s,best_action] = 1   #indicate the best action

    return policy, V, epochs


# ## MC agent


# This class define the Monte-Carlo agent

class MC_agent(object):
  
  # [Action required]
  # WARNING: make sure this function can be called by the auto-marking script
  def solve(self, env):
    """
    Solve a given Maze environment using Monte Carlo learning
    input: env {Maze object} -- Maze to solve
    output: 
      - policy {np.array} -- Optimal policy found to solve the given Maze environment 
      - values {list of np.array} -- List of successive value functions for each episode 
      - total_rewards {list of float} -- Corresponding list of successive total non-discounted sum of reward for each episode 
    """

    # Initialisation (can be edited)
    Q = np.random.rand(env.get_state_size(), env.get_action_size()) 
    V = np.zeros(env.get_state_size())
    policy = np.zeros((env.get_state_size(), env.get_action_size()))
    values = [V]  #values at each episode
    total_rewards = []

    num_episodes = 3000
    epsilon = 1

    #### 
    # Add your code here
    # WARNING: this agent only has access to env.reset() and env.step()
    # You should not use env.get_T(), env.get_R() or env.get_absorbing() to compute any value
    ####

    
    Returns = {(s, a): [] for s in range(env.get_state_size()) for a in range(env.get_action_size())}
    for i in range(num_episodes):
      epsilon = np.exp(-((i)/2000)) #epsilon decay formula at episode i

      episode = self.generate_episode(env, policy, epsilon)  #create trace (state, action, reward, timestep)

      G = 0 #discounted reward
      states_actions_visited = set()  #resume of visited pairs (state,action)

      episode_reward = 0  #Total of non-discounted reward for an episode

      for step in reversed(range(len(episode))):
        current_state = episode[step][0]
        current_action = episode[step][1]
        current_reward = episode[step][2]

        episode_reward = episode_reward + current_reward
        G = env.get_gamma()*G + current_reward

        if (current_state, current_action) not in states_actions_visited:
            
          Returns[(current_state, current_action)].append(G)

          states_actions_visited.add((current_state, current_action))

          Q[current_state, current_action] = np.mean(Returns[((current_state, current_action))])

          best_action = np.argmax(Q[current_state])

          for a in range(env.get_action_size()):
            if a == best_action:
              policy[current_state,a] = 1 - epsilon + epsilon/env.get_action_size()
            else:
              policy[current_state,a] = epsilon/env.get_action_size()

      total_rewards.append(episode_reward)  #add non discounted reward

      # Update value function and total rewards for the episode
      values.append(np.max(Q, axis=1))
    
    return policy, values, total_rewards
  
  # Function for selecting an action following an epsilon greedy policy.
  def select_action_soft(self, env, policy, epsilon, state):
    if random.uniform(0, 1) < epsilon:  # Case of random action
      action_true = np.random.choice(env.get_action_size())
      return action_true
    else:
      action_true = np.argmax(policy[state])  #best action
      return action_true


  #Episode generation => create a trace
  def generate_episode(self, env, policy, epsilon):
    episode_trajectory = []
        
    timestep, state, reward, done = env.reset()  #random starting point, with done=False
    nb_step=0
    while done == False and nb_step <= 500: #limit the number of step at 500
      action = self.select_action_soft(env, policy, epsilon, state) #give the index of the action

      timestep, next_state, next_reward, done = env.step(action)

      episode_trajectory.append((state, action, next_reward))

      state = next_state

      nb_step = nb_step+1   #increase number of step to stay below 500
            
    return episode_trajectory
    

  #Policy evaluation version:
  def policy_evaluation(self, env, episode_nb): #add policy as input
    V = np.zeros(env.get_state_size())
    
    policy = np.zeros((env.get_state_size(), env.get_action_size()))
    policy[:,0] = 1 #initialize policy function for taking only action 0

    returns = [[] for s in range(env.get_state_size())]  #List containing the return for each state

    for i in range(episode_nb):
      episode = self.generate_episode(env, policy)  #create trace (state, action, reward, timestep)

      G = 0 #discounted reward

      states_visited = set()  #resume of visited states

      for step in range(len(episode)-1, -1, -1):  #inverse path from T-1 to 0
        current_state = episode[step][0]
        current_reward = episode[step][2]

        G = env.get_gamma() * G + current_reward  #extract the reward of time step t+1

        if current_state not in states_visited :  #just verify if the state has been visited and not pair (St, At) caused we are in first visit
          returns[current_state].append(G)
          states_visited.add(current_state)

          V[current_state] = np.mean(returns[current_state])

      

    return V, policy  #delete policy
  


  def solve_batch(self, env, Q, V, policy, nb_episode_batch, nb_batch, epsilon):
    tt_reward = []  #reward per batch
    val = [V]  #values at each batch

    for batch in range(nb_batch):
      Returns = {(s, a): [] for s in range(env.get_state_size()) for a in range(env.get_action_size())}
      for i in range(nb_episode_batch):
        epsilon = np.exp(-((batch)/2000))
        episode = self.generate_episode(env, policy, epsilon)  #create trace (state, action, reward, timestep)

        G = 0 #discounted reward
        states_actions_visited = set()  #resume of visited pairs (state,action)

        episode_reward = 0  #Total of non-discounted reward for an episode

        for step in reversed(range(len(episode))):
          current_state = episode[step][0]
          current_action = episode[step][1]
          current_reward = episode[step][2]

          episode_reward = episode_reward + current_reward
          G = env.get_gamma()*G + current_reward

          if (current_state, current_action) not in states_actions_visited:
              
            Returns[(current_state, current_action)].append(G)

            states_actions_visited.add((current_state, current_action))

        tt_reward.append(episode_reward)

      for s in range(env.get_state_size()):
        for a in range(env.get_action_size()):
          if Returns[(s, a)]:
            Q[s, a] = np.mean(Returns[(s, a)])

        best_action = np.argmax(Q[s])

        for a in range(env.get_action_size()):
          if a == best_action:
            policy[s,a] = 1 - epsilon + epsilon/env.get_action_size()
          else:
            policy[s,a] = epsilon/env.get_action_size()


      # Update value function and total rewards for the episode
      val.append(np.max(Q, axis=1))

    return policy, val, tt_reward


# ## TD agent


# This class define the Temporal-Difference agent

class TD_agent(object):

  # [Action required]
  # WARNING: make sure this function can be called by the auto-marking script
  def solve(self, env):
    """
    Solve a given Maze environment using Temporal Difference learning
    input: env {Maze object} -- Maze to solve
    output: 
      - policy {np.array} -- Optimal policy found to solve the given Maze environment 
      - values {list of np.array} -- List of successive value functions for each episode 
      - total_rewards {list of float} -- Corresponding list of successive total non-discounted sum of reward for each episode 
    """

    # Initialisation (can be edited)
    #Q = np.random.rand(env.get_state_size(), env.get_action_size())
    Q = np.zeros((env.get_state_size(), env.get_action_size()))
    V = np.zeros(env.get_state_size())
    policy = np.zeros((env.get_state_size(), env.get_action_size())) 
    values = [V]
    total_rewards = []

    epsilon = 1
    alpha = 0.25
    gamma = env.get_gamma()
    nb_episodes = 2000

    #### 
    # Add your code here
    # WARNING: this agent only has access to env.reset() and env.step()
    # You should not use env.get_T(), env.get_R() or env.get_absorbing() to compute any value
    ####

    for i in range(nb_episodes):
      
      epsilon = np.exp(-((i)/1000))
      Q, non_discounted_reward_ep = self.generate_episode_updating_Qlearning(env, Q, epsilon, alpha, gamma)
      
      #Policy Update with respect to Q(s,a)
      for s in range(env.get_state_size()):
        best_action = np.argmax(Q[s,:])
        policy[s,:] = 0
        policy[s, best_action] = 1

      # Update of V based on Q function for each state
      V = np.max(Q, axis=1)
      values.append(V)
      total_rewards.append(non_discounted_reward_ep)

    return policy, values, total_rewards
  


  #Select an action following epsilon greedy approach (here behaviour policy = epsilon greedy of target policy)
  def select_action_soft(self, env, epsilon, Q_func, state):
    if random.uniform(0, 1) < epsilon:  #random exploration
      action_true = np.random.choice(env.get_action_size())
      return action_true
    else: #best action derived from Q
      action_true = np.argmax(Q_func[state,:])
      return action_true
    

  #Generating an episode for Q-learning approach
  def generate_episode_updating_Qlearning(self, env, Q_func, epsilon, alpha, gamma):
    timestep, state, reward, done = env.reset()  #random starting point, with done=False
    non_discounted_reward_ep = 0
    nb_step=0

    while done == False and nb_step <= 500:
      action = self.select_action_soft(env, epsilon, Q_func, state) #choose action using policy derived from Q

      #Take action_true and observe the associated reward and next state
      timestep, next_state, next_reward, done = env.step(action)

      #Update of the policy (greedy) with respect to Q(s,a)
      next_best_action = np.argmax(Q_func[next_state, :])
      #Update the Q regarding the alternative better solution
      Q_func[state, action] = Q_func[state, action] + alpha * (next_reward + gamma*Q_func[next_state, next_best_action] - Q_func[state, action])

      non_discounted_reward_ep += next_reward
      state = next_state

      nb_step = nb_step+1   #increase number of step to stay below 500
            
    return Q_func, non_discounted_reward_ep
  


  #Generating an episode for SARSA approach
  def generate_episode_updating_SARSA(self, env, Q_func, epsilon, alpha, gamma):
    timestep, state, reward, done = env.reset()  #random starting point, with done=False
    non_discounted_reward_ep = 0
    nb_step=0

    action = self.select_action_soft(env, epsilon, Q_func, state)

    while done == False and nb_step <= 500:
      #Take action_true and observe the associated reward and next state
      timestep, next_state, next_reward, done = env.step(action)

      next_action = self.select_action_soft(env, epsilon, Q_func, next_state) #choose action using policy derived from Q

      #Update the Q: update of the policy prime (eps-greedy) with respect to Q(s,a)
      Q_func[state, action] = Q_func[state, action] + alpha * (next_reward + gamma*Q_func[next_state, next_action] - Q_func[state, action])

      non_discounted_reward_ep += next_reward
      state = next_state
      action = next_action

      nb_step = nb_step+1   #increase number of step to stay below 500
            
    return Q_func, non_discounted_reward_ep

