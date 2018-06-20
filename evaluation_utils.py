from pommerman.agents import SimpleAgent, RandomAgent, PlayerAgent, BaseAgent
from pommerman.configs import ffa_v0_fast_env
from pommerman.configs import ffa_v0_fast_env as ffa_v0_env
from pommerman.envs.v0 import Pomme
from pommerman.characters import Bomber
from pommerman import utility
from pommerman.constants import *

from matplotlib import pyplot as plt
import numpy as np
import time

from keras.models import load_model
import tensorflow as tf
from tqdm import tqdm_notebook as tqdm

view_size = BOARD_SIZE * 2 - 1  # 21
history_length = 4
n_channels = 18
ACTIONS = ['stop', 'up', 'down', 'left', 'right', 'bomb']

def make_centered(board, position, view_size=BOARD_SIZE * 2 - 1, crop=False):
    # If it is a perk channel, just return the resized full array
    if np.all(board == 1):
        if crop:
            view_size = BOARD_SIZE
        return np.full((view_size, view_size, 1), 1)
    
    # make sure of odd view_size
    view_size = view_size + 1 if view_size % 2 == 0 else view_size 
    
    # TODO: what should be the value of the off world? maybe defining an edge channel?
    agent_view = np.zeros((view_size, view_size, 1)) # agent centric full-world coverage
    center = (view_size // 2 + 1, view_size // 2 + 1)
    
    # copy board to the new view
    offset_y = center[0] - position[0] - 1
    offset_x = center[1] - position[1] - 1
    agent_view[offset_y:offset_y+BOARD_SIZE, offset_x:offset_x+BOARD_SIZE, :] = board
    
    # finalize view size
    r = BOARD_SIZE // 2
    start, end = center[0]-r-1, center[0]+r
    if crop:
        agent_view = agent_view[start:end, start:end, :]
    
    return np.array(agent_view, dtype=np.float32)

def featurize(obs, center=True, crop=False):
    shape = (BOARD_SIZE, BOARD_SIZE, 1)

    def get_matrix(board, key):
        res = board[key]
        return res.reshape(shape).astype(np.float32)

    def get_map(board, item):
        map = np.zeros(shape)
        map[board == item] = 1
        return map

    board = get_matrix(obs, 'board')

    path_map       = get_map(board, 0)          # Empty space
    rigid_map      = get_map(board, 1)          # Rigid = 1
    wood_map       = get_map(board, 2)          # Wood = 2
    bomb_map       = get_map(board, 3)          # Bomb = 3
    flames_map     = get_map(board, 4)          # Flames = 4
    fog_map        = get_map(board, 5)          # TODO: not used for first two stages Fog = 5
    extra_bomb_map = get_map(board, 6)          # ExtraBomb = 6
    incr_range_map = get_map(board, 7)          # IncrRange = 7
    kick_map       = get_map(board, 8)          # Kick = 8
    skull_map      = get_map(board, 9)          # Skull = 9

    position = obs["position"]
    my_position = np.zeros(shape)
    my_position[position[0], position[1], 0] = 1

    team_mates = get_map(board, obs["teammate"].value) # TODO during documentation it should be an array

    enemies = np.zeros(shape)
    for enemy in obs["enemies"]:
        enemies[board == enemy.value] = 1

    bomb_blast_strength = get_matrix(obs, 'bomb_blast_strength')
    bomb_life           = get_matrix(obs, 'bomb_life')

    ammo           = np.full((BOARD_SIZE, BOARD_SIZE, 1), obs["ammo"])
    blast_strength = np.full((BOARD_SIZE, BOARD_SIZE, 1), obs["blast_strength"])
    can_kick       = np.full((BOARD_SIZE, BOARD_SIZE, 1), int(obs["can_kick"]))
    
    maps = [my_position, enemies, team_mates, path_map, rigid_map, 
                          wood_map, bomb_map, flames_map, fog_map, extra_bomb_map,
                          incr_range_map, kick_map, skull_map, bomb_blast_strength,
                          bomb_life, ammo, blast_strength, can_kick]
    
    if center:
        maps = [make_centered(m, position, crop=crop) for m in maps]
    
    obs = np.concatenate(maps, axis=2)
    return obs.astype(np.uint8)

def run_episode(agent, config, env, agent_id=0):
    # K.clear_session()
    # Add 3 random agents and one trained
    agents = [agent if i == agent_id else SimpleAgent(config["agent"](i, config["game_type"])) for i in range(4)]
    env.set_agents(agents)
    env.set_init_game_state(None)

    # Seed and reset the environment
    env.seed(0)
    obs = env.reset()

    # Run the agents until we're done
    done = False
    lens = [None] * 4
    t = 0
    while not done:
    #     env.render()
        actions = env.act(obs)
        obs, reward, done, info = env.step(actions)
        for j in range(4):
            if lens[j] is None and reward[j] != 0:
                lens[j] = t
        t += 1
            
    env.render(close=True)
    env.close()
    return info, reward, lens

def plot_statistics(agent, info, selected_labels=None, agent_id=0, iterations=100):
    actions_list = ACTIONS
    plot_labels = np.any(selected_labels)
    fig, ax = plt.subplots(ncols=3 if plot_labels else 2, figsize=(18, 5))
    
    wins = [h['winners'][0] for h in info if 'winners' in h]
    agent_ids, agent_win_counts = np.unique(wins, return_counts=True)
    agent_win_counts = np.array([agent_win_counts[agent_ids == i][0] if i in agent_ids else 0 for i in np.arange(4)])
    agent_ids = np.arange(4)
    win_proportions = [np.around(x, 2) for x in np.append(agent_win_counts, len(info) - len(wins)) / len(info) * 100]
    agent_labels = ['\n'.join(pair) for pair in zip(np.append(np.arange(4), 'Tie'), ['({}%)'.format(p) for p in win_proportions])]
    
    ax[0].bar(np.append(agent_ids, 4), np.append(agent_win_counts, len(info) - len(wins)),
              color=['red' if i == agent_id else 'blue' for i in range(5)])
    ax[0].set_xticks(np.arange(5))
    ax[0].set_xticklabels(agent_labels)
    ax[0].set_title('Win counts ({} games, avg.reward: {})'.format(len(info), np.round(2*agent_win_counts/len(info) - 1, 2)))
    ax[0].set_xlabel('Agent ID')
    

    # Agent movements over all episodes
    all_movements = np.concatenate(agent.actions_history)
    movements, movements_count = np.unique(all_movements, return_counts=True)
    performed_actions = np.array(actions_list)[movements.astype(np.uint8)]
    movement_proportions = np.round(movements_count / len(all_movements) * 100, 2)
    movement_labels = ['\n'.join(pair) for pair in zip(performed_actions, ['({}%)'.format(p) for p in movement_proportions])]
    
    ax[1].bar(movements, movements_count, color='blue')
    ax[1].set_title('Agent movements ({} steps)'.format(len(all_movements)))
    ax[1].set_xticks(movements)
    ax[1].set_xticklabels(movement_labels)
    ax[1].set_xlabel('Movement')

    if plot_labels:
        tr_movements, tr_movements_count = np.unique(np.argmax(selected_labels, axis=1), return_counts=True)
        tr_movement_proportions = np.round(tr_movements_count / len(selected_labels) * 100, 2)
        tr_movement_labels = ['\n'.join(pair) for pair in zip(actions_list, ['({}%)'.format(p) for p in tr_movement_proportions])]

        ax[2].bar(tr_movements, tr_movements_count, color='blue')
        ax[2].set_title('Train set movements')
        ax[2].set_xticks(np.arange(6))
        ax[2].set_xticklabels(tr_movement_labels)
        ax[2].set_xlabel('Movement')
    
    plt.suptitle("Test runs for agent {}, {} iterations".format(agent_id, iterations))
    plt.show()
    
    max_steps = np.max(list(map(len, agent.actions_history)))
    history = np.zeros((6, max_steps))
    for episode in agent.actions_history:
        for i in range(max_steps):
            if i < len(episode):
                history[episode[i], i] += 1
    
    fig = plt.figure(figsize=(18, 10))
    bars = [plt.bar(np.arange(max_steps), 
                    history[action, :], 
                    bottom = np.repeat(0, max_steps) if action == 0 else np.sum(history[:action, :], axis=0),
                    edgecolor='white',
                    linewidth=.1,
                    alpha=1) 
            for action in range(history.shape[0])]

    plt.xlim(0, max_steps)
    plt.legend(bars, actions_list)
    plt.show()
    
	
def evaluate_agent(agent, config, selected_labels=None, agent_id=0, iterations=100, plot=True):
    # Instantiate the environment
    env = Pomme(**config["env_kwargs"])
    info = []
    rewards = np.zeros((iterations, 4))
    lengths = np.zeros((iterations, 4))
    
    if isinstance(agent, EvaluatorAgent):
        agent.reset_run()
    start_time = time.time()
    for i in tqdm(range(iterations)):
        # print('{}/{}'.format(i+1, iterations), end='\r')
        info_ep, reward, lens = run_episode(agent, config, env, agent_id)
        info.append(info_ep)
        rewards[i] = reward
        lengths[i] = lens
        if isinstance(agent, EvaluatorAgent):
            agent.end_episode()
    
    if plot:
        plot_statistics(agent, info, selected_labels, agent_id, iterations)
    elapsed = time.time() - start_time
    return info, rewards, lengths, elapsed
    
    
class Logger(object):
    """Logging in tensorboard without tensorflow ops."""

    def __init__(self, log_dir):
        """Creates a summary writer logging to log_dir."""
        self.writer = tf.summary.FileWriter(log_dir)

    def log_scalar(self, tag, value, step):
        """Log a scalar variable.
        Parameter
        ----------
        tag : basestring
            Name of the scalar
        value
        step : int
            training iteration
        """
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag,
                                                     simple_value=value)])
        self.writer.add_summary(summary, step)

# Base evaluator 
class EvaluatorAgent(BaseAgent):
    def __init__(self, n_actions, character, 
                 evaluation_model=None, evaluation_model_path=None, 
                 # Set agent properties to preprocess observations
                 use_history=True,    # Use previous observations for predictions
                 use_2d=True,         # Use 2d convolutions
                 patient=True,        # Wait to make initial observations (you don't need it if you don't use history)
                 center_view=True,    # Use centering
                 original_view=False, # Use 11x11 board, if false, use 21x21
                 verbose=False        # Comment actions
                ):
        super(EvaluatorAgent, self).__init__(character=character)
        
        # Properties
        self.use_history = use_history
        self.use_2d = use_2d
        self.patient = patient
        self.center_view = center_view
        self.original_view = original_view
        self.verbose = verbose
        
        # Acting history for the evaluation
        self.actions_history = []
        self.observations_history = []
        self.episode_count = 0
        self.steps = 0
        
        self.n_actions = n_actions
        
        self.simple_agent = SimpleAgent(character=character)
        # Load any custom model
        self.evaluation_model = None
        if evaluation_model:
            self.evaluation_model = evaluation_model
            if evaluation_model_path:
                try:
                    self.evaluation_model.load_weights(evaluation_model_path)
                except:
                    print('Weights load failed')
        elif evaluation_model_path:
            try:
                self.evaluation_model = load_model(evaluation_model_path)
            except:
                print('Model load failed')
        else:
            print('Use SimpleAgent')
        
    
    # Featurization
    def featurize(self, obs):
        return featurize(obs, center=self.center_view, crop=self.original_view)
    
    # Acting
    def act(self, obs, action_space=None):
        # Initialize new episode
        if self.steps == 0:
            self.actions_history.append([])
        
        # Create observation, merge with the predecessors
        obs_f = self.featurize(obs)
        
        # If our agent is patient, wait for the first 3 steps to make observations
        if self.patient and len(self.observations_history) < history_length - 1:
            self.observations_history.append(obs_f)
            self.actions_history[self.episode_count].append(0)
            return 0

        if self.use_history:
            obs_history = self.make_observation(obs_f, self.steps, self.use_2d)
        else:
            obs_history = obs_f
            
        self.observations_history.append(obs_f) # Append current observation after the merge
        
        # Predict action
        if self.evaluation_model is not None:
            res = self.evaluation_model.predict(obs_history.reshape((1,) + obs_history.shape))[0]
            res = np.argmax(res)
        else:
            res = self.simple_agent.act(obs, action_space)
        if self.verbose:
            print(res, end='; ')
        
#        # In the dueling DQN the first output relates to the advantage
#        if len(res) > self.n_actions:
#            res = res[1:]
        
        self.actions_history[self.episode_count].append(res)
         
        if self.verbose:
            print(ACTIONS[res])
            
        self.steps += 1
        return res
    
    def make_observation(self, obs, i, use_2d=True):
        if i == 0: # If it is a first observation
            res = np.array([obs for _ in range(history_length)])
        elif i < history_length - 1: # If there are less than 3 observations in a history
            n_first = history_length - 1 - i
            res = np.concatenate([np.array([self.observations_history[0] for _ in range(n_first)]), # Repeat the first observation
                                  np.array(self.observations_history[:i]).reshape(i, view_size, view_size, n_channels), # Add next observations
                                  obs.reshape(1, view_size, view_size, n_channels)], # Current observation
                                  axis=0)
        else:
            res = np.concatenate([np.array(self.observations_history[i-history_length+1:i]).reshape(history_length-1, view_size, view_size, n_channels), # Add next observations
                                  obs.reshape(1, view_size, view_size, n_channels)], # Current observation
                                  axis=0)
        if use_2d:
            res = np.concatenate(res, axis=-1)
        return res
    
    # Evaluation
    def end_episode(self):
        self.steps = 0
        self.episode_count += 1
        self.observations_history = []
        
    def reset_run(self):
        self.actions_history = []
        self.episode_count = 0
        self.steps = 0
    
    def close(self):
        pass
    
    def run_episode(self, config, env):
        return run_episode(self, config, env, self.agent_id)

    def plot_statistics(self, info, selected_labels):
        return plot_statistics(self, info, selected_labels)

    def evaluate_agent(self, selected_labels, iterations=100, plot=True):
        return evaluate_agent(self, selected_labels, self.agent_id, iterations, plot)
