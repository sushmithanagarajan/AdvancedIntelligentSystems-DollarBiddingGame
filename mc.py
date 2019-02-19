from __future__ import division

import numpy as np


class MonteCarloLearning(object):
    """Monte Carlo Q-learning.

    Optimal policy is initialized randomly.

    Args:
        num_states (int): number of states in the game being played
        num_actions (int): number of actions in the game being played

    Attributes:
        Q (array): action-value function: expected reward from taking action while in state
        optimal_policy (array): dictates which action is best to take for each state
        state_action_reward_sum (array): sum of rewards for each state-action pair
        state_action_count (array): number of times each state-action pair has been seen
        states_seen (list): all states seen by the agent during the current game

    """

    def __init__(self, num_states, num_actions):
        """Initialize Monte Carlo Q-learning."""
        assert num_states > 0, 'Number of game states must be greater than zero.'
        assert num_actions > 0, 'Number of possible actions must be greater than zero.'
        self.num_states = num_states
        self.num_actions = num_actions

        self.Q = np.zeros((self.num_states, self.num_actions))
        self.optimal_policy = np.random.randint(self.num_actions, size=self.num_states)
        self.state_action_reward_sum = np.zeros((self.num_states, self.num_actions))
        self.state_action_count = np.zeros((self.num_states, self.num_actions))
        self.states_seen = []

    def update(self, state_index, action_index, reward):
        """Update statistics for action value function Q.

        Args:
            state_index (int): array index of state
            action_index (int): array index of action
            reward (int): reward -1, 0, 1 corresponds to losing, drawing, winning the game

        """
        assert state_index < self.num_states, 'Invalid state (does not exist).'
        assert action_index < self.num_actions, 'Invalid action (does not exist).'
        self.state_action_count[state_index, action_index] += 1
        self.state_action_reward_sum[state_index, action_index] += reward
        self.Q[state_index, action_index] = (self.state_action_reward_sum[state_index, action_index]
                                             / self.state_action_count[state_index, action_index])

        self.optimal_policy[state_index] = np.argmax(self.Q[state_index])
        return self.optimal_policy

    def record_state_seen(self, game_state):
        """Add game_state to list of states seen by player.

        Args:
            game_state (list): [card_showing, smallest, median, largest]

        """
        self.states_seen.append(np.array(game_state))

    def clear_states_seen(self):
        """Clear list of states seen."""
        self.states_seen = []

    def save_learning(self, episode):
        """Save current information about Monte Carlo learning to .txt files.

        Args:
            episode (int): number of training episodes elapsed

        Outputs:
            Q: action values
            optimal_policy: best optimal_policy
            state_action_count: number of times each state-action pair has been encountered
            state_action_reward_sum: sum of rewards for each state-action pair

        """
        np.savetxt('Q-%i.txt' % episode, self.Q, fmt='%.8f')
        np.savetxt('optimal_policy-%i.txt' % episode, self.optimal_policy, fmt='%i')
        np.savetxt('state_action_count-%i.txt' % episode, self.state_action_count, fmt='%i')
        np.savetxt('state_action_reward_sum-%i.txt' % episode,
                   self.state_action_reward_sum, fmt='%i')
