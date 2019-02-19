from __future__ import division

import math

import numpy as np


class Deck(object):
    """Deck.

    Args:
        cards (dict): {card_value: unique_cards}

    Attributes:
        cards (dict): {card_value: num_cards}
        unique_cards (int): number of unique card values
        deck_size (int): total number of cards in the deck
        current_deck (list): full list of cards remaining in deck

    """

    def __init__(self, cards):
        """Initialize deck of cards."""
        self.cards = cards
        self.unique_cards = len(self.cards)
        self.deck_size = sum(self.cards.values())
        self.current_deck = self.shuffle_deck()

    def shuffle_deck(self):
        """Return shuffled deck of all cards."""
        d = []
        for card, num in zip(self.cards.keys(), self.cards.values()):
            d += [card] * num
        return np.random.permutation(np.array(d)).tolist()

    def deal_cards(self, num_cards_to_deal):
        """Deal N cards from top of deck."""
        assert len(self.current_deck) >= num_cards_to_deal, \
            'Not enough cards left in deck to deal those cards.'
        dealt_cards = self.current_deck[:num_cards_to_deal]
        self.current_deck = self.current_deck[num_cards_to_deal:]
        return dealt_cards


class CardGame(object):
    """Card game.

    Args:
        num_players (int): number of players playing the game
        deck (Deck): deck of cards
        actions (list): list of action names
        hand_size (int): number of cards a player holds in their hand

    Attributes:
        num_players (int): number of players playing the game
        deck (Deck): deck of cards
        actions (list): list of action names
        num_actions (int): number of unique actions
        num_states (int): number of possible game states
        hand_size (int): number of cards a player holds in their hand
        num_rounds (int): number of rounds that are played in one game

    """

    def __init__(self, deck, num_players, actions, hand_size):
        """Initialize card game."""
        self.num_players = num_players
        self.deck = deck
        self.actions = actions
        self.num_actions = len(actions)
        self.num_states = int((self.deck.unique_cards + 1)
                              * (math.factorial(self.num_actions + self.deck.unique_cards - 1))
                              / (math.factorial(self.num_actions)
                                 * math.factorial(self.deck.unique_cards - 1)))
        self.hand_size = hand_size
        self.num_rounds = 1 + (self.deck.deck_size
                               - (self.num_players * self.hand_size)) // self.num_players
        self.true_state_index = self._true_state_index()

    def _true_state_index(self):
        """Return the true index in list of unique states for each permutation.

        For a potential game state permutation [card_showing, smallest, median, largest],
        if smallest <= median <= largest does not hold, the permutation is an invalid game state
        and the true state index should be a -1. For all valid permutations, the true state index
        should be sequentially increasing.

        Returns:
            (list): true state index of valid permutations

        """
        states = []
        unique_cards = self.deck.unique_cards
        for card_showing in xrange(unique_cards + 1):
            for smallest in xrange(unique_cards):
                for median in xrange(unique_cards):
                    for largest in xrange(unique_cards):
                        states.append(np.array([card_showing, smallest, median, largest]))

        true_state_index = []
        true_index_counter = 0
        for state in states:
            if np.all(state[1:-1] <= state[2:]):
                true_state_index.append(true_index_counter)
                true_index_counter += 1
            else:
                true_state_index.append(-1)

        return true_state_index


class Player(object):
    """Player of card game.

    Args:
        policy (list): an initial policy

    Attributes:
        policy (list): the player's optimal policy for choosing action when in state
        hand (list): the player's cards in hand
        game_state (array): the current state of the game
        next_action: the player's chosen action to play next turn
        last_card_played (float): the value of the last card played
        total_score (float): the player's total score
        wins (int): the player's total number of wins

    """

    def __init__(self, policy):
        """Initialize player."""
        self.policy = policy
        self.hand = []
        self.game_state = None
        self.next_action = None
        self.last_card_played = None
        self.total_score = 0
        self.wins = 0

    def update_policy(self, new_policy):
        """Change player's optimal policy to new policy."""
        self.policy = new_policy

    def pick_up_cards(self, cards):
        """Add cards (list or tuple) to player's hand."""
        assert isinstance(cards, list) or isinstance(cards, tuple)
        self.hand = np.sort(self.hand + cards).tolist()

    def play_card(self, card_position_in_hand):
        """Play specific card in hand."""
        card_value = self.hand[card_position_in_hand]
        self.hand = np.delete(self.hand, 0).tolist()
        self.last_card_played = card_value
        return card_value

    def set_game_state(self, card_showing):
        """TODO: Remove this method; not appropriate for player class."""
        median_card_index = len(self.hand) // 2
        self.game_state = [card_showing, self.hand[0], self.hand[median_card_index], self.hand[-1]]

    def reset_score(self):
        """Reset player's score to zero."""
        self.total_score = 0

    def reset_wins(self):
        """Reset player's win count to zero."""
        self.wins = 0
