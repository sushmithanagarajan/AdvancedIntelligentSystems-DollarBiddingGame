"""Monte Carlo agent learns to play divide-the-dollar."""
from __future__ import division

import numpy as np
#import xrange

from game import CardGame, Deck, Player
from mc import MonteCarloLearning

CARDS_IN_DECK = {0.25: 16, 0.50: 28, 0.75: 16}
NUM_PLAYERS = 2
HAND_SIZE = 5
ACTIONS = ['small_spoil', 'median', 'large_max']

NUM_GAMES_TO_PLAY = 2000000

deck = Deck(CARDS_IN_DECK)
card_game = CardGame(deck, NUM_PLAYERS, ACTIONS, HAND_SIZE)


def play_action(card_showing, player):
    if card_showing == 0:  # player goes first
        if player.next_action == ACTIONS.index('small_spoil'):
            card_value = player.play_card(0)
        elif player.next_action == ACTIONS.index('large_max'):
            card_value = player.play_card(-1)
        else:
            card_value = player.play_card(card_game.hand_size // 2)
    else:  # opponent went first, player's turn
        if player.next_action == ACTIONS.index('small_spoil'):
            for c, card in enumerate(player.hand):
                if card + card_showing > 1.0:  # can spoil, play this card
                    card_value = player.play_card(c)
                    break
                elif c == len(player.hand) - 1:  # can't spoil, play largest card
                    card_value = player.play_card(-1)
        elif player.next_action == ACTIONS.index('large_max'):
            for c, card in enumerate(np.flipud(player.hand)):
                if card + card_showing <= 1.0:  # can maximize, play this card
                    card_value = player.play_card(len(player.hand) - 1 - c)
                    break
                elif len(player) - c == 0:  # can't maximize, play smallest card
                    card_value = player.play_card(0)
        else:
            card_value = player.play_card(card_game.hand_size // 2)
    return card_value


def take_turn(player, round_index, card_showing, monte_carlo=False):
    player.set_game_state(card_showing)
    if monte_carlo:
        q_learning.record_state_seen(player.game_state)

    policy_index = int(card_game.true_state_index[int(np.ravel_multi_index(
        player.game_state, dims=(deck.num_unique_cards + 1, deck.num_unique_cards,
                                 deck.num_unique_cards, deck.num_unique_cards)))])

    if monte_carlo and (round_index <= 1):  # exploring starts
        player.next_action = np.random.choice(card_game.num_actions)
    else:
        player.next_action = player.policy[policy_index]

    return play_action(card_showing, player)


q_learning = MonteCarloLearning(card_game.num_states, card_game.num_actions)
monte = Player()
opponent = Player()

for episode_index in xrange(NUM_GAMES_TO_PLAY):
    deck.shuffle_deck()

    monte.pick_up_cards(deck.deal_cards(card_game.hand_size))
    opponent.pick_up_cards(deck.deal_cards(card_game.hand_size))

    q_learning.clear_states_seen()

    for round_index in xrange(card_game.num_rounds):
        sum_of_cards = 0.

        if round_index % 2 == 0:
            sum_of_cards = take_turn(monte, round_index, sum_of_cards, monte_carlo=True)
            sum_of_cards += take_turn(opponent, round_index, sum_of_cards)
        else:
            card_showing = take_turn(opponent, round_index, sum_of_cards)
            sum_of_cards = take_turn(monte, round_index, sum_of_cards, monte_carlo=True)

        if monte.last_card_played + opponent.last_card_played <= 1:
            monte.total_score += monte.last_card_played
            opponent.total_score += opponent.last_card_played

        monte.pick_up_cards(deck.deal_cards(1))
        opponent.pick_up_cards(deck.deal_cards(1))

    reward = 0
    if monte.total_score > opponent.total_score:
        reward = +1
        monte.wins += 1
    elif monte.total_score < opponent.total_score:
        reward = -1
        opponent.wins += 1

    for state in xrange(len(q_learning.states_seen)):
        state_index = int(card_game.true_state_index[int(np.ravel_multi_index(
            q_learning.states_seen[state], dims=(deck.num_unique_cards + 1,
                                                 deck.num_unique_cards,
                                                 deck.num_unique_cards,
                                                 deck.num_unique_cards)))])
        action_index = int(card_game.true_state_index[int(q_learning.policy[state_index])])
        q_learning.update(state_index, action_index, reward)

q_learning.save_learning(NUM_GAMES_TO_PLAY)
