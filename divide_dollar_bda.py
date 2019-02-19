from __future__ import division

import copy
import time

import bda
import numpy as np
import scipy.stats as st

"""Evolving BDA agents to play divide-the-dollar."""

start = time.clock()

# Parameters for divide-the-dollar game ##
cards = [0.25, 0.50, 0.75]  # specifies the unique cards in the deck: indexed as [0,1,2]
num_of_unique_cards = [16, 28, 16]  # specifies the total number of each of the unique cards
num_cards = len(cards)  # number of unique cards
deck_size = sum(num_of_unique_cards)  # total number of cards in the deck
hand_size = 5  # number of cards in a player's hand--must be odd
num_players = 2  # number of players
num_rounds = 1+(deck_size-(num_players*hand_size))//num_players  # number of hands to play (until deck runs out)
num_episodes = 5  # number of games to play between two players (to ensure fair deck shuffling over time)

# Parameters for BDA specification
bda_states = 8

actions = {'small_spoil': 0, 'median': 1, 'large_max': 2}
num_actions = len(actions)
assert num_actions == bda.NUM_ACTIONS, "num_actions=%i does not match bda.NUM_ACTIONS=%i" % (num_actions, bda.NUM_ACTIONS)

# Parameters for evolution
pop_size = 15
rand_pop_size = 25
t_size = 11
max_mutations = 9
num_gens = 250
num_runs = 100


def init_pop():
    pop = []
    for i in xrange(pop_size+rand_pop_size):
        pop.append(bda.BDA(bda_states))
        pop[i].randomize()
    return pop


def load_deck():
    d = []
    for card, num in enumerate(num_of_unique_cards):
        d += [card]*num
    return np.array(d)


def play_action(card_showing, player, player_action):
    if card_showing == num_cards: # player's going first
        if player_action == actions['small_spoil']:
            player_card_value = cards[player[0]] # play smallest card
            card_showing = player[0] # update card showing
            player = np.delete(player, 0) # remove card from player's hand
        elif player_action == actions['large_max']:
            player_card_value = cards[player[-1]] # play largest card
            card_showing = player[-1] # update card showing
            player = np.delete(player, -1) # remove card from player's hand
        else:
            player_card_value = cards[player[hand_size//2]] # play median card
            card_showing = player[hand_size//2] # update card showing
            player = np.delete(player, hand_size//2) # remove card from player's hand
    else: # opponent went first, player's turn
        if player_action == actions['small_spoil']: # spoil with smallest card
            for c, pcard in enumerate(player):
                if cards[pcard] + cards[card_showing] > 1.0: # can spoil, play this card
                    player_card_value = cards[player[c]]
                    player = np.delete(player, c) # remove card from player's hand
                    break
                elif c == len(player)-1: # can't spoil, play largest card
                    player_card_value = cards[player[-1]]
                    player = np.delete(player, -1) # remove card from player's hand
        elif player_action == actions['large_max']: # maximize score with largest card
            for c, pcard in enumerate(np.flipud(player)):
                if cards[pcard] + cards[card_showing] <= 1.0: # can maximize, play this card
                    player_card_value = cards[player[len(player)-1-c]]
                    player = np.delete(player, len(player)-1-c) # remove card from player's hand
                    break
                elif len(player)-c == 0: # can't maximize, play smallest card
                    player_card_value = cards[player[0]]
                    player = np.delete(player, 0) # remove card from player's hand
        else:
            player_card_value = cards[player[hand_size//2]] # play median card
            player = np.delete(player, hand_size//2) # remove card from player's hand
    return card_showing, player, player_card_value


def save_pop(run, pop, fit):
    pop_file = open('pop-%i.txt' % run, 'w')
    first = True
    for i in np.argsort(fit)[0:][::-1]:
        if first:
            pop_file.write('%s\n\n' % bda_pop[i].write_bda())
            first = False
        pop_file.write('%.6f -fitness\n%s\n\n' % (fit[i],bda_pop[i].print_bda()))
    pop_file.close()


def report_fit_stats(stats_file, run, fit):
    mean = np.mean(fit)
    ci = st.t.interval(0.95, len(fit)-1, loc=mean, scale=st.sem(fit))
    std = np.std(fit)
    best = np.amax(fit)
    stats_file.write('%.6f %.6f %.6f %.6f\n' % (mean, ci[1], std, best))


for run in xrange(0,num_runs):
    print 'run %i' % run
    win_percen_file = open('win_percen-%i.txt' % run, 'w')
    plus_minus_file = open('plus_minus-%i.txt' % run, 'w')
    score_earned_file = open('score_earned-%i.txt' % run, 'w')
    score_diff_file = open('score_diff-%i.txt' % run, 'w')
    bda_pop = init_pop()
    dx = np.array([i for i in xrange(pop_size)])  # sorting index
    for gen in xrange(num_gens):
        #print 'gen %i' % gen

        if gen != 0:
            for i in xrange(pop_size,pop_size+rand_pop_size):
                bda_pop[i].randomize()

        # (fitness) score-keeping
        wins = np.array([0 for i in xrange(pop_size+rand_pop_size)])
        losses = np.array([0 for i in xrange(pop_size+rand_pop_size)])
        plus_minus = np.array([0 for i in xrange(pop_size+rand_pop_size)])
        score_earned = np.array([0 for i in xrange(pop_size+rand_pop_size)])
        score_diff = np.array([0 for i in xrange(pop_size+rand_pop_size)])

        ## Round-robin Match-ups ##
        for p1_index in xrange(pop_size): # Player 1 - evolving
            for p2_index in xrange(pop_size,pop_size+rand_pop_size):  # Player 2 - random
                for ep in xrange(num_episodes):
                    # Load and shuffle deck
                    deck = load_deck()
                    np.random.shuffle(deck)

                    bda_pop[p1_index].reset()
                    bda_pop[p2_index].reset()
                    p1_total_score = 0
                    p2_total_score = 0
                    num_deals = 0  # number of times a round resulted in a positive score for both players

                    # Deal initial hands
                    p1_cards = np.sort(deck[:hand_size])
                    deck = deck[hand_size:]
                    p2_cards = np.sort(deck[:hand_size])
                    deck = deck[hand_size:]

                    for round_index in xrange(num_rounds):
                        p1_card_value = 0
                        p2_card_value = 0

                        # Determine the value of the card showing (0 if playing first; opponent's pick if playing second)
                        card_showing = num_cards # an index of 'num_cards' corresponds to no card showing (i.e. zero)
                        if round_index % 2 == 0:
                            # Player 1 goes first
                            p1_game_state = [0, p1_cards[0], p1_cards[hand_size//2], p1_cards[-1], num_deals/(round_index+1), 0]
                            p1_action = bda_pop[p1_index].run(p1_game_state)
                            card_showing, p1_cards, p1_card_value = play_action(card_showing, p1_cards, p1_action)

                            # Player 2 goes second
                            p2_game_state = [cards[card_showing], p2_cards[0], p2_cards[hand_size//2], p2_cards[-1], num_deals/(round_index+1), 1]
                            p2_action = bda_pop[p2_index].run(p2_game_state)
                            card_showing, p2_cards, p2_card_value = play_action(card_showing, p2_cards, p2_action)
                        else:
                            # Player 2 goes first
                            p2_game_state = [0, p2_cards[0], p2_cards[hand_size//2], p2_cards[-1], num_deals/(round_index+1), 0]
                            p2_action = bda_pop[p2_index].run(p2_game_state)
                            card_showing, p2_cards, p2_card_value = play_action(card_showing, p2_cards, p2_action)

                            # Player 1 goes second
                            p1_game_state = [cards[card_showing], p1_cards[0], p1_cards[hand_size//2], p1_cards[-1], num_deals/(round_index+1), 1]
                            p1_action = bda_pop[p1_index].run(p1_game_state)
                            card_showing, p1_cards, p1_card_value = play_action(card_showing, p1_cards, p1_action)

                        # Determine score for playing this hand
                        if p1_card_value + p2_card_value <= 1:
                            p1_total_score += p1_card_value
                            p2_total_score += p2_card_value
                            num_deals += 1

                        # If deck isn't empty, pick up new cards
                        if len(deck) != 0:
                            p1_cards = np.sort(np.append(p1_cards, deck[:1]))
                            deck = deck[1:]
                        if len(deck) != 0:
                            p2_cards = np.sort(np.append(p2_cards, deck[:1]))
                            deck = deck[1:]

                    # Determine final winner of the game and give out reward (+score keeping)
                    if p1_total_score > p2_total_score:
                        wins[p1_index] += 1
                        losses[p2_index] += 1
                        plus_minus[p1_index] += 1
                        plus_minus[p2_index] -= 1
                    elif p1_total_score < p2_total_score:
                        losses[p1_index] += 1
                        wins[p2_index] += 1
                        plus_minus[p2_index] += 1
                        plus_minus[p1_index] -= 1
                    score_earned[p1_index] += p1_total_score
                    score_earned[p2_index] += p2_total_score
                    score_diff[p1_index] += p1_total_score - p2_total_score
                    score_diff[p2_index] += p2_total_score - p1_total_score

        fit = wins[0:pop_size]/(rand_pop_size*num_episodes) # choose fitness measure (i.e. wins, plus_minus, score_earned, score_diff)
        report_fit_stats(win_percen_file, run, wins[0:pop_size]/(rand_pop_size*num_episodes)) # save information about fitness for this generation
        report_fit_stats(plus_minus_file, run, plus_minus[0:pop_size])
        report_fit_stats(score_earned_file, run, score_earned[0:pop_size])
        report_fit_stats(score_diff_file, run, score_diff[0:pop_size])
        if gen == num_gens-1:
            #save_pop(run, bda_pop, fit)
            pop_file = open('pop-%i.txt' % run, 'w')
            for i in np.argsort(fit)[0:][::-1]:
                pop_file.write('%.6f -fitness (%i %.2f %.2f)\n%s\n\n' % (fit[i], plus_minus[i], score_earned[i], score_diff[i], bda_pop[i].print_bda()))
            pop_file.close()
        else: ## Evolution time ##
            # Choose and sort the mating tournament participants
            dx = np.random.permutation(len(fit)) # sorting index
            dx[:t_size] = dx[:t_size][fit[dx][:t_size].argsort()]

            # Crossover (replace worst two with crossover result of best two)
            bda_pop[dx[0]] = copy.deepcopy(bda_pop[dx[t_size-1]])
            bda_pop[dx[1]] = copy.deepcopy(bda_pop[dx[t_size-2]])
            bda_pop[dx[0]].two_point_crossover(bda_pop[dx[1]])
            # Mutation
            for m in xrange(max_mutations):
                bda_pop[dx[0]].mutate()
                bda_pop[dx[1]].mutate()

    win_percen_file.close()
    plus_minus_file.close()
    score_earned_file.close()
    score_diff_file.close()


end = time.clock()
print "%.2f minutes" % ((end-start)/60)
