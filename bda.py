from __future__ import division

import random
from io import StringIO

NUM_ACTIONS = 3
NUM_INPUTS = 6
NUM_TESTS = 3
NEAR = 0.05
MAX_TRANSITIONS = 5


class State(object):
    def __init__(self):
        self.decision_index = 0
        self.decision_type = 0
        self.threshold_val = 0.5
        self.transitions = [0, 0]
        self.actions = [0, -1]


class BDA(object):
    def __init__(self, ns):
        self.num_states = ns
        self.states = [State() for n in xrange(self.num_states)]
        self.current_state = 0

    def randomize(self):
        for n in xrange(self.num_states):
            self.states[n].decision_index = random.randint(0,NUM_INPUTS-1)
            self.states[n].decision_type = random.randint(0,NUM_TESTS-1)
            self.states[n].threshold_val = random.randint(0,1000)/1000
            for i in xrange(2):
                self.states[n].actions[i] = random.randint(0,NUM_ACTIONS-1)
                self.states[n].transitions[i] = random.randint(0,self.num_states-1)

    def reset(self):
        self.current_state = 0

    def run(self, sim_state): # run on a given simulator state, return action
        ## sim_state = [total_played, low_card, median_card, high_card, fraction_of_deals, first_player?]
        bd = 1  # binary decision (bd=0 means if statement is TRUE, bd=1 means if statement is FALSE)
        it = 0 # number of internal transitions
        while bd == 1 and it <= MAX_TRANSITIONS:
            cdv = self.states[self.current_state].decision_index # index of current decision variable
            sdt = self.states[self.current_state].decision_type # state decision type
            if sdt == 0:
                if sim_state[cdv] > self.states[self.current_state].threshold_val:
                    bd = 0
                    break
            elif sdt == 1:
                if sim_state[cdv] < self.states[self.current_state].threshold_val:
                    bd = 0
                    break
            elif sdt == 2:
                if abs(sim_state[cdv] - self.states[self.current_state].threshold_val) < NEAR:
                    bd = 0
                    break
            self.current_state = self.states[self.current_state].transitions[bd]
            it += 1

        return_action = self.states[self.current_state].actions[bd]
        self.current_state = self.states[self.current_state].transitions[bd] # transition to new state

        return return_action # tell user what the action to take is

    def two_point_crossover(self, other):
        crossover_pt1 = random.randint(0,self.num_states-1)
        crossover_pt2 = random.randint(0,self.num_states-1)
        if crossover_pt1 > crossover_pt2:
            c = crossover_pt1
            crossover_pt1 = crossover_pt2
            crossover_pt2 = c
        for i in xrange(crossover_pt1,crossover_pt2): # loop over positions to swap
            # swap decision index
            sw = self.states[i].decision_index
            self.states[i].decision_index = other.states[i].decision_index
            other.states[i].decision_index = sw

            # swap decision type
            sw = self.states[i].decision_type
            self.states[i].decision_type = other.states[i].decision_type
            other.states[i].decision_type = sw

            # swap threshold
            sw = self.states[i].threshold_val
            self.states[i].threshold_val = other.states[i].threshold_val
            other.states[i].threshold_val = sw

            for j in xrange(2):
                # swap transitions
                sw = self.states[i].transitions[j]
                self.states[i].transitions[j] = other.states[i].transitions[j]
                other.states[i].transitions[j] = sw

                # swap actions
                sw = self.states[i].actions[j]
                self.states[i].actions[j] = other.states[i].actions[j]
                other.states[i].actions[j] = sw

    def mutate(self):
        q = random.randint(0,self.num_states-1) # select state to mutate
        m = random.randint(0,6) # pick an object to mutate
        if m == 0:
            self.states[q].decision_index = random.randint(0,NUM_INPUTS-1) # new input
        elif m == 1:
            self.states[q].decision_type = random.randint(0,NUM_TESTS-1) # new decision type
        elif m == 2:
            self.states[q].threshold_val = random.randint(0,1000)/1000 # new threshold
        elif m == 3:
            self.states[q].transitions[0] = random.randint(0,self.num_states-1) # new first transition
        elif m == 4:
            self.states[q].transitions[1] = random.randint(0,self.num_states-1) # new second transition
        elif m == 5:
            self.states[q].actions[0] = random.randint(0,NUM_ACTIONS-1) # new first action
        elif m == 6:
            self.states[q].actions[1] = random.randint(0,NUM_ACTIONS-1) # new second action

    def write_bda(self):
        #output = '%i\n' % self.num_states
        output = ''
        for n in xrange(self.num_states):
            output += '%i %i %.3f ' % (self.states[n].decision_index, self.states[n].decision_type, self.states[n].threshold_val)
            for i in xrange(2):
                output += '%i %i ' % (self.states[n].actions[i], self.states[n].transitions[i])
            output += '\n'
        return output

    def read_bda(self, bda_array):
        self.__init__(int(len(bda_array)))
        for n, state_data in enumerate(bda_array):
            self.states[n].decision_index = int(state_data[0])
            self.states[n].decision_type = int(state_data[1])
            self.states[n].threshold_val = state_data[2]
            self.states[n].actions[0] = int(state_data[3])
            self.states[n].transitions[0] = int(state_data[4])
            self.states[n].actions[1] = int(state_data[5])
            self.states[n].transitions[1] = int(state_data[6])

    def print_bda(self): # human-readable form
        action_text = ('SmlSpl', 'Median', 'LrgMax')
        input_text = ('Ttl', 'Sml', 'Med', 'Lrg', 'Coop', 'Idx')
        decision_text = ('>', '<', 'near')

        bda_output = '%i states\n' % self.num_states
        for i in xrange(self.num_states):
            bda_output += '%i) if(%s %s %.3f) ' % (i, input_text[self.states[i].decision_index], decision_text[self.states[i].decision_type], self.states[i].threshold_val)
            bda_output += '%s-> %i else %s-> %i\n' % (action_text[self.states[i].actions[0]], self.states[i].transitions[0], action_text[self.states[i].actions[1]], self.states[i].transitions[1])
        return bda_output
