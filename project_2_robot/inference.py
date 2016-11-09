#!/usr/bin/env python
# inference.py
# Base code by George H. Chen (georgehc@mit.edu) -- updated 10/18/2016
import collections
import sys

import graphics
import numpy as np
import robot


# Throughout the code, we use these variables.
# Do NOT change these (but you'll need to use them!):
# - all_possible_hidden_states: a list of possible hidden states
# - all_possible_observed_states: a list of possible observed states
# - prior_distribution: a distribution over states
# - transition_model: a function that takes a hidden state and returns a
#     Distribution for the next state
# - observation_model: a function that takes a hidden state and returns a
#     Distribution for the observation from that hidden state
all_possible_hidden_states = robot.get_all_hidden_states()
all_possible_observed_states = robot.get_all_observed_states()
prior_distribution = robot.initial_distribution()
transition_model = robot.transition_model
observation_model = robot.observation_model


# You may find this function helpful for computing logs without yielding a
# NumPy warning when taking the log of 0.
def careful_log(x):
    # computes the log of a non-negative real number
    if x == 0:
        return -np.inf
    else:
        return np.log(x)


# returns the reversed observation model with observations mapping to states
def compute_reverse_observation_model():
    # initialize final distribution
    reverse_observations = {}

    for observation in all_possible_observed_states:
        reverse_observations[observation] = robot.Distribution()

    # collect (unnormalized) observation-probabilities
    for state in all_possible_hidden_states:
        possible_observations = observation_model(state)

        for observation in possible_observations:
            reverse_observations[observation][state] = \
                possible_observations[observation]

    return reverse_observations

# returns the reversed observation model with a ln-transformation
def compute_log_reverse_observation_model():
    # initialize final distribution
    reverse_observations = {}

    for observation in all_possible_observed_states:
        reverse_observations[observation] = robot.Distribution()

    # collect (unnormalized) observation-probabilities
    for state in all_possible_hidden_states:
        possible_observations = observation_model(state)

        for observation in possible_observations:
            reverse_observations[observation][state] = \
                careful_log(possible_observations[observation])

    return reverse_observations

# returns the reversed transition model given probs of previous states
def compute_reverse_transition_model():
    # initialize final distribution
    reverse_transitions = {}

    for state in all_possible_hidden_states:
        reverse_transitions[state] = robot.Distribution()

    # collect (unnormalized) reverse transition probabilities
    for start_state in all_possible_hidden_states:
        possible_goal_states = transition_model(start_state)

        for goal_state in possible_goal_states:
            reverse_transitions[goal_state][start_state] = \
                possible_goal_states[goal_state]

    return reverse_transitions


# returns a Distribution with one entry for every state with equal prob
def generate_uniform_message():
    message = robot.Distribution()

    for state in all_possible_hidden_states:
        message[state] = 1 / len(all_possible_hidden_states)

    return message

# -----------------------------------------------------------------------------
# Functions for you to implement
#

def forward_backward(observations):
    """
    Input
    -----
    observations: a list of observations, one per hidden state
        (a missing observation is encoded as None)

    Output
    ------
    A list of marginal distributions at each time step; each distribution
    should be encoded as a Distribution (see the Distribution class in
    robot.py and see how it is used in both robot.py and the function
    generate_data() above, and the i-th Distribution should correspond to time
    step i
    """

    num_time_steps = len(observations)
    forward_messages = [None] * num_time_steps
    forward_messages[0] = prior_distribution
    reverse_observation_model = compute_reverse_observation_model()

    # Compute the forward messages
    #print("  - compute forward messages...")

    # apply for every timestep
    for time_step in range(1, num_time_steps):
        forward_messages[time_step] = robot.Distribution()
        observation = observations[time_step - 1]

        # loop through all possible values of next state
        for goal_state in all_possible_hidden_states:
            probability_sum = 0

            if observation:
                # loop through all non-zero values of previous state
                for current_state in reverse_observation_model[observation]:
                    # multiply obs-model, trans-model, message
                    probability_sum += \
                        reverse_observation_model[observation][current_state] * \
                        transition_model(current_state)[goal_state] * \
                        forward_messages[time_step - 1][current_state]
            else:
                for current_state in forward_messages[time_step -1]:
                    probability_sum += \
                        transition_model(current_state)[goal_state] * \
                        forward_messages[time_step - 1][current_state]

            if probability_sum > 0:
                forward_messages[time_step][goal_state] = probability_sum

    # Compute the backward messages
    # note: order will be reverse! backward_messages[0] will hold the message
    #       coming in to the *last* hidden state
    #print("  - compute backward messages")

    backward_messages = [None] * num_time_steps
    backward_messages[0] = generate_uniform_message()
    reverse_transition_model = compute_reverse_transition_model()

    # apply for every time step
    for time_step in range(1, num_time_steps):
        backward_messages[time_step] = robot.Distribution()
        observation = observations[num_time_steps - time_step]

        # create a table-entry for every possible value
        for goal_state in all_possible_hidden_states:
            probability_sum = 0

            if observation:
                # look only at non-zero values allowed by the observation
                for current_state in reverse_observation_model[observation]:
                    # multiply obs-model, reverse-trans-model, back-message
                    probability_sum += \
                        reverse_observation_model[observation][current_state] * \
                        reverse_transition_model[current_state][goal_state] * \
                        backward_messages[time_step - 1][current_state]
            else:
                for current_state in backward_messages[time_step - 1]:
                    probability_sum += \
                        reverse_transition_model[current_state][goal_state] * \
                        backward_messages[time_step - 1][current_state]

            if probability_sum > 0:
                backward_messages[time_step][goal_state] = probability_sum

    # Compute the marginals
    #print("  - compute marginals")

    marginals = [None] * num_time_steps # remove this

    for time_step in range(0, num_time_steps):
        marginals[time_step] = robot.Distribution()
        observation = observations[time_step]

        for state in all_possible_hidden_states:
            if observation:
                probability = \
                    reverse_observation_model[observation][state] * \
                    forward_messages[time_step][state] * \
                    backward_messages[num_time_steps - 1 - time_step][state]
            else:
                probability = \
                    forward_messages[time_step][state] * \
                    backward_messages[num_time_steps - 1 - time_step][state]

            if probability > 0:
                marginals[time_step][state] = probability

    for px in marginals:
        px.renormalize()

    return marginals


# returns the state with maximum probability as a single-element list
def MAP_estimate(observation):
    # without observation, all prior states are equally likely
    if not observation[0]:
        return list(prior_distribution.keys())[0]

    reverse_observation_model = compute_reverse_observation_model()

    max_val = -1
    argmax = None

    for state in reverse_observation_model[observation[0]]:
        potential = \
            reverse_observation_model[observation[0]][state] * \
            prior_distribution[state]

        if potential > max_val:
            max_val = potential
            argmax = state

    return [argmax]

# returns a uniform singleton potential over all states
def uniform_potential():
    potential = robot.Distribution()

    for state in all_possible_hidden_states:
        potential[state] = 0.1

    return potential

# returns a list of node-potentials derived from the given observations
def precompute_singleton_potentials(observations):
    num_time_steps = len(observations)
    phi = [None] * num_time_steps

    reverse_observation_model = compute_reverse_observation_model()

    for time_step in range(num_time_steps):
        phi[time_step] = robot.Distribution()
        observation = observations[time_step]

        if not observation:
            phi[time_step] = uniform_potential()
            continue

        for state in reverse_observation_model[observation]:
            potential = reverse_observation_model[observation][state]

            # time step 0 takes into account the prior dist.
            if time_step == 0:
                potential *= prior_distribution[state]

            if potential > 0:
                phi[time_step][state] = -careful_log(potential)

    return phi


def Viterbi(observations):
    """
    Input
    -----
    observations: a list of observations, one per hidden state
        (a missing observation is encoded as None)

    Output
    ------
    A list of esimated hidden states, each encoded as a tuple
    (<x>, <y>, <action>)
    """
    num_time_steps = len(observations)
    if num_time_steps == 1: return MAP_estimate(observations)

    estimated_hidden_states = [None] * num_time_steps

    # fold in observations
    phi = precompute_singleton_potentials(observations)

    # compute forward-messages
    forward_messages = [None] * (num_time_steps - 1)
    traceback_messages = [None] * (num_time_steps - 1)

    for time_step in range(num_time_steps - 1):
        forward_messages[time_step] = robot.Distribution()
        traceback_messages[time_step] = robot.Distribution()

        for goal_state in all_possible_hidden_states:
            min_val = np.inf
            argmin = None

            for current_state in phi[time_step]:
                current_val = \
                    phi[time_step][current_state] + \
                    -careful_log(transition_model(current_state)[goal_state])

                if time_step > 0:
                    current_val += forward_messages[time_step - 1][current_state]

                if current_val < min_val:
                    min_val = current_val
                    argmin = current_state

            forward_messages[time_step][goal_state] = min_val
            traceback_messages[time_step][goal_state] = argmin

    # compute maximum value at the root
    min_val = np.inf
    argmin = None

    for state in phi[num_time_steps - 1]:
        current_val = \
            phi[num_time_steps - 1][state] + \
            forward_messages[num_time_steps - 2][state]

        if current_val < min_val:
            min_val = current_val
            argmin = state

    estimated_hidden_states[num_time_steps - 1] = argmin

    # follow the traceback
    for time_step in range(num_time_steps - 2, -1, -1):
        estimated_hidden_states[time_step] = \
            traceback_messages[time_step][estimated_hidden_states[time_step + 1]]

    return estimated_hidden_states


def test():
    obs = [(2, 0), (2, 0), (3, 0), (4, 0), (4, 0), (6, 0), (6, 1), (5, 0), (6, 0), (6, 2)]
    assert Viterbi(obs) == [
        (1, 0, "stay"), (2, 0, "right"), (3, 0, "right"), (4, 0, "right"),
        (5, 0, "right"), (6, 0, "right"), (6, 0, "stay"), (6, 0, "stay"),
        (6, 1, "down"), (6, 2, "down")]
    print('autograder testcase: OK')

    obs = [(1, 0)]
    assert MAP_estimate(obs) == [(0, 0, 'stay')]
    print('MAP, 1 observation: OK')

    obs = [(1, 0)]
    assert Viterbi(obs) == [(0, 0, 'stay')]
    print('Single case: OK')

    obs = [(0, 1), (3, 1)]
    assert Viterbi(obs) == [(1, 1, 'stay'), (2, 1, 'right')]
    print('two-node case: ok')

    obs = [(2, 5), (3, 3), (2, 1)]
    assert Viterbi(obs) == [(2, 4, 'stay'), (2, 3, 'up'), (2, 2, 'up')]
    print("Three-node case OK")

    obs = [(0, 1), None, (4, 1)]
    assert Viterbi(obs) == [(1, 1, 'stay'), (2, 1, 'right'), (3, 1, 'right')]
    print('three-node case with missing observation: ok')

    obs = [(0, 0), (1, 2), (2, 0), (4, 1)]
    assert Viterbi(obs) == [(0, 1, 'stay'), (1, 1, 'right'), (2, 1, 'right'), (3, 1, 'right')]
    print("Four-node case OK")

    obs = [(0, 0), (1, 2), (2, 0), (4, 1), (5, 1)]
    assert Viterbi(obs) == [(0, 1, 'stay'), (1, 1, 'right'), (2, 1, 'right'), (3, 1, 'right'), (4, 1, 'right')]
    print("Five-node case OK")


def display(dictionary, title=""):
    print("\n", title, ":")

    for key in dictionary:
        print(' ', key, ":", dictionary[key])


def second_best(observations):
    """
    Input
    -----
    observations: a list of observations, one per hidden state
        (a missing observation is encoded as None)

    Output
    ------
    A list of esimated hidden states, each encoded as a tuple
    (<x>, <y>, <action>)
    """
    num_time_steps = len(observations)
    if num_time_steps == 1: return MAP_estimate(observations)

    reverse_transition_model = compute_reverse_transition_model()
    estimated_hidden_states = [None] * num_time_steps

    # fold in observations
    phi = precompute_singleton_potentials(observations)

    # compute forward-messages
    print('compute forward messages')
    forward_messages = [None] * (num_time_steps - 1)
    traceback_messages = [None] * (num_time_steps - 1)

    for time_step in range(num_time_steps - 1):
        forward_messages[time_step] = robot.Distribution()
        traceback_messages[time_step] = robot.Distribution()

        for goal_state in all_possible_hidden_states:
            min_val = np.inf
            argmin = None

            for current_state in phi[time_step]:
                current_val = \
                    phi[time_step][current_state] + \
                    -careful_log(transition_model(current_state)[goal_state])

                if time_step > 0:
                    current_val += forward_messages[time_step - 1][current_state]

                if current_val < min_val:
                    min_val = current_val
                    argmin = current_state

            forward_messages[time_step][goal_state] = min_val
            traceback_messages[time_step][goal_state] = argmin

    # compute backward-messages
    print('compute backward messages')
    backward_messages = [None] * (num_time_steps - 1)
    backward_traceback_messages = [None] * (num_time_steps - 1)

    for time_step in range(num_time_steps - 2, -1, -1):
        backward_messages[time_step] = robot.Distribution()
        backward_traceback_messages[time_step] = robot.Distribution()

        for goal_state in all_possible_hidden_states:
            min_val = np.inf
            argmin = None

            for current_state in phi[time_step + 1]:
                current_val = \
                    phi[time_step + 1][current_state] + \
                    -careful_log(reverse_transition_model[current_state][goal_state])

                if time_step < num_time_steps - 2:
                    current_val += backward_messages[time_step + 1][current_state]

                if current_val < min_val:
                    min_val = current_val
                    argmin = current_state

            backward_messages[time_step][goal_state] = min_val
            backward_traceback_messages[time_step][goal_state] = argmin


    # compute minimum value at first node
    min_val = np.inf
    argmin = None

    for state in phi[0]:
        current_val = phi[0][state] + backward_messages[0][state]

        if current_val < min_val:
            min_val = current_val
            argmin = state

    estimated_hidden_states[0] = argmin

    # follow the traceback
    for time_step in range(1, num_time_steps):
        estimated_hidden_states[time_step] = \
            backward_traceback_messages[time_step - 1][estimated_hidden_states[time_step - 1]]



#     # compute maximum value at the root
#     min_val = np.inf
#     argmin = None
# 
#     for state in phi[num_time_steps - 1]:
#         current_val = \
#             phi[num_time_steps - 1][state] + \
#             forward_messages[num_time_steps - 2][state]
# 
#         if current_val < min_val:
#             min_val = current_val
#             argmin = state
# 
#     estimated_hidden_states[num_time_steps - 1] = argmin
# 
#     # follow the traceback
#     for time_step in range(num_time_steps - 2, -1, -1):
#         estimated_hidden_states[time_step] = \
#             traceback_messages[time_step][estimated_hidden_states[time_step + 1]]

    print('***', estimated_hidden_states)
    return estimated_hidden_states





def test2():
    obs = [(2, 0), (2, 0), (3, 0), (4, 0), (4, 0), (6, 0), (6, 1), (5, 0), (6, 0), (6, 2)]
    assert second_best(obs) == [
        (1, 0, "stay"), (2, 0, "right"), (3, 0, "right"), (4, 0, "right"),
        (5, 0, "right"), (6, 0, "right"), (6, 0, "stay"), (6, 0, "stay"),
        (6, 1, "down"), (6, 2, "down")]
    print('autograder testcase: OK')

#     obs = [(8, 2), (8, 1), (10, 0), (10, 0), (10, 1),
#            (11, 0), (11, 0), (11, 1), (11, 2), (11, 2)]
#     assert second_best(obs) == \
#         [[9, 2, "stay"], [9, 1, "up"], [9, 0, "up"],
#          [9, 0, "stay"], [10, 0, "right"], [11, 0, "right"],
#          [11, 0, "stay"], [11, 0, "stay"], [11, 1, "down"],
#          [11, 2, "down"]]
#     print('grader case 1: OK')
# 
#     obs = [(1, 4), (1, 5), (1, 5), (1, 6), (0, 7),
#            (1, 7), (3, 7), (4, 7), (4, 7), (4, 7)]
#     assert second_best(obs) == \
#         [[1, 4, "stay"], [1, 5, "down"], [1, 6, "down"],
#          [1, 7, "down"], [1, 7, "stay"], [1, 7, "stay"],
#          [2, 7, "right"], [3, 7, "right"], [4, 7, "right"],
#          [5, 7, "right"]]
#     print('grader case 2: OK')




# -----------------------------------------------------------------------------
# Generating data from the hidden Markov model
#

def generate_data(num_time_steps, make_some_observations_missing=False,
                  random_seed=None):
    # generate samples from this project's hidden Markov model
    hidden_states = []
    observations = []

    # if the random seed is not None, then this makes the randomness
    # deterministic, which may be helpful for debug purposes
    np.random.seed(random_seed)

    # draw initial state and emit an observation
    initial_state = prior_distribution.sample()
    initial_observation = observation_model(initial_state).sample()

    hidden_states.append(initial_state)
    observations.append(initial_observation)

    for time_step in range(1, num_time_steps):
        # move the robot
        prev_state = hidden_states[-1]
        new_state = transition_model(prev_state).sample()

        # maybe emit an observation
        if not make_some_observations_missing:
            new_observation = observation_model(new_state).sample()
        else:
            if np.random.rand() < .1:  # 0.1 prob. of observation being missing
                new_observation = None
            else:
                new_observation = observation_model(new_state).sample()

        hidden_states.append(new_state)
        observations.append(new_observation)

    return hidden_states, observations


# -----------------------------------------------------------------------------
# Main
#

def main():
    test2()
    exit(1)

    # flags
    make_some_observations_missing = False
    use_graphics = True
    need_to_generate_data = True

    # parse command line arguments
    for arg in sys.argv[1:]:
        if arg == '--missing':
            make_some_observations_missing = True
        elif arg == '--nographics':
            use_graphics = False
        elif arg.startswith('--load='):
            filename = arg[7:]
            hidden_states, observations = robot.load_data(filename)
            need_to_generate_data = False
            num_time_steps = len(hidden_states)

    # if no data is loaded, then generate new data
    if need_to_generate_data:
        num_time_steps = 100
        hidden_states, observations = \
            generate_data(num_time_steps,
                          make_some_observations_missing)

    print('Running forward-backward...')
    marginals = forward_backward(observations)
    print("\n")

    timestep = 2
    #timestep = 99
    print("Most likely parts of marginal at time %d:" % (timestep))
    if marginals[timestep] is not None:
        print(sorted(marginals[timestep].items(),
                     key=lambda x: x[1],
                     reverse=True)[:10])
    else:
        print('*No marginal computed*')
    print("\n")


    print('Running Viterbi...')
    estimated_states = Viterbi(observations)
    print("\n")

    print("Last 10 hidden states in the MAP estimate:")
    for time_step in range(num_time_steps - 10 - 1, num_time_steps):
        if estimated_states[time_step] is None:
            print('Missing')
        else:
            print(estimated_states[time_step])
    print("\n")


    print('Finding second-best MAP estimate...')
    estimated_states2 = second_best(observations)
    print("\n")

    print("Last 10 hidden states in the second-best MAP estimate:")
    for time_step in range(num_time_steps - 10 - 1, num_time_steps):
        if estimated_states2[time_step] is None:
            print('Missing')
        else:
            print(estimated_states2[time_step])
    print("\n")

    difference = 0
    difference_time_steps = []
    for time_step in range(num_time_steps):
        if estimated_states[time_step] != hidden_states[time_step]:
            difference += 1
            difference_time_steps.append(time_step)
    print("Number of differences between MAP estimate and true hidden " +
          "states:", difference)
    if difference > 0:
        print("Differences are at the following time steps: " +
              ", ".join(["%d" % time_step
                         for time_step in difference_time_steps]))
    print("\n")

    difference = 0
    difference_time_steps = []
    for time_step in range(num_time_steps):
        if estimated_states2[time_step] != hidden_states[time_step]:
            difference += 1
            difference_time_steps.append(time_step)
    print("Number of differences between second-best MAP estimate and " +
          "true hidden states:", difference)
    if difference > 0:
        print("Differences are at the following time steps: " +
              ", ".join(["%d" % time_step
                         for time_step in difference_time_steps]))
    print("\n")

    difference = 0
    difference_time_steps = []
    for time_step in range(num_time_steps):
        if estimated_states[time_step] != estimated_states2[time_step]:
            difference += 1
            difference_time_steps.append(time_step)
    print("Number of differences between MAP and second-best MAP " +
          "estimates:", difference)
    if difference > 0:
        print("Differences are at the following time steps: " +
              ", ".join(["%d" % time_step
                         for time_step in difference_time_steps]))
    print("\n")

    # display
    if use_graphics:
        app = graphics.playback_positions(hidden_states,
                                          observations,
                                          estimated_states,
                                          marginals)
        app.mainloop()


if __name__ == '__main__':
    main()
