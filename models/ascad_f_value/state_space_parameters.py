output_states = 256  # Number of Classes/Output layer size
input_size = 700  # Amount of measurements before they enter network/input layer size

layer_limit = 14  # Max number of layers

# Transition Options
possible_conv_depths = [2, 4, 8, 16, 32, 64, 128]  # Choices for number of filters in a convolutional layer
possible_conv_sizes = [1, 2, 3, 25, 50, 75, 100]  # Choices for kernel size
possible_pool_sizes = [2, 4, 7, 25, 50, 75, 100]  # Choices for filter_size for an average pooling layer
possible_pool_strides = possible_pool_sizes  # Choices for stride for an average pooling layer
max_fc = 3  # Maximum number of fully connected layers (excluding final FC layer for softmax output)
# Possible number of neurons in a fully connected layer
possible_fc_sizes = [2, 4, 10, 15, 20, 30]

allow_initial_pooling = False  # Allow pooling as the first layer
init_utility = 0.3  # Set this to around the performance of an average model. It is better to undershoot this
allow_consecutive_pooling = False  # Allow a pooling layer to follow a pooling layer

conv_padding = 'SAME'  # set to 'SAME' (recommended) to pad convolutions so input and output dimension are the same
# set to 'VALID' to not pad convolutions


# Epsilon schedule for q learning agent.
# Format : [[epsilon, # unique models]]
# Epsilon = 1.0 corresponds to fully random, 0.0 to fully greedy
epsilon_schedule = [[1.0, 1500],
                    [0.9, 100],
                    [0.8, 100],
                    [0.7, 100],
                    [0.6, 150],
                    [0.5, 150],
                    [0.4, 150],
                    [0.3, 150],
                    [0.2, 150],
                    [0.1, 150]]

# Q-Learning Hyper parameters
# Q Learning omega polynomial parameter (α = 1 / t^ω) where t is the iteration step and α is the learning rate from Eq 3
# This learning rate was based on theoretical and experimental results (Even-Dar and Mansour, 2003)
learning_rate_omega = 0.85
discount_factor = 1.0  # Q Learning discount factor (gamma from Equation 3)
replay_number = 128  # Number trajectories to sample for replay at each iteration
