from .state_enumerator import State, StateEnumerator


class StateStringUtils(object):
    """ Contains all functions dealing with converting nets to net strings
        and net strings to state lists.
    """

    def __init__(self, state_space_parameters):
        self.input_size = state_space_parameters.input_size
        self.output_number = state_space_parameters.output_states
        self.enum = StateEnumerator(state_space_parameters)

    @staticmethod
    def add_drop_out_states(state_list):
        """ Add drop out every 2 layers and after each fully connected layer
        Sets dropout rate to be between 0 and 0.5 at a linear rate
        """
        new_state_list = []
        number_fc = len([state for state in state_list if state.layer_type == 'fc'])
        number_gap = len([state for state in state_list if state.layer_type == 'gap'])
        number_drop_layers = (len(state_list) - number_gap - number_fc) / 2 + number_fc
        drop_number = 1
        for i in range(len(state_list)):
            new_state_list.append(state_list[i])
            if ((((i + 1) % 2 == 0 and i != 0) or state_list[i].layer_type == 'fc')
                    and state_list[i].terminate != 1
                    and state_list[i].layer_type != 'gap'
                    and drop_number <= number_drop_layers):
                drop_state = state_list[i].copy()
                drop_state.dropout_rate = 0.5 * float(drop_number) / number_drop_layers
                drop_state.layer_type = 'dropout'
                drop_number += 1
                new_state_list.append(drop_state)

        return new_state_list

    @staticmethod
    def remove_drop_out_states(state_list):
        new_state_list = []
        for state in state_list:
            if state.layer_type != 'dropout':
                new_state_list.append(state)
        return new_state_list

    def state_list_to_string(self, state_list):
        """Convert the list of strings to a string we can train from according to the grammar"""
        strings = []
        i = 0
        while i < len(state_list):
            state = state_list[i]
            if self.state_to_string(state):
                strings.append(self.state_to_string(state))
            i += 1
        return str('[' + ', '.join(strings) + ']')

    @staticmethod
    def state_to_string(state):
        """ Returns the string asociated with state.
        """
        if state.terminate == 1:
            return 'SM(%i)' % state.fc_size
        elif state.layer_type == 'start':
            return 'I(%i,1)' % state.feature_size
        elif state.layer_type == 'conv':
            return 'C(%i,%i,%i)' % (state.filter_depth, state.filter_size, state.stride)
        elif state.layer_type == 'gap':
            return 'GAP(%i)' % state.feature_size
        elif state.layer_type == 'pool':
            return 'P(%i,%i)' % (state.filter_size, state.stride)
        elif state.layer_type == 'fc':
            return 'FC(%i)' % state.fc_size
        elif state.layer_type == 'batch_norm':
            return 'BN'
        elif state.layer_type == 'flatten':
            return 'FLAT(%i)' % state.feature_size
        elif state.layer_type == 'dropout':
            return 'D(%i)' % state.dropout_ratio
        return None

    def convert_model_string_to_states(self, parsed_list, start_state=None):
        """Takes a parsed model string and returns a recursive list of states."""

        states = [start_state] if start_state else [State('start', 0, 1, 0, 0, self.input_size, 0, 0, False)]

        for layer in parsed_list:
            if layer[0] == 'conv':
                states.append(State(layer_type='conv',
                                    layer_depth=states[-1].layer_depth + 1,
                                    filter_depth=layer[1],
                                    filter_size=layer[2],
                                    stride=layer[3],
                                    feature_size=states[-1].feature_size,
                                    fc_size=0,
                                    terminate=False))
            elif layer[0] == 'gap':
                states.append(State(layer_type='gap',
                                    layer_depth=states[-1].layer_depth + 1,
                                    filter_depth=0,
                                    filter_size=0,
                                    stride=0,
                                    feature_size=layer[1],
                                    fc_size=0,
                                    terminate=False))
            elif layer[0] == 'pool':
                states.append(State(layer_type='pool',
                                    layer_depth=states[-1].layer_depth + 1,
                                    filter_depth=0,
                                    filter_size=layer[1],
                                    stride=layer[2],
                                    feature_size=self.enum.calc_new_feature_size(states[-1].feature_size, layer[1],
                                                                                 layer[2]),
                                    fc_size=0,
                                    terminate=False))
            elif layer[0] == 'fc':
                states.append(State(layer_type='fc',
                                    layer_depth=states[-1].layer_depth + 1,
                                    filter_depth=len([state for state in states if state.layer_type == 'fc']),
                                    filter_size=0,
                                    stride=0,
                                    feature_size=0,
                                    fc_size=layer[1],
                                    terminate=False))
            elif layer[0] == 'dropout':
                drop_state = states[-1].copy()
                drop_state.dropout_rate = layer[1]
                drop_state.layer_type = 'dropout'
                states.append(drop_state)
            elif layer[0] == 'bn':
                states.append(State(layer_type='batch_norm',
                                    layer_depth=states[-1].layer_depth + 1,
                                    feature_size=states[-1].feature_size,
                                    stride=0))
            elif layer[0] == 'flat':
                states.append(State(layer_type='flatten',
                                    layer_depth=states[-1].layer_depth + 1,
                                    filter_depth=0,
                                    filter_size=0,
                                    stride=0,
                                    feature_size=layer[1],
                                    fc_size=0))
            elif layer[0] == 'softmax':
                termination_state = states[-1].copy() if states[-1].layer_type != 'dropout' else states[-2].copy()
                termination_state.layer_type = 'softmax'
                termination_state.terminate = True
                termination_state.feature_size = 0
                termination_state.fc_size = self.output_number
                termination_state.layer_depth += 1
                states.append(termination_state)

        return states
