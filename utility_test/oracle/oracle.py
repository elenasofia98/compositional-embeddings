class Oracle:
    """
    Class that hold ordered info from oracle file.
    The Oracle keeps info about a generic couple of variables (first and second)
    and the value the Oracle associates to the couple
    """
    def __init__(self):
        self.correlations = {}

    def add_correlations(self, value, first, second):
        self.correlations[len(self.correlations)] = {'value': value, 'first': first, 'second': second}


class POSAwareOracle(Oracle):
    """
    The POSAwareOracle keeps info about POS tag for words
    """
    def __init__(self):
        super(POSAwareOracle, self).__init__()

    def add_correlations(self, value, first, second, target_pos, w1_pos, w2_pos):
        self.correlations[len(self.correlations)] = {'value': value, 'first': first, 'second': second,
                                                     'target_pos': target_pos, 'w1_pos': w1_pos, 'w2_pos': w2_pos}