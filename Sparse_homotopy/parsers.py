import argparse


class BaseParser():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.parser.add_argument('--imgdir', required=True)
        self.parser.add_argument('--target_class', default=-1, type=int)##changed to -1, doesn't matter for untargeted
        self.parser.add_argument('--batchSize', type=int, default=1)
        self.parser.add_argument('--dec_factor', type=float, default=0.90)##def - 0.90
        self.parser.add_argument('--val_c', type=float, default=7)## default - 3, changing to 7
        self.parser.add_argument('--val_w1', type=float, default=1e-3)
        self.parser.add_argument('--val_w2', type=float, default=1e-5)
        self.parser.add_argument('--val_gamma', default=0.96, type=float)
        self.parser.add_argument('--max_update', default=10, type=int)## default - 200, changing to 10
        self.parser.add_argument('--max_epsilon', default=0.05, type=float)
        self.parser.add_argument('--maxiter', default=10, type=int)
        self.parser.add_argument('--name', type=str, default='homotopy')

    def parse(self):
        args = self.parser.parse_args()
        self.args = args
        return self.args

