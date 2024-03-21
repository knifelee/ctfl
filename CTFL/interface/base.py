from utils import config
from utils.args import args
import time
from interface.valuation import Valuation
from abc import ABC, abstractmethod


class Base(ABC):
    def __init__(self, method, v: Valuation):
        self.start_time = time.process_time()
        self.scores = [0 for _ in range(config.NUM_PARTS)]
        self.method = method
        self.v = v

    # compute the scores of all participants
    @abstractmethod
    def phi(self):
        raise 'error to execute the abstract phi method'

    def eval_score(self, mode):
        rank = sorted(range(len(self.scores)), key=lambda i: self.scores[i], reverse=True)
        print(f"Participants' contribution scores by {self.method}_{mode}, in descending order")
        for i in rank:
            print(f'PID-{i}, score = {self.scores[i]:.2f}')
        print(f'\nContribution estimation time cost is {self.time_cost:.0f} seconds')

    def run(self):
        self.scores_dict = self.phi()
        self.time_cost = time.process_time() - self.start_time
        for mode, self.scores in self.scores_dict.items():
            self.eval_score(mode)
