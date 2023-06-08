import time


class Timer:

    def __init__(self):
        self.timer_dict = {}
        self.stop_dict = {}

    def tic(self, name):
        self.timer_dict[name] = time.time()

    def toc(self, name):
        assert name in self.timer_dict
        elapsed = time.time() - self.timer_dict[name]
        del self.timer_dict[name]
        return elapsed

    def stop(self, name):
        self.stop_dict[name] = time.time()

    def resume(self, name):
        if name not in self.timer_dict:
            del self.stop_dict[name]
            return
        elapsed = time.time() - self.stop_dict[name]
        self.timer_dict[name] = self.timer_dict[name] + elapsed
        del self.stop_dict[name]
