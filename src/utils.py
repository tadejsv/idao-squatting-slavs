from dataclasses import dataclass


@dataclass
class ExponentialAverage:
    beta: float = 0.98
    running_avg: float = None

    def __call__(self, new_val: float) -> float:

        if self.running_avg is None:
            self.running_avg = new_val
        else:
            self.running_avg = self.running_avg * self.beta + new_val * (1 - self.beta)

        return self.running_avg
