
from collections import namedtuple
import random

import numpy as np


LinearRange = namedtuple("LinearRange", ["start", "end"])
LogRange = namedtuple("LogRange", ["start", "end"])


ranges = {
  "embedding_dim": set([32, 64, 128]),
  "policy_dims": set(["32", "64", "128"]),
  "critic_dims": set(["", "32", "64", "128", "32,32", "64,64"]),

  "embedding_init_range": LinearRange(0.01, 0.1),

  "batch_size": set([32, 64, 128, 256]),
  "policy_lr": LogRange(0.000001, 0.1),
  "critic_lr": LogRange(0.000001, 0.1),
  "gamma": LinearRange(0.95, 0.99)
}


def make_params():
  params = {}
  for key, spec in ranges.items():
    if isinstance(spec, set):
      params[key] = random.choice(list(spec))
    elif isinstance(spec, (LogRange, LinearRange)):
      if isinstance(spec[0], int):
        # LogRange not valid for ints
        params[key] = random.randint(spec[0], spec[1])
      elif isinstance(spec[0], float):
        start, end = spec
        if isinstance(spec, LogRange):
          start, end = np.log10(start), np.log10(end)

        choice = np.random.uniform(start, end)

        if isinstance(spec, LogRange):
          choice = np.exp(choice)

        params[key] = choice

  return params


if __name__ == "__main__":
  params = make_params()
  for param, value in params.items():
    print "--%s=%s" % (param, value)
