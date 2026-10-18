import numpy as np
from scipy import sparse
from .ials import IALS


class RatingRecencyALS(IALS):
    def __init__(self, half_life=730, **kwargs):
        super().__init__(**kwargs)
        self.half_life = half_life

    def make_target(self, frame, shape, reference_time):
        age = (reference_time - frame.timestamp.to_numpy()) / 86400
        if (age < 0).any():
            raise ValueError('Training event occurs after recommendation time')
        recency = .5 + .5 * np.exp(-np.log(2) * age / self.half_life)
        values = (frame.rating.to_numpy() - 3) / 2 * recency
        return sparse.csr_matrix((values, (frame.user, frame.item)), shape=shape)

    def fit(self, x, frame, reference_time):
        return super().fit(x, self.make_target(frame, x.shape, reference_time))
