import numpy as np
from scipy.signal import convolve2d


class Descriptor:
    def __init__(self, configuration, width):
        self.config = configuration
        self.shape_init = configuration.shape
        self.width = width
        self.initial_num = configuration.sum()
        # fix size of config if smaller than width of cutoff
        for i, (w, s) in enumerate(zip(self.width, self.shape_init)):
            if s < 2 * w + 1:
                self.config = np.tile(
                    self.config,
                    (1 if j != i else -((2 * w + 1) // -s) for j in range(3)),
                )
        self.shape = self.config.shape
        # setup mask
        self.mask = self.setup_mask()

    def setup_mask(self):

        mask = np.ones(self.shape)
        mask[1 + self.width[0] : -self.width[0], :, :] = 0
        mask[:, 1 + self.width[1] : -self.width[1], :] = 0
        return np.array([mask for _ in range(8)], dtype=bool)

    def get_descs(self, diffuse=False, iterations=3, alpha=0.2):
        descs = []
        types = []
        idxs = np.array((self.config == 1).nonzero()).T

        # clean idxs
        idxs = idxs[idxs[:, 0] < self.shape_init[0]]
        idxs = idxs[idxs[:, 1] < self.shape_init[1]]

        if diffuse:
            self.config = self.diffuse(self.config, iterations, alpha)
        else:
            self.config = self.config.copy()

        for idx in idxs:
            tmp = np.roll(self.config, (-idx[0], -idx[1]), axis=(0, 1))
            descs.append(tmp[self.mask[idx[2]]])
            types.append(idx[2])

        # descs = [np.resize(d, np.max(self.mask.sum(axis=(0, 1, 2)))) for d in descs]

        return np.array(descs, dtype=float), types

    def diffuse(self, config, iterations, alpha):
        diffused = config.copy()

        kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])

        for _ in range(iterations):
            diffused += np.moveaxis(
                [
                    convolve2d(
                        diffused[:, :, i], alpha * kernel, mode="same", boundary="wrap"
                    )
                    for i in range(8)
                ],
                0,
                2,
            )

        return diffused
