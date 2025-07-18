import pickle

import numpy as np
from scipy.signal import convolve2d


def extract_data(path, cell, partitions):
    configurations = []
    energies = []

    with open(path, "rb") as f:
        data = pickle.load(f)

    for x, y, mag, defect in zip(data["pos"], data["en"], data["mag"], data["defect"]):

        # transform coordinates to grid and fill with magnetization
        # of Ti sites
        interface = Cell_grid_interface(cell, positions=x, partitions=partitions)
        interface.get_indices(x, frac=[0.4, 0.1, -0.1])
        configuration = interface.fill_grid(mag[:240], (0, 240))[:, :, 3:6]

        # transform coordinates to grid and fill with occupation matrices
        interface = Cell_grid_interface(
            cell, (partitions[0], partitions[1], 10), (2, 5, 5)
        )
        interface.get_indices(x, frac=[0.4, 0.1, -0.1])

        # similarly transform coordinates of oxygen vacancies to grid
        interface = Cell_grid_interface(
            cell,
            (partitions[0], partitions[1], 1),
        )
        idx = interface.get_indices(defect, frac=0.1)
        for i in idx:
            configuration[i[0], i[1], -1] = 1

        configurations.append(configuration)
        energies.append(y)

        print(len(configurations), " extracted", end="\r")

    # detect polarons where magnetization is greater than 0.5
    configurations = np.array(configurations)
    configurations = np.abs(configurations) > 0.5
    energies = np.array(energies)
    return configurations, energies


def extract_data_firat(positions, energies, mags, cell, partitions=(12, 8)):
    configurations = []
    energies = []
    #
    # with open(path, 'rb') as f:
    #     data = pickle.load(f)

    for x, y, mag in zip(positions, energies, mags):
        # transform coordinates to grid and fill with magnetization
        # of Ti sites
        interface = Cell_grid_interface(cell, (partitions[0], partitions[1], 10))
        interface.get_indices(x, frac=[0.4, 0.1, -0.1])
        configuration = interface.fill_grid(mag[432:], (0, 216))[:, :, :]

        # transform coordinates to grid and fill with occupation matrices
        # interface = Cell_grid_interface(cell, (partitions[0], partitions[1], 10), (2, 5, 5))
        # interface.get_indices(x, frac=[0.4, 0.1, -0.1])

        # similarly transform coordinates of oxygen vacancies to grid
        # interface = Cell_grid_interface(cell, (partitions[0], partitions[1], 1), )
        # idx = interface.get_indices(defect, frac=0.1)
        # for i in idx:
        #     configuration[i[0], i[1], -1] = 1

        configurations.append(configuration)
        energies.append(y)

        print(len(configurations), " extracted", end="\r")

    # detect polarons where magnetization is greater than 0.5
    configurations = np.array(configurations)
    configurations = np.abs(configurations) > 0.5
    energies = np.array(energies)
    return configurations, energies


def augment(configurations, energies):
    new_conf = []
    new_en = []
    for configuration, energy in zip(configurations, energies):
        # identity
        new_conf.append(configuration)
        new_en.append(energy)

        # mirror x
        new_conf.append(np.roll(configuration[::-1, :], -1, axis=0))
        new_en.append(energy)

        # mirror y
        new_conf.append(np.roll(configuration[:, ::-1], -1, axis=1))
        new_en.append(energy)

        # mirror x and y
        new_conf.append(np.roll(configuration[::-1, ::-1], (-1, -1), axis=(0, 1)))
        new_en.append(energy)

    configurations = np.array(new_conf, dtype=float)
    energies = np.array(new_en)
    return configurations, energies


class Scaler:
    def __init__(self, energies, invert_sign=True):
        self.invert = invert_sign
        tmp = energies
        if self.invert:
            tmp = -tmp
        self.min, self.max = np.nanmin(tmp), np.nanmax(tmp)

    def scale(self, energies):
        scaled = energies
        if self.invert:
            scaled = -scaled
        scaled = (scaled - self.min) / (self.max - self.min)
        return scaled

    def unscale(self, predictions):
        unscaled = predictions * (self.max - self.min) + self.min
        if self.invert:
            unscaled = -unscaled
        return unscaled


def convert_to_desc(
    configurations,
    energies,
    desc_scaling=None,
    en_scaling=None,
    size=(6, 4),
    diffuse=True,
    iterations=8,
    alpha=0.1,
    n_desc=12,
):
    X = []
    pol_type = []
    Y = energies

    for i, configuration in enumerate(configurations):
        d = Descriptor(configuration, size)
        feature, pol = d.get_descs(diffuse=diffuse, iterations=iterations, alpha=alpha)
        rescale_factor = feature.shape[0] // n_desc
        if rescale_factor != 1:
            d, idxs, c = np.unique(
                feature,
                return_index=True,
                return_counts=True,
                axis=0,
            )
            feature = np.repeat(d, c // rescale_factor, axis=0)
            pol = np.repeat(np.array(pol)[idxs], c // rescale_factor, axis=0)
        X.append(feature)
        pol_type.append(pol)
        print(i, "/", len(configurations), end="\r")
    X = np.array(X)
    pol_type = np.array(pol_type)

    if desc_scaling is not None:
        desc_scaler = desc_scaling
    else:
        desc_scaler = Scaler(X.flatten(), invert_sign=False)
    X = desc_scaler.scale(X)

    if en_scaling is not None:
        en_scaler = en_scaling
    else:
        en_scaler = Scaler(
            Y,
        )
    Y = en_scaler.scale(Y)

    return X, pol_type, Y, desc_scaler, en_scaler


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

        def checkerboard(shape):
            return np.array(np.indices(shape).sum(axis=0) % 2, dtype=bool)

        def make_mask(shape):
            mask = checkerboard(shape)
            mask[:, 1::2, 0] = 0
            mask[:, 1::2, 1] = 0
            mask[:, 0::2, 2] = 0
            return mask

        mask_S1 = np.roll(make_mask(self.shape), -1, axis=0)
        mask_S1[1 + self.width[0] : -self.width[0], :, :] = 0
        mask_S1[:, 1 + self.width[1] : -self.width[1], :] = 0

        mask_S0 = make_mask(self.shape)
        mask_S0[1 + self.width[0] : -self.width[0], :, :] = 0
        mask_S0[:, 1 + self.width[1] : -self.width[1], :] = 0

        mask_VO = np.roll(make_mask(self.shape), -1, axis=1)
        mask_VO[1 + self.width[0] : -self.width[0], :, :] = 0
        mask_VO[:, 1 + self.width[1] : -self.width[1], :] = 0

        return np.array([mask_S1, mask_S0, mask_VO], dtype=bool)

    def get_descs(self, diffuse=False, iterations=3, alpha=0.2):
        descs = []
        types = []
        idxs = np.array((self.config == 1).nonzero()).T

        # clean idxs
        idxs = idxs[idxs[:, 0] < self.shape_init[0]]
        idxs = idxs[idxs[:, 1] < self.shape_init[1]]

        if diffuse:
            config = self.diffuse(self.config, iterations, alpha)
        else:
            config = self.config.copy()

        for idx in idxs:
            tmp = np.roll(config, (-idx[0], -idx[1]), axis=(0, 1))
            descs.append(tmp[self.mask[idx[2]]])
            types.append(idx[2])

        descs = [np.resize(d, np.max(self.mask.sum(axis=(0, 1, 2)))) for d in descs]

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
                    for i in range(3)
                ],
                0,
                2,
            )

        return diffused


class Cell_grid_interface:
    """Assignment of cartesian coordinates and associated values
    to discretized grid.

    Partitioning of cell of given dimensions into discretized bins.
    The discretized bins can additionally be linked to the cartesian
    coordinates of sites in the cell, such that values corresponding
    to specific sites (e.g., the site magnetization) can be filled in
    the grid

    Attributes
    ----------
    cell : array
        dimensions of the cell with 3 dimesions
    grid : array
        the discretized grid which can be filled with values
        corresponding to the values of sites within the grid cell
    bins : list
        list with three entries that correspond to the bins and their
        spatial extent in direct coordinates of the corresponding cell
    dx : array
        length of the bins in the x, y and z dimension
    indices : None
        Stores the indices of the grid cell corresponding to a cartesian
        coordinate if self.get_indices(...) is called

    Parameters
    ----------
    cell : array
        dimensions of the cell with 3 dimesions
    partitions : tuple
        tuple consisting of three integers that define the partitioning
        of the cell

    Methods
    -------
    get_indices(positions, shift=True, frac=0)
        Returns the indices of grid cells of positions
    fill_grid(values, sites)
        Assigns values to given sites stored in indices

    """

    def __init__(
        self, cell: np.array, positions, partitions: tuple, value_shape: tuple = ()
    ):
        self.cell = cell
        self.positions = positions
        self.grid = np.zeros(partitions + value_shape)
        self.bins = [
            np.linspace(0, 1, partitions[0] + 1),
            np.linspace(0, 1, partitions[1] + 1),
            np.linspace(
                0,
                1,
                partitions[2] + 1,
            ),
        ]
        self.dx = np.array([[tmp[1] - tmp[0] for tmp in self.bins]])
        self.indices = None

    def get_indices(self, shift=True, frac=0.0):
        """Gives the indices of grid cells in which cartesian
        coordinates of positions lie.

        Parameters
        ----------
        positions : array
            (n, 3) shaped array, where n is the number of points
            and 3 corresponds to the cartesian coordinates
        shift : bool, default=True
            Whether cartesian coordinates are shifted relative to
            their origin. Might be needed for alignment purposes.
        frac : float or list of floats
            Fraction of grid cell size that the positions should
            be shifted

        Returns
        -------
        indices : array
            Contains the indices of the grid cells in which entry of
            positions lies.
        """
        positions = self.positions / self.cell
        positions += self.dx * frac
        positions = np.where(positions > 1, positions - 1, positions)
        self.indices = np.array(
            [
                np.digitize(position, bin_)
                for position, bin_ in zip(positions.T, self.bins)
            ]
        ).T
        self.indices -= 1
        return self.indices

    def fill_grid(self, values, sites=(0, -1)):
        """Assigns values to grid.

        Uses indices determined in get_indices to fill values into grid.
        For example take a list of cartesian coordinates of sites in the
        simulation cell and a list of their corresponding site
        magnetizations. By using get_indices, the grid cells corresponding
        to each site are determined and with fill_grid the corresponding
        magnetizations can be assigned to the grid cells.

        Parameters
        ----------
        values : array
            values that should be assigned to the indices.
        sites : tuple, default=(0,-1)
            can be used to only assign value to a subset of indices
        Returns
        -------
        grid : array
            The grid filled with the values
        """
        for index, value in zip(self.indices[sites[0] : sites[1]], values):
            self.grid[tuple(index)] = value
        return self.grid
