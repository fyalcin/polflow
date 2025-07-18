from collections import defaultdict

import numpy as np
from scipy.signal import convolve2d

from caching import cache_results
from configurational_ML.dataset import Cell_grid_interface
from polarflow.tools.misc import ConfigurationGenerator


def get_polaron_grid(structure, polaron_type, interface, partitions):
    interface.get_indices(frac=[0.4, 0.1, -0.1])
    polaron_indices = [
        index
        for index, value in enumerate(structure.site_properties["polaron"])
        if value == polaron_type
    ]
    configuration_polaron = np.zeros(partitions)
    for polaron_index in polaron_indices:
        idx = interface.indices[polaron_index]
        configuration_polaron[idx[0], idx[1], idx[2]] = 1
    return configuration_polaron


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

    @staticmethod
    def diffuse(config, iterations, alpha):
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


import pickle


def generate_move_mask(width, restrict_ov=False, restrict_nb=False, restrict_pol=False):
    # load the base_move_mask from base_move_mask.pkl
    with open("base_move_masks.pkl", "rb") as f:
        base_move_masks = pickle.load(f)

    # make movement local
    base_move_masks[:, 1 + width[0] : -width[0], :, :] = False
    base_move_masks[:, :, 1 + width[1] : -width[1], :] = False

    # base_move_masks is a 4D array. First dimension is the defect index, second and third are the x and y
    # coordinates, and the fourth is the layer index.
    # now, we need to restrict the movement of the defects to the other positions
    if restrict_ov:
        # oxygen vacancy is the first defect, and we only want to keep it fixed to the
        # 0, 0 x,y coordinate and the 0th layer
        base_move_masks[0, 1:, :, :] = 0
        base_move_masks[0, :, 1:, :] = 0
        base_move_masks[0, :, :, 1:] = 0

    if restrict_nb:
        # Nb is the second and the third defect, and we only want to keep it fixed to the
        # 0, 0 x,y coordinate and either the first or the second layer, depending on the defect index
        base_move_masks[1, 1:, :, :] = 0
        base_move_masks[1, :, 1:, :] = 0
        base_move_masks[1, :, :, 0] = 0
        base_move_masks[1, :, :, 2] = 0

        base_move_masks[2, 1:, :, :] = 0
        base_move_masks[2, :, 1:, :] = 0
        base_move_masks[2, :, :, 0] = 0
        base_move_masks[2, :, :, 1] = 0

    if restrict_pol:
        # polaron is the fourth, fifth, sixth, seventh, and eighth defect, but this time we don't restrict x and y,
        # only the layer, and layer restriction is as follows: 4th layer to 4th, 5th layer to 5th and 6th,
        # 6th layer to 5th and 6th, 7th layer to 7th and 8th, 8th layer to 7th and 8th
        base_move_masks[3, :, :, :3] = 0
        base_move_masks[3, :, :, 4:] = 0

        base_move_masks[4, :, :, :4] = 0
        base_move_masks[4, :, :, 6:] = 0

        base_move_masks[5, :, :, :4] = 0
        base_move_masks[5, :, :, 6:] = 0

        base_move_masks[6, :, :, :6] = 0
        base_move_masks[7, :, :, :6] = 0

    return base_move_masks


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
    desc_size=(6, 4),
    diffuse=True,
    iterations=8,
    alpha=0.1,
    n_desc=8,
):
    X = []
    pol_type = []
    Y = energies

    for i, configuration in enumerate(configurations):
        d = Descriptor(configuration, desc_size)
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
        # print(i, '/', len(configurations), end='\r')

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


class ConfigurationGridInterface:
    def __init__(self, partitions, base_structure):
        self.partitions = partitions
        self.base_structure = base_structure
        self.positions = base_structure.cart_coords
        self.cell = base_structure.lattice.abc

    @cache_results("grid_from_config_dict.pkl")
    def get_grid_from_config_dict(self, conf_dict):
        base_structure = self.base_structure
        cell = base_structure.lattice.abc
        positions = self.positions

        interface = Cell_grid_interface(cell, positions, self.partitions)

        vacancy_indices = conf_dict["vacancy"]
        dopant_indices = conf_dict["dopant"]
        polaron_indices = conf_dict["polaron"]

        conf_arr = [
            {"type": "vacancy", "indices": vacancy_indices},
            {"type": "dopant", "indices": dopant_indices, "element": "Nb"},
            {"type": "polaron", "indices": polaron_indices},
        ]
        confgen = ConfigurationGenerator()
        confgen.modifications = conf_arr
        structure = base_structure.copy()
        confgen.decorate_structure(structure)

        configuration = {}
        # vO configuration
        interface.get_indices(positions, frac=0.1)
        configuration_vo = np.zeros(self.partitions)
        for vo_index in conf_dict["vacancy"]:
            idx = interface.indices[vo_index]
            configuration_vo[idx[0], idx[1], idx[2]] = 1
        configuration["vO"] = configuration_vo

        # dopant configuration
        configuration_dopant = np.zeros(self.partitions)
        for dopant_index in conf_dict["dopant"]:
            idx = interface.indices[dopant_index]
            configuration_dopant[idx[0], idx[1], idx[2]] = 1
        configuration["dopant"] = configuration_dopant

        all_polaron_types = ["S0H", "S1H", "S1V", "S2H", "S2V"]

        for polaron_type in all_polaron_types:
            configuration_polaron = get_polaron_grid(
                structure, polaron_type, interface, self.partitions
            )
            configuration[polaron_type] = configuration_polaron

        conf_vO = np.asarray(configuration["vO"])[:, :, 9:10]
        conf_dopant = np.asarray(configuration["dopant"])[:, :, 6:8]
        conf_s0h = np.asarray(configuration["S0H"])[:, :, 8:9]
        conf_s1h = np.asarray(configuration["S1H"])[:, :, 7:8]
        conf_s1v = np.asarray(configuration["S1V"])[:, :, 7:8]
        conf_s2h = np.asarray(configuration["S2H"])[:, :, 6:7]
        conf_s2v = np.asarray(configuration["S2V"])[:, :, 6:7]
        self.conf_grid_dict = {
            "vO": configuration["vO"],
            "dopant": configuration["dopant"],
            "S0H": configuration["S0H"],
            "S1H": configuration["S1H"],
            "S1V": configuration["S1V"],
            "S2H": configuration["S2H"],
            "S2V": configuration["S2V"],
        }
        # concatenate all configurations
        conf_concat = np.concatenate(
            [conf_vO, conf_dopant, conf_s0h, conf_s1h, conf_s1v, conf_s2h, conf_s2v],
            axis=-1,
        )

        return conf_concat

    def get_config_dict_from_grid(self, config_grids):
        base_structure = self.base_structure
        cell = base_structure.lattice.abc
        positions = self.positions
        partitions = self.partitions
        defect_to_layer = {
            "vO": [9],
            "dopant1": [6],
            "dopant2": [7],
            "S0H": [8],
            "S1H": [7],
            "S1V": [7],
            "S2H": [6],
            "S2V": [6],
        }
        config_grid_layers = [
            "vO",
            "dopant1",
            "dopant2",
            "S0H",
            "S1H",
            "S1V",
            "S2H",
            "S2V",
        ]

        conf_dicts = []

        for index, grid in enumerate(config_grids):
            conf_grid_dict = {}
            # then, we will fill the array with the configurations from the config_grid
            for i, defect in enumerate(config_grid_layers):
                conf_grid = np.zeros((12, 6, 12))
                layers = defect_to_layer[defect]
                for layer in layers:
                    conf_grid[:, :, layer] = grid[:, :, i]

                conf_grid_dict[defect] = conf_grid

            site_indices = [index for index, site in enumerate(base_structure)]
            defect_dict = defaultdict(list)
            interface = Cell_grid_interface(cell, positions, partitions)
            # turn conf_grid_dict values into boolean arrays

            for defect_type, grid in conf_grid_dict.items():
                shift = (
                    0.1
                    if defect_type in ["vO", "dopant1", "dopant2"]
                    else [0.4, 0.1, -0.1]
                )
                interface.get_indices(positions, frac=shift)
                valgrid = interface.fill_grid(values=site_indices)
                defect_dict[defect_type].extend(sorted(valgrid[grid == 1].tolist()))
                # turn the floats into integers
                defect_dict[defect_type] = [
                    int(site) for site in defect_dict[defect_type]
                ]

            conf_dict_reconstructed = {
                "vacancy": sorted(defect_dict["vO"]),
                "dopant": [*defect_dict["dopant1"], *defect_dict["dopant2"]],
                "polaron": sorted(
                    [
                        *defect_dict["S0H"],
                        *defect_dict["S1H"],
                        *defect_dict["S1V"],
                        *defect_dict["S2H"],
                        *defect_dict["S2V"],
                    ]
                ),
            }
            conf_dicts.append(conf_dict_reconstructed)

        return conf_dicts


@cache_results("grid_from_config_dict_sep.pkl")
def get_grid_from_config_dict_sep(base_structure, conf_dict, partitions):
    cell = base_structure.lattice.abc
    positions = base_structure.cart_coords

    interface = Cell_grid_interface(cell, positions, partitions)

    vacancy_indices = conf_dict.get("vacancy", [])
    dopant_indices = conf_dict.get("dopant", [])
    polaron_indices = conf_dict.get("polaron", [])

    # conf_arr = [
    #     {"type": "vacancy", "indices": vacancy_indices},
    #     {"type": "dopant", "indices": dopant_indices, "element": "Nb"},
    #     {"type": "polaron", "indices": polaron_indices},
    # ]
    conf_arr = []
    if vacancy_indices:
        conf_arr.append({"type": "vacancy", "indices": vacancy_indices})
    if dopant_indices:
        conf_arr.append({"type": "dopant", "indices": dopant_indices, "element": "Nb"})
    if polaron_indices:
        conf_arr.append({"type": "polaron", "indices": polaron_indices})

    confgen = ConfigurationGenerator()
    confgen.modifications = conf_arr
    structure = base_structure.copy()
    confgen.decorate_structure(structure)

    configuration = {}
    # vO configuration
    interface.get_indices(positions, frac=0.1)
    configuration_vo = np.zeros(partitions)
    for vo_index in vacancy_indices:
        idx = interface.indices[vo_index]
        configuration_vo[idx[0], idx[1], idx[2]] = 1
    configuration["vO"] = configuration_vo

    # dopant configuration
    configuration_dopant = np.zeros(partitions)
    for dopant_index in dopant_indices:
        idx = interface.indices[dopant_index]
        configuration_dopant[idx[0], idx[1], idx[2]] = 1
    configuration["dopant"] = configuration_dopant

    all_polaron_types = ["S0H", "S1H", "S1V", "S2H", "S2V"]

    for polaron_type in all_polaron_types:
        configuration_polaron = get_polaron_grid(
            structure, polaron_type, interface, partitions
        )
        configuration[polaron_type] = configuration_polaron

    conf_vO = np.asarray(configuration["vO"])[:, :, 9:10]
    conf_dopant = np.asarray(configuration["dopant"])[:, :, 6:8]
    conf_s0h = np.asarray(configuration["S0H"])[:, :, 8:9]
    conf_s1h = np.asarray(configuration["S1H"])[:, :, 7:8]
    conf_s1v = np.asarray(configuration["S1V"])[:, :, 7:8]
    conf_s2h = np.asarray(configuration["S2H"])[:, :, 6:7]
    conf_s2v = np.asarray(configuration["S2V"])[:, :, 6:7]
    # concatenate all configurations
    conf_concat = np.concatenate(
        [conf_vO, conf_dopant, conf_s0h, conf_s1h, conf_s1v, conf_s2h, conf_s2v],
        axis=-1,
    )

    return conf_concat


def predict(
    configurations, model, weights, scaler, pcas, radius=(6, 4), iterations=8, alpha=0.1
):
    """Uses model to predict energy of intermediate configuration
    representation.

    Parameters
    ----------
    configuration : array
        Intermediate configuration representation (24,6)
    model : model
        Above defined model function
    weights : list
        List containing the weights of the model.
    augment : bool, default=True
        Whether augmentation should be used to enhance prediction
        quality
    radius : int, default=4
        Employed radius in local environment descriptor calculation

    Returns
    -------
    Energy : float
        Energy of the give configuration or mean of all symmetrically
        equivalent configurations.
    """
    configs = []
    # augment data to account for all symmetrically
    # equivalent configurations
    for configuration in configurations:
        configs.append(configuration)
        configs.append(np.roll(configuration[::-1, :], -1, axis=0))
        configs.append(np.roll(configuration[:, ::-1], -1, axis=1))
        configs.append(np.roll(configuration[::-1, ::-1], (-1, -1), axis=(0, 1)))
    # calculate descriptors
    descriptors = []
    pol_type = []
    for conf in configs:
        d = Descriptor(conf, radius)
        feature, pol = d.get_descs(diffuse=True, iterations=iterations, alpha=alpha)
        descriptors.append(feature)
        pol_type.append(pol)
    descriptors = np.array(descriptors)
    descriptors = scaler.scale(descriptors)
    pol_type = np.array(pol_type)

    n_comp = 120

    # create a new array with the same dimensions as X, except the third dimension is n_comp
    X_pca = np.zeros((descriptors.shape[0], descriptors.shape[1], n_comp))
    for index, pca in enumerate(pcas):
        j, k = (pol_type == index).nonzero()
        try:
            transformed = pca.transform(descriptors[j, k])
            X_pca[j, k] = transformed
        except ValueError:
            pass
        # remap the transformed array to the original indices and place it in X_pca

    descriptors = X_pca
    # predict and return mean of all symmetrically equivalent configurations
    # energy
    return (
        model(descriptors, pol_type, weights[0], weights[1], weights[2])
        .reshape(-1, 4)
        .mean(axis=1)
    )
