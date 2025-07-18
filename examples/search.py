import warnings
from copy import deepcopy

import numpy as np

from configurational_ML.ML import model
from configurational_ML.firat import Descriptor
from misc import generate_move_mask

# Load the data from the database

# suppress warnings
warnings.filterwarnings("ignore")


# with open('parameters/tmp_weights_.pkl', 'rb') as f:
#     weights = pickle.load(f)
#
# with open('dataset/data_firat_reduced.pkl', 'rb') as f:
#     X, pol_type, Y, desc_scaler, en_scaler = pickle.load(f)
#
# db_file = "auto"
# high_level = "polaron_test"
#
# db = VaspDB(db_file, high_level)
#
# results_coll = "results_test"
# data = list(db.find_many_data(collection=results_coll, fltr={}))
#
# df = pd.DataFrame(data)
#
# configurations = df["configuration"].values
#
# config_dict = configurations[0]
#
# poscar_path = 'dataset/POSCAR_pristine'
# base_structure = Structure.from_file(poscar_path)
# cell = base_structure.lattice.abc
# positions = base_structure.cart_coords
#
# partitions = (12, 6, 12)
# cgi = ConfigurationGridInterface(partitions=partitions, base_structure=base_structure)
#
# configuration = cgi.get_grid_from_config_dict(config_dict)


def predict(
    configurations,
    model,
    weights,
    scaler,
    pcas,
    n_comp,
    radius=(6, 4),
    iterations=8,
    alpha=0.1,
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


# seed = 42
# np.random.seed(seed=seed)
#
# desc_size = (6, 3)
#
# with open("pcas.pkl", "rb") as f:
#     pcas = pickle.load(f)
#
# energy = predict(configuration[np.newaxis], model, weights, desc_scaler, pcas=pcas, radius=desc_size)


def move(configuration, masks):
    # if restrict_ov:
    #     masks[2, :, [2, -2], :] = 0
    # if restrict_pol:
    #     masks[0, :, :, [1, 2]] = 0
    #     masks[1, :, :, [0, 2]] = 0
    # masks[0, :, :, 2] = 0
    # masks[1, :, :, 2] = 0
    # masks[2, :, :, :2] = 0

    shape = configuration.shape
    # get indices of defects
    idxs = np.array(configuration.nonzero()).T
    # choose random element to move
    idx = idxs[np.random.randint(0, idxs.shape[0])]

    # shift mask to defect which should be moved
    mask = np.roll(masks[idx[2]], (idx[0], idx[1]), axis=(0, 1))
    # remove already occupied sites from mask
    mask = np.logical_and(mask, np.logical_xor(configuration, mask))

    # select new position
    new_idxs = np.array(mask.nonzero()).T
    if len(new_idxs) != 0:
        new_idx = new_idxs[np.random.randint(0, new_idxs.shape[0])]
    else:
        new_idx = idx

    # remove old defect and put new one
    new_configuration = configuration.copy()
    new_configuration[idx[0], idx[1], idx[2]] = 0
    new_configuration[new_idx[0], new_idx[1], new_idx[2]] = 1

    return np.array([new_configuration])


def simulated_annealing(
    temps,
    configuration,
    model,
    weights,
    desc_scaler,
    pcas,
    n_comp,
    desc_size,
    restrict_ov=True,
    restrict_nb=True,
    restrict_pol=False,
    print_freq=100,
):
    configurations = []
    energies = []

    configurations.append(configuration)
    energy = predict(
        np.array([configuration]),
        model,
        weights,
        desc_scaler,
        pcas,
        n_comp,
        radius=desc_size,
    )[0]
    energies.append(energy.item())

    accepted = 0
    tot = 0

    masks = generate_move_mask(
        (2, 2),
        restrict_ov=restrict_ov,
        restrict_nb=restrict_nb,
        restrict_pol=restrict_pol,
    )
    for i, temp in enumerate(temps):
        tot += 1

        new_configurations = move(configuration, deepcopy(masks))
        new_energies = predict(
            new_configurations,
            model,
            weights,
            desc_scaler,
            pcas,
            n_comp,
            radius=desc_size,
        )
        idx = np.argmax(new_energies)

        # acceptance prob
        prob = np.exp((new_energies[idx] - energy) / temp)
        # metropolis criterion
        if prob > np.random.rand():
            accepted += 1
            configuration = new_configurations[idx].copy()
            energy = new_energies[idx].copy()

        if not np.any(np.all(configuration == configurations, axis=(1, 2, 3))):
            configurations.append(configuration.copy())
            energies.append(energy.item())

        if i % print_freq == 0:
            print(
                i,
                "\t",
                "{:.4e}".format(np.max(energies)),
                "\t",
                "{:.4e}".format(np.mean(energies[-print_freq:])),
                "\t",
                "{:.1e}".format(temp),
                "\t",
                "{:.2e}".format(accepted / tot),
            )

    return configurations, energies, configuration, energy


# proposed_configuration, proposed_energy = simulated_annealing(
#     np.concatenate([np.ones(1000) * 1.6e-5, np.ones(1000) * 1e-5]),
#     configuration,
#     model,
#     weights,
#     desc_scaler,
#     pcas,
#     desc_size,
#     restrict_ov=True,
#     restrict_nb=True,
#     print_freq=100)
#
# poscar_path = 'dataset/POSCAR_pristine'
# base_structure = Structure.from_file(poscar_path)
#
# partitions = (12, 6, 12)
# cgi = ConfigurationGridInterface(partitions=partitions, base_structure=base_structure)
#
# conf_dicts = cgi.get_config_dict_from_grid([proposed_configuration])
