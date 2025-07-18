import numpy as np

from .ML import model
from .dataset import Descriptor


def move(configuration, restrict_ov=False, restrict_pol=False, random=0.0):
    if np.random.rand() < random:
        masks = Descriptor(
            configuration,
            (int(configuration.shape[0] / 2 - 1), int(configuration.shape[1] / 2 - 1)),
        ).mask
    # generate masks for allowed movement of defects
    else:
        masks = Descriptor(configuration, (2, 2)).mask
        if restrict_ov:
            masks[2, :, [2, -2], :] = 0
        if restrict_pol:
            masks[0, :, :, [1, 2]] = 0
            masks[1, :, :, [0, 2]] = 0
    masks[0, :, :, 2] = 0
    masks[1, :, :, 2] = 0
    masks[2, :, :, :2] = 0

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


def predict(
    configurations, model, weights, scaler, radius=(6, 4), iterations=8, alpha=0.1
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

    # predict and return mean of all symmetrically equivalent configurations
    # energy
    return (
        model(descriptors, pol_type, weights[0], weights[1], weights[2])
        .reshape(-1, 4)
        .mean(axis=1)
    )


def generate_random_config(size, c_S1, seed=1234):
    np.random.seed(seed=seed)

    def checkerboard(shape):
        return np.array(np.indices(shape).sum(axis=0) % 2, dtype=bool)

    def make_mask(shape):
        mask = checkerboard(shape)
        mask[:, 1::2, 0] = 0
        mask[:, 1::2, 1] = 0
        mask[:, 0::2, 2] = 0
        return mask

    config = np.zeros((size[0] * 2, size[1] * 2, 3))
    n_VO = int(round(1 / 6 * size[0] * size[1], 0))
    n_pol = 2 * n_VO
    n_S1 = int(round(n_pol * c_S1))
    n_S0 = n_pol - n_S1

    mask = make_mask(config.shape)
    for i, (n) in enumerate([n_S1, n_S0, n_VO]):
        idxs = np.array(mask[:, :, i].nonzero()).T
        choice = np.random.choice(np.arange(idxs.shape[0]), n, replace=False)
        idxs = idxs[choice]
        config[idxs[:, 0], idxs[:, 1], i] = 1
    return config


def sim_anneal(
    configuration,
    weights,
    desc_scaler,
    seed=1234,
    temps=np.ones(5) * 0.01,
    restrict_ov=False,
    restrict_pol=False,
    random=0.01,
    print_freq=100,
):
    np.random.seed(seed=seed)

    accepted = 0
    tot = 0

    configurations = []
    energies = []

    desc_size = (6, 4)

    energy = predict(
        configuration[np.newaxis], model, weights, desc_scaler, radius=desc_size
    )
    configurations.append(configuration)
    energies.append(energy.item())

    for i, temp in enumerate(temps):
        tot += 1

        new_configurations = move(
            configuration,
            restrict_ov=restrict_ov,
            restrict_pol=restrict_pol,
            random=random,
        )
        new_energies = predict(
            new_configurations, model, weights, desc_scaler, radius=desc_size
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
                seed,
                "\t",
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

    return np.array(configurations), np.array(energies)
