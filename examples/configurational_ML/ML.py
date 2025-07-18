import pickle as pickle

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm


def init_weights(pol_types, in_features, hidden_features_1, hidden_features_2):
    w_1 = np.random.normal(
        size=(pol_types, in_features + 1, hidden_features_1), scale=1e-1
    )
    w_2 = np.random.normal(
        size=(pol_types, hidden_features_1 + 1, hidden_features_2), scale=1e-1
    )
    w_3 = np.random.normal(size=(pol_types, hidden_features_2 + 1, 1), scale=1e-1)
    return w_1, w_2, w_3


@jax.jit
def model(x, pol, w_1, w_2, w_3):
    """Neural network architecture for energy prediction.

    The neural network consists of three multi-layer perceptrons.
    For each type of defect (S1A, S0A, VO) a specific MLP is used.
    Each MLP consists of three layers with a leaky-relu activation
    and predicts the contribution of a defect to the total energy
    of system. Finally, the predictions of all MLPs are summed up
    to give the total energy of the system.

    Parameters
    ----------
    x : array
        Descriptors of the polaron configuration of shape
        (n_samples, n_pols, n_features), where n_samples is the
        number of configurations to predict, n_pols the number
        of polarons in each configuration and n_features the
        number of sites that described the local environment
        of each polaron (in our case 6*(2*radius)+1 where radius
        is defined in the function definition of the descriptor).
    pol : array
        This array contains information on the type of defect
        ("vO", "dopant1", "dopant2", "S0H", "S1H", "S1V", "S2H", "S2V")
        and is used to invoke a specific MLP for each defect. Has shape (n_samples, n_pols)
    w_1 : array
        Weights of the first layers of the MLPs with shape
        (3, in_features+1, out_features). Three (in_features x out_features)
        matrices and a bias vector (+1)  for each type of defect
    w_2 : array
        Weights of the second layers of the MLPs with shape
        (3, in_features+1, out_features). Three (in_features x out_features)
        matrices and a bias vector (+1)  for each type of defect
    w_3 : array
        Weights of the third layers of the MLPs with shape
        (3, in_features+1, out_features). Three (in_features x out_features)
        matrices and a bias vector (+1)  for each type of defect

    Returns
    -------
    Energy : float
    """
    x = jax.nn.leaky_relu(jnp.einsum("ijk,ijkl->ijl", x, w_1[pol, :-1]) + w_1[pol, -1])
    x = jax.nn.leaky_relu(jnp.einsum("ijk,ijkl->ijl", x, w_2[pol, :-1]) + w_2[pol, -1])
    x = jnp.einsum("ijk,ijkl->ijl", x, w_3[pol, :-1]) + w_3[pol, -1]
    return jnp.squeeze(jnp.sum(x, axis=1))


def train(
    X_train,
    Y_train,
    pol_train,
    X_test,
    Y_test,
    pol_test,
    initial_weights,
    epochs=10000,
    eta=0.005,
    batch_size=16,
    store_path="parameters/tmp_weights.pkl",
    no_update_epochs=100,
):
    @jax.jit
    def loss(x, pol, y, w1, w2, w3):
        pred = model(x, pol, w1, w2, w3)
        return jnp.mean((pred - y) ** 2)

    import wandb

    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="Polarons on Nb-doped-TiO2",
        # track hyperparameters and run metadata
        config={
            "learning_rate": 0.005,
            "architecture": "Don't Know",
            "dataset": "Polaron",
            "epochs": 50000,
        },
    )

    loss_grad = jax.grad(loss, (3, 4, 5))

    w_1, w_2, w_3 = initial_weights

    test_loss = []
    train_loss = []
    lowest = 1000
    count = 0
    no_update_count = 0

    from time import perf_counter

    start = perf_counter()

    print("#Epoch \t Train \t\t Test \t\t #Updates")

    for epoch in tqdm(range(epochs), total=epochs):
        # shuffle indices of training set
        idxs = np.arange(X_train.shape[0])
        np.random.shuffle(idxs)

        # use stochastic batch gradient descent
        for i in range(0, X_train.shape[0], batch_size):
            grad = loss_grad(
                X_train[idxs[i : i + batch_size]],
                pol_train[idxs[i : i + batch_size]],
                Y_train[idxs[i : i + batch_size]],
                w_1,
                w_2,
                w_3,
            )

            # update regression weights
            w_1 -= eta * grad[0]
            w_2 -= eta * grad[1]
            w_3 -= eta * grad[2]

        # store loss for visualization
        train_loss.append(loss(X_train, pol_train, Y_train, w_1, w_2, w_3))
        test_loss.append(loss(X_test, pol_test, Y_test, w_1, w_2, w_3))

        wandb.log({"train_loss": train_loss[-1], "test_loss": test_loss[-1]})

        # store model if lowest loss on test set
        if test_loss[-1] < lowest:
            count += 1
            lowest = test_loss[-1].copy()
            with open(store_path, "wb") as f:
                pickle.dump([w_1, w_2, w_3], f)
            no_update_count = 0
        else:
            no_update_count += 1

        if no_update_count >= no_update_epochs:
            print(
                "Training halted: No updates in the last {} epochs.".format(
                    no_update_epochs
                )
            )
            break

        if (epoch + 1) % 100 == 0:
            intermediate = perf_counter()
            print(
                "{:d} \t {:.6f} \t {:.6f} \t {} \t {}".format(
                    epoch + 1,
                    np.mean(train_loss[-1:]),
                    np.mean(test_loss[-1:]),
                    count,
                    intermediate - start,
                ),
                flush=True,
            )
            count = 0

    end = perf_counter()
    print(f"Time elapsed: {end - start}")
    # pickle the train and test loss
    with open("train_test_loss.pkl", "wb") as f:
        pickle.dump([train_loss, test_loss], f)

    return train_loss, test_loss, X_train, X_test, Y_train, Y_test, pol_train, pol_test
