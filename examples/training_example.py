import pickle
import subprocess

import numpy as np
import pandas as pd
from fireworks import LaunchPad
from pymatgen.core.structure import Structure
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from caching import cache_results
from configurational_ML.ML import init_weights, train
from configurational_ML.dataset import augment
from misc import ConfigurationGridInterface
from misc import convert_to_desc, get_grid_from_config_dict_sep
from polflow.database.db_tools import VaspDB
from polflow.tools.misc import ConfigurationGenerator
from search import model


def word_in_file(word, file_path):
    try:
        output = subprocess.check_output(f"grep -c '{word}' {file_path}", shell=True)
        count = int(output.decode().strip())
        return count > 0
    except subprocess.CalledProcessError:
        return False


def get_pca_num_components(X, n_desc, pol_type):
    num_components = 0
    for i in range(n_desc):
        pca = PCA()
        j, k = (pol_type == i).nonzero()
        print(X[j, k].shape)

        pca.fit(X[j, k])
        x_99 = np.isclose(
            np.cumsum(pca.explained_variance_ratio_), 0.99, atol=1e-3
        ).nonzero()[0][0]
        num_components = max(num_components, x_99)
        print(num_components)
    num_components = int(1.05 * num_components)
    return num_components


def reduce_with_pca(X, n_desc, num_components, pol_type):
    X_pca = np.zeros((X.shape[0], X.shape[1], num_components))
    pcas = []
    for i in range(n_desc):
        pca = PCA(n_components=num_components)
        j, k = (pol_type == i).nonzero()
        pca.fit(X[j, k])
        transformed = pca.transform(X[j, k])
        X_pca[j, k] = transformed
        pcas.append(pca)
    return X_pca, pcas


@cache_results("train_model.pkl")
def train_model(
    X_train,
    Y_train,
    pol_train,
    X_test,
    Y_test,
    pol_test,
    num_components,
    epochs=50000,
    eta=0.005,
    batch_size=16,
    store_path="parameters/tmp_weights.pkl",
    no_update_epochs=2000,
):
    initial_weights = init_weights(8, num_components, 30, 10)

    res = train(
        X_train,
        Y_train,
        pol_train,
        X_test,
        Y_test,
        pol_test,
        initial_weights,
        epochs=epochs,
        eta=eta,
        batch_size=batch_size,
        store_path=store_path,
        no_update_epochs=no_update_epochs,
    )

    with open(store_path, "rb") as weights_file:
        weights = pickle.load(weights_file)

    return res, weights


def calculate_mse(X, Y, pol_type, weights):
    predictions = model(X, pol_type, *weights)
    mse = np.mean((predictions - Y) ** 2)
    return mse


# Define the path to the database file
db_file = "db.json"
# Define the name of the high level database
high_level = "polaron_test"

# Initialize a database query object
db = VaspDB(db_file, high_level)

# Define the collection from which to extract the data
results_coll = "results_test"

# We need the base structure to be able to create the cell grid interface
poscar_path = "dataset/POSCAR_pristine"
base_structure = Structure.from_file(poscar_path)

# Create the cell grid interface using the base structure and the partitions that will be used to discretize the cell
partitions = (12, 6, 12)
cgi = ConfigurationGridInterface(partitions=partitions, base_structure=base_structure)

# Initialize the launchpad
lpad = LaunchPad.auto_load()

with open("unique_deloc_confs.pkl", "rb") as f:
    all_structures = pickle.load(f)

deloc_structs_df = pd.DataFrame(all_structures)

conf_arr_random = [
    {"type": "vacancy", "indices": [0, 5]},
    {"type": "dopant", "indices": [473], "element": "Nb"},
]

confgen = ConfigurationGenerator()
confgen.modifications = conf_arr_random
confgen.decorate_structure(base_structure)
possible_polaronic_sites = [
    index
    for index, site in enumerate(base_structure)
    if site.properties.get("polaron_type")
]

n_desc = 8
desc_size = (6, 3)
diffuse = True
iterations = 8
alpha = 0.1

try:
    df_saved = pd.read_pickle("active_learning_df.pkl")
except FileNotFoundError:
    df_saved = pd.DataFrame()

try:
    with open("active_uuids.pkl", "rb") as f:
        active_annealing_uuids, active_regular_uuids = pickle.load(f)
except FileNotFoundError:
    active_annealing_uuids = []
    active_regular_uuids = []

# let's get the configurations already in the dataframe, or an empty list if the dataframe is empty
try:
    configurations = df_saved.configuration.tolist()
except AttributeError:
    configurations = []

# Query the data from the database all except those whose metadata.uuid fields are in the active uuids and those
# with configurations already in the dataframe
data = list(
    db.find_many_data(
        collection=results_coll,
        fltr={
            "status": "success",
            "configuration": {"$nin": configurations},
            "metadata.uuid": {"$nin": active_annealing_uuids + active_regular_uuids},
        },
        projection={
            "path": 1,
            "configuration": 1,
            "final_energy": 1,
            "status": 1,
            "metadata": 1,
        },
    )
)

# add the queried data to the dataframe
# df = pd.concat([df, pd.DataFrame(data)], ignore_index=True)
df = pd.DataFrame(data)

# check if metadata column exists, if not, create it
if "metadata" not in df.columns:
    df["metadata"] = [{} for _ in range(df.shape[0])]

# Perform an additional check to see if the required accuracy was reached
df["required_accuracy_reached"] = df.path.apply(
    lambda x: word_in_file("required accuracy", f"{x}/vasp.out")
)

# Only keep the rows where the required accuracy was reached
df = df[df.required_accuracy_reached]

grid_configurations = []
# Iterate over the rows of the dataframe
for i, row in tqdm(df.iterrows(), total=df.shape[0]):
    # Get the configuration array from the configuration dictionary
    grid_configuration = get_grid_from_config_dict_sep(
        base_structure=base_structure,
        conf_dict=row.configuration,
        partitions=partitions,
    )

    grid_configurations.append(grid_configuration)

df["grid_configuration"] = grid_configurations

# print(f"Processed {len(configurations)} configurations.")
print(f"Processed {df.shape[0]} configurations.")

# # Convert the list of configurations to a numpy array
# configurations = np.array(configurations)
#
# # Create a mask to filter out the configurations that do not contain 8 defects
# mask = configurations.sum(axis=(1, 2, 3)) == 8
# # Apply the mask to the configurations
# configurations = configurations[mask]

# also apply the same filter to the dataframe and drop the rows that do not contain 8 defects
df["num_defects"] = df.grid_configuration.apply(lambda x: x.sum())
df = df[df.num_defects == 8]

# print(f"Of which {len(configurations)} configurations contain 8 defects.")
print(f"Of which {df.shape[0]} configurations contain 8 defects.")

# # Extract the final energies from the dataframe and filter them using the mask
# energies = np.array(df.final_energy.tolist())
# energies = np.array(energies)[mask]

# Augment the configurations and energies by adding the inverse and mirror configurations
# configurations, energies = augment(configurations, energies)

augmented_confs_list = []
augmented_energies_list = []
for index, row in df.iterrows():
    configuration = row.grid_configuration
    energy = row.final_energy
    augmented_confs, augmented_energies = augment([configuration], [energy])
    augmented_confs_list.append(augmented_confs)
    augmented_energies_list.append(augmented_energies)

# set the augmented configurations and energies for the missing indices
df["augmented_configurations"] = augmented_confs_list
df["augmented_energies"] = augmented_energies_list

# print(f"Augmented the dataset to {len(configurations)} configurations.")
# the total number of elements in the augmented configurations gives the total number of configurations
print(
    f"Augmented the dataset to {df.augmented_configurations.apply(len).sum()} configurations."
)

# concatenate the two dataframes
combined_df = pd.concat([df_saved, df], ignore_index=True)

configs_all = []
energies_all = []
for index, row in tqdm(combined_df.iterrows(), total=combined_df.shape[0]):
    configs_all.extend(row.augmented_configurations)
    energies_all.extend(row.augmented_energies)

# convert to numpy arrays
configs_all = np.array(configs_all)
energies_all = np.array(energies_all)

_, _, _, desc_scaler, en_scaler = convert_to_desc(
    configs_all,
    energies_all,
    desc_scaling=None,
    en_scaling=None,
    desc_size=desc_size,
    diffuse=diffuse,
    iterations=iterations,
    alpha=alpha,
    n_desc=n_desc,
)

# pickle the en_scaler and desc_scaler
with open("parameters/scalers.pkl", "wb") as f:
    pickle.dump([desc_scaler, en_scaler], f)

# set the empty metadata values to an empty dictionary
df.metadata = df.metadata.apply(lambda x: {} if pd.isna(x) else x)

annealed_train_confs = []
annealed_train_energies = []
annealed_test_confs = []
annealed_test_energies = []
regular_confs = []
regular_energies = []
regular_conf_dicts = []
train_conf_dicts = []
test_conf_dicts = []
for index, row in tqdm(df.iterrows(), total=df.shape[0]):
    source = row.metadata.get("source", "regular")
    if source in ["simulated_annealing", "phase_2", "phase_3"]:
        annealed_train_confs.extend(row.augmented_configurations[:3])
        annealed_train_energies.extend(row.augmented_energies[:3])
        annealed_test_confs.extend(row.augmented_configurations[3:])
        annealed_test_energies.extend(row.augmented_energies[3:])
        train_conf_dicts.extend([row.configuration] * 3)
        test_conf_dicts.extend([row.configuration])
    else:
        regular_confs.extend(row.augmented_configurations)
        regular_energies.extend(row.augmented_energies)
        regular_conf_dicts.extend(
            [row.configuration] * len(row.augmented_configurations)
        )

# train_test_split the regular configurations and energies
try:
    (
        regular_train_confs,
        regular_test_confs,
        regular_train_energies,
        regular_test_energies,
        regular_train_conf_dicts,
        regular_test_conf_dicts,
    ) = train_test_split(
        regular_confs, regular_energies, regular_conf_dicts, test_size=0.20
    )
except ValueError:
    regular_train_confs = regular_confs
    regular_train_energies = regular_energies
    regular_test_confs = []
    regular_test_energies = []
else:
    train_conf_dicts.extend(regular_train_conf_dicts)
    test_conf_dicts.extend(regular_test_conf_dicts)

# try to open train_test_conf_dicts.pkl and load the data from it
try:
    with open("parameters/train_test_conf_dicts.pkl", "rb") as f:
        a, b = pickle.load(f)
except FileNotFoundError:
    a = []
    b = []
a.extend(train_conf_dicts)
b.extend(test_conf_dicts)
with open("parameters/train_test_conf_dicts.pkl", "wb") as f:
    pickle.dump([a, b], f)

# concatenate the annealed and regular configurations and energies
if annealed_train_confs and regular_train_confs:
    train_confs = np.concatenate([annealed_train_confs, regular_train_confs])
    train_energies = np.concatenate([annealed_train_energies, regular_train_energies])
elif regular_train_confs:
    train_confs = np.array(regular_train_confs)
    train_energies = np.array(regular_train_energies)
else:
    train_confs = np.array(annealed_train_confs)
    train_energies = np.array(annealed_train_energies)

if annealed_test_confs and regular_test_confs:
    test_confs = np.concatenate([annealed_test_confs, regular_test_confs])
    test_energies = np.concatenate([annealed_test_energies, regular_test_energies])
elif regular_test_confs:
    test_confs = np.array(regular_test_confs)
    test_energies = np.array(regular_test_energies)
else:
    test_confs = np.array(annealed_test_confs)
    test_energies = np.array(annealed_test_energies)

# convert the configurations to descriptors
X_train_up, pol_train_up, Y_train_up, _, _ = convert_to_desc(
    train_confs,
    train_energies,
    desc_scaling=desc_scaler,
    en_scaling=en_scaler,
    desc_size=desc_size,
    diffuse=diffuse,
    iterations=iterations,
    alpha=alpha,
    n_desc=n_desc,
)

X_test_up, pol_test_up, Y_test_up, _, _ = convert_to_desc(
    test_confs,
    test_energies,
    desc_scaling=desc_scaler,
    en_scaling=en_scaler,
    desc_size=desc_size,
    diffuse=diffuse,
    iterations=iterations,
    alpha=alpha,
    n_desc=n_desc,
)

# let's check if a parameters/train_test_data.pkl file exists and load the data from it. If it contains data,
# we add the unprocessed data to the training and test data
# otherwise, we create the file and add the unprocessed data to the training and test data
try:
    with open("parameters/train_test_data.pkl", "rb") as f:
        X_train, X_test, Y_train, Y_test, pol_train, pol_test = pickle.load(f)
except FileNotFoundError:
    X_train = X_train_up
    X_test = X_test_up
    Y_train = Y_train_up
    Y_test = Y_test_up
    pol_train = pol_train_up
    pol_test = pol_test_up
else:
    # concatenate the unprocessed data to the processed data only if the unprocessed data is not empty
    if X_train_up.size != 0:
        X_train = np.concatenate([X_train, X_train_up])
        Y_train = np.concatenate([Y_train, Y_train_up])
        pol_train = np.concatenate([pol_train, pol_train_up])
        X_test = np.concatenate([X_test, X_test_up])
        Y_test = np.concatenate([Y_test, Y_test_up])
        pol_test = np.concatenate([pol_test, pol_test_up])

# pickle the training and test data
with open("parameters/train_test_data.pkl", "wb") as f:
    pickle.dump([X_train, X_test, Y_train, Y_test, pol_train, pol_test], f)

num_components = get_pca_num_components(X_train, n_desc, pol_train)
X_train_pca, pcas = reduce_with_pca(X_train, n_desc, num_components, pol_train)

# pickle the pca components
with open("parameters/pca_components.pkl", "wb") as f:
    pickle.dump(pcas, f)

# let's apply the pca to the test data
X_test_pca = np.zeros((X_test.shape[0], X_test.shape[1], num_components))
for i in range(n_desc):
    transformed = pcas[i].transform(X_test[pol_test == i])
    X_test_pca[pol_test == i] = transformed

# pickle the reduced training data and the principal components
with open("parameters/train_test_data_reduced.pkl", "wb") as f:
    pickle.dump([X_train_pca, X_test_pca, Y_train, Y_test, pol_train, pol_test], f)

print(
    f"Training the model with {X_train_pca.shape[0]} configurations, first training before annealing."
)
res, weights = train_model(
    X_train=X_train_pca,
    Y_train=Y_train,
    pol_train=pol_train,
    X_test=X_test_pca,
    Y_test=Y_test,
    pol_test=pol_test,
    epochs=50000,
    num_components=num_components,
    no_update_epochs=3000,
)

test_predictions = model(X_test_pca, pol_test, *weights)

# calculate the mean squared error
test_mse = np.mean((test_predictions - Y_test) ** 2)
