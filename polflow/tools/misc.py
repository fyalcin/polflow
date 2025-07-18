import hashlib
import json
import os
import pickle
from datetime import datetime

from htflow_utils.shaper import Shaper
from monty.io import zopen
from pymatgen.core import Structure
from pymatgen.core.periodic_table import DummySpecies

from polflow.tools.occmat_gen import write_occmat

polaron_rename_dict = {"Ti": DummySpecies("X1"), "Nb": DummySpecies("X2")}


def dict_to_hash(my_dict: dict) -> str:
    """
    Creates a hash from the given dictionary to be used as a unique ID.

    :param my_dict: Dictionary, should not be nested.
    :type my_dict: dict

    :return: Hash value of the given dictionary in string format.
    :rtype: str
    """
    sorted_items = sorted(my_dict.items())
    str_repr = repr(sorted_items).encode("utf-8")
    hash_object = hashlib.md5(str_repr)
    hash_str = hash_object.hexdigest()
    return hash_str


def replace_in_file(file_path, old_str, new_str):
    """
    Replaces a string in a file with a new string.

    :param file_path: Path to the file.
    :type file_path: str

    :param old_str: String to be replaced.
    :type old_str: str

    :param new_str: String to replace the old string with.
    :type new_str: str
    """
    with zopen(file_path, "r") as f:
        file_str = f.read()
    file_str = file_str.replace(old_str, new_str)
    with zopen(file_path, "w") as f:
        f.write(file_str)


def reset_launchpad(launchpad):
    """
    Resets the launchpad by deleting all the workflows and fireworks in it.

    :param launchpad: Launchpad object.
    :type launchpad: LaunchPad
    """
    today = datetime.today().strftime("%Y-%m-%d")
    launchpad.reset(today)
    print("Launchpad reset for date: ", today)


def get_polaron_conf_from_outcar(outcar, mag_threshold=0.5):
    mag_arr = [mag["tot"] for mag in outcar.magnetization]
    polaron_conf = [index for index, mag in enumerate(mag_arr) if mag > mag_threshold]
    return polaron_conf


class ConfigurationGenerator:
    def __init__(self):
        cwd = os.path.dirname(os.path.abspath(__file__))
        occmat_path = os.path.join(cwd, "occmats.pkl")
        with open(occmat_path, "rb") as handle:
            occmat_dict = pickle.load(handle)
        self.occmat_dict = occmat_dict
        self._modifications = []

    @property
    def modifications(self):
        return self._modifications

    @modifications.setter
    def modifications(self, modifications):
        for mod in modifications:
            mod["indices"].sort()
        self._modifications = modifications

    def initialize_site_properties(self, structure):
        mods = self._modifications
        for mod in mods:
            mod_type = mod["type"]
            structure.add_site_property(mod_type, [None] * len(structure))

    def add_modification(
        self, modification_type: str, indices: list = None, element: str = None
    ):
        modification = {
            "type": modification_type,
            "indices": indices,
            "element": element,
        }
        self._modifications.append(modification)

    def get_conf_dict(self):
        conf_dict = {mod["type"]: mod["indices"] for mod in self._modifications}
        return conf_dict

    def get_conf_str(self):
        conf_dict = self.get_conf_dict()
        conf_str = "_".join([f"{k}{v}" for k, v in conf_dict.items()])
        return conf_str

    def assign_polaron_types(self, structure, r=2.50):
        occmat_dict = self.occmat_dict
        neighbors_all = structure.get_all_neighbors(r=r)
        layers = Shaper.get_layers(structure, tol=0.45)

        # sort the layers by z
        layers = dict(sorted(layers.items(), reverse=True))

        layers_by_species = {
            k: [structure[site_index].species_string for site_index in v]
            for k, v in layers.items()
        }
        layers_ti = [k for k, v in layers_by_species.items() if "Ti" in v]

        ti_by_layer = {}
        for layer_index, layer in enumerate(layers_ti):
            for site_index in layers[layer]:
                ti_by_layer[site_index] = layer_index

        polaron_types = []
        for site_index, site in enumerate(structure):
            if site.species_string == "O":
                polaron_types.append(None)
                continue
            polaron_type = f"S{ti_by_layer[site_index]}"

            neighbors = neighbors_all[site_index]
            abs_diff_z_arr = [abs(site.z - neighbor.z) for neighbor in neighbors]
            # if any of the abs_diff_z_arr is greater than 1.5, then it is an A site
            if any([abs_diff_z > 1.5 for abs_diff_z in abs_diff_z_arr]):
                polaron_type += "H"
            else:
                polaron_type += "V"

            if polaron_type in occmat_dict:
                polaron_types.append(polaron_type)
            else:
                polaron_types.append(None)

        structure.add_site_property("polaron_type", polaron_types)

    def decorate_structure(self, structure):
        self.initialize_site_properties(structure=structure)
        self.assign_polaron_types(structure=structure)
        # iterate through the modifications and add site properties to the structure
        for modification in self._modifications:
            modification_type = modification["type"]
            indices = modification["indices"]
            site_property = structure.site_properties.get(modification_type, [])
            if modification_type == "dopant":
                element: str = modification["element"]
                for index in indices:
                    site_property[index] = element
            elif modification_type == "polaron":
                for index in indices:
                    site_property[index] = structure[index].properties["polaron_type"]
            elif modification_type == "vacancy":
                for index in indices:
                    site_property[index] = True

            structure.add_site_property(modification_type, site_property)
        # add an original index property to the structure
        original_indices = list(range(len(structure)))
        structure.add_site_property("original_index", original_indices)

    def apply_modifications(
        self,
        structure: Structure,
        to_file=False,
        sort_structure=True,
        group_polarons=True,
        rename_polarons=True,
    ):
        if "vacancy" in structure.site_properties:
            indices_to_remove = [
                i for i, v in enumerate(structure.site_properties["vacancy"]) if v
            ]
            structure.remove_sites(indices_to_remove)

        if "dopant" in structure.site_properties:
            indices_to_replace = [
                i for i, v in enumerate(structure.site_properties["dopant"]) if v
            ]
            for index in indices_to_replace:
                site_props = structure[index].properties
                structure.replace(
                    idx=index, species=site_props["dopant"], properties=site_props
                )

        if sort_structure:
            structure.sort(
                key=lambda x: {"O": 1, "Nb": 2, "Ti": 3}.get(x.species_string, 6)
            )

        if group_polarons and "polaron" in structure.site_properties:
            structure.sort(key=lambda x: bool(x.properties["polaron"]))

        if rename_polarons:
            if "polaron" in structure.site_properties:
                polaron_indices = [
                    i for i, v in enumerate(structure.site_properties["polaron"]) if v
                ]
                for index in polaron_indices:
                    site = structure[index]
                    structure.replace(
                        idx=index,
                        species=polaron_rename_dict[site.species_string],
                        properties=site.properties,
                    )

        if to_file:
            write_occmat(structure=structure, occmat_dict=self.occmat_dict)


# class PolaronStructure(Structure):
#     # let's use a setter to set the polaron indices
#     @property
#     def polaron_indices(self):
#         return self._polaron_indices
#
#     @polaron_indices.setter
#     def polaron_indices(self, indices):
#         self._polaron_indices = sorted(indices)
#
#     def assign_polaron_types(self):
#         # get the directory of this file
#         cwd = os.path.dirname(os.path.abspath(__file__))
#         occmat_path = os.path.join(cwd, "occmats.pkl")
#         with open(occmat_path, 'rb') as handle:
#             occmat_dict = pickle.load(handle)
#
#         # first, decorate the structure
#         assign_polaron_types(self, occmat_dict=occmat_dict)
#         return self


def check_input(input_dict: dict, required_keys: list[str]) -> dict:
    """
    Checks if the input dictionary contains all the required keys and replaces the ones
    missing with the default values.

    :param input_dict: Dictionary containing all the input parameters needed
        to start a subworkflow.
    :type input_dict: dict

    :param required_keys: List of required keys.
    :type required_keys: list[str]

    :return: Input dictionary with the required keys.
    :rtype: dict
    """
    defaults = load_defaults("polflow")
    input_params_dict = {}
    for key in required_keys:
        default = defaults.get(key, None)
        user_input = input_dict.get(key)
        try:
            input_params_dict[key] = {**default, **user_input}
        except TypeError:
            input_params_dict[key] = user_input or default
    return input_params_dict


def load_defaults(module_name):
    """
    Loads the defaults from a module's defaults.json file and updates them

    :param module_name: name of the module
    :type module_name: str

    :return: defaults
    :rtype: dict
    """
    config_file = os.environ.get("FW_CONFIG_FILE")
    config_dir = os.path.dirname(config_file) if config_file else None
    try:
        with open(os.path.join(config_dir, "user_defaults.json")) as f:
            user_defaults = json.load(f)
    except (FileNotFoundError, TypeError):
        user_defaults = {}

    # import the module and get the path
    module = __import__(module_name)
    module_path = os.path.dirname(module.__file__)
    json_path = os.path.join(module_path, "defaults.json")
    with open(json_path, "r") as f:
        defaults_dict = json.load(f)
    defaults_dict.update(user_defaults)
    return defaults_dict
