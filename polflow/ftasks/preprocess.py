import pickle

import numpy as np
from fireworks import FiretaskBase, FWAction
from fireworks import explicit_serialize

from viktor.configurational_ML.dataset import extract_data, augment, convert_to_desc


@explicit_serialize
class ExtractData(FiretaskBase):
    _fw_name = "FT_ExtractData"
    required_params = ["load_path", "poscar_path", "save_path"]
    optional_params = []

    def run_task(self, fw_spec):
        load_path = self.get("load_path")
        poscar_path = self.get("poscar_path")

        cell = np.diag(np.loadtxt(poscar_path, skiprows=2, max_rows=3))

        print("-" * 50)
        print("Extracting data:")
        configurations, energies = extract_data(load_path, cell, partitions=(12, 8))
        print(len(configurations), " extracted")

        save_path = self.get("save_path")
        data = [configurations, energies]
        with open(save_path, "wb") as f:
            pickle.dump(data, f)

        fw_spec.update({"save_path": save_path})
        return FWAction(update_spec=fw_spec)


@explicit_serialize
class AugmentData(FiretaskBase):
    _fw_name = "FT_AugmentData"
    required_params = []
    optional_params = []

    def run_task(self, fw_spec):
        save_path = fw_spec["save_path"]
        load_path = save_path
        with open(load_path, "rb") as f:
            data = pickle.load(f)

        configurations, energies = data

        print("-" * 50)
        print("Augmenting data:")
        configurations, energies = augment(configurations, energies)
        print(len(configurations), " configurations after augmentation")

        data = [configurations, energies]
        with open(save_path, "wb") as f:
            pickle.dump(data, f)

        fw_spec.update({"save_path": save_path})
        return FWAction(update_spec=fw_spec)


@explicit_serialize
class ConvertToDescriptor(FiretaskBase):
    _fw_name = "Convert the configurations to descriptors"
    required_params = []
    optional_params = []

    def run_task(self, fw_spec):
        save_path = fw_spec["save_path"]
        load_path = fw_spec["save_path"]
        with open(load_path, "rb") as f:
            data = pickle.load(f)

        configurations, energies = data

        print("-" * 50)
        print("Calculating Descriptor:")
        x, pol_type, y, desc_scaler, en_scaler = convert_to_desc(
            configurations, energies
        )

        print("-" * 50)
        print("Storing Data")
        with open(save_path, "wb") as f:
            pickle.dump([x, pol_type, y, desc_scaler, en_scaler], f)

        return FWAction(update_spec=fw_spec)
