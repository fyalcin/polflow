import numpy as np
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from polflow.tools.misc import ConfigurationGenerator


class PolaronicStructure(Structure):
    def __init__(
        self,
        structure: Structure,
        vacancy_indices: list = None,
        dopant_indices: list = None,
        polaron_indices: list = None,
        sort_structure: bool = True,
        group_polarons: bool = True,
        rename_polarons: bool = True,
        **kwargs,
    ) -> None:
        confgen = ConfigurationGenerator()
        mods = [
            {"type": "vacancy", "indices": vacancy_indices},
            {"type": "dopant", "indices": dopant_indices, "element": "Nb"},
            {"type": "polaron", "indices": polaron_indices},
        ]
        self.mods = mods
        confgen.modifications = mods
        confgen.decorate_structure(structure=structure)
        self.decorated_structure = structure.copy()
        confgen.apply_modifications(
            structure=structure,
            to_file=False,
            sort_structure=sort_structure,
            group_polarons=group_polarons,
            rename_polarons=rename_polarons,
        )
        self.confgen = confgen
        self.polaron_types = structure.site_properties["polaron_type"]
        self.original_indices = structure.site_properties["original_index"]
        self.vacancies = structure.site_properties.get("vacancy", [])
        self.dopants = structure.site_properties.get("dopant", [])
        self.polarons = structure.site_properties.get("polaron", [])

        super().__init__(
            lattice=structure.lattice,
            species=structure.species,
            coords=structure.frac_coords,
            coords_are_cartesian=True,
            site_properties=structure.site_properties,
            to_unit_cell=False,
            validate_proximity=False,
            **kwargs,
        )


def get_equivalent_configurations(base_structure, config_dict, tol=1e-2):
    sga = SpacegroupAnalyzer(base_structure)

    symm_ops = sga.get_symmetry_operations()
    frac_coords = base_structure.frac_coords.round(3) % 1

    coords_dict = {}
    for defect_type, indices in config_dict.items():
        coords_dict[defect_type] = []
        for site in indices:
            coords_dict[defect_type].append(frac_coords[site])

    equivalent_confs = []
    for op in symm_ops:
        indices_dict = {}
        for defect_type, coords in coords_dict.items():
            symm_coords = [op.operate(coord).round(3) % 1 for coord in coords]
            defect_indices = []
            for coord in symm_coords:
                # get the index of the vacancy site by searching for the vacancy site with the same coordinates as
                # coord in the numpy array of fractional coordinates
                matches = np.all(np.abs(frac_coords - coord) < tol, axis=1)
                defect_indices.append(int(np.where(matches)[0][0]))
            indices_dict[defect_type] = sorted(defect_indices)

        equivalent_confs.append(indices_dict)

    return equivalent_confs
