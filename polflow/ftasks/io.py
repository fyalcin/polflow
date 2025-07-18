import os

from atomate.common.firetasks.glue_tasks import get_calc_loc
from atomate.utils.fileio import FileClient
from fireworks import explicit_serialize, FiretaskBase, FWAction
from pymatgen.io.vasp import Poscar, Potcar, Incar, Kpoints

from polflow.ftasks.vasp import POTCAR_MAP, INPUT_TEMPLATES_PATH, LDA_MAP, occmat_dict
from polflow.tools.misc import replace_in_file, polaron_rename_dict
from polflow.tools.occmat_gen import write_occmat


@explicit_serialize
class WritePoscar(FiretaskBase):
    _fw_name = "WritePoscar"
    required_params = ["calc_type"]
    optional_params = ["structure", "from_prev_calc"]

    def run_task(self, fw_spec):
        calc_type = self.get("calc_type")
        from_prev_calc = self.get("from_prev_calc", False)
        if from_prev_calc:
            prev_calc_loc = get_calc_loc(from_prev_calc, fw_spec["calc_locs"])
            poscar = Poscar.from_file(os.path.join(prev_calc_loc["path"], "CONTCAR"))
        else:
            structure = self.get("structure")
            if calc_type == "pol_rel_mlff":
                polaron_indices = [
                    i for i, v in enumerate(structure.site_properties["polaron"]) if v
                ]
                tmp_struct = structure.copy()
                for index in polaron_indices:
                    site = structure[index]
                    tmp_struct.replace(
                        idx=index,
                        species=polaron_rename_dict[site.species_string],
                        properties=site.properties,
                    )
            else:
                tmp_struct = structure
            poscar = Poscar(structure=tmp_struct)

        poscar.write_file("POSCAR")

        if calc_type.endswith("mlff"):
            replace_in_file("POSCAR", "X1", "T2")
            replace_in_file("POSCAR", "X2", "N2")

        return FWAction(update_spec=fw_spec)


@explicit_serialize
class WritePrevCalcLoc(FiretaskBase):
    _fw_name = "WritePrevCalcLoc"
    required_params = ["prev_calc_name"]
    optional_params = []

    def run_task(self, fw_spec):
        prev_calc_name = self.get("prev_calc_name")

        prev_calc_loc = get_calc_loc(prev_calc_name, fw_spec["calc_locs"])
        print("-" * 50)
        print(prev_calc_loc)
        print("-" * 50)

        return FWAction(update_spec=fw_spec)


@explicit_serialize
class WriteInputs(FiretaskBase):
    _fw_name = "WriteInputs"
    required_params = ["structure", "calc_type"]
    optional_params = ["user_incar_settings", "magmom_arr"]

    def run_task(self, fw_spec):
        structure = self.get("structure")
        calc_type = self.get("calc_type")
        magmom_arr = self.get("magmom_arr", None)
        user_incar_settings = self.get("user_incar_settings", None)
        self.write_inputs(
            structure=structure,
            calc_type=calc_type,
            user_incar_settings=user_incar_settings,
            magmom_arr=magmom_arr,
        )

        return FWAction(update_spec=fw_spec)

    @staticmethod
    def write_inputs(structure, calc_type, user_incar_settings=None, magmom_arr=None):
        if user_incar_settings is None:
            user_incar_settings = {}
        if magmom_arr is None:
            magmom_arr = []
        if calc_type == "pol_rel_mlff":
            polaron_indices = [
                i for i, v in enumerate(structure.site_properties["polaron"]) if v
            ]
            tmp_struct = structure.copy()
            for index in polaron_indices:
                site = structure[index]
                tmp_struct.replace(
                    idx=index,
                    species=polaron_rename_dict[site.species_string],
                    properties=site.properties,
                )
        else:
            tmp_struct = structure
        poscar = Poscar(structure=tmp_struct)
        potcar = Potcar(symbols=[POTCAR_MAP[el] for el in poscar.site_symbols])
        incar = Incar.from_file(
            os.path.join(INPUT_TEMPLATES_PATH, f"INCAR_{calc_type}")
        )
        incar["LDAUL"] = [LDA_MAP["LDAUL"][el] for el in poscar.site_symbols]
        incar["LDAUU"] = [LDA_MAP["LDAUU"][el] for el in poscar.site_symbols]
        incar["LDAUJ"] = [LDA_MAP["LDAUJ"][el] for el in poscar.site_symbols]
        incar.update(user_incar_settings)
        potcar.write_file("POTCAR")

        if magmom_arr:
            incar["MAGMOM"] = magmom_arr

        incar.write_file("INCAR")
        if calc_type.endswith("mlff"):
            fc = FileClient()
            replace_in_file("INCAR", "Run", "run")
            dest = os.getcwd()
            fc.copy(
                os.path.join(INPUT_TEMPLATES_PATH, "ML_FF.SVD"),
                os.path.join(dest, "ML_FF"),
            )

        kpoints = Kpoints(comment="Gamma centered k-points")
        kpoints.write_file("KPOINTS")

        if calc_type.endswith("occmat"):
            write_occmat(structure=structure, occmat_dict=occmat_dict)
