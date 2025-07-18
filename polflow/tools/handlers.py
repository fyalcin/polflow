import numpy as np
from custodian.ansible.actions import FileActions
from custodian.ansible.interpreter import Modder
from custodian.custodian import ErrorHandler
from custodian.utils import backup
from custodian.vasp.handlers import VASP_BACKUP_FILES
from custodian.vasp.interpreter import VaspModder
from pymatgen.io.vasp.inputs import VaspInput
from pymatgen.io.vasp.outputs import Outcar, Vasprun

from polflow.tools.misc import get_polaron_conf_from_outcar


class UnconvergedErrorHandlerCustom(ErrorHandler):
    """Check if a run is converged."""

    is_monitor = False

    def __init__(self, output_filename="vasprun.xml"):
        """
        Initializes the handler with the output file to check.

        Args:
            output_vasprun (str): Filename for the vasprun.xml file. Change
                this only if it is different from the default (unlikely).
        """
        self.output_filename = output_filename

    def check(self):
        """Check for error."""
        try:
            v = Vasprun(self.output_filename)
            if not v.converged:
                return True
        except Exception:
            pass
        return False

    def correct(self):
        """Perform corrections."""
        v = Vasprun(self.output_filename)
        algo = v.incar.get("ALGO", "Normal").lower()
        actions = []
        if not v.converged_electronic:
            # NOTE: This is the amin error handler
            # Sometimes an AMIN warning can appear with large unit cell dimensions, so we'll address it now
            if (
                np.max(v.final_structure.lattice.abc) > 50.0
                and v.incar.get("AMIN", 0.1) > 0.01
            ):
                actions.append({"dict": "INCAR", "action": {"_set": {"AMIN": 0.01}}})

            if (
                v.incar.get("ISMEAR", -1) >= 0
                or not 50 <= v.incar.get("IALGO", 38) <= 59
                and v.incar.get("METAGGA", "--") != "--"
                and algo != "all"
            ):
                # If meta-GGA, go straight to Algo = All only if ISMEAR is greater or equal 0.
                # Algo = All is recommended in the VASP manual and some meta-GGAs explicitly
                # say to set Algo = All for proper convergence. I am using "--" as the check
                # for METAGGA here because this is the default in the vasprun.xml file
                actions.append({"dict": "INCAR", "action": {"_set": {"ALGO": "All"}}})

            # If a hybrid is used, do not set Algo = Fast or VeryFast. Hybrid calculations do not
            # support these algorithms, but no warning is printed.
            if v.incar.get("LHFCALC", False):
                if (
                    v.incar.get("ISMEAR", -1) >= 0
                    or not 50 <= v.incar.get("IALGO", 38) <= 59
                ):
                    if algo != "all":
                        actions.append(
                            {"dict": "INCAR", "action": {"_set": {"ALGO": "All"}}}
                        )
                    # See the VASP manual section on LHFCALC for more information.
                    elif algo != "damped":
                        actions.append(
                            {
                                "dict": "INCAR",
                                "action": {"_set": {"ALGO": "Damped", "TIME": 0.5}},
                            }
                        )
                else:
                    actions.append(
                        {"dict": "INCAR", "action": {"_set": {"ALGO": "Normal"}}}
                    )

            # Ladder from VeryFast to Fast to Normal to All
            # (except for meta-GGAs and hybrids).
            # These progressively switch to more stable but more
            # expensive algorithms.
            if len(actions) == 0:
                if algo == "veryfast":
                    actions.append(
                        {"dict": "INCAR", "action": {"_set": {"ALGO": "Fast"}}}
                    )
                elif algo == "fast":
                    actions.append(
                        {"dict": "INCAR", "action": {"_set": {"ALGO": "Normal"}}}
                    )
                elif algo == "normal" and (
                    v.incar.get("ISMEAR", -1) >= 0
                    or not 50 <= v.incar.get("IALGO", 38) <= 59
                ):
                    actions.append(
                        {"dict": "INCAR", "action": {"_set": {"ALGO": "All"}}}
                    )
                else:
                    # Try mixing as last resort
                    new_settings = {
                        "ISTART": 1,
                        "ALGO": "Normal",
                        "NELMDL": -6,
                        "BMIX": 0.001,
                        "AMIX_MAG": 0.8,
                        "BMIX_MAG": 0.001,
                    }

                    if not all(
                        v.incar.get(k, "") == val for k, val in new_settings.items()
                    ):
                        actions.append(
                            {"dict": "INCAR", "action": {"_set": new_settings}}
                        )

        elif not v.converged_ionic:
            # Just continue optimizing and let other handlers fix ionic
            # optimizer parameters
            actions += [
                {"dict": "INCAR", "action": {"_set": {"IBRION": 2}}},
                {"file": "CONTCAR", "action": {"_file_copy": {"dest": "POSCAR"}}},
            ]

        if actions:
            vi = VaspInput.from_directory(".")
            backup(VASP_BACKUP_FILES)
            VaspModder(vi=vi).apply_actions(actions)
            return {"errors": ["Unconverged"], "actions": actions}

        # Unfixable error. Just return None for actions.
        return {"errors": ["Unconverged"], "actions": None}


class PolaronHandler(ErrorHandler):
    """ """

    is_monitor = True

    # The PolaronHandler should not terminate as we want VASP to terminate
    # itself naturally with the STOPCAR.
    is_terminating = False

    # This handler will be unrecoverable, but custodian shouldn't raise an
    # error
    raises_runtime_error = False

    def __init__(self, num_polarons_expected, electronic_step_stop=False):
        """
        Initializes the handler with the input and output files to check.

        Args:
            outcar_path (str): path to the OUTCAR file.
            pol_conf_expected (list): list of indices of atoms that should be
                polarons.
            electronic_step_stop (bool): if True, the electronic step will be
                stopped, otherwise the whole calculation will be stopped.
        """
        self.num_polarons_expected = num_polarons_expected
        self.electronic_step_stop = electronic_step_stop

    def check(self):
        """Check for error."""
        try:
            outcar = Outcar("OUTCAR")
        except Exception:
            # Can't perform check if Outcar not valid
            return False
        # if there are no ionic steps in the OUTCAR, the calculation is not
        # finished. we can check this by looking at outcar.magnetization to see if it is empty
        if not outcar.magnetization:
            return False
        current_pol_conf = get_polaron_conf_from_outcar(outcar)
        if len(current_pol_conf) < self.num_polarons_expected:
            print("Polaron delocalization, aborting with STOPCAR")
            return True

        return False

    def correct(self):
        """Perform corrections."""
        content = (
            "LSTOP = .TRUE." if not self.electronic_step_stop else "LABORT = .TRUE."
        )
        # Write STOPCAR
        actions = [
            {"file": "STOPCAR", "action": {"_file_create": {"content": content}}}
        ]

        m = Modder(actions=[FileActions])
        for a in actions:
            m.modify(a["action"], a["file"])

        return {
            "errors": [
                "Current polaron configuration differs from the initial one, aborting!"
            ],
            "actions": None,
        }
