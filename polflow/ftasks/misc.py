import os

from atomate.utils.utils import env_chk
from fireworks import explicit_serialize, FiretaskBase, FWAction


# import polflow.fworks.misc as misc


@explicit_serialize
class GenerateRandomNumber(FiretaskBase):
    """
    Generates a random number and stores it in the fw_spec.
    """

    _fw_name = "GenerateRandomNumber"
    required_params = ["key"]

    def run_task(self, fw_spec):
        key = self["key"]
        import random

        random_number = random.random()
        fw_spec[key] = random_number

        return FWAction(update_spec=fw_spec)


# @explicit_serialize
# class FinishUp(FiretaskBase):
#     """
#     Generates a random number and stores it in the fw_spec.
#     """
#
#     _fw_name = "FinishUp"
#     required_params = ["base_name", "calc_type"]
#
#     def run_task(self, fw_spec):
#         base_name = self["base_name"]
#         calc_type = self["calc_type"]
#
#         if calc_type == "mlff":
#             fw = misc.CustomFW(base_name=base_name, calc_type="pol_rel", spec=fw_spec)
#             wf = Workflow([fw])
#             return FWAction(additions=wf)
#         elif calc_type == "pol_rel":
#             initial_random_number = fw_spec[f"{base_name}_{calc_type}_initial"]
#             final_random_number = fw_spec[f"{base_name}_{calc_type}_final"]
#             # check if both are below 0.5
#             if initial_random_number < 1 and final_random_number < 1:
#                 fw = misc.CustomFW(base_name=base_name, calc_type="pol_rel_hq", spec=fw_spec)
#                 wf = Workflow([fw])
#                 return FWAction(additions=wf)
#             else:
#                 return FWAction(stored_data={"pol_rel": "failed"})
#         elif calc_type == "pol_rel_hq":
#             initial_random_number = fw_spec[f"{base_name}_{calc_type}_initial"]
#             final_random_number = fw_spec[f"{base_name}_{calc_type}_final"]
#             # check if both are below 0.5
#             if initial_random_number < 0.5 and final_random_number < 0.5:
#                 return FWAction(stored_data={"pol_rel_hq": "failed"})
#             else:
#                 return FWAction(stored_data={"pol_rel_hq": "success"})


@explicit_serialize
class UpdateCalcLocs(FiretaskBase):
    """
    Passes information about where the current calculation is located
    for the next FireWork. This is achieved by passing a key to
    the fw_spec called "calc_data" with this information.

    Required params:
        name (str): descriptive name for this calculation file/dir

    Optional params:
        filesystem (str or custom user format): name of filesystem. Supports env_chk.
            defaults to None
        path (str): The path to the directory containing the calculation. defaults to
            current working directory.
    """

    required_params = ["current_fw_name", "saved_calc_data"]
    optional_params = []

    def run_task(self, fw_spec):
        # required_params
        current_fw_name = self["current_fw_name"]
        saved_calc_data = self["saved_calc_data"]

        calc_locs = list(fw_spec.get("calc_locs", []))
        calc_locs.append(
            {
                "name": current_fw_name,
                "filesystem": saved_calc_data.get("filesystem", None),
                "path": saved_calc_data.get("path", None),
            }
        )
        fw_spec["calc_locs"] = calc_locs
        fw_spec[current_fw_name] = saved_calc_data

        return FWAction(update_spec=fw_spec)


@explicit_serialize
class PassCalcLocs(FiretaskBase):
    """
    Passes information about where the current calculation is located
    for the next FireWork. This is achieved by passing a key to
    the fw_spec called "calc_data" with this information.

    Required params:
        name (str): descriptive name for this calculation file/dir

    Optional params:
        filesystem (str or custom user format): name of filesystem. Supports env_chk.
            defaults to None
        path (str): The path to the directory containing the calculation. defaults to
            current working directory.
    """

    required_params = ["name"]
    optional_params = ["filesystem", "path"]

    def run_task(self, fw_spec):
        # print("PassCalcLocs: setting calc_data")
        calc_locs = list(fw_spec.get("calc_locs", []))
        calc_locs.append(
            {
                "name": self["name"],
                "filesystem": env_chk(self.get("filesystem", None), fw_spec),
                "path": self.get("path", os.getcwd()),
            }
        )
        fw_spec["calc_locs"] = calc_locs

        return FWAction(update_spec=fw_spec)


@explicit_serialize
class PrintSpec(FiretaskBase):
    _fw_name = "PrintSpec"
    required_params = []
    optional_params = ["vasp_cmd", "key"]

    def run_task(self, fw_spec):
        if self.get("key"):
            to_print = fw_spec[self.get("key")]
        else:
            to_print = fw_spec
        vasp_cmd = env_chk(self.get("vasp_cmd", ">>vasp_cmd<<"), fw_spec)
        print("-" * 50)
        print(to_print)
        print("Running VASP using {}:".format(vasp_cmd))
        print("-" * 50)
        return FWAction(update_spec=fw_spec)
