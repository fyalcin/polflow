from fireworks import Firework, Workflow

import polflow.ftasks.polaron as polaron_tasks
from polflow.tools.misc import ConfigurationGenerator

required_params = ["base_structure", "config_arr", "method"]
optional_params = ["db_file", "high_level_db", "calc_data_coll", "results_coll"]


def polaron_wf(
    base_structure,
    config_arr,
    method="mlff",
    db_file="auto",
    high_level_db="polaron_test",
    calc_data_coll="calc_data",
    polaron_results_coll="polaron_results",
    deloc_results_coll="deloc_results",
    metadata=None,
):
    if metadata is None:
        metadata = {}
    deloc_confgen = ConfigurationGenerator()
    deloc_confgen.modifications = [c for c in config_arr if not c["type"] == "polaron"]
    deloc_config_str = deloc_confgen.get_conf_str()
    fws = []

    fw1 = Firework(
        [
            polaron_tasks.StartDelocWF(
                base_structure=base_structure,
                config_arr=config_arr,
                db_file=db_file,
                high_level_db=high_level_db,
                calc_data_coll=calc_data_coll,
                results_coll=deloc_results_coll,
            )
        ],
        name=f"Deloc WF: {deloc_config_str}",
    )
    fws.append(fw1)

    pol_confgen = ConfigurationGenerator()
    pol_confgen.modifications = config_arr
    pol_config_str = pol_confgen.get_conf_str()

    fw2 = Firework(
        [
            polaron_tasks.StartPolaronWF(
                base_structure=base_structure,
                config_arr=config_arr,
                method=method,
                db_file=db_file,
                high_level_db=high_level_db,
                calc_data_coll=calc_data_coll,
                results_coll=polaron_results_coll,
                metadata=metadata,
            )
        ],
        name=f"Polaron WF: {pol_config_str}",
    )
    fws.append(fw2)

    wf = Workflow(fws, name=f"Polaron WF: {pol_config_str}")
    return wf
