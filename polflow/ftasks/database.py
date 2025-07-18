import os
from datetime import datetime

from atomate.common.firetasks.glue_tasks import get_calc_loc
from fireworks import explicit_serialize, FiretaskBase, FWAction
from pymatgen.io.vasp import Outcar

from polflow.database.db_tools import VaspDB
from polflow.tools.misc import get_polaron_conf_from_outcar


@explicit_serialize
class CalcDataToDb(FiretaskBase):
    _fw_name = "CalcDataToDb"
    required_params = ["structure", "calc_name", "calc_type", "config_dict"]
    optional_params = [
        "method",
        "metadata",
        "db_file",
        "high_level_db",
        "calc_data_coll",
    ]

    def run_task(self, fw_spec):
        # required parameters
        print(fw_spec)
        structure = self.get("structure")
        calc_name = self.get("calc_name")
        calc_type = self.get("calc_type")
        config_dict = self.get("config_dict")

        # optional parameters
        method = self.get("method", "mlff")
        metadata = self.get("metadata", {})
        db_file = self.get("db_file", "auto")
        high_level = self.get("high_level_db", "polaron_test")
        calc_data_coll = self.get("calc_data_coll", "calc_data")

        calc_loc_data = get_calc_loc(calc_name, fw_spec.get("calc_locs", []))
        outcar = Outcar(os.path.join(calc_loc_data["path"], "OUTCAR"))
        final_energy = outcar.final_energy
        vasp_db = VaspDB(db_file=db_file, high_level=high_level)

        fltr = {"configuration": config_dict, "method": method, "calc_type": calc_type}

        db_entry = {
            "path": calc_loc_data["path"],
            "final_energy": final_energy,
            "filesystem": calc_loc_data["filesystem"],
            "fw_name": calc_loc_data["name"],
            "time_added": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
            "run_stats": outcar.run_stats,
            **metadata,
        }

        if method is not None:
            final_polarons = get_polaron_conf_from_outcar(outcar=outcar)
            final_polarons = sorted(
                [structure.site_properties["original_index"][i] for i in final_polarons]
            )
            db_entry.update(
                {
                    "initial_polarons": config_dict["polaron"],
                    "final_polarons": final_polarons,
                }
            )

        vasp_db.update_data(
            collection=calc_data_coll,
            fltr=fltr,
            new_values={"$set": db_entry},
            upsert=True,
        )

        fw_spec[calc_name] = db_entry

        return FWAction(update_spec=fw_spec)
