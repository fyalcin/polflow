import json
import os
from typing import Union

from atomate.vasp.database import VaspCalcDb

from polflow.structure.polaron import get_equivalent_configurations


class VaspDB:
    """
    Class to interact with the VASP database.
    """

    def __init__(self, db_file: str = "auto", high_level: Union[str, bool] = False):
        """
        Class to interact with the VASP database.

        :param db_file: Full path of the db.json file. Defaults to 'auto', in which case the path
            is checked from the environment variable FW_CONFIG_FILE.
        :type db_file: str, optional

        :param high_level: Name of the high level database to use. If set to False, the low level database is used.
            Defaults to True, in which case the value in the db.json file is used.
        :type high_level: bool, str, optional
        """
        if db_file == "auto":
            config_path = os.environ.get("FW_CONFIG_FILE")
            if not config_path:
                raise Exception(
                    "The environmental variable FW_CONFIG_FILE "
                    "is not set. Please set it to the path of "
                    "your db.json file."
                )

            db_file = config_path.replace("FW_config.yaml", "db.json")

        self.db_file = db_file
        self.high_level = high_level

        with open(db_file, "r") as f:
            db_dict = json.load(f)
        self.db_dict = db_dict

        self.vasp_calc_db = VaspCalcDb.from_db_file(db_file, admin=True)

        try:
            db = self.vasp_calc_db.connection[high_level]
        except TypeError:
            if high_level:
                db = self.vasp_calc_db.connection[db_dict.get("high_level")]
            else:
                db = self.vasp_calc_db.db
        self.db = db

    def find_data(self, collection: str, fltr: dict, projection: dict = None):
        return self.db[collection].find_one(filter=fltr, projection=projection)

    def find_many_data(self, collection: str, fltr: dict, projection: dict = None):
        return self.db[collection].find(filter=fltr, projection=projection)

    def update_data(
        self, collection: str, fltr: dict, new_values: dict, upsert: bool = True
    ):
        self.db[collection].update_one(fltr, new_values, upsert=upsert)

    def insert_data(self, collection: str, data: dict):
        self.db[collection].insert_one(data)

    def delete_data(self, collection: str, fltr: dict):
        self.db[collection].delete_one(fltr)

    def find_calc_in_db(
        self,
        base_structure,
        calc_type: str,
        config_dict: dict,
        method: str,
        calc_data_coll: str,
    ):
        equiv_confs = get_equivalent_configurations(
            base_structure=base_structure, config_dict=config_dict
        )
        data = self.find_data(
            collection=calc_data_coll,
            fltr={
                "calc_type": calc_type,
                "method": method,
                "configuration": {"$in": equiv_confs},
            },
            projection={"_id": 0},
        )
        return data

    def find_result_in_db(
        self,
        base_structure,
        config_dict: dict,
        results_coll: str,
        projection: dict = None,
    ):
        if projection is None:
            projection = {"_id": 0}
        else:
            projection.update({"_id": 0})
        equiv_confs = get_equivalent_configurations(
            base_structure=base_structure, config_dict=config_dict
        )
        data = self.find_data(
            collection=results_coll,
            fltr={"configuration": {"$in": equiv_confs}},
            projection=projection,
        )
        return data
