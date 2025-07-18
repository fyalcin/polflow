from fireworks import FiretaskBase, Firework, FWAction, Workflow
from fireworks import explicit_serialize

from polflow.database.db_tools import VaspDB
from polflow.fworks import vasp as vasp_fworks
from polflow.tools.handlers import PolaronHandler, UnconvergedErrorHandlerCustom
from polflow.tools.misc import ConfigurationGenerator

fallbacks = {"mlff": "occmat"}


@explicit_serialize
class VerifyPolarons(FiretaskBase):
    _fw_name = "VerifyPolarons"
    required_params = [
        "base_structure",
        "decorated_structure",
        "config_dict",
        "config_arr",
        "prev_calc_name",
        "method",
    ]
    optional_params = [
        "db_file",
        "high_level_db",
        "calc_data_coll",
        "results_coll",
        "metadata",
    ]

    def run_task(self, fw_spec):
        # required parameters
        base_structure = self["base_structure"]
        decorated_structure = self["decorated_structure"]
        config_dict = self["config_dict"]
        config_arr = self["config_arr"]
        prev_calc_name = self["prev_calc_name"]
        method = self["method"]

        num_polarons = len(config_dict["polaron"])

        # optional parameters
        db_file = self.get("db_file", "auto")
        high_level_db = self.get("high_level_db", "polaron_test")
        calc_data_coll = self.get("calc_data_coll", "calc_data")
        results_coll = self.get("results_coll", "results")
        metadata = self.get("metadata", {})

        db = VaspDB(db_file=db_file, high_level=high_level_db)

        config_str = "_".join([f"{k}{v}" for k, v in config_dict.items()])

        initial_polaron_conf = config_dict["polaron"]
        calc_data = fw_spec[prev_calc_name]
        final_polaron_conf = calc_data["final_polarons"]

        if initial_polaron_conf == final_polaron_conf:
            # we perform the final calculation
            calc_type = "pol_rel_hq"
            fw1 = vasp_fworks.PolaronicVaspFW(
                structure=decorated_structure,
                base_structure=base_structure,
                name=f"{config_str}_{method}_{calc_type}",
                method=method,
                calc_type=calc_type,
                config_dict=config_dict,
                prev_calc_name=prev_calc_name,
                db_file=db_file,
                handlers=[
                    PolaronHandler(num_polarons_expected=num_polarons),
                    UnconvergedErrorHandlerCustom(),
                ],
                spec=fw_spec,
                high_level_db=high_level_db,
                calc_data_coll=calc_data_coll,
            )

            fw2 = Firework(
                [
                    FinalizePolaronicFW(
                        base_structure=base_structure,
                        prev_calc_name=f"{config_str}_{method}_{calc_type}",
                        method=method,
                        config_dict=config_dict,
                        db_file=db_file,
                        high_level_db=high_level_db,
                        results_coll=results_coll,
                        metadata=metadata,
                    )
                ],
                name=f"FinalizePolaronicFW_{config_str}_{method}",
                parents=[fw1],
            )
            wf = Workflow(
                [fw1, fw2], name=f"Polaron {method} workflow for {config_str}"
            )
            return FWAction(detours=wf, update_spec=fw_spec)
        else:
            fallback_method = fallbacks.get(method)

            final_config_dict = config_dict.copy()
            final_config_dict["polaron"] = final_polaron_conf
            final_config_str = "_".join(
                [f"{k}{v}" for k, v in final_config_dict.items()]
            )

            final_config_in_db = db.find_result_in_db(
                base_structure=base_structure,
                config_dict=final_config_dict,
                results_coll=results_coll,
            )
            if final_config_in_db:
                print(
                    f"Calculation for {final_config_str} and method {method} already exists in the database."
                )
                return FWAction(update_spec=fw_spec)

            if len(initial_polaron_conf) == len(final_polaron_conf):
                print("Polaron migration")
                calc_type = "pol_rel_hq"

                fws = []
                fw1 = vasp_fworks.PolaronicVaspFW(
                    structure=decorated_structure,
                    base_structure=base_structure,
                    name=f"{final_config_str}_{method}_{calc_type}",
                    method=method,
                    calc_type=calc_type,
                    config_dict=final_config_dict,
                    spec=fw_spec,
                    handlers=[
                        PolaronHandler(num_polarons_expected=5),
                        UnconvergedErrorHandlerCustom(),
                    ],
                    prev_calc_name=prev_calc_name,
                    db_file=db_file,
                    high_level_db=high_level_db,
                    calc_data_coll=calc_data_coll,
                )
                fws.append(fw1)

                fw2 = Firework(
                    [
                        FinalizePolaronicFW(
                            base_structure=base_structure,
                            prev_calc_name=fw1.name,
                            method=method,
                            config_dict=final_config_dict,
                            db_file=db_file,
                            high_level_db=high_level_db,
                            results_coll=results_coll,
                        )
                    ],
                    name=f"FinalizePolaronicFW_{final_config_str}_{method}",
                    parents=[fw1],
                )
                fws.append(fw2)

                if fallback_method is not None:
                    fw3 = Firework(
                        [
                            StartPolaronWF(
                                base_structure=base_structure,
                                config_arr=config_arr,
                                method=fallback_method,
                                db_file=db_file,
                                high_level_db=high_level_db,
                                calc_data_coll=calc_data_coll,
                                results_coll=results_coll,
                            )
                        ],
                        name=f"StartPolaronWF_{config_str}",
                    )
                    fws.append(fw3)
                else:
                    to_write = calc_data
                    to_write["status"] = "fail"
                    to_write["metadata"] = metadata
                    to_write["note"] = "Polaron migration without fallback method"
                    db.update_data(
                        collection=results_coll,
                        fltr={"configuration": config_dict, "method": method},
                        new_values={"$set": to_write},
                        upsert=True,
                    )

                wf = Workflow(fws, name="Follow-up workflow for polaron migration")
                return FWAction(detours=wf, update_spec=fw_spec)
            else:
                print("Polaron delocalization")
                if fallback_method is None:
                    print("No fallback method found")
                    to_write = calc_data
                    to_write["status"] = "fail"
                    to_write["note"] = "Polaron delocalization without fallback method"

                    db.update_data(
                        collection=results_coll,
                        fltr={"configuration": config_dict, "method": method},
                        new_values={"$set": to_write},
                        upsert=True,
                    )

                    return FWAction(update_spec=fw_spec)
                else:
                    print(f"Falling back to {fallback_method}")
                    fw = Firework(
                        [
                            StartPolaronWF(
                                base_structure=base_structure,
                                config_arr=config_arr,
                                method=fallback_method,
                                db_file=db_file,
                                high_level_db=high_level_db,
                                calc_data_coll=calc_data_coll,
                                results_coll=results_coll,
                            )
                        ],
                        name=f"StartPolaronWF_{config_str}",
                    )
                    wf = Workflow(
                        [fw],
                        name=f"Polaron {fallback_method} workflow for {config_str}",
                    )
                    return FWAction(detours=wf, update_spec=fw_spec)


@explicit_serialize
class FinalizePolaronicFW(FiretaskBase):
    _fw_name = "FinalizePolaronicFW"
    required_params = ["base_structure", "prev_calc_name", "method", "config_dict"]
    optional_params = ["db_file", "high_level_db", "results_coll", "metadata"]

    def run_task(self, fw_spec):
        # required parameters
        base_structure = self["base_structure"]
        prev_calc_name = self["prev_calc_name"]
        method = self["method"]
        config_dict = self["config_dict"]

        # optional parameters
        db_file = self.get("db_file", "auto")
        high_level_db = self.get("high_level_db", "polaron_test")
        results_coll = self.get("results_coll", "results")
        metadata = self.get("metadata", {})

        db = VaspDB(db_file=db_file, high_level=high_level_db)

        data = fw_spec[prev_calc_name]
        initial_polarons = config_dict["polaron"]
        final_polarons = data["final_polarons"]

        fltr = {"configuration": config_dict, "method": method}

        to_write = data

        if final_polarons == initial_polarons:
            result_in_db = db.find_result_in_db(
                base_structure=base_structure,
                config_dict=config_dict,
                results_coll=results_coll,
            )
            result_conf = result_in_db["configuration"] if result_in_db else None
            if result_conf:
                to_write["symm_equiv_to"] = result_conf
            to_write["status"] = "success"
            to_write["note"] = "no polaron migration or delocalization after hq relax"
        elif len(final_polarons) == len(initial_polarons):
            final_config = config_dict.copy()
            final_config["polaron"] = final_polarons
            result_in_db_final = db.find_result_in_db(
                base_structure=base_structure,
                config_dict=final_config,
                results_coll=results_coll,
            )
            result_conf_final = (
                result_in_db_final["configuration"] if result_in_db_final else None
            )
            if result_conf_final:
                to_write["symm_equiv_to"] = result_conf_final
            to_write["status"] = "fail"
            to_write["note"] = "polaron migration with hq relax - should not happen"
        else:
            to_write["status"] = "fail"
            to_write["note"] = (
                "polaron delocalization with hq relax - should not happen"
            )

        to_write["metadata"] = metadata
        db = VaspDB(db_file=db_file, high_level=high_level_db)
        db.update_data(
            collection=results_coll,
            fltr=fltr,
            new_values={"$set": to_write},
            upsert=True,
        )


@explicit_serialize
class StartPolaronWF(FiretaskBase):
    _fw_name = "StartPolaronWF"
    required_params = ["base_structure", "config_arr", "method"]
    optional_params = [
        "db_file",
        "high_level_db",
        "calc_data_coll",
        "results_coll",
        "metadata",
    ]

    def run_task(self, fw_spec):
        # required parameters
        base_structure = self["base_structure"]
        config_arr = self["config_arr"]
        method = self.get("method")

        # optional parameters
        db_file = self.get("db_file", "auto")
        high_level_db = self.get("high_level_db", "polaron_test")
        calc_data_coll = self.get("calc_data_coll", "calc_data")
        results_coll = self.get("results_coll", "results")
        metadata = self.get("metadata", {})

        confgen = ConfigurationGenerator()
        confgen.modifications = config_arr
        structure = base_structure.copy()
        confgen.decorate_structure(structure)
        confgen.apply_modifications(
            structure, sort_structure=True, group_polarons=True, rename_polarons=False
        )

        config_dict = confgen.get_conf_dict()
        num_polarons = len(config_dict["polaron"])
        config_str = confgen.get_conf_str()

        fws = []

        db = VaspDB(db_file=db_file, high_level=high_level_db)

        polaron_data = db.find_result_in_db(
            base_structure=base_structure,
            config_dict=config_dict,
            results_coll=results_coll,
        )

        if polaron_data:
            print(
                f"Calculation for {config_str} and method {method} already exists in the database."
            )
            return FWAction()

        calc_type = f"pol_rel_{method}"

        fw1 = vasp_fworks.PolaronicVaspFW(
            structure=structure,
            base_structure=base_structure,
            name=f"{config_str}_{method}_{calc_type}",
            method=method,
            calc_type=calc_type,
            config_dict=config_dict,
            db_file=db_file,
            high_level_db=high_level_db,
            calc_data_coll=calc_data_coll,
            additional_files_from_prev_calcs=None,
            user_incar_settings=None,
        )
        fws.append(fw1)

        calc_type = "pol_rel_rough"
        additional_files = (
            {fw1.name: ["CHGCAR", "WAVECAR"]} if method == "occmat" else None
        )
        fw2 = vasp_fworks.PolaronicVaspFW(
            structure=structure,
            base_structure=base_structure,
            name=f"{config_str}_{method}_{calc_type}",
            method=method,
            calc_type=calc_type,
            config_dict=config_dict,
            prev_calc_name=fw1.name,
            db_file=db_file,
            handlers=[
                PolaronHandler(num_polarons_expected=num_polarons),
                UnconvergedErrorHandlerCustom(),
            ],
            high_level_db=high_level_db,
            calc_data_coll=calc_data_coll,
            additional_files_from_prev_calcs=additional_files,
            user_incar_settings=None,
            parents=[fw1],
        )
        fws.append(fw2)

        fw3 = vasp_fworks.VerifyPolaronsFW(
            base_structure=base_structure,
            decorated_structure=structure,
            config_arr=config_arr,
            config_dict=config_dict,
            method=method,
            prev_calc_name=f"{config_str}_{method}_{calc_type}",
            db_file=db_file,
            high_level_db=high_level_db,
            calc_data_coll=calc_data_coll,
            results_coll=results_coll,
            metadata=metadata,
            parents=[fw2],
        )
        fws.append(fw3)

        wf = Workflow(fireworks=fws, name=f"Polaron {method} workflow for {config_str}")
        return FWAction(detours=[wf])


@explicit_serialize
class StartDelocWF(FiretaskBase):
    _fw_name = "StartDelocWF"
    required_params = ["base_structure", "config_arr"]
    optional_params = ["db_file", "high_level_db", "calc_data_coll", "results_coll"]

    def run_task(self, fw_spec):
        # required parameters
        base_structure = self["base_structure"]
        config_arr = self["config_arr"]

        # optional parameters
        db_file = self.get("db_file", "auto")
        high_level_db = self.get("high_level_db", "polaron_test")
        calc_data_coll = self.get("calc_data_coll", "calc_data")
        results_coll = self.get("results_coll", "deloc_results")

        # remove the polarons from config_arr
        config_arr = [mod for mod in config_arr if mod["type"] != "polaron"]

        confgen = ConfigurationGenerator()
        confgen.modifications = config_arr
        structure = base_structure.copy()
        confgen.decorate_structure(structure)
        confgen.apply_modifications(
            structure, sort_structure=True, group_polarons=True, rename_polarons=True
        )

        config_dict = confgen.get_conf_dict()
        config_str = confgen.get_conf_str()

        db = VaspDB(db_file=db_file, high_level=high_level_db)

        deloc_data = db.find_result_in_db(
            base_structure=base_structure,
            config_dict=config_dict,
            results_coll=results_coll,
        )

        if deloc_data:
            print(
                f"Deloc. Calculation for {config_str} already exists in the database."
            )
            return FWAction()

        fws = []
        calc_type = "deloc_rel_rough"
        fw1 = vasp_fworks.PolaronicVaspFW(
            structure=structure,
            base_structure=base_structure,
            name=f"{config_str}_{calc_type}",
            calc_type=calc_type,
            config_dict=config_dict,
            db_file=db_file,
            high_level_db=high_level_db,
            calc_data_coll=calc_data_coll,
        )
        fws.append(fw1)

        calc_type = "deloc_rel_hq"
        fw2 = vasp_fworks.PolaronicVaspFW(
            structure=structure,
            base_structure=base_structure,
            name=f"{config_str}_{calc_type}",
            calc_type=calc_type,
            config_dict=config_dict,
            prev_calc_name=fw1.name,
            db_file=db_file,
            high_level_db=high_level_db,
            calc_data_coll=calc_data_coll,
            parents=[fw1],
        )
        fws.append(fw2)

        fw3 = Firework(
            [
                FinalizeDelocFW(
                    prev_calc_name=fw2.name,
                    config_dict=config_dict,
                    db_file=db_file,
                    high_level_db=high_level_db,
                    results_coll=results_coll,
                )
            ],
            name=f"FinalizeDelocFW_{config_str}",
            parents=[fw2],
        )
        fws.append(fw3)

        wf = Workflow(fireworks=fws, name=f"Delocalization workflow for {config_str}")
        return FWAction(detours=[wf])


@explicit_serialize
class FinalizeDelocFW(FiretaskBase):
    _fw_name = "FinalizeDelocFW"
    required_params = ["prev_calc_name", "config_dict"]
    optional_params = ["db_file", "high_level_db", "results_coll"]

    def run_task(self, fw_spec):
        # required parameters
        prev_calc_name = self["prev_calc_name"]
        config_dict = self["config_dict"]

        # optional parameters
        db_file = self.get("db_file", "auto")
        high_level_db = self.get("high_level_db", "polaron_test")
        results_coll = self.get("results_coll", "deloc_results")

        data = fw_spec[prev_calc_name]
        fltr = {"configuration": config_dict}

        to_write = data

        db = VaspDB(db_file=db_file, high_level=high_level_db)
        db.update_data(
            collection=results_coll,
            fltr=fltr,
            new_values={"$set": to_write},
            upsert=True,
        )
        return FWAction()
