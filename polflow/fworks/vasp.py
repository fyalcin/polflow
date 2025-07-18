import psutil
from atomate.common.firetasks.glue_tasks import CopyFilesFromCalcLoc
from atomate.common.firetasks.glue_tasks import PassCalcLocs
from fireworks import Firework

import polflow.ftasks.polaron as polaron_tasks
from polflow.database.db_tools import VaspDB
from polflow.ftasks.database import CalcDataToDb
from polflow.ftasks.io import WritePoscar, WriteInputs
from polflow.ftasks.misc import PassCalcLocs, UpdateCalcLocs
from polflow.ftasks.vasp import RunVaspCustodianCustom

vasp_executables = {"occmat": "vasp.5.4.4_occmat_gam_michele"}


class PolaronicVaspFW(Firework):
    def __init__(
        self,
        structure,
        base_structure,
        name,
        calc_type,
        config_dict,
        db_file: str,
        high_level_db: str,
        calc_data_coll: str,
        method=None,
        spec=None,
        parents=None,
        prev_calc_name=None,
        handlers=None,
        additional_files_from_prev_calcs: dict = None,
        user_incar_settings=None,
        metadata=None,
        **kwargs,
    ):
        """
        Standard static calculation Firework - either from a previous location or from a structure.

        Args:
            structure (Structure): Input structure. Note that for prev_calc_loc jobs, the structure
                is only used to set the name of the FW and any structure with the same composition
                can be used.
            name (str): Name for the Firework.
            vasp_input_set (VaspInputSet): input set to use (for jobs w/no parents)
                Defaults to MPStaticSet() if None.
            vasp_input_set_params (dict): Dict of vasp_input_set kwargs.
            vasp_cmd (str): Command to run vasp.
            prev_calc_loc (bool or str): If true (default), copies outputs from previous calc. If
                a str value, retrieves a previous calculation output by name. If False/None, will create
                new static calculation using the provided structure.
            prev_calc_dir (str): Path to a previous calculation to copy from
            db_file (str): Path to file specifying db credentials.
            parents (Firework): Parents of this particular Firework. FW or list of FWS.
            vasptodb_kwargs (dict): kwargs to pass to VaspToDb
            additional_files_from_prev_calcs (list o str): Copy additional files other than
                POSCAR, POTCAR, KPOINTS, INCAR, vasprun.xml, like WAVECAR or CHGCAR.
            **kwargs: Other kwargs that are passed to Firework.__init__.
        """
        self.structure = structure
        self.calc_type = calc_type
        if user_incar_settings is None:
            user_incar_settings = {}
        if additional_files_from_prev_calcs is None:
            additional_files_from_prev_calcs = {}
        if metadata is None:
            metadata = {}

        db = VaspDB(db_file=db_file, high_level=high_level_db)

        data = db.find_calc_in_db(
            base_structure=base_structure,
            config_dict=config_dict,
            method=method,
            calc_type=calc_type,
            calc_data_coll=calc_data_coll,
        )
        t = []
        if data:
            # filesystem, path = data["filesystem"], data["path"]
            print(
                "*********** Found calc_loc, only passing calc_ data... **************"
            )
            # we still need to pass the calc_data to the next firework
            # in case that one has not been run yet
            t.append(UpdateCalcLocs(current_fw_name=name, saved_calc_data=data))
        else:
            # copy over input files from a predefined directory
            # t.append(CopyFiles(files_to_copy=["INCAR", "KPOINTS", "POTCAR"],
            #                    from_dir=INPUT_DIRS[calc_type]))
            if calc_type in ["pol_rel_hq", "pol_rel_rough"]:
                pol_indices = [
                    structure.site_properties["original_index"].index(i)
                    for i in config_dict["polaron"]
                ]
                magmom_arr = [
                    1.0 if i in pol_indices else 0 for i in range(len(structure))
                ]
            else:
                magmom_arr = None
            t.append(
                WriteInputs(
                    structure=structure,
                    calc_type=calc_type,
                    magmom_arr=magmom_arr,
                    user_incar_settings=user_incar_settings,
                )
            )

            if prev_calc_name:
                # t.append(WritePrevCalcLoc(prev_calc_name=parent.name))
                t.append(
                    WritePoscar(from_prev_calc=prev_calc_name, calc_type=calc_type)
                )

                if additional_files_from_prev_calcs:
                    for (
                        parent_fw_name,
                        filenames,
                    ) in additional_files_from_prev_calcs.items():
                        t.append(
                            CopyFilesFromCalcLoc(
                                calc_loc=parent_fw_name, filenames=filenames
                            )
                        )
            else:
                t.append(WritePoscar(calc_type=calc_type, structure=structure))

            handlers = handlers if handlers else []
            job_type = "mlff" if calc_type.endswith("mlff") else "normal_no_backup"
            vasp_executable = vasp_executables.get(method, "vasp.6.4.2_gam")
            cpu_count = psutil.cpu_count(logical=False)
            if cpu_count >= 64:
                cpu_count = int(cpu_count / 2)
            else:
                cpu_count = int(cpu_count)
            vasp_cmd = f"mpirun -np {cpu_count} {vasp_executable}"

            t.append(
                RunVaspCustodianCustom(
                    vasp_cmd=vasp_cmd,
                    job_type=job_type,
                    max_errors=2,
                    # auto_npar=">>auto_npar<<",
                    handler_group=handlers,
                    gzip_output=False,
                )
            )
            t.append(PassCalcLocs(name=name))
            t.append(
                CalcDataToDb(
                    structure=structure,
                    calc_name=name,
                    calc_type=calc_type,
                    method=method,
                    config_dict=config_dict,
                    metadata=metadata,
                    db_file=db_file,
                    high_level_db=high_level_db,
                    calc_data_coll=calc_data_coll,
                )
            )
        super().__init__(t, parents=parents, spec=spec, name=name, **kwargs)  # type: ignore


class VerifyPolaronsFW(Firework):
    def __init__(
        self,
        base_structure,
        decorated_structure,
        config_arr,
        config_dict,
        method,
        prev_calc_name,
        db_file="auto",
        high_level_db="polaron_test",
        calc_data_coll="calc_data",
        results_coll="results",
        metadata=None,
        spec=None,
        parents=None,
        **kwargs,
    ):
        """
        Standard static calculation Firework - either from a previous location or from a structure.

        Args:
            structure (Structure): Input structure. Note that for prev_calc_loc jobs, the structure
                is only used to set the name of the FW and any structure with the same composition
                can be used.
            name (str): Name for the Firework.
            vasp_input_set (VaspInputSet): input set to use (for jobs w/no parents)
                Defaults to MPStaticSet() if None.
            vasp_input_set_params (dict): Dict of vasp_input_set kwargs.
            vasp_cmd (str): Command to run vasp.
            prev_calc_loc (bool or str): If true (default), copies outputs from previous calc. If
                a str value, retrieves a previous calculation output by name. If False/None, will create
                new static calculation using the provided structure.
            prev_calc_dir (str): Path to a previous calculation to copy from
            db_file (str): Path to file specifying db credentials.
            parents (Firework): Parents of this particular Firework. FW or list of FWS.
            vasptodb_kwargs (dict): kwargs to pass to VaspToDb
            additional_files_from_prev_calc (list o str): Copy additional files other than
                POSCAR, POTCAR, KPOINTS, INCAR, vasprun.xml, like WAVECAR or CHGCAR.
            **kwargs: Other kwargs that are passed to Firework.__init__.
        """
        if metadata is None:
            metadata = {}
        t = [
            polaron_tasks.VerifyPolarons(
                base_structure=base_structure,
                decorated_structure=decorated_structure,
                config_arr=config_arr,
                config_dict=config_dict,
                prev_calc_name=prev_calc_name,
                method=method,
                db_file=db_file,
                high_level_db=high_level_db,
                calc_data_coll=calc_data_coll,
                results_coll=results_coll,
                metadata=metadata,
            )
        ]
        super().__init__(
            t, parents=parents, spec=spec, name="VerifyPolaronsFW", **kwargs
        )  # type: ignore
