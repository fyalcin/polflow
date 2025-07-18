from fireworks import Firework

from polflow.ftasks.misc import GenerateRandomNumber, PassCalcLocs, PrintSpec, FinishUp


class CustomFW(Firework):
    def __init__(self, base_name, calc_type, spec=None, parents=None, **kwargs):
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
        name = f"{base_name}_{calc_type}"
        t = []
        t.append(GenerateRandomNumber(key=f"{name}_initial"))
        t.append(PassCalcLocs(name=name))
        t.append(GenerateRandomNumber(key=f"{name}_final"))
        t.append(PrintSpec())
        t.append(FinishUp(base_name=base_name, calc_type=calc_type))
        super().__init__(t, parents=parents, spec=spec, name=name, **kwargs)
