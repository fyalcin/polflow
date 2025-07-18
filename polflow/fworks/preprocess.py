from fireworks.core.firework import Firework

from polflow.ftasks.preprocess import ExtractData, AugmentData, ConvertToDescriptor


class PreProcessData(Firework):
    def __init__(
        self,
        load_path,
        save_path,
        poscar_path,
        db_file=None,
        high_level=None,
        name="PreProcessData",
        parents=None,
    ):
        tasks = [
            ExtractData(
                load_path=load_path, save_path=save_path, poscar_path=poscar_path
            ),
            AugmentData(),
            ConvertToDescriptor(),
        ]

        super().__init__(tasks=tasks, parents=parents, name=name)
