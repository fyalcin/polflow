import numpy as np
from pymatgen.core import Structure
from pymatgen.core.surface import Slab


def slab_from_structure(miller: list | tuple, structure: Structure) -> Slab:
    """Returns a pymatgen.core.surface.Slab from a pymatgen structure.

    Parameters
    ----------
    miller : list of int
        Miller indices given as a list of integers
    structure : pymatgen.core.structure.Structure
        Structure to be converted into a Slab

    Returns
    -------
    pymatgen.core.surface.Slab
        The input structure converted to a Slab

    """
    return Slab(
        lattice=structure.lattice,
        species=structure.species_and_occu,
        coords=structure.frac_coords,
        miller_index=miller,
        oriented_unit_cell=structure,
        shift=0,
        scale_factor=np.eye(3, dtype=int),
        site_properties=structure.site_properties,
    )
