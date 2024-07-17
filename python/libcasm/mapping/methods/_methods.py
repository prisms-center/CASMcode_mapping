import libcasm.mapping.info as mapinfo
import libcasm.xtal as xtal
import numpy as np


def map_lattices_without_reorientation(
    lattice1: xtal.Lattice,
    lattice2: xtal.Lattice,
) -> mapinfo.LatticeMapping:
    """Map lattices without reorienting the child.

    This function may be used to find the lattice mapping from
    an ideal structure to a deformed structure during geometric
    relaxation via DFT.

    Parameters
    ----------
    lattice1: xtal.Lattice
        The parent crystal lattice.
    lattice2: xtal.Lattice
        The child crystal lattice.

    Returns
    -------
    lattice_mapping: mapinfo.LatticeMapping
        The lattice mapping from the parent to the child.
    """
    # calculate deformation gradient
    l1 = lattice1.column_vector_matrix()
    l2 = lattice2.column_vector_matrix()
    deformation_gradient = l2 @ np.linalg.inv(l1)
    lattice_mapping = mapinfo.LatticeMapping(
        deformation_gradient=deformation_gradient,
        transformation_matrix_to_super=np.eye(3, dtype=int),
        reorientation=np.eye(3, dtype=float),
    )
    return lattice_mapping
