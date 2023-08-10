import math

import numpy as np

import libcasm.mapping.methods as mapmethods
import libcasm.xtal as xtal


def test_map_lattices_0():
    """Map to self: max_cost==0.0 -> # of mappings == 48"""
    lattice1 = xtal.Lattice(np.eye(3))
    lattice2 = xtal.Lattice(
        np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        ).transpose()
    )

    lattice_mappings = mapmethods.map_lattices(lattice1, lattice2, max_cost=0.0)
    assert len(lattice_mappings) == 48


def test_map_lattices_1():
    """Map to Ezz: max_cost==0.0 -> # mappings == 0"""
    lattice1 = xtal.Lattice(np.eye(3))
    lattice2 = xtal.Lattice(
        np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.2],
            ]
        ).transpose()
    )

    lattice_mappings = mapmethods.map_lattices(lattice1, lattice2, max_cost=0.0)
    assert len(lattice_mappings) == 0


def test_map_lattices_2():
    """Map to Ezz: Use k_best=1 -> # of tied-for-lowest cost mappings == 48"""
    L1 = np.eye(3)
    lattice1 = xtal.Lattice(L1)

    L2 = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.2],
        ]
    ).transpose()
    lattice2 = xtal.Lattice(L2)

    lattice_mappings = mapmethods.map_lattices(
        lattice1, lattice2, max_cost=1.0, k_best=1
    )
    for lattice_mapping in lattice_mappings:
        Q = lattice_mapping.isometry()
        U = lattice_mapping.right_stretch()
        T = lattice_mapping.transformation_matrix_to_super()
        N = lattice_mapping.reorientation()
        assert np.allclose(Q @ U @ L1 @ T @ N, L2)
        assert math.isclose(lattice_mapping.lattice_cost(), 0.007434763583127571)
    assert len(lattice_mappings) == 48


def test_map_lattices_3():
    """Map to Ezz

    Use lattice1 point group -> Number of tied-for-lowest cost mappings == 1
    """
    lattice1 = xtal.Lattice(np.eye(3))
    lattice2 = xtal.Lattice(
        np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.2],
            ]
        ).transpose()
    )

    lattice1_point_group = xtal.make_point_group(lattice1)
    lattice_mappings = mapmethods.map_lattices(
        lattice1,
        lattice2,
        max_cost=1.0,
        lattice1_point_group=lattice1_point_group,
        k_best=1,
    )
    assert len(lattice_mappings) == 1
