from math import sqrt

import numpy as np

import libcasm.mapping.methods as mapmethods
import libcasm.xtal as xtal
import libcasm.xtal.lattices as xtal_lattices

# testing


def test_interpolate_bcc_fcc_lattices():
    bcc_lattice = xtal_lattices.BCC(r=1.0)
    bcc_point_group = xtal.make_point_group(bcc_lattice)

    fcc_lattice = xtal_lattices.FCC(r=1.0)

    lattice_mapping = mapmethods.map_lattices(
        bcc_lattice,
        fcc_lattice,
        lattice1_point_group=bcc_point_group,
        max_cost=1e20,
        min_cost=0.0,
    )[0]

    # It is not guaranteed that the particular mapping
    # orientation tested for here is always the first mapping
    # result. These tests may need to change.

    # f=0.0, end point == parent (bcc)
    lattice = mapmethods.make_mapped_lattice(
        bcc_lattice, lattice_mapping.interpolated(0.0)
    )
    assert np.allclose(
        lattice.column_vector_matrix(),
        np.array(
            [
                [2.0 / sqrt(3.0), -2.0 / sqrt(3.0), -2.0 / sqrt(3.0)],
                [2.0 / sqrt(3.0), 2.0 / sqrt(3.0), -2.0 / sqrt(3.0)],
                [4.0 / sqrt(3.0), 0.0, 0.0],
            ]
        ).transpose(),
    )

    # f=0.2
    lattice = mapmethods.make_mapped_lattice(
        bcc_lattice, lattice_mapping.interpolated(0.2)
    )
    print(lattice.column_vector_matrix().transpose())
    assert np.allclose(
        lattice.column_vector_matrix(),
        np.array(
            [
                [1.12376043, -1.12376043, -1.20660314],
                [1.12376043, 1.12376043, -1.20660314],
                [2.24752086, 0.0, 0.0],
            ]
        ).transpose(),
    )

    # f=1.0, end point = child (fcc)
    lattice = mapmethods.make_mapped_lattice(
        bcc_lattice, lattice_mapping.interpolated(1.0)
    )
    print(lattice.column_vector_matrix().transpose())
    assert np.allclose(
        lattice.column_vector_matrix(),
        np.array(
            [[1.0, -1, -sqrt(2.0)], [1.0, 1, -sqrt(2.0)], [2.0, 0.0, 0.0]]
        ).transpose(),
    )


def test_interpolate_bcc_hcp_lattices():
    bcc_lattice = xtal_lattices.BCC(r=1.0)
    bcc_point_group = xtal.make_point_group(bcc_lattice)

    hcp_lattice = xtal_lattices.HCP(r=1.0)

    # vol=2 supercell of bcc maps to hcp
    T = np.array([[-1, 1, -1], [-1, 0, 0], [-1, 1, 1]])

    lattice_mapping = mapmethods.map_lattices(
        bcc_lattice,
        hcp_lattice,
        transformation_matrix_to_super=T,
        lattice1_point_group=bcc_point_group,
        max_cost=1e20,
        min_cost=0.0,
    )[0]

    # It is not guaranteed that the particular mapping
    # orientation tested for here is always the first mapping
    # result. These tests may need to change.

    # f=0.0, end point == parent (bcc)
    lattice = mapmethods.make_mapped_lattice(
        bcc_lattice, lattice_mapping.interpolated(0.0)
    )
    assert np.allclose(
        lattice.column_vector_matrix(),
        np.array(
            [
                [2.0 / sqrt(3.0), 2.0 / sqrt(3.0), 2.0 / sqrt(3.0)],
                [-2.0 / sqrt(3.0), 2.0 / sqrt(3.0), -2.0 / sqrt(3.0)],
                [-4.0 / sqrt(3.0), 0.0, 4.0 / sqrt(3.0)],
            ]
        ).transpose(),
    )

    # f=0.2
    lattice = mapmethods.make_mapped_lattice(
        bcc_lattice, lattice_mapping.interpolated(0.2)
    )
    assert np.allclose(
        lattice.column_vector_matrix(),
        np.array(
            [
                [1.1687094, 1.12376043, 1.1687094],
                [-1.1687094, 1.12376043, -1.1687094],
                [-2.30940108, 0.0, 2.30940108],
            ]
        ).transpose(),
    )

    # f=1.0, end point == child (hcp)
    lattice = mapmethods.make_mapped_lattice(
        bcc_lattice, lattice_mapping.interpolated(1.0)
    )
    assert np.allclose(
        lattice.column_vector_matrix(),
        np.array(
            [
                [sqrt(3.0 / 2.0), 1.0, sqrt(3.0 / 2.0)],
                [-sqrt(3.0 / 2.0), 1.0, -sqrt(3.0 / 2.0)],
                [-sqrt(16.0 / 3.0), 0.0, sqrt(16.0 / 3.0)],
            ]
        ).transpose(),
    )
