from math import sqrt

import numpy as np

import libcasm.mapping.methods as mapmethods
import libcasm.xtal as xtal
import libcasm.xtal.prims as xtal_prims
import libcasm.xtal.structures as xtal_structures

# testing


def test_interpolate_bcc_fcc_structures():
    prim = xtal_prims.BCC(r=1.0, occ_dof=["A", "B", "Va"])
    prim_factor_group = xtal.make_factor_group(prim)

    fcc_structure = xtal_structures.FCC(r=1.0, atom_type="A")

    structure_mapping = mapmethods.map_structures(
        prim,
        fcc_structure,
        prim_factor_group=prim_factor_group,
        max_vol=2,
        max_cost=1e20,
        min_cost=0.0,
    )[0]

    # It is not guaranteed that the particular mapping
    # orientation tested for here is always the first mapping
    # result. These tests may need to change.

    # f=0.0, end point == parent
    structure = mapmethods.make_mapped_structure(
        structure_mapping.interpolated(0.0), fcc_structure
    )
    assert np.allclose(
        structure.atom_coordinate_frac(), np.array([[0.0, 0.0, 0.0]]).transpose()
    )
    assert np.allclose(
        structure.lattice().column_vector_matrix(),
        np.array(
            [
                [0.0, 0.0, -4.0 / sqrt(3.0)],
                [2.0 / sqrt(3.0), 2.0 / sqrt(3.0), -2.0 / sqrt(3.0)],
                [-2.0 / sqrt(3.0), 2.0 / sqrt(3.0), -2.0 / sqrt(3.0)],
            ]
        ).transpose(),
    )

    # f=0.2, interpolated
    structure = mapmethods.make_mapped_structure(
        structure_mapping.interpolated(0.2), fcc_structure
    )
    assert np.allclose(
        structure.atom_coordinate_frac(), np.array([[0.0, 0.0, 0.0]]).transpose()
    )
    assert np.allclose(
        structure.lattice().column_vector_matrix(),
        np.array(
            [
                [0.0, 0.0, -2.2475208614068025],
                [1.1237604307034013, 1.2066031431780204, -1.1237604307034013],
                [-1.1237604307034013, 1.2066031431780204, -1.1237604307034013],
            ]
        ).transpose(),
    )

    # f=1.0, end point == child
    structure = mapmethods.make_mapped_structure(
        structure_mapping.interpolated(1.0), fcc_structure
    )
    assert np.allclose(
        structure.atom_coordinate_frac(), np.array([[0.0, 0.0, 0.0]]).transpose()
    )
    assert np.allclose(
        structure.lattice().column_vector_matrix(),
        np.array(
            [
                [0.0, 0.0, -2.0],
                [1.0, sqrt(2.0), -1.0],
                [-1.0, sqrt(2.0), -1.0],
            ]
        ).transpose(),
    )


def test_interpolate_bcc_hcp_structures():
    prim = xtal_prims.BCC(r=1.0, occ_dof=["A", "B", "Va"])
    prim_factor_group = xtal.make_factor_group(prim)

    hcp_structure = xtal_structures.HCP(r=1.0, atom_type="A")

    structure_mapping = mapmethods.map_structures(
        prim,
        hcp_structure,
        prim_factor_group=prim_factor_group,
        max_vol=2,
        max_cost=1e20,
        min_cost=0.0,
    )[0]

    # It is not guaranteed that the particular mapping
    # orientation tested for here is always the first mapping
    # result. These tests may need to change.

    # f=0.0, end point == parent
    structure = mapmethods.make_mapped_structure(
        structure_mapping.interpolated(0.0), hcp_structure
    )
    assert np.allclose(
        structure.atom_coordinate_frac(),
        np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.5, 0.5],
            ]
        ).transpose(),
    )
    assert np.allclose(
        structure.atom_coordinate_cart(),
        np.array(
            [
                [0.0, 0.0, 0.0],
                [2 / sqrt(3.0), 2 / sqrt(3.0), -2 / sqrt(3.0)],
            ]
        ).transpose(),
    )
    assert np.allclose(
        structure.lattice().column_vector_matrix(),
        np.array(
            [
                [-2.0 / sqrt(3.0), -2.0 / sqrt(3.0), -2.0 / sqrt(3.0)],
                [0.0, 4.0 / sqrt(3.0), 0.0],
                [4.0 / sqrt(3.0), 0.0, -4.0 / sqrt(3.0)],
            ]
        ).transpose(),
    )

    # f=0.2, interpolated
    structure = mapmethods.make_mapped_structure(
        structure_mapping.interpolated(0.2), hcp_structure
    )
    assert np.allclose(
        structure.atom_coordinate_frac(),
        np.array(
            [
                [-3.33333333e-02, -1.66666667e-02, 0.0],
                [3.33333333e-02, 5.16666667e-01, 0.5],
            ]
        ).transpose(),
    )
    assert np.allclose(
        structure.atom_coordinate_cart(),
        np.array(
            [[0.0389569802, 0.0, 0.0389569802], [1.11574356, 1.12376043, -1.19365752]]
        ).transpose(),
    )
    assert np.allclose(
        structure.lattice().column_vector_matrix(),
        np.array(
            [
                [-1.1687094049817186, -1.123760430703401, -1.1687094049817184],
                [0.0, 2.2475208614068025, -4.440892098500626e-16],
                [2.309401076758503, 4.440892098500626e-16, -2.309401076758503],
            ]
        ).transpose(),
    )

    # f=1.0, end point == child
    structure = mapmethods.make_mapped_structure(
        structure_mapping.interpolated(1.0), hcp_structure
    )
    assert np.allclose(
        structure.atom_coordinate_frac(),
        np.array(
            [[-1.0 / 6.0, -1.0 / 12.0, 0.0], [1.0 / 6.0, 7.0 / 12.0, 1.0 / 2.0]]
        ).transpose(),
    )
    assert np.allclose(
        structure.atom_coordinate_cart(),
        np.array(
            [[0.204124145, 0.0, 0.204124145], [0.950576393, 1.0, -1.35882468]]
        ).transpose(),
    )
    assert np.allclose(
        structure.lattice().column_vector_matrix(),
        np.array(
            [
                [-sqrt(3.0 / 2.0), -1.0, -sqrt(3.0 / 2.0)],  # a
                [0.0, 2.0, 0.0],  # a
                [4 / sqrt(3.0), 0.0, -4 / sqrt(3.0)],  # c
            ]
        ).transpose(),
    )


def test_interpolate_bcc_hcp_poscars():
    prim = xtal_prims.BCC(r=1.0, occ_dof=["A", "B", "Va"])
    prim_factor_group = xtal.make_factor_group(prim)

    hcp_structure = xtal_structures.HCP(r=1.0, atom_type="A")

    structure_mapping = mapmethods.map_structures(
        prim,
        hcp_structure,
        prim_factor_group=prim_factor_group,
        max_vol=2,
        max_cost=1e20,
        min_cost=0.0,
    )[0]

    for x in np.linspace(0.0, 1.0, 5):
        structure = mapmethods.make_mapped_structure(
            structure_mapping.interpolated(x), hcp_structure
        )
        assert isinstance(structure, xtal.Structure)
