"""Test structure mapping"""

import math

import numpy as np

import libcasm.mapping.info as mapinfo
import libcasm.mapping.methods as mapmethods
import libcasm.xtal as xtal
import libcasm.xtal.prims as xtal_prims
import libcasm.xtal.structures as xtal_structures


def make_geometric_atom_cost(
    supercell_lattice_column_vector_matrix,
    displacement,
):
    """Calculate geometric atom cost from scratch"""
    S = supercell_lattice_column_vector_matrix
    n_site = displacement.shape[1]
    volume_per_site = np.abs(np.linalg.det(S)) / n_site
    displacement_squaredNorm = np.sum(displacement**2)
    geometric_atom_cost = (
        math.pow(3 * volume_per_site / (4.0 * np.pi), -2.0 / 3.0)
        * displacement_squaredNorm
        / n_site
    )
    # print("volume_per_site:", volume_per_site)
    # print("displacement_squaredNorm:", displacement_squaredNorm)
    return geometric_atom_cost


def make_isotropic_atom_cost(
    prim_lattice_column_vector_matrix,
    lattice_mapping,
    displacement,
):
    """Calculate isotropic atom cost from scratch"""
    L1 = prim_lattice_column_vector_matrix
    T = lattice_mapping.transformation_matrix_to_super()
    N = lattice_mapping.reorientation()
    U = lattice_mapping.right_stretch()

    S1 = L1 @ T @ N
    L2 = U @ S1
    d = displacement
    d_reverse = -U @ d

    geometric_atom_cost_forward = make_geometric_atom_cost(S1, d)
    geometric_atom_cost_reverse = make_geometric_atom_cost(L2, d_reverse)
    isotropic_atom_cost = (
        geometric_atom_cost_forward + geometric_atom_cost_reverse
    ) / 2.0
    # print("geometric_atom_cost_forward:", geometric_atom_cost_forward)
    # print("geometric_atom_cost_reverse:", geometric_atom_cost_reverse)
    return isotropic_atom_cost


def as_int(a: float):
    """Round floating point arrays that are approximately integer to integer arrays"""
    b = np.rint(a)
    if not np.allclose(a, b):
        raise Exception("Error converting to integer array: not approximately integer")
    return np.array(b, dtype=int)


def check_mapping(
    prim: xtal.Prim,
    structure: xtal.Structure,
    structure_mapping: mapinfo.ScoredStructureMapping,
):
    """Check that a structure mapping does map between a structure and a superstructure
    of a prim
    """
    # print("structure:")
    # print("lattice_column_vector_matrix:\n",
    #       structure.lattice().column_vector_matrix())
    # print("atom_coordinate_frac:\n", structure.atom_coordinate_frac().transpose())
    # print("atom_type:", structure.atom_type())
    mapped_structure = mapmethods.make_mapped_structure(structure_mapping, structure)
    mapped_structure_L = mapped_structure.lattice().column_vector_matrix()
    mapped_structure_atom_type = mapped_structure.atom_type()
    mapped_structure_atom_coordinate_frac = mapped_structure.atom_coordinate_frac()
    mapped_structure_atom_coordinate_cart = (
        mapped_structure_L @ mapped_structure_atom_coordinate_frac
    )

    # print("mapped_structure:")
    # print("lattice_column_vector_matrix:\n",
    #       mapped_structure.lattice().column_vector_matrix())
    # print("atom_coordinate_frac:\n",
    #       mapped_structure.atom_coordinate_frac().transpose())
    # print("atom_type:", mapped_structure_atom_type)

    # lattice mapping relation:
    # Q * U * L1 * T * N = L2
    lmap = structure_mapping.lattice_mapping()
    L1 = prim.lattice().column_vector_matrix()
    L2 = structure.lattice().column_vector_matrix()
    F = lmap.deformation_gradient()
    Q = lmap.isometry()
    U = lmap.right_stretch()
    T = as_int(lmap.transformation_matrix_to_super())
    N = as_int(lmap.reorientation())
    # print("F:\n", F)
    # print("Q:\n", Q)
    # print("U:\n", U)
    # print("T:\n", T)
    # print("N:\n", N)
    # print("L1 @ T @ N:\n", L1 @ T @ N)
    # print("U @ L1 @ T @ N:\n", U @ L1 @ T @ N)
    assert np.allclose(Q @ U @ L1 @ T @ N, L2)
    assert np.allclose(
        U @ L1 @ T @ N, mapped_structure.lattice().column_vector_matrix()
    )

    # atom mapping relation:
    # F ( r1(i) + disp(i) ) = r2(perm[i]) + trans
    # print("Check atom mapping:")
    prim_occ_dof = prim.occ_dof()
    prim_structure = xtal.Structure(
        lattice=prim.lattice(),
        atom_coordinate_frac=prim.coordinate_frac(),
        atom_type=[x[0] for x in prim_occ_dof],
    )

    ideal_superstructure = xtal.make_superstructure(T @ N, prim_structure)
    amap = structure_mapping.atom_mapping()
    r1 = ideal_superstructure.atom_coordinate_cart()
    r2 = structure.atom_coordinate_cart()
    disp = amap.displacement()
    perm = amap.permutation()
    trans = amap.translation()
    # print("disp:\n", disp.transpose())
    # print("perm:", perm)
    # print("trans:", trans)
    for i in range(r1.shape[1]):
        b = i % len(prim_occ_dof)
        assert mapped_structure_atom_type[i] in prim_occ_dof[b]
        if perm[i] >= r2.shape[1]:
            # implied vacancy
            assert mapped_structure_atom_type[i] == "Va"
        else:
            # check unmapped structure vs mapping with Q
            x1 = F @ (r1[:, i] + disp[:, i])
            x2 = r2[:, perm[i]] + trans
            d = xtal.min_periodic_displacement(structure.lattice(), x1, x2)
            assert math.isclose(np.linalg.norm(d), 0.0, abs_tol=1e-10)

            # check unmapped structure vs mapping without Q
            x1 = U @ (r1[:, i] + disp[:, i])
            x2 = mapped_structure_atom_coordinate_cart[:, i]
            d = xtal.min_periodic_displacement(structure.lattice(), x1, x2)
            assert math.isclose(np.linalg.norm(d), 0.0, abs_tol=1e-10)

    # atom mapping cost:
    # isotropic_atom_cost = make_isotropic_atom_cost(L1, lmap, disp)
    # print("isotropic_atom_cost:", isotropic_atom_cost)


def test_map_structures_0():
    """Map to self: max_cost==0.0 -> # of mappings == 48"""
    prim = xtal_prims.cubic(a=1.0, occ_dof=["A"])

    structure = xtal.Structure(
        lattice=prim.lattice(),
        atom_coordinate_frac=prim.coordinate_frac(),
        atom_type=["A"],
    )

    structure_mappings = mapmethods.map_structures(
        prim, structure, max_vol=1, max_cost=0.0, min_cost=0.0
    )
    assert len(structure_mappings) == 48
    for smap in structure_mappings:
        check_mapping(prim, structure, smap)


def test_map_structures_1():
    """Map to self: Use prim factor group -> # of tied-for-lowest cost mappings == 1"""
    prim = xtal_prims.cubic(a=1.0, occ_dof=["A"])
    prim_factor_group = xtal.make_factor_group(prim)

    structure = xtal.Structure(
        lattice=prim.lattice(),
        atom_coordinate_frac=prim.coordinate_frac(),
        atom_type=["A"],
    )

    structure_mappings = mapmethods.map_structures(
        prim,
        structure,
        prim_factor_group=prim_factor_group,
        max_vol=1,
        max_cost=0.0,
        min_cost=0.0,
    )
    assert len(structure_mappings) == 1
    for smap in structure_mappings:
        check_mapping(prim, structure, smap)


def test_map_structures_2():
    """Map to superstructure of self: 2 equally perfect permutations"""
    prim = xtal_prims.cubic(a=1.0, occ_dof=["A"])
    prim_factor_group = xtal.make_factor_group(prim)

    unit_structure = xtal.Structure(
        lattice=prim.lattice(),
        atom_coordinate_frac=prim.coordinate_frac(),
        atom_type=["A"],
    )

    T = np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 2],
        ],
        dtype=int,
    )
    structure = xtal.make_superstructure(T, unit_structure)

    structure_mappings = mapmethods.map_structures(
        prim,
        structure,
        prim_factor_group=prim_factor_group,
        max_vol=2,
        max_cost=0.0,
        min_cost=0.0,
    )
    assert len(structure_mappings) == 2  # should this be 1?
    for smap in structure_mappings:
        check_mapping(prim, structure, smap)


def test_map_structures_3():
    """Map ordered structure to self: 1 perfect mapping"""
    prim_lattice = xtal.Lattice(
        np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 2.0],
            ]
        ).transpose()
    )
    coordinate_frac = np.array(
        [
            [0.0, 0.25, 0.25],
            [0.0, 0.25, 0.75],
        ]
    ).transpose()
    occ_dof = [
        ["A"],
        ["B"],
    ]
    prim = xtal.Prim(
        lattice=prim_lattice, coordinate_frac=coordinate_frac, occ_dof=occ_dof
    )
    prim_factor_group = xtal.make_factor_group(prim)

    structure = xtal.Structure(
        lattice=prim.lattice(),
        atom_coordinate_frac=prim.coordinate_frac(),
        atom_type=["A", "B"],
    )

    structure_mappings = mapmethods.map_structures(
        prim,
        structure,
        prim_factor_group=prim_factor_group,
        max_vol=2,
        max_cost=0.0,
        min_cost=0.0,
    )
    assert len(structure_mappings) == 1
    check_mapping(prim, structure, structure_mappings[0])


def test_map_structures_4():
    """Map to Ezz && Eyz"""
    prim = xtal_prims.cubic(a=1.0, occ_dof=["A"])
    prim_factor_group = xtal.make_factor_group(prim)

    structure_lattice = xtal.Lattice(
        np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.1],
                [0.0, 0.0, 1.1],
            ]
        ).transpose()
    )
    structure = xtal.Structure(
        lattice=structure_lattice,
        atom_coordinate_frac=prim.coordinate_frac(),
        atom_type=["A"],
    )

    structure_mappings = mapmethods.map_structures(
        prim,
        structure,
        prim_factor_group=prim_factor_group,
        max_vol=1,
        max_cost=0.1,
        min_cost=0.0,
    )
    assert len(structure_mappings) == 1
    structure_mapping = structure_mappings[0]
    U = structure_mapping.lattice_mapping().right_stretch()
    assert np.allclose(
        U,
        np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.00362465, 0.05232166],
                [0.0, 0.05232166, 1.09875495],
            ]
        ),
    )

    check_mapping(prim, structure, structure_mappings[0])


def test_map_structures_5():
    """Map to BCC: 1 implicit vacancy"""
    prim = xtal_prims.BCC(r=1.0, occ_dof=["A", "B", "Va"])
    prim_factor_group = xtal.make_factor_group(prim)
    prim_occ_dof = prim.occ_dof()
    L1 = prim.lattice().column_vector_matrix()

    # from scipy.spatial.transform import Rotation
    # Qi = Rotation.from_euler("z", 30, degrees=True).as_matrix()
    Qi = np.array([[0.8660254, -0.5, 0.0], [0.5, 0.8660254, 0.0], [0.0, 0.0, 1.0]])
    Ui = np.array([[1.01, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    Fi = Qi @ Ui

    T = np.eye(3, dtype=int) * 2
    prim_structure = xtal.Structure(
        lattice=prim.lattice(),
        atom_coordinate_frac=prim.coordinate_frac(),
        atom_type=[x[0] for x in prim_occ_dof],
    )
    ideal_superstructure = xtal.make_superstructure(T, prim_structure)

    structure_lattice = xtal.Lattice(Fi @ L1 @ T)
    disp_frac = np.array(
        [
            [0.01, -0.01, 0.01],
            [0.00, 0.01, -0.01],
            [0.01, 0.00, -0.01],
            [-0.01, 0.01, 0.0],
            [-0.01, 0.00, 0.01],
            [0.0, 0.00, -0.01],
            [0.01, 0.00, 0.0],
        ]
    ).transpose()
    atom_coordinate_frac = (
        ideal_superstructure.atom_coordinate_frac()[:, 1:] + disp_frac
    )
    structure = xtal.Structure(
        lattice=structure_lattice,
        atom_coordinate_frac=atom_coordinate_frac,
        atom_type=["A"] * 7,
    )

    structure_mappings = mapmethods.map_structures(
        prim,
        structure,
        prim_factor_group=prim_factor_group,
        max_vol=8,
        min_vol=8,
        max_cost=1e20,
        min_cost=0.0,
    )

    assert len(structure_mappings) == 1
    check_mapping(prim, structure, structure_mappings[0])


def test_bcc_fcc_mapping():
    prim = xtal_prims.BCC(r=1.0, occ_dof=["A", "B", "Va"])
    prim_factor_group = xtal.make_factor_group(prim)

    fcc_structure = xtal_structures.FCC(r=1.0)

    structure_mappings = mapmethods.map_structures(
        prim,
        fcc_structure,
        prim_factor_group=prim_factor_group,
        max_vol=2,
        max_cost=1e20,
        min_cost=0.0,
    )

    assert len(structure_mappings)
    for i, smap in enumerate(structure_mappings):
        assert math.isclose(smap.lattice_cost(), 0.027319751852797797)
        assert math.isclose(smap.atom_cost(), 0.0)
        check_mapping(prim, fcc_structure, smap)


def test_bcc_hcp_mapping():
    prim = xtal_prims.BCC(r=1.0, occ_dof=["A", "B", "Va"])
    prim_factor_group = xtal.make_factor_group(prim)

    hcp_structure = xtal_structures.HCP(r=1.0, atom_type="A")

    structure_mappings = mapmethods.map_structures(
        prim,
        hcp_structure,
        prim_factor_group=prim_factor_group,
        max_vol=4,
        max_cost=1e20,
        min_cost=0.0,
    )

    assert len(structure_mappings)
    for i, smap in enumerate(structure_mappings):
        check_mapping(prim, hcp_structure, smap)
        assert math.isclose(smap.lattice_cost(), 0.007297079413597657)
        assert math.isclose(smap.atom_cost(), 0.06274848406141671)


def test_make_mapped_structure_0(shared_datadir):
    import json

    # set up prim
    with open(shared_datadir / "AB_test" / "prim.json", "r") as f:
        xtal_prim = xtal.Prim.from_dict(json.load(f))
    prim_factor_group = xtal.make_factor_group(xtal_prim)

    # get the relaxed structure (includes properties such as energy)
    relaxed_properties_json = shared_datadir / "AB_test" / "properties.calc.json"
    with open(relaxed_properties_json, "r") as f:
        props = json.load(f)
        relaxed_structure = xtal.Structure.from_dict(props)

    # get lattice mapping (should have 0 cost) and 2 atom mappings
    # then, generate the 2 equivalent structures from the 2 atom mappings
    transformation_matrix_to_super = xtal.make_transformation_matrix_to_super(
        superlattice=relaxed_structure.lattice(), unit_lattice=xtal_prim.lattice()
    )
    lattice_mapping = mapmethods.map_lattices_without_reorientation(
        xtal_prim.lattice(),
        relaxed_structure.lattice(),
        transformation_matrix_to_super=transformation_matrix_to_super,
    )
    atom_mapping = mapmethods.map_atoms(
        prim=xtal_prim,
        structure=relaxed_structure,
        lattice_mapping=lattice_mapping,
        prim_factor_group=prim_factor_group,
    )
    ## 0 lattice cost
    assert np.isclose(
        mapinfo.isotropic_strain_cost(lattice_mapping.deformation_gradient()),
        0,
        atol=1e-10,
    )
    ## 2 atom mappings
    assert len(atom_mapping) == 2
    ## check that atom mappings have the same cost, and then make the mapped structures
    _atom_cost = atom_mapping[0].atom_cost()
    for i in atom_mapping:
        assert np.isclose(_atom_cost, i.atom_cost(), atol=1e-10)
        structure_mapping = mapinfo.StructureMapping(
            prim=xtal_prim, lattice_mapping=lattice_mapping, atom_mapping=i
        )

    # print(structure_mapping.to_dict())

    # attempt to make the mapped structure
    mapped_structure = mapmethods.make_mapped_structure(
        structure_mapping=structure_mapping,
        unmapped_structure=relaxed_structure,
    )

    assert isinstance(mapped_structure, xtal.Structure)

    # make sure the forces in the mapped_structure are non-zero
    assert not np.allclose(
        np.zeros(mapped_structure.atom_properties()["force"].shape),
        mapped_structure.atom_properties()["force"],
    )
