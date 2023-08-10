Usage
=====

The :func:`~libcasm.mapping.methods.map_structures` method finds mappings of a crystal structure (:class:`~libcasm.xtal.Structure`) with atomic occupants to a CASM :class:`~libcasm.xtal.Prim`, which specifies a primitive crystal structure and allowed occupation degrees of freedom (DoF).

The result of a mapping is a :class:`~libcasm.mapping.info.StructureMappingResults` object, giving possible structure mappings, sorted by total cost. Each structure mapping consists of a lattice mapping (see :class:`~libcasm.mapping.info.LatticeMapping`) with a lattice mapping cost, and an atom mapping (see :class:`~libcasm.mapping.info.AtomMapping`) with an atom mapping cost. The total cost is a weighted sum of the lattice and atom mapping costs.

The section provides a few example use cases. To see all the options for the standard structure, lattice, and atom mapping methods, see the reference pages:

- :func:`~libcasm.mapping.methods.map_structures`
- :func:`~libcasm.mapping.methods.map_lattices`
- :func:`~libcasm.mapping.methods.map_atoms`


Example 1: A perfect mapping, no symmetry information provided
--------------------------------------------------------------

This example maps a simple cubic structure that is simply rotated relative to the prim definition. No symmetry information is provided to :func:`~libcasm.mapping.methods.map_structures`, so it finds each of the 48 symmetrically equivalent ways to map the structure back to the prim.

.. code-block:: Python

    import libcasm.mapping.methods as mapmethods
    import libcasm.xtal as xtal
    import libcasm.xtal.prims as xtal_prims
    from scipy.spatial.transform import Rotation

    prim = xtal_prims.cubic(a=1.0, occ_dof=["A"])
    print("prim:", prim.to_dict())

    Qi = Rotation.from_euler("z", 30, degrees=True).as_matrix()
    structure_lattice = xtal.Lattice(Qi @ prim.lattice().column_vector_matrix())

    structure = xtal.Structure(
        lattice=structure_lattice,
        atom_coordinate_frac=prim.coordinate_frac(),
        atom_type=["A"],
    )
    print("structure:", structure.to_dict())

    structure_mappings = mapmethods.map_structures(
        prim, structure, max_vol=1, max_cost=0.0, min_cost=0.0
    )
    print(len(structure_mappings))
    for i, m in enumerate(structure_mappings):
        print("mapping:", i, "total_cost:", m.total_cost())


::

    prim: {'basis': [{'coordinate': [0.0, 0.0, 0.0], 'occupants': ['A']}], 'coordinate_mode': 'Fractional', 'lattice_vectors': [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], 'title': 'prim'}
    structure: {'atom_coords': [[0.0, 0.0, 0.0]], 'atom_type': ['A'], 'coordinate_mode': 'Cartesian', 'lattice_vectors': [[0.8660254037844387, 0.49999999999999994, 0.0], [-0.49999999999999994, 0.8660254037844387, 0.0], [0.0, 0.0, 1.0]]}
    48
    mapping: 0 total_cost: 0.0
    mapping: 1 total_cost: 0.0
    mapping: 2 total_cost: 0.0
    mapping: 3 total_cost: 0.0
    ...
    mapping: 45 total_cost: 0.0
    mapping: 46 total_cost: 0.0
    mapping: 47 total_cost: 0.0


Example 2: A perfect mapping, with symmetry information provided
----------------------------------------------------------------

This example also maps a simple cubic structure that is simply rotated relative to the prim definition, but now the prim factor group is provided to :func:`~libcasm.mapping.methods.map_structures`, reducing the number of checks that need to be made to find a mapping. Now :func:`~libcasm.mapping.methods.map_structures` returns just one of the 48 symmetrically equivalent mapping results found in the first example.

.. code-block:: Python

    import libcasm.mapping.methods as mapmethods
    import libcasm.xtal as xtal
    import libcasm.xtal.prims as xtal_prims
    from scipy.spatial.transform import Rotation

    prim = xtal_prims.cubic(a=1.0, occ_dof=["A"])
    prim_factor_group = xtal.make_factor_group(prim)
    print("prim:", prim.to_dict())

    Qi = Rotation.from_euler("z", 30, degrees=True).as_matrix()
    structure_lattice = xtal.Lattice(Qi @ prim.lattice().column_vector_matrix())

    structure = xtal.Structure(
        lattice=structure_lattice,
        atom_coordinate_frac=prim.coordinate_frac(),
        atom_type=["A"],
    )
    print("structure:", structure.to_dict())

    structure_mappings = mapmethods.map_structures(
        prim,
        structure,
        prim_factor_group=prim_factor_group,
        max_vol=1,
        max_cost=0.0,
        min_cost=0.0,
    )
    print(len(structure_mappings))
    for i, m in enumerate(structure_mappings):
        print("mapping:", i, "total_cost:", m.total_cost())

::

    prim: {'basis': [{'coordinate': [0.0, 0.0, 0.0], 'occupants': ['A']}], 'coordinate_mode': 'Fractional', 'lattice_vectors': [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], 'title': 'prim'}
    structure: {'atom_coords': [[0.0, 0.0, 0.0]], 'atom_type': ['A'], 'coordinate_mode': 'Cartesian', 'lattice_vectors': [[0.8660254037844387, 0.49999999999999994, 0.0], [-0.49999999999999994, 0.8660254037844387, 0.0], [0.0, 0.0, 1.0]]}
    1
    mapping: 0 total_cost: 0.0


Example 3: Mapping an HCP structure to a BCC prim
-------------------------------------------------

This example maps an HCP structure to a BCC prim. The transformation pathway is known as the Burgers path. In this example, the structure (which has 2 atoms) maps to a supercell of the prim (which has 1 basis site). As a result, multiple mappings with equivalent mappings scores are returned which map the structure to different supercells of the prim.

For the definition of the mapping results, see:

- :func:`~libcasm.mapping.info.ScoredStructureMapping`

.. code-block:: Python

    import libcasm.mapping.methods as mapmethods
    import libcasm.xtal as xtal
    import libcasm.xtal.prims as xtal_prims
    import libcasm.xtal.structures as xtal_structures

    prim = xtal_prims.BCC(r=1.0, occ_dof=["A", "B", "Va"])
    prim_factor_group = xtal.make_factor_group(prim)

    hcp_structure = xtal_structures.HCP(r=1.0, atom_type="A")

    structure_mappings = mapmethods.map_structures(
        prim,
        hcp_structure,
        prim_factor_group=prim_factor_group,
        max_vol=4,
        k_best=1,
    )

    print(len(structure_mappings))
    for i, m in enumerate(structure_mappings):
        print("mapping:", i, "total_cost:", m.total_cost())
        data = m.to_dict()
        for key in data:
            print(key, data[key])
        print()

::

    4
    mapping: 0 total_cost: 0.035022781737507204
    atom_cost 0.0627484840614167
    deformation_gradient [[-0.6495190528383288, -0.43301270189221924, -0.6495190528383289], [-0.3749999999999999, 0.7499999999999996, -0.37499999999999967], [0.7071067811865471, -2.294561174923997e-16, -0.7071067811865472]]
    displacement [[0.19245008972987537, -4.440892098500626e-16, 0.1924500897298752], [-0.19245008972987537, 4.440892098500626e-16, -0.1924500897298752]]
    isometry [[-0.6123724356957944, -0.5, -0.6123724356957944], [-0.3535533905932736, 0.8660254037844388, -0.3535533905932736], [0.7071067811865475, -2.5807503295505684e-17, -0.7071067811865476]]
    lattice_cost 0.0072970794135977035
    left_stretch [[1.012001479780975, 0.0842793267718456, 1.6653345369377348e-16], [0.08427932677184571, 0.9146840957832839, -1.1102230246251565e-16], [1.6653345369377348e-16, -1.3877787807814457e-16, 0.9999999999999996]]
    permutation [0, 1]
    reorientation [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    right_stretch [[1.0303300858899105, -6.001097268289729e-17, 0.030330085889910874], [-6.001097268289739e-17, 0.8660254037844382, 2.328813733397368e-16], [0.030330085889910874, 2.328813733397366e-16, 1.0303300858899105]]
    total_cost 0.035022781737507204
    transformation_matrix_to_supercell [[-1, 1, -1], [-1, 0, 0], [-1, 1, 1]]
    translation [-0.24999999999999967, -1.299038105676658, -0.8164965809277257]

    mapping: 1 total_cost: 0.0350227817375071
    atom_cost 0.06274848406141649
    deformation_gradient [[-0.6495190528383288, 0.43301270189221924, -0.6495190528383289], [0.3749999999999999, 0.7499999999999996, 0.37499999999999967], [0.7071067811865471, 2.294561174923997e-16, -0.7071067811865472]]
    displacement [[-0.19245008972987498, 5.551115123125783e-16, -0.19245008972987498], [0.19245008972987498, -5.551115123125783e-16, 0.19245008972987498]]
    isometry [[-0.6123724356957944, 0.5, -0.6123724356957944], [0.3535533905932736, 0.8660254037844388, 0.3535533905932736], [0.7071067811865475, 2.5807503295505684e-17, -0.7071067811865476]]
    lattice_cost 0.0072970794135977035
    left_stretch [[1.012001479780975, -0.0842793267718456, 1.6653345369377348e-16], [-0.08427932677184571, 0.9146840957832839, 1.1102230246251565e-16], [1.6653345369377348e-16, 1.3877787807814457e-16, 0.9999999999999996]]
    permutation [0, 1]
    reorientation [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    right_stretch [[1.0303300858899105, 6.001097268289729e-17, 0.030330085889910874], [6.001097268289739e-17, 0.8660254037844382, -2.328813733397368e-16], [0.030330085889910874, -2.328813733397366e-16, 1.0303300858899105]]
    total_cost 0.0350227817375071
    transformation_matrix_to_supercell [[0, 1, -1], [-1, 1, 0], [0, 1, 1]]
    translation [0.24999999999999964, -1.2990381056766571, -0.8164965809277259]

    mapping: 2 total_cost: 0.035022781737507114
    atom_cost 0.06274848406141657
    deformation_gradient [[-0.0, 0.8660254037844385, 0.0], [-0.7500000000000001, 0.0, -0.7499999999999999], [0.7071067811865474, -0.0, -0.7071067811865474]]
    displacement [[0.19245008972987487, 0.0, 0.19245008972987532], [-0.19245008972987487, 0.0, -0.19245008972987532]]
    isometry [[0.0, 1.0000000000000002, 0.0], [-0.7071067811865478, 0.0, -0.7071067811865476], [0.7071067811865476, 0.0, -0.7071067811865476]]
    lattice_cost 0.007297079413597657
    left_stretch [[0.8660254037844387, 0.0, 0.0], [0.0, 1.0606601717798214, -1.1102230246251565e-16], [0.0, -2.220446049250313e-16, 0.9999999999999998]]
    permutation [1, 0]
    reorientation [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    right_stretch [[1.0303300858899105, 0.0, 0.030330085889910707], [0.0, 0.8660254037844385, 0.0], [0.030330085889910707, 0.0, 1.0303300858899105]]
    total_cost 0.035022781737507114
    transformation_matrix_to_supercell [[1, -1, -1], [0, -1, 0], [1, -1, 1]]
    translation [-1.0, -0.8660254037844383, -2.4494897427831774]

    mapping: 3 total_cost: 0.03502278173750718
    atom_cost 0.06274848406141671
    deformation_gradient [[-0.0, 0.8660254037844385, 0.0], [-0.7500000000000001, 0.0, -0.7499999999999999], [0.7071067811865474, -0.0, -0.7071067811865474]]
    displacement [[-0.1924500897298751, 4.440892098500626e-16, -0.19245008972987554], [0.1924500897298751, -4.440892098500626e-16, 0.19245008972987554]]
    isometry [[0.0, 1.0000000000000002, 0.0], [-0.7071067811865478, 0.0, -0.7071067811865476], [0.7071067811865476, 0.0, -0.7071067811865476]]
    lattice_cost 0.007297079413597657
    left_stretch [[0.8660254037844387, 0.0, 0.0], [0.0, 1.0606601717798214, -1.1102230246251565e-16], [0.0, -2.220446049250313e-16, 0.9999999999999998]]
    permutation [0, 1]
    reorientation [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    right_stretch [[1.0303300858899105, 0.0, 0.030330085889910707], [0.0, 0.8660254037844385, 0.0], [0.030330085889910707, 0.0, 1.0303300858899105]]
    total_cost 0.03502278173750718
    transformation_matrix_to_supercell [[1, -1, -1], [0, -1, 0], [1, -1, 1]]
    translation [3.845925372767127e-16, -0.8660254037844384, -0.8164965809277255]


Example 4: Interpolating between an HCP structure and the BCC prim
------------------------------------------------------------------

Given a structure mapping, CASM can generate interpolated structures between the parent (prim) and child (unmapped structure). The following example generates the parent structure (BCC), child structure (HCP), and an intermediate structure from the BCC to HCP structure mapping and prints a VASP POSCAR for each.

.. code-block :: Python

    import libcasm.mapping.methods as mapmethods
    import libcasm.xtal as xtal
    import libcasm.xtal.prims as xtal_prims
    import libcasm.xtal.structures as xtal_structures

    prim = xtal_prims.BCC(r=1.0, occ_dof=["A", "B", "Va"])
    prim_factor_group = xtal.make_factor_group(prim)

    hcp_structure = xtal_structures.HCP(r=1.0, atom_type="A")

    structure_mappings = mapmethods.map_structures(
        prim,
        hcp_structure,
        prim_factor_group=prim_factor_group,
        max_vol=4,
        k_best=1,
    )

    m = structure_mappings[0]

    # f=0.0, end point == parent
    structure = mapmethods.make_mapped_structure(
        m.interpolated(0.0), hcp_structure
    )
    print("2-atom bcc:")
    print(structure.to_poscar_str())

    # f=1.0, end point == child
    print("hcp:")
    structure = mapmethods.make_mapped_structure(
        m.interpolated(1.0), hcp_structure
    )
    print(structure.to_poscar_str())

    # f=0.5, intermediate point
    print("2-atom intermediate:")
    structure = mapmethods.make_mapped_structure(
        m.interpolated(0.5), hcp_structure
    )
    print(structure.to_poscar_str())

::

    2-atom bcc:
    <title>
    1.00000000
    -1.15470054 -1.15470054 -1.15470054
    0.00000000 2.30940108 0.00000000
    2.30940108 0.00000000 -2.30940108
    A
    2
    Direct
    0.00000000 0.00000000 0.00000000 A
    0.00000000 0.50000000 0.50000000 A


    hcp:
    <title>
    1.00000000
    -1.22474487 -1.00000000 -1.22474487
    0.00000000 2.00000000 0.00000000
    2.30940108 -0.00000000 -2.30940108
    A
    2
    Direct
    -0.16666667 -0.08333333 0.00000000 A
    0.16666667 0.58333333 0.50000000 A


    2-atom intermediate:
    <title>
    1.00000000
    -1.18972270 -1.07735027 -1.18972270
    0.00000000 2.15470054 0.00000000
    2.30940108 -0.00000000 -2.30940108
    A
    2
    Direct
    -0.08333333 -0.04166667 0.00000000 A
    0.08333333 0.54166667 0.50000000 A
