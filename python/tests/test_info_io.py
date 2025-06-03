import numpy as np
import pytest

import libcasm.mapping.info as info
import libcasm.xtal as xtal
import libcasm.xtal.prims as xtal_prims
import libcasm.xtal.structures as xtal_structures
from libcasm.mapping.methods import map_atoms, map_lattices, map_structures


def test_structure_mapping_io():
    prim = xtal_prims.BCC(r=1.0, occ_dof=["A", "B", "Va"])
    prim_factor_group = xtal.make_factor_group(prim)

    fcc_structure = xtal_structures.FCC(r=1.0, atom_type="A")

    structure_mappings = map_structures(
        prim,
        fcc_structure,
        prim_factor_group=prim_factor_group,
        max_vol=2,
        max_cost=1e20,
        min_cost=0.0,
    )

    # StructureMappingResults
    assert len(structure_mappings) == 3
    assert isinstance(structure_mappings, info.StructureMappingResults)
    data = structure_mappings.to_dict()
    assert isinstance(data, list)
    obj = info.StructureMappingResults.from_dict(data=data, prim=prim)
    assert isinstance(obj, info.StructureMappingResults)
    assert len(obj) == 3

    # ScoredStructureMapping
    scored = structure_mappings[0]
    assert isinstance(scored, info.ScoredStructureMapping)
    data = scored.to_dict()
    assert isinstance(data, dict)
    obj = info.ScoredStructureMapping.from_dict(data=data, prim=prim)
    assert isinstance(obj, info.ScoredStructureMapping)

    # StructureMapping
    smap = info.StructureMapping(prim, scored.lattice_mapping(), scored.atom_mapping())
    assert isinstance(smap, info.StructureMapping)
    data = smap.to_dict()
    assert isinstance(data, dict)
    obj = info.StructureMapping.from_dict(data=data, prim=prim)
    assert isinstance(obj, info.StructureMapping)

    # LatticeMapping
    lmap = scored.lattice_mapping()
    assert isinstance(lmap, info.LatticeMapping)
    data = lmap.to_dict()
    assert isinstance(data, dict)
    obj = info.LatticeMapping.from_dict(data)
    assert isinstance(obj, info.LatticeMapping)

    # AtomMapping
    amap = scored.atom_mapping()
    assert isinstance(amap, info.AtomMapping)
    data = amap.to_dict()
    assert isinstance(data, dict)
    obj = info.AtomMapping.from_dict(data)
    assert isinstance(obj, info.AtomMapping)

    # Do not segfault on out of bounds index
    with pytest.raises(IndexError):
        structure_mappings[len(structure_mappings)]


def test_lattice_and_atom_mapping_io():
    prim = xtal_prims.BCC(r=1.0, occ_dof=["A", "B", "Va"])
    prim_factor_group = xtal.make_factor_group(prim)
    point_group = xtal.make_crystal_point_group(prim)

    hcp_structure = xtal_structures.HCP(r=1.0, atom_type="A")

    # vol=2 supercell of bcc maps to hcp
    T = np.array([[-1, 1, -1], [-1, 0, 0], [-1, 1, 1]])

    lattice_mappings = map_lattices(
        lattice1=prim.lattice(),
        lattice2=hcp_structure.lattice(),
        transformation_matrix_to_super=T,
        lattice1_point_group=point_group,
        max_cost=1e20,
        min_cost=0.0,
        k_best=1,
    )

    # LatticeMappingResults
    assert len(lattice_mappings) == 3
    assert isinstance(lattice_mappings, info.LatticeMappingResults)
    data = lattice_mappings.to_dict()
    assert isinstance(data, list)
    obj = info.LatticeMappingResults.from_dict(data)
    assert isinstance(obj, info.LatticeMappingResults)
    assert len(obj) == 3

    # ScoredLatticeMapping
    scored = lattice_mappings[0]
    assert isinstance(scored, info.ScoredLatticeMapping)
    data = scored.to_dict()
    assert isinstance(data, dict)
    obj = info.ScoredLatticeMapping.from_dict(data)
    assert isinstance(obj, info.ScoredLatticeMapping)

    # Do not segfault on out of bounds index
    with pytest.raises(IndexError):
        lattice_mappings[len(lattice_mappings)]

    atom_mappings = map_atoms(
        prim=prim,
        structure=hcp_structure,
        lattice_mapping=lattice_mappings[0],
        prim_factor_group=prim_factor_group,
        atom_cost_method="isotropic_disp_cost",
        max_cost=1e20,
        min_cost=0.0,
        k_best=10,
    )

    # AtomMappingResults
    assert len(atom_mappings) == 2
    assert isinstance(atom_mappings, info.AtomMappingResults)
    data = atom_mappings.to_dict()
    assert isinstance(data, list)
    obj = info.AtomMappingResults.from_dict(data)
    assert isinstance(obj, info.AtomMappingResults)
    assert len(obj) == 2

    # ScoredAtomMapping
    scored = atom_mappings[0]
    assert isinstance(scored, info.ScoredAtomMapping)
    data = scored.to_dict()
    assert isinstance(data, dict)
    obj = info.ScoredAtomMapping.from_dict(data)
    assert isinstance(obj, info.ScoredAtomMapping)

    # Do not segfault on out of bounds index
    with pytest.raises(IndexError):
        atom_mappings[len(lattice_mappings)]
