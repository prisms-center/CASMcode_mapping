# Changelog

All notable changes to `libcasm-mapping` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.2.0] - 2025-08-14

### Changed

- Set pybind11~=3.0


## [2.1.0] - 2025-08-07

### Changed

- Build Linux wheels using manylinux_2_28 (previously manylinux2014)
- Removed Cirrus CI testing


## [2.0.1] - 2025-06-03

### Fixed

- Fixed `StructureMapping.from_dict`, `ScoredStructureMapping.from_dict`, `StructureMappingResults.from_dict` so that `prim` and `data` argument names are no longer mixed up


## [2.0.0] - 2025-05-02

### Fixed

- Fixed a bug in `make_mapped_structure` which was always returning zero for atom properties

### Changed

- Build for Python 3.13
- Restrict requires-python to ">=3.9,<3.14"
- Run CI tests using Python 3.13
- Build MacOS arm64 wheels using MacOS 15
- Build Linux wheels using Ubuntu 24.04


## [v2.0a6] - 2024-09-05

### Fixed

- Fixed `make_mapped_structure` to properly handle atom properties (i.e. forces) with implied vacancies.


## [v2.0a5] - 2024-08-18

### Added

- Added `isotropic_strain_cost` and `symmetry_breaking_strain_cost` to `libcasm.mapping.info`.
- Added `map_lattices_without_reorientation` to `libcasm.mapping.methods`.

### Fixed

- Fixed the displacement cost descriptions in `libcasm.mapping.methods.map_atoms`.


## [v2.0a4] - 2024-07-12

### Added

- Added `cost` methods to IsotropicAtomCost, SymmetryBreakingAtomCost, and WeightedTotalCost equivalent to the __call__ operators to improve documentation.

### Changed

- Changed default `atom_cost_f` for libcasm.mapping.mapsearch.MappingSearch to IsotropicAtomCost.
- Changed MappingSearch node insertion so that nodes are only added to the queue if the total cost is less than or equal to the current max_cost instead of always.
- Wheels compiled with numpy>=2.0.0

### Fixed

- Documentation errors
- Renamed QueueConstraints constructor parameter which was misnamed `min_queue_size` to the correct name `max_queue_size`
- Fixed MappingSearch node insertion so that solutions are only added to the results if the total cost is less than or equal to the current max_cost even if the current number of results is less than k_best.


## [v2.0a3] - 2024-03-14

### Added

- Build wheels for python3.12

### Changed

- Updated libcasm-global dependency version to 2.0.4
- Updated libcasm-xtal dependency version to 2.0a9 


## [v2.0a2] - 2023-12-01

### Fixed

- Fix setting atom coordinates in CASM::mapping::make_mapped_structure (Fixes #13) and update tests

### Changed

- Updated libcasm-xtal dependency version to 2.0a8
- Update tests to stricter libcasm.xtal integer array parameter type 


## [v2.0a1] - 2023-08-17

This release separates out mapping methods from CASM crystallography v1, adds data structures for holding results, adds additional methods for working with mapping results, and provides an alternative mapping search implementation that can be used to construct custom searches. It creates a Python package, libcasm.mapping, that enables using the mapping methods and may be installed via pip install, using scikit-build, CMake, and pybind11. This release also includes usage examples and API documentation for using libcasm.mapping, built using Sphinx.

### Added

- Added the casm/mapping C++ module with new data structures and methods in namespace CASM::mapping, and with CASM v1 mapping code under namespace CASM::mapping::impl
- Added mapping::LatticeMapping, mapping::AtomMapping, and mapping::StructureMapping data structures for holding individual mapping results
- Added mapping::ScoredLatticeMapping, mapping::ScoredAtomMapping, and mapping::ScoredStructureMapping data structures for holding mapping results along with mapping costs
- Added mapping::LatticeMappingResults, mapping::AtomMappingResults, and mapping::StructureMappingResults data structures for holding multiple mapping results along with mapping costs
- Added mapping::interpolated_mapping for constructing structures that are interpolated between two structures along a particular mapping path
- Added mapping::make_mapping_to_equivalent_structure to generate symmetrically equivalent mappings
- Added mapping::make_mapping_to_equivalent_superlattice to generate mappings to a different, but symmetrically equivalent superlattice of the prim
- Added mapping::make_mapped_structure for applying a mapping::StructureMapping result to an unmapped structure to construct the mapped structure aligned to the parent crystal structure, with implied vacancies, strain, and atomic displacement; This can be used be copied to configuration degree of freedom (DoF) values and calculated properties.
- Added mapping::MappingSearch class and related data structures and methods to perform the mapping search method
- Added Python package libcasm.mapping.info for mapping data structures
- Added Python package libcasm.mapping.methods for the existing lattice, atom, and structure mapping methods
- Added Python package libcasm.mapping.mapsearch for the customizeable mapping search data structures and methods
- Added GitHub Actions for unit testing
- Added GitHub Action build_wheels.yml for Python wheel building using cibuildwheel
- Added Python documentation


### Removed

- Removed autotools build process
- Removed boost dependencies
