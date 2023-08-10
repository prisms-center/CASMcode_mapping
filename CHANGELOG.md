# Changelog

All notable changes to `libcasm-mapping` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
