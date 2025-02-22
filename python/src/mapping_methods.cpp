#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "casm/crystallography/BasicStructure.hh"
#include "casm/crystallography/Lattice.hh"
#include "casm/crystallography/SimpleStructure.hh"
#include "casm/crystallography/SymType.hh"
#include "casm/mapping/AtomMapping.hh"
#include "casm/mapping/LatticeMapping.hh"
#include "casm/mapping/StructureMapping.hh"
#include "casm/mapping/map_atoms.hh"
#include "casm/mapping/map_lattices.hh"
#include "casm/mapping/map_structures.hh"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

/// CASM - Python binding code
namespace CASMpy {

using namespace CASM;
using namespace CASM::mapping;

}  // namespace CASMpy

PYBIND11_MODULE(_mapping_methods, m) {
  using namespace CASMpy;

  m.doc() = R"pbdoc(
        Lattice, atom, and structure mapping methods

        libcasm.mapping.methods
        -----------------------

        The libcasm.mapping.methods module contains lattice, atom, and
        structure mapping methods.

    )pbdoc";
  py::module_::import("libcasm.xtal");
  py::module_::import("libcasm.mapping.info");

  m.def("map_lattices", &map_lattices, R"pbdoc(
      Find mappings between two lattices

      This method finds mappings from a superlattice of a reference "parent"
      lattice to a "child" lattice. The lattice mappings have the form:

      .. math::

          F L_1 T N = L_2

      where:

      - :math:`L_1` is a shape=(3,3) matrix with columns containing the
        reference "parent" lattice vectors.
      - :math:`L_2` is a shape=(3,3) matrix with columns containing the
        "child" lattice vectors.
      - :math:`F` is the parent-to-child deformation gradient tensor,
        a shape=(3,3) matrix. :math:`F` can be decomposed as:

        .. math::

            F = Q U = V Q

        into a pure rigid transformation component, :math:`Q`, (called
        an isometry) and either a right stretch tensor, :math:`U`, or
        a left stretch tensor, :math:`V`. :math:`U` and :math:`V` are
        both positive definite and symmetric and include all of the
        non-rigid lattice deformation. Isometries, :math:`Q`, have the
        property that :math:`Q^{-1} = Q^{\mathsf{T}}`, where :math:`^{\mathsf{T}}`
        is used to indicate matrix transpose.
      - :math:`T` is an integer transformation matrix that generates a
        superlattice of :math:`L_1`.
      - :math:`N` is a unimodular reorientation matrix that generates a
        lattice equivalent to :math:`L_1 T` with reoriented lattice
        vectors.

      Lattice mapping costs are calculated and ranked according to one of:

      - "isotropic_strain_cost": a strain cost, calculated to be
        volume-normalized and invariant to which structure is the
        parent/child. Calculated as:

        .. math::

            \mathrm{cost} &= \frac{1}{2}*\left( \frac{1}{3}*\mathrm{tr}\left(\tilde{B}_{c \to p}^{2} \right) + \frac{1}{3}*\mathrm{tr}\left(\tilde{B}_{p \to c}^{2} \right) \right)

            \tilde{F} &= \frac{1}{\det{F}^{1/3}} F = \tilde{Q} \tilde{U}

            \tilde{B} &= \tilde{U} - I,

        where :math:`F` is the deformation gradient tensor, which can be defined
        in either the parent-to-child (:math:`p \to c`) sense as
        :math:`F_{p \to c} L_1 T N = L_2`, or in the child-to-parent sense
        (:math:`c \to p`) according to :math:`F_{c \to p} = F_{p \to c}^{-1}`,
        :math:`B` is the Biot strain tensor, and the use of (:math:`\tilde{X}`)
        indicates that the quantity has been normalized to be volume
        invariant.

      - "symmetry_breaking_strain_cost": a strain cost, including only
        the components of the strain that break the symmetry of the parent
        point group. Calculated as:

        .. math::

            \mathrm{cost} &= \frac{1}{2}*\left( \frac{1}{3}*\mathrm{tr}\left( B_{sym-break, p \to c}^2 \right) + \frac{1}{3}*\mathrm{tr}\left( B_{sym-break, c \to p}^2 \right) \right)

            B_{sym-break} &= B - B_{sym}

            B_{sym} &= \frac{1}{N_{G_1}} * \sum_i ( G_1(i) * B * G_1^{\mathsf{T}}(i) )

        where :math:`B_{sym-break}`, is the symmetry-breaking Biot strain,
        :math:`G_1(i)` are parent point group operations, and
        :math:`N_{G_1}` is the total number of operations. Similar relations hold
        for calculating :math:`B_{sym-break, p \to c}` and
        :math:`B_{sym-break, c \to p}` from :math:`B`.

      For more details, see :cite:t:`THOMAS2021a`.

      Notes
      -----
      Deformations can be validly defined as parent-to-child or
      child-to-parent. Be careful as to which convention is being used.
      Lattice mappings use the parent-to-child definition.


      Parameters
      ----------
      lattice1 : libcasm.xtal.Lattice
          The reference "parent" lattice, :math:`L_1`.
      lattice2 : libcasm.xtal.Lattice
          The "child" lattice, :math:`L_2`.
      transformation_matrix_to_super : Optional[array_like, shape=(3,3)], optional
          An approximately integer transformation matrix that generates a
          superlattice of :math:`L_1`. The default value is the identity
          matrix.
      lattice1_point_group : List[libcasm.xtal.SymOp], optional
          Used to skip reorientation matrices that result in symmetrically
          equivalent mappings. The default (empty), is equivalent to only
          including the identity operation.
      lattice2_point_group : List[libcasm.xtal.SymOp], optional
          Used to skip reorientation matrices that result in symmetrically
          equivalent mappings. The default (empty), is equivalent to just
          including the identity operation.
      min_cost : float, default=0.
          Keep lattice mappings with cost >= min_cost
      max_cost : float, default=1e20
          Keep results with cost <= max_cost
      cost_method : str, default="isotropic_strain_cost"
          Selects the method used to calculate lattice mapping costs. One of
          "isotropic_strain_cost" or "symmetry_breaking_strain_cost".
      k_best : Optional[int], default=None
          If not None, then only keep the k-best results (i.e. k lattice mappings
          with minimum cost) satisfying the min_cost and max_cost constraints.
          If there are approximate ties, those will also be kept.
      reorientation_range : int, default=1
          The absolute value of the maximum element in the lattice mapping
          reorientation matrix, :math:`N`. This determines how many
          equivalent lattice vector reorientations are checked. Increasing
          the value results in more checks. The value 1 is expected to be
          sufficient because reduced cell lattices are compared internally.
      cost_tol : float, default=1e-5
          Tolerance for checking if lattice mapping costs are approximately
          equal.

      Returns
      -------
      lattice_mappings : ~libcasm.mapping.info.LatticeMappingResults
          A :class:`~libcasm.mapping.info.LatticeMappingResults` object,
          giving possible lattice mappings, sorted by lattice mapping cost.
      )pbdoc",
        py::arg("lattice1"), py::arg("lattice2"),
        py::arg("transformation_matrix_to_super") = std::nullopt,
        py::arg("reorientation_range") = 1,
        py::arg("lattice1_point_group") = std::vector<xtal::SymOp>{},
        py::arg("lattice2_point_group") = std::vector<xtal::SymOp>{},
        py::arg("min_cost") = 0.0, py::arg("max_cost") = 1e20,
        py::arg("cost_method") = std::string("isotropic_strain_cost"),
        py::arg("k_best") = std::nullopt, py::arg("cost_tol") = 1e-5);

  m.def("map_structures", &map_structures, R"pbdoc(
      Find mappings between two structures

      This method finds mappings from a superstructure of a reference "parent"
      structure to a "child" structure. It works by finding lattice mappings
      (:class:`~libcasm.mapping.info.LatticeMapping`) for symmetrically unique
      superlattices of the "parent" lattice for a range of supercell volumes,
      and for each potential lattice mapping finding atom mappings
      (:class:`~libcasm.mapping.info.AtomMapping`).

      The total structure mapping cost, total_cost, is a weighted mixture of
      the lattice mapping cost, lattice_cost, and the atom mapping cost,
      atom_cost:

      .. code-block:: Python

          total_cost = lattice_cost_weight*lattice_cost + (1.0 - lattice_cost_weight)*atom_cost

      where lattice_cost_weight is an input parameter.

      For more details, see :cite:t:`THOMAS2021a`.

      Parameters
      ----------
      prim : libcasm.xtal.Prim
          The reference "parent" structure, with lattice :math:`L_1`,
          represented as Prim, with occupation DoF indicating which atom
          types are allowed to map to each basis site.
      structure : libcasm.xtal.Structure
          The "child" structure, with lattice :math:`L_2`.
      max_vol : int
          The maximum parent superstructure volume to consider, as a
          multiple of the parent structure volume.
      prim_factor_group : List[libcasm.xtal.SymOp], optional
          Used to skip symmetrically equivalent mappings. The default
          (empty), is equivalent to only including the identity operation.
      structure_factor_group : List[libcasm.xtal.SymOp], optional
          Used to skip symmetrically equivalent mappings. The default
          (empty), is equivalent to just including the identity operation.
      min_vol : int, default=1
          The minimum parent superstructure volume to consider, as a
          multiple of the parent structure volume.
      min_cost : float, default=0.
          Keep structure mappings with cost >= min_cost
      max_cost : float, default=1e20
          Keep structure mappings with cost <= max_cost
      lattice_cost_weight : float, default=0.5
          The fraction of the total cost due to the lattice strain cost.
          The remaining fraction (1.-lattice_cost_weight) is due to the
          atom cost.
      lattice_cost_method : str, default="isotropic_strain_cost"
          Selects the method used to calculate lattice mapping costs. One of
          "isotropic_strain_cost" or "symmetry_breaking_strain_cost".
      atom_cost_method : str, default="isotropic_disp_cost"
          Selects the method used to calculate atom mapping costs. One of
          "isotropic_disp_cost" or "symmetry_breaking_disp_cost".
      k_best : int, default=1
          Only keep the k-best results (i.e. k mappings with minimum cost)
          satisfying the min_cost and max_cost constraints. If there are
          approximate ties, those will also be kept.
      cost_tol : float, default=1e-5
          Tolerance for checking if structure mappings costs are approximately
          equal.

      Returns
      -------
      structure_mappings : ~libcasm.mapping.info.StructureMappingResults
          A :class:`~libcasm.mapping.info.StructureMappingResults` object,
          giving possible structure mappings, sorted by total cost.
      )pbdoc",
        py::arg("prim"), py::arg("structure"), py::arg("max_vol"),
        py::arg("prim_factor_group") = std::vector<xtal::SymOp>{},
        py::arg("structure_factor_group") = std::vector<xtal::SymOp>{},
        py::arg("min_vol") = 1, py::arg("min_cost") = 0.0,
        py::arg("max_cost") = 1e20, py::arg("lattice_cost_weight") = 0.5,
        py::arg("lattice_cost_method") = std::string("isotropic_strain_cost"),
        py::arg("atom_cost_method") = std::string("isotropic_disp_cost"),
        py::arg("k_best") = 1, py::arg("cost_tol") = 1e-5);

  m.def("map_atoms", &map_atoms, R"pbdoc(
      Find atom mappings between two structures, given a particular lattice mapping

      This method finds atom mappings from a superstructure of a reference
      "parent" structure to a "child" structure. It works by checking
      atom mappings (:class:`~libcasm.mapping.info.AtomMapping`) given a particular
      lattice mapping (:class:`~libcasm.mapping.info.LatticeMapping`).

      Atom mapping costs are calculated and ranked according to one of:

      - "isotropic_disp_cost": A displacement cost that is a volume-normalized and does
        not depend on which structure is "parent" and which is "child". It is calculated
        as

        .. math::

            \mathrm{cost} &= \frac{1}{2}(c^{g}(L_1 T, D) + c^g(L_2, D^{rev}) \\
            c^{g}(S, D) &= \left( \frac{3v(S)}{4\pi} \right)^{-2/3} \frac{\sum^N_i \vec{d}(i)^2}{N}

        where:

        - :math:`\vec{d}(i)`: The site-to-atom displacement, as defined in
          :class:`libcasm.mapping.info.AtomMapping`,
        - :math:`D`: The shape=(3, N) matrix with column `i` being the
          displacement :math:`\vec{d}(i)` associated with the atom at the
          i-th site in the parent superstructure.
        - :math:`D^{rev}`: The shape=(3, n_sites) matrix corresponding to the
          displacements associated with the reverse mapping,
          :math:`D^{rev} = -U * D`, :math:`U` being the right stretch tensor
          of the parent-to-child lattice mapping, as defined in
          :class:`libcasm.mapping.info.LatticeMapping`,
        - v(S): The volume per site for the superlattice :math:`S`
        - :math:`c^{g}(S,D)` is the "geometric atom cost" for the displacements
          in superlattice :math:`S`, and

      - "symmetry_breaking_disp_cost": A displacement cost, based on the
        symmetry-breaking displacements that are left after removing
        displacement modes which preserve the symmetry of the parent point
        group. The symmetry-breaking displacement cost is
        calculated using the same equation as the isotropic displacement cost,
        with the substitutions :math:`D \to D^{sym-break}` and
        :math:`D^{rev} \to D^{sym-break, rev}`, the matrices holding the
        symmetry-breaking displacements for the parent-to-child and reverse
        atom mappings.

      For more details, see :cite:t:`THOMAS2021a`.

      Parameters
      ----------
      prim : libcasm.xtal.Prim
          The reference "parent" structure, with lattice :math:`L_1`,
          represented as Prim, with occupation DoF indicating which atom
          types are allowed to map to each basis site.
      structure : libcasm.xtal.Structure
          The "child" structure, with lattice :math:`L_2`.
      lattice_mapping : ~libcasm.mapping.info.LatticeMapping
          Defines the lattice mapping from the "parent" structure to
          the "child" structure.
      prim_factor_group : List[libcasm.xtal.SymOp], optional
          Used to skip symmetrically equivalent mappings. The default
          (empty), is equivalent to only including the identity operation.
      min_cost : float, default=0.
          Keep atom mappings with cost >= min_cost
      max_cost : float, default=1e20
          Keep atom mappings with cost <= max_cost
      atom_cost_method : str, default="isotropic_disp_cost"
          Selects the method used to calculate atom mapping costs. One of
          "isotropic_disp_cost" or "symmetry_breaking_disp_cost".
      k_best : int, default=1
          Only keep the k-best results (i.e. k mappings with minimum cost)
          satisfying the min_cost and max_cost constraints. If there are
          approximate ties, those will also be kept.
      cost_tol : float, default=1e-5
          Tolerance for checking if atom mapping costs are approximately
          equal.

      Returns
      -------
      atom_mappings : ~libcasm.mapping.info.AtomMappingResults
          A :class:`~libcasm.mapping.info.AtomMappingResults` object,
          giving possible atom mappings, sorted by atom mapping cost.
      )pbdoc",
        py::arg("prim"), py::arg("structure"), py::arg("lattice_mapping"),
        py::arg("prim_factor_group") = std::vector<xtal::SymOp>{},
        py::arg("min_cost") = 0.0, py::arg("max_cost") = 1e20,
        py::arg("atom_cost_method") = std::string("isotropic_disp_cost"),
        py::arg("k_best") = 1, py::arg("cost_tol") = 1e-5);

  // Apply structure mapping
  m.def("make_mapped_lattice", &mapping::make_mapped_lattice, R"pbdoc(
      Return the mapped lattice

      The mapped lattice is constructed from the parent lattice
      by applying the lattice mapping without isometry. It has
      the lattice vector column matrix:

      .. math::

          L_m = U L_1 T N

      where :math:`L_1` is the reference "parent" lattice vectors, and
      :math:`L_m` is the "mapped" lattice vectors, as columns of
      shape=(3,3) matrices. The othe shape=(3,3) matrices are:

      - :math:`U`, the shape=(3,3) right stretch tensor of the
        parent-to-child deformation gradient tensor
      - :math:`T`, an integer transformation matrix that generates a
        superlattice of :math:`L_1`
      - :math:`N`, a unimodular reorientation matrix that generates a
        lattice equivalent to :math:`L_1 T` with reoriented lattice
        vectors

      Parameters
      ----------
      parent_lattice : ~libcasm.xtal.Lattice
          The reference "parent" lattice.
      lattice_mapping : ~libcasm.mapping.info.LatticeMapping
          The lattice mapping transformation

      Returns
      -------
      mapped_lattice : libcasm.xtal.Lattice
          The mapped lattice that results from transforming the parent
          lattice according to the lattice mapping
      )pbdoc",
        py::arg("parent_lattice"), py::arg("lattice_mapping"));

  // Apply structure mapping
  m.def("make_mapped_structure", &mapping::make_mapped_structure, R"pbdoc(
      Return the mapped structure, with implied vacancies, strain, and
      atomic displacement

      The "mapped structure" lattice and site coordinates are constructed
      from the parent structure by applying the lattice and atom mappings
      without isometry. Atom names and atom properties are
      determined from the unmapped structure, permutation, and
      inverse isometry. Global properties are determined
      from the unmapped global properties and inverse isometry. Strain,
      using the right stretch tensor as the strain metric, is stored as
      the global property "Ustrain". Displacement is stored as
      atom properties.

      Notes:

      - This method is only implemented for atomic structures, not
        molecular structures
      - Implicit vacancies are added as "Va", with value 0.0 any atom
        properties present in the unmapped_structure (i.e. force)
      - Raises if unmapped_structure already has a strain or disp
        property


      Parameters
      ----------
      structure_mapping : ~libcasm.mapping.info.StructureMapping
          The structure mapping transformation
      unmapped_structure : ~libcasm.xtal.Structure
          The unmapped "child" structure

      Returns
      -------
      mapped_structure : ~libcasm.xtal.Structure
          The mapped structure, with implied vacancies, strain, and
          atomic displacement
      )pbdoc",
        py::arg("structure_mapping"), py::arg("unmapped_structure"));

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif
}
