#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// nlohmann::json binding
#define JSON_USE_IMPLICIT_CONVERSIONS 0
#include "casm/casm_io/container/json_io.hh"
#include "casm/casm_io/json/jsonParser.hh"
#include "casm/crystallography/BasicStructure.hh"
#include "casm/mapping/AtomMapping.hh"
#include "casm/mapping/LatticeMapping.hh"
#include "casm/mapping/MappingSearch.hh"
#include "casm/mapping/StructureMapping.hh"
#include "casm/mapping/io/json_io.hh"
#include "pybind11_json/pybind11_json.hpp"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

/// CASM - Python binding code
namespace CASMpy {

using namespace CASM;
using namespace CASM::mapping;

MappingSearch make_MappingSearch(
    double _min_cost, double _max_cost, int _k_best,
    std::optional<AtomCostFunction> _atom_cost_f,
    std::optional<TotalCostFunction> _total_cost_f,
    std::optional<DispOnlyAtomToSiteCostFunction> _atom_to_site_cost_f,
    bool _enable_remove_mean_displacement, double _infinity, double _cost_tol,
    std::optional<AtomToSiteCostFunction> _atom_to_site_cost_future_f) {
  if (!_atom_cost_f) {
    _atom_cost_f = IsotropicAtomCost();
  }
  if (!_total_cost_f) {
    _total_cost_f = WeightedTotalCost(0.5);
  }
  AtomToSiteCostFunction f;
  if (_atom_to_site_cost_future_f) {
    f = _atom_to_site_cost_future_f.value();
  } else {
    if (!_atom_to_site_cost_f) {
      _atom_to_site_cost_f =
          DispOnlyAtomToSiteCostFunction(make_atom_to_site_cost);
    }
    // convert to AtomToSiteCostFunction
    f = [_atom_to_site_cost_f](
            xtal::Lattice const &lattice, Eigen::Vector3d const &displacement,
            std::string const &atom_type,
            std::vector<std::string> const &allowed_atom_types,
            double infinity) {
      return _atom_to_site_cost_f.value()(displacement, atom_type,
                                          allowed_atom_types, infinity);
    };
  }
  return MappingSearch(_min_cost, _max_cost, _k_best, _atom_cost_f.value(),
                       _total_cost_f.value(), f,
                       _enable_remove_mean_displacement, _infinity, _cost_tol);
}

std::shared_ptr<AtomMappingSearchData> make_AtomMappingSearchData(
    std::shared_ptr<LatticeMappingSearchData const> lattice_mapping_data,
    Eigen::Vector3d const &trial_translation_cart,
    std::optional<DispOnlyAtomToSiteCostFunction> _atom_to_site_cost_f,
    double infinity,
    std::optional<AtomToSiteCostFunction> _atom_to_site_cost_future_f) {
  AtomToSiteCostFunction f;
  if (_atom_to_site_cost_future_f) {
    f = _atom_to_site_cost_future_f.value();
  } else {
    if (!_atom_to_site_cost_f) {
      _atom_to_site_cost_f =
          DispOnlyAtomToSiteCostFunction(make_atom_to_site_cost);
    }
    // convert to AtomToSiteCostFunction
    f = [_atom_to_site_cost_f](
            xtal::Lattice const &lattice, Eigen::Vector3d const &displacement,
            std::string const &atom_type,
            std::vector<std::string> const &allowed_atom_types,
            double infinity) {
      return _atom_to_site_cost_f.value()(displacement, atom_type,
                                          allowed_atom_types, infinity);
    };
  }
  return std::make_shared<AtomMappingSearchData>(
      lattice_mapping_data, trial_translation_cart, f, infinity);
}

}  // namespace CASMpy

PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);

PYBIND11_MODULE(_mapping_mapsearch, m) {
  using namespace CASMpy;

  m.doc() = R"pbdoc(
        Components for implementing custom structure mapping searches

        libcasm.mapping.mapsearch
        -------------------------

        The libcasm.mapping.mapsearch module contains data structures and methods
        used to perform structure mapping searches.
        )pbdoc";
  py::module_::import("libcasm.xtal");
  py::module_::import("libcasm.mapping.info");

  py::class_<PrimSearchData, std::shared_ptr<PrimSearchData>>(
      m, "PrimSearchData", R"pbdoc(
      Prim-related data used for mapping searches

      This object holds shared data for use by all mappings to a single
      Prim.

      )pbdoc")
      .def(py::init<std::shared_ptr<xtal::BasicStructure const>,
                    std::optional<std::vector<xtal::SymOp>>, bool>(),
           py::arg("prim"),
           py::arg("override_prim_factor_group") =
               std::optional<std::vector<xtal::SymOp>>(),
           py::arg("enable_symmetry_breaking_atom_cost") = true,
           R"pbdoc(
          .. rubric:: Constructor

          Parameters
          ----------

          prim : libcasm.xtal.Prim
              A primitive reference "parent" structure that a structure
              may be mapped to
          override_prim_factor_group : Optional[List[libcasm.xtal.SymOp]] = None
              Optional, allows explicitly setting the symmetry operations
              used to skip symmetrically equivalent structure mappings. The
              default (None), uses the prim factor group as generated by
              `libcasm.xtal.make_prim_factor_group`. The first symmetry
              operation should always be the identity operation. Will raise
              if an empty vector is provided.
          enable_symmetry_breaking_atom_cost : bool = True
              If symmetry_breaking_atom_cost is intended to be used, setting
              this to true will generate the symmetry-invariant displacement
              modes required for the calculation using this object's
              prim_factor_group.
          )pbdoc")
      .def(
          "prim", [](PrimSearchData const &m) { return m.prim; },
          "Returns the prim.")
      .def(
          "prim_lattice",
          [](PrimSearchData const &m) { return m.prim_lattice; },
          "Returns the lattice of the prim.")
      .def(
          "N_prim_site", [](PrimSearchData const &m) { return m.N_prim_site; },
          "Returns the number of sites in the prim.")
      .def(
          "prim_site_coordinate_cart",
          [](PrimSearchData const &m) { return m.prim_site_coordinate_cart; },
          "Returns the Cartesian coordinates of prim sites as columns of a "
          "shape=(3,N_prim_site) matrix.")
      .def(
          "prim_allowed_atom_types",
          [](PrimSearchData const &m) { return m.prim_allowed_atom_types; },
          "Returns a size=N_prim_site array of arrays with the names of atoms "
          "allowed on each site in the prim.")
      .def(
          "prim_factor_group",
          [](PrimSearchData const &m) { return m.prim_factor_group; },
          "Returns symmetry operations of the prim that may be used to skip "
          "symmetrically equivalent structure mappings.")
      .def(
          "prim_crystal_point_group",
          [](PrimSearchData const &m) { return m.prim_crystal_point_group; },
          "Returns point group operations of the prim that may be used to skip "
          "symmetrically equivalent lattice mappings.")
      .def(
          "prim_sym_invariant_displacement_modes",
          [](PrimSearchData const &m) {
            return m.prim_sym_invariant_displacement_modes;
          },
          "Returns a size=N_mode vector with shape=(3,N_prim_site) matrices, "
          "giving the symmetry invariant displacement modes. Columns of the "
          "matrices are the displacements associated with each site for a "
          "given mode.");

  py::class_<StructureSearchData, std::shared_ptr<StructureSearchData>>(
      m, "StructureSearchData", R"pbdoc(
      Struture-related data used for mapping searches

      This object holds shared data for use by all mappings of a
      single libcasm.xtal.Structure. Mapping may be performed of:

      - lattice orientation
      - lattice strain
      - site permutation
      - site atomic occupant type
      - site displacements

      Mapping of structures with molecular occupants and other types
      of properties (magentic spin, etc.) is not currently supported.

      )pbdoc")
      .def(py::init<xtal::Lattice const &, Eigen::MatrixXd const &,
                    std::vector<std::string>,
                    std::optional<std::vector<xtal::SymOp>>>(),
           py::arg("lattice"), py::arg("atom_coordinate_cart"),
           py::arg("atom_type"),
           py::arg("override_structure_factor_group") =
               std::optional<std::vector<xtal::SymOp>>(),
           R"pbdoc(
          .. rubric:: Constructor

          Parameters
          ----------
          lattice : libcasm.xtal.Lattice
              The lattice of the structure being mapped
          atom_coordinate_cart : array_like, shape (3, n)
            Atom positions, as columns of a matrix, in Cartesian
            coordinates. May included coordinates of explicitly specified
            vacancies.
          atom_type : List[str], size=n
            Atom type names. May include explicitly specified vacancies,
            which should be named "Va", "va", or "VA".
          override_structure_factor_group : List[libcasm.xtal.SymOp], optional
              Optional, allows explicitly setting the symmetry operations
              of the structure which may be used to skip symmetrically
              equivalent structure mappings. The default (None), uses the
              factor group as generated by
              `libcasm.xtal.make_structure_factor_group` for the structure
              specified by `lattice`, `atom_coordinate_cart`, and
              `atom_type`. The first symmetry operation should always be
              the identity operation. Will raise if an empty vector is
              provided.
          )pbdoc")
      .def(
          "lattice", [](StructureSearchData const &m) { return m.lattice; },
          "Returns the lattice of the structure being mapped.")
      .def(
          "N_atom", [](StructureSearchData const &m) { return m.N_atom; },
          "Returns the number of atoms (and explicitly included vacancies) in "
          "the structure being mapped.")
      .def(
          "atom_coordinate_cart",
          [](StructureSearchData const &m) { return m.atom_coordinate_cart; },
          "Returns the Cartesian coordinates, as columns of a shape=(3,N_atom) "
          "matrix, of atoms (and explicitly included vacancies) in the "
          "structure being mapped.")
      .def(
          "atom_type", [](StructureSearchData const &m) { return m.atom_type; },
          "Returns a size=N_atom array of with the name of the atom (or "
          "explicitly included vacancy) at each site. Explicit specified "
          "vacancies should be given the name \"Va\", \"VA\", or \"va\".")
      .def(
          "structure_factor_group",
          [](StructureSearchData const &m) { return m.structure_factor_group; },
          "Returns symmetry operations of the structure being mapped that may "
          "be used to skip symmetrically equivalent structure mappings.")
      .def(
          "structure_crystal_point_group",
          [](StructureSearchData const &m) {
            return m.structure_crystal_point_group;
          },
          "Returns point group operations of the structure being mapped that "
          "may be used to skip symmetrically equivalent lattice mappings.")
      .def(
          "is_superstructure",
          [](StructureSearchData const &m) {
            return m.prim_structure_data != nullptr;
          },
          "Returns True if this is a superstructure, else returns False.")
      .def(
          "prim_structure_data",
          [](StructureSearchData const &m) {
            if (m.prim_structure_data != nullptr) {
              return m.prim_structure_data;
            }
            throw std::runtime_error("Not a superstructure");
          },
          "Returns the primitive structure, if this is a superstructure, else "
          "raises.")
      .def(
          "transformation_matrix_to_super",
          [](StructureSearchData const &m) {
            return m.transformation_matrix_to_super;
          },
          "Returns the shape=(3,3) integer transformation matrix that "
          "generates this structure's lattice from the primitive structure "
          "lattice, if this is a superstructure, else returns the identity "
          "matrix.");

  m.def(
      "make_superstructure_data",
      [](std::shared_ptr<StructureSearchData const> prim_structure_data,
         Eigen::Matrix3l const &transformation_matrix_to_super) {
        return std::make_shared<StructureSearchData const>(
            std::move(prim_structure_data), transformation_matrix_to_super);
      },
      py::arg("prim_structure_data"), py::arg("transformation_matrix_to_super"),
      R"pbdoc(
        Construct StructureSearchData for a superstructure

        Parameters
        ----------
        prim_structure_data : ~libcasm.mapping.mapsearch.StructureSearchData
            Search data for the primitive structure being mapped
        transformation_matrix_to_super : array_like, shape=(3,3)
            Integer transformation matrix, :math:`T`, that generates the
            superstructure lattice vectors, as columns of a matrix, :math:`S`,
            from the primitive structure lattice vectors, as columns of a
            matrix, :math:`L`, according to :math:`S = L T`.

        Returns
        -------
        superstructure_data : ~libcasm.mapping.mapsearch.StructureSearchData
            Search data for a super structure.
        )pbdoc");

  py::class_<LatticeMappingSearchData,
             std::shared_ptr<LatticeMappingSearchData>>(
      m, "LatticeMappingSearchData", R"pbdoc(
      Lattice mapping-related data used for mapping searches

      This object holds shared data for use by all structure mappings
      in the context of a single lattice mapping
      (:class:`~libcasm.mapping.info.LatticeMapping`).
      )pbdoc")
      .def(py::init<std::shared_ptr<PrimSearchData const>,
                    std::shared_ptr<StructureSearchData const>,
                    LatticeMapping>(),
           py::arg("prim_data"), py::arg("structure_data"),
           py::arg("lattice_mapping"),
           R"pbdoc(
          .. rubric:: Constructor

          Parameters
          ----------

          prim_data : ~libcasm.mapping.mapsearch.PrimSearchData
              Search data for a prim being mapped to
          structure_data : ~libcasm.mapping.mapsearch.StructureSearchData
              Search data for the structure being mapped
          lattice_mapping : ~libcasm.mapping.info.LatticeMapping
              Lattice mapping between the prim being mapped to and
              the structure being mapped
          )pbdoc")
      .def(
          "prim_data",
          [](LatticeMappingSearchData const &m) { return m.prim_data; },
          "Returns the search data for the prim being mapped to.")
      .def(
          "structure_data",
          [](LatticeMappingSearchData const &m) { return m.structure_data; },
          "Returns the search data for the structure being mapped.")
      .def(
          "lattice_mapping",
          [](LatticeMappingSearchData const &m) { return m.lattice_mapping; },
          "Returns the lattice mapping between the prim being mapped to and "
          "the structure being mapped.")
      .def(
          "transformation_matrix_to_super",
          [](LatticeMappingSearchData const &m) {
            return m.transformation_matrix_to_super;
          },
          R"pbdoc(
          Returns the transformation matrix to the ideal superstructure lattice

          The transformation matrix that gives the ideal
          superstructure lattice, for this lattice mapping,
          from the prim lattice (i.e. T*N of the lattice
          mapping).

          This is equivalent to:

              lround(lattice_mapping.transformation_matrix_to_super *
                     lattice_mapping.reorientation)

          )pbdoc")
      .def(
          "supercell_lattice",
          [](LatticeMappingSearchData const &m) { return m.supercell_lattice; },
          R"pbdoc(
          Returns the lattice of the ideal supercell.

          The lattice of the ideal supercell is :math:`S_1 = L_1 * T * N`
          as defined in :class:`~libcasm.mapping.info.LatticeMapping`).
          )pbdoc")
      .def(
          "N_supercell_site",
          [](LatticeMappingSearchData const &m) { return m.N_supercell_site; },
          "Returns the number of sites in the ideal superstructure specified "
          "by "
          "the lattice_mapping.")
      .def(
          "atom_coordinate_cart_in_supercell",
          [](LatticeMappingSearchData const &m) {
            return m.atom_coordinate_cart_in_supercell;
          },
          R"pbdoc(
          Returns atom coordinates mapped to the lattice of the ideal supercell

          Returns
          -------
          atom_coordinate_cart_in_supercell : numpy.ndarray[numpy.float64[m, n]]
              This is :math:`F^{-1}\vec{r_2}`, as defined in
              :class:`~libcasm.mapping.info.AtomMapping`, a shape=(3,N_atom)
              matrix with columns containing the Cartesian coordinates of the
              structure's atoms in the state after the inverse lattice
              mapping deformation is applied.

              The "supercell" refers to the ideal supercell, with superlattice,
              :math:`S_1 = L_1 * T * N`, as defined in
              :class:`~libcasm.mapping.info.LatticeMapping`.
          )pbdoc")
      .def(
          "supercell_site_coordinate_cart",
          [](LatticeMappingSearchData const &m) {
            return m.supercell_site_coordinate_cart;
          },
          "Returns the Cartesian coordinates of sites in the ideal "
          "supersuperstructure, as columns of a shape=(3,N_supercell_site) "
          "matrix.")
      .def(
          "supercell_allowed_atom_types",
          [](LatticeMappingSearchData const &m) {
            return m.supercell_allowed_atom_types;
          },
          "Returns a size=N_supercell_site array of arrays with the names of "
          "atoms allowed on each site in the ideal superstructure.");

  m.def("make_trial_translations", &make_trial_translations,
        py::arg("lattice_mapping_data"),
        R"pbdoc(
        Returns translations that bring atoms into registry with ideal \
        superstructure sites.

        This function returns a minimal set of trial translations by finding
        an atom type with the fewest valid atom type -> allowed site
        translations and returning the corresponding translations. For each
        trial translation at least one site displacement is of length zero.

        See :class:`~libcasm.mapping.mapsearch.AtomMappingSearchData` for a
        description of how the trial translation is used when finding an
        atom mapping and associated displacements.

        Parameters
        ----------
        lattice_mapping_data : libcasm.mapping.mapsearch.LatticeMappingSearchData
            Data describing a lattice mapping between a prim and a structure

        Returns
        -------
        trial_translations_cart : List[numpy.ndarray[numpy.float64[3, 1]]]
            An array holding a minimal set of trial translations, in
            Cartesian coordinates, which bring atoms of the structure
            being mapped into alignment with sites in the ideal
            superstructure of the prim they are being mapped to.
        )pbdoc");

  m.def("make_atom_to_site_cost", &make_atom_to_site_cost,
        py::arg("displacement"), py::arg("atom_type"),
        py::arg("allowed_atom_types"), py::arg("infinity"),
        R"pbdoc(
        Returns the cost for mapping a particular atom to a particular site


        .. deprecated:: 2.3.0
            The :func:`make_atom_to_site_cost_future` method, which takes
            `lattice` as a parameter, is planned to replace this
            method in libcasm-mapping>=3.0.0.

        The mapping cost:

        - of a vacancy to any site that allows vacancies is set to
          0.0.
        - of an atom to a site that does not allow the atom type is
          infinity
        - otherwise, the mapping cost is equal to displacement length
          squared

        Notes
        -----
        Atoms are treated as vacancies if they are named \"Va\", \"VA\",
        or \"va\".

        Parameters
        ----------
        displacement : array_like, shape=(3,)
            The minimum length displacement, accounting for periodic
            boundaries, from the site to the atom.
        atom_type : str,
            The atom (or vacancy) type.
        allowed_atom_types : List[str]
            The atom (or vacancy) types allowed on the site.
        infinity: float
            The value to use for the cost of unallowed mappings

        Returns
        -------
        cost : float
            The atom (or vacancy) mapping cost.
        )pbdoc");

  m.def("make_atom_to_site_cost_future", &make_atom_to_site_cost_future,
        py::arg("lattice"), py::arg("displacement"), py::arg("atom_type"),
        py::arg("allowed_atom_types"), py::arg("infinity"),
        R"pbdoc(
        Returns the cost for mapping a particular atom to a particular site

        .. deprecated:: 2.3.0
            This method is planned to replace :func:`make_atom_to_site_cost` in
            libcasm-mapping>=3.0.0, and this method will be removed.

        The mapping cost:

        - of a vacancy to any site that allows vacancies is set to
          0.0.
        - of an atom to a site that does not allow the atom type is
          infinity
        - of a displacement on the lattice voronoi cell boundary is
          infinity (the displacement is ambiguous as to which periodic
          image it should map to)
        - otherwise, the mapping cost is equal to displacement length
          squared

        Notes
        -----
        Atoms are treated as vacancies if they are named \"Va\", \"VA\",
        or \"va\".

        Parameters
        ----------
        lattice : libcasm.xtal.Lattice
            The lattice in which the displacements are calculated under
            periodic boundary conditions.
        displacement : array_like, shape=(3,)
            The minimum length displacement, accounting for periodic
            boundaries, from the site to the atom.
        atom_type : str,
            The atom (or vacancy) type.
        allowed_atom_types : List[str]
            The atom (or vacancy) types allowed on the site.
        infinity: float
            The value to use for the cost of unallowed mappings

        Returns
        -------
        cost : float
            The atom (or vacancy) mapping cost.
        )pbdoc");

  py::class_<AtomMappingSearchData, std::shared_ptr<AtomMappingSearchData>>(
      m, "AtomMappingSearchData", R"pbdoc(
      Atom mapping-related data used for mapping searches

      Atom mapping assignment is made by optimizing a cost that depends on
      site-to-atom displacements, :math:`\vec{d}^{*}(i)`, that are calculated
      using the minimum length displacements under periodic boundary
      conditions that satisfy a proposed atom assignment.

      Each atom assignment proposal is made in the context of a lattice
      mapping, with deformation gradient, :math:`F`, and a trial translation,
      :math:`\vec{t}^{*}`, according to

      .. math::

          \left(\vec{r_1}(i) + \vec{d}^{*}(i) \right) = F^{-1} \vec{r_2}(p_i) + \vec{t}^{*},

      using the same definitions as :class:`~libcasm.mapping.info.AtomMapping`.

      Given a proposed assignment, the mean displacement,
      :math:`\langle\vec{d}^{*}\rangle` may optionally be removed and the
      assignment cost re-calculated.

      If the mean displacement is removed, then the
      :class:`~libcasm.mapping.info.AtomMapping` displacements are
      :math:`\vec{d}(i) = \vec{d}^{*}(i) - \langle\vec{d^{*}}\rangle`,
      and the :class:`~libcasm.mapping.info.AtomMapping` translation,
      :math:`\vec{t}`, is related to the trial translation according to
      :math:`\vec{t} = F (\vec{t}^{*} - \langle\vec{d^{*}}\rangle)`.

      If the mean displacement is not removed, then
      :math:`\vec{d}(i) = \vec{d}^{*}(i)` and :math:`\vec{t} = F \vec{t}^{*}`.

      This object stores a `trial_translation_cart`, `site_displacements`,
      and resulting `cost_matrix` used to find optimal atom-to-site
      assignment solutions in the context of a particular lattice mapping.
      )pbdoc")
      .def(py::init<>(&make_AtomMappingSearchData),
           py::arg("lattice_mapping_data"), py::arg("trial_translation_cart"),
           py::arg("atom_to_site_cost_f") = std::nullopt,
           py::arg("infinity") = 1e20,
           py::arg("atom_to_site_cost_future_f") = std::nullopt,
           R"pbdoc(
          .. rubric:: Constructor

          Parameters
          ----------

          lattice_mapping_data : ~libcasm.mapping.mapsearch.LatticeMappingSearchData
              Search data for a particular lattice mapping between a prim and
              the structure being mapped.
          trial_translation_cart : array_like, shape=(3,)
              A Cartesian translation applied to atom coordinates in the
              ideal superstructure setting (i.e.
              atom_coordinate_cart_in_supercell) to bring the atoms and sites
              into alignment.
          atom_to_site_cost_f : Optional[Callable] = None
              A function used to calculate the cost of mapping an atom to a
              particular site. Expected to match the same signature as
              :func:`~libcasm.mapping.mapsearch.make_atom_to_site_cost`, which
              is the default method.

              .. deprecated:: 2.3.0
                  The signature of
                  :func:`~libcasm.mapping.mapsearch.make_atom_to_site_cost`
                  will change in libcasm-mapping>=3.0.0 to accept the lattice
                  used for finding the displacements under periodic boundary
                  conditions. The function
                  :func:`~libcasm.mapping.mapsearch.make_atom_to_site_cost_future`
                  will be used as the default.

          infinity : float = 1e20
              The value to use for the cost of unallowed mappings.

          atom_to_site_cost_future_f : Optional[Callable] = None
              A function used to calculate the cost of mapping an atom to a
              particular site. Expected to match the same signature as
              :func:`~libcasm.mapping.mapsearch.make_atom_to_site_cost_future`.
              If provided, this function will be used with priority over
              `atom_to_site_cost_f`.

              .. deprecated:: 2.3.0
                  This argument will be removed in libcasm-mapping>=3.0.0.


          )pbdoc")
      .def(
          "lattice_mapping_data",
          [](AtomMappingSearchData const &m) { return m.lattice_mapping_data; },
          "Returns the search data for the lattice mapping.")
      .def(
          "trial_translation_cart",
          [](AtomMappingSearchData const &m) {
            return m.trial_translation_cart;
          },
          "Returns the Cartesian translation applied to atom coordinates in "
          "the "
          "ideal superstructure setting (i.e. "
          "atom_coordinate_cart_in_supercell) to bring the atoms into "
          "alignment with ideal superstructure sites.")
      .def(
          "site_displacements",
          [](AtomMappingSearchData const &m) { return m.site_displacements; },
          R"pbdoc(
          Returns the site-to-atom displacements of minimum length under periodic boundary conditions of the ideal superstructure.

          The displacements are indexed using
          `site_displacements[site_index][atom_index]`, where the `site_index`
          and `atom_index` are indices into the columns of
          `lattice_mapping_data.supercell_site_coordinate_cart()` and
          `lattice_mapping_data.atom_coordinate_cart_in_supercell()`,
          respectively.
          )pbdoc")
      .def(
          "cost_matrix",
          [](AtomMappingSearchData const &m) { return m.cost_matrix; },
          R"pbdoc(
          Returns a shape=(N_supercell_site, N_supercell_site) cost matrix used in the atom to site assignment problem.

          The element `cost_matrix(site_index, atom_index)` is set to the
          cost of mapping a particular atom onto a particular site, where
          the indices are into the columns of
          `lattice_mapping_data.supercell_site_coordinate_cart()` and
          `lattice_mapping_data.atom_coordinate_cart_in_supercell()`,
          respectively.
          )pbdoc");

  py::class_<IsotropicAtomCost>(m, "IsotropicAtomCost", R"pbdoc(
      A functor for calculating the isotropic atom mapping cost

      .. rubric:: Special methods

      :class:`~libcasm.mapping.mapsearch.IsotropicAtomCost` has a call
      operator which is equivalent to the
      :func:`~libcasm.mapping.mapsearch.IsotropicAtomCost.cost` method.

      )pbdoc")
      .def(py::init(), R"pbdoc(

        .. rubric:: Constructor

        Default constructor only.
        )pbdoc")
      .def("__call__", &IsotropicAtomCost::operator(),
           py::arg("lattice_mapping_data"), py::arg("atom_mapping_data"),
           py::arg("atom_mapping"), "Calculate the isotropic atom mapping cost")
      .def("cost", &IsotropicAtomCost::operator(),
           py::arg("lattice_mapping_data"), py::arg("atom_mapping_data"),
           py::arg("atom_mapping"),
           R"pbdoc(
           Calculate the isotropic atom mapping cost

           Notes
           -----

           - An call operator exists which is equivalent to this method.

           Parameters
           ----------
           lattice_mapping_data: libcasm.mapping.mapsearch.LatticeMappingSearchData
               The lattice mapping search data.
           atom_mapping_data: libcasm.mapping.mapsearch.LatticeMappingSearchData
               The atom mapping search data.
           atom_mapping: libcasm.mapping.info.AtomMapping
               The atom mapping.

           Returns
           -------
           atom_cost: float
               The isotropic atom mapping cost.

           )pbdoc");

  py::class_<SymmetryBreakingAtomCost>(m, "SymmetryBreakingAtomCost", R"pbdoc(
     A functor for calculating the symmetry-breaking atom mapping cost

     .. rubric:: Special methods

     :class:`~libcasm.mapping.mapsearch.SymmetryBreakingAtomCost` has a call
     operator which is equivalent to the
     :func:`~libcasm.mapping.mapsearch.SymmetryBreakingAtomCost.cost` method.

     )pbdoc")
      .def(py::init(), R"pbdoc(

        .. rubric:: Constructor

        Default constructor only.
        )pbdoc")
      .def("__call__", &SymmetryBreakingAtomCost::operator(),
           py::arg("lattice_mapping_data"), py::arg("atom_mapping_data"),
           py::arg("atom_mapping"),
           "Calculate the symmetry-breaking atom mapping cost")
      .def("cost", &SymmetryBreakingAtomCost::operator(),
           py::arg("lattice_mapping_data"), py::arg("atom_mapping_data"),
           py::arg("atom_mapping"),
           R"pbdoc(
           Calculate the symmetry-breaking atom mapping cost

           Notes
           -----

           - An call operator exists which is equivalent to this method.

           Parameters
           ----------
           lattice_mapping_data: libcasm.mapping.mapsearch.LatticeMappingSearchData
               The lattice mapping search data.
           atom_mapping_data: libcasm.mapping.mapsearch.LatticeMappingSearchData
               The atom mapping search data.
           atom_mapping: libcasm.mapping.info.AtomMapping
               The atom mapping.

           Returns
           -------
           atom_cost: float
               The symmetry-breaking atom mapping cost.

           )pbdoc");

  py::class_<WeightedTotalCost>(m, "WeightedTotalCost", R"pbdoc(
     A functor for calculating the total mapping cost as a weighted average of
     the lattice and atom mapping costs

     The total mapping cost is calculated as

     .. code-block:: Python

         total_cost = lattice_cost_weight*lattice_cost + (1.0 - lattice_cost_weight)*atom_cost

     .. rubric:: Special methods

     :class:`~libcasm.mapping.mapsearch.WeightedTotalCost` has a call
     operator which is equivalent to the
     :func:`~libcasm.mapping.mapsearch.WeightedTotalCost.cost` method.

     )pbdoc")
      .def(py::init<double>(), py::arg("lattice_cost_weight"), R"pbdoc(

        .. rubric:: Constructor

        Parameters
        ----------
        lattice_cost_weight : float
            The weight given to the lattice cost in the total mapping cost.
        )pbdoc")
      .def("__call__", &WeightedTotalCost::operator(), py::arg("lattice_cost"),
           py::arg("lattice_mapping_data"), py::arg("atom_cost"),
           py::arg("atom_mapping_data"), py::arg("atom_mapping"),
           R"pbdoc(
           Calculate the total mapping cost as a weighted average of the \
           lattice and atom mapping costs
           )pbdoc")
      .def("cost", &WeightedTotalCost::operator(), py::arg("lattice_cost"),
           py::arg("lattice_mapping_data"), py::arg("atom_cost"),
           py::arg("atom_mapping_data"), py::arg("atom_mapping"),
           R"pbdoc(
           Calculate the total mapping cost as a weighted average of the \
           lattice and atom mapping costs

           Notes
           -----

           - An call operator exists which is equivalent to this method.

           Parameters
           ----------
           lattice_cost: float
               The lattice mapping cost.
           lattice_mapping_data: libcasm.mapping.mapsearch.LatticeMappingSearchData
               The lattice mapping search data.
           atom_cost: float
               The atom mapping cost.
           atom_mapping_data: libcasm.mapping.mapsearch.LatticeMappingSearchData
               The atom mapping search data.
           atom_mapping: libcasm.mapping.info.AtomMapping
               The atom mapping.

           Returns
           -------
           total_cost: float
               The total mapping cost is calculated as

               .. code-block:: Python

                   total_cost = lattice_cost_weight*lattice_cost + (1.0 - lattice_cost_weight)*atom_cost


           )pbdoc");

  py::class_<MappingSearch> pyMappingSearch(m, "MappingSearch", R"pbdoc(
      Used to perform structure mapping searches

      The MappingSearch class holds parameters, data,
      and methods used to search for low cost structure mappings.

      It holds a queue of :class:`~libcasm.mapping.mapsearch.MappingNode`,
      which encode a particular structure mapping, and the data
      necessary to start from that structure mapping and find
      sub-optimal atom mappings as part of a search using the Murty
      Algorithm to find sub-optimal atom-to-site assignments.

      It also holds the best results found so far which satisfy
      some acceptance criteria:

      - min_cost: Keep mappings with total cost >= min_cost
      - max_cost: Keep mappings with total cost <= max_cost
      - k_best: Keep the k_best mappings with lowest total cost
        that also satisfy the min/max cost criteria. Approximate
        ties with the current k_best result are also kept.

      Overview of methods:

      - To begin,
        :func:`~libcasm.mapping.mapsearch.MappingSearch.make_and_insert_mapping_node`
        is called one or more times to generate initial structure mapping
        solutions given a particular lattice mapping and choice of trial
        translation to bring atoms into alignment with sites that they might be
        mapped to. Each call adds one node (think one structure mapping) to the
        MappingSearch queue and, potentially, to the MappingSearch results (if
        the cost range and k-best acceptance criterais are satisfied).
      - Then, :func:`~libcasm.mapping.mapsearch.MappingSearch.partition` is
        called repeatedly to search for sub-optimal cost mapping solutions.
        Each partition creates 0 or more nodes (structure mappings with
        sub-optimal atom assignment solutions) which are inserted into the
        MappingSearch queue and, potentially, to the MappingSearch results (if
        the cost range and k-best acceptance criterais are satisfied).
      - The methods :func:`~libcasm.mapping.mapsearch.MappingSearch.front`,
        :func:`~libcasm.mapping.mapsearch.MappingSearch.back`,
        :func:`~libcasm.mapping.mapsearch.MappingSearch.pop_front`,
        :func:`~libcasm.mapping.mapsearch.MappingSearch.pop_back`, and
        :func:`~libcasm.mapping.mapsearch.MappingSearch.size` allow managing
        the MapppingSearch queue.
      - The method :func:`~libcasm.mapping.mapsearch.MappingSearch.results`
        returns the current mapping results (including approximate ties with
        the current k_best result).


      Notes
      -----

      - The :class:`~libcasm.mapping.mapsearch.MappingSearch` constructor
        parameters `min_cost` and `max_cost` set bounds on mappings stored
        in the search :func:`~libcasm.mapping.mapsearch.MappingSearch.results`,
        but not on the :class:`~libcasm.mapping.mapsearch.MappingNode` (used to
        find next-best costing assignments) stored in the
        :class:`~libcasm.mapping.mapsearch.MappingSearch` queue.
      - The :class:`~libcasm.mapping.mapsearch.QueueConstraints` class is an
        example of an approach to manage the MappingSearch queue during a
        search.

      )pbdoc");

  py::class_<MappingNode>(m, "MappingNode", R"pbdoc(
      A node in the search for optimal structure mappings

      This encodes a particular prim, lattice mapping, and atom mapping,
      and includes the information needed to continue searching for
      suboptimal assignments. In normal usage, it is constructed by the
      :class:`~libcasm.mapping.mapsearch.MappingSearch.make_and_insert_mapping_node`
      method and not on its own.
      )pbdoc")
      .def(py::init(&make_mapping_node), py::arg("search"),
           py::arg("lattice_cost"), py::arg("lattice_mapping_data"),
           py::arg("trial_translation_cart"),
           py::arg("forced_on") = std::map<Index, Index>(),
           py::arg("forced_off") = std::vector<std::pair<Index, Index>>(),
           R"pbdoc(
          .. rubric:: Constructor

          Given any assignment constraints (`forced_on` and `forced_off`), and

          Parameters
          ----------
          search : ~libcasm.mapping.mapsearch.MappingSearch
              A :class:`~libcasm.mapping.mapsearch.MappingSearch` method.
          lattice_cost : float
              The lattice mapping cost.
          lattice_mapping_data : ~libcasm.mapping.mapsearch.LatticeMappingSearchData
              Search data for a particular lattice mapping between a prim and
              the structure being mapped.
          atom_mapping_data : ~libcasm.mapping.mapsearch.AtomMappingSearchData
              Search data for a particular lattice mapping and choice of
              trial translation between a prim and the structure being mapped.
          forced_on : Dict[int, int]
              A map of assignments `site_index: atom_index` that are forced
              on.
          forced_off : List[Tuple[int, int]]
              A list of tuples of assignments `(site_index, atom_index)` that
              are forced off.
          )pbdoc")
      .def(
          "lattice_cost", [](MappingNode const &m) { return m.lattice_cost; },
          "Returns the lattice mapping cost.")
      .def(
          "lattice_mapping_data",
          [](MappingNode const &m) { return m.lattice_mapping_data; },
          "Returns the search data for the lattice mapping.")
      .def(
          "atom_cost", [](MappingNode const &m) { return m.atom_cost; },
          "Returns the atom mapping cost.")
      .def(
          "atom_mapping_data",
          [](MappingNode const &m) { return m.atom_mapping_data; },
          "Returns the search data for a particular lattice mapping and choice "
          "of trial translation between a prim and the structure being mapped.")
      .def(
          "atom_mapping", [](MappingNode const &m) { return m.atom_mapping; },
          "Returns the atom mapping transformation.")
      .def(
          "forced_on",
          [](MappingNode const &m) { return m.assignment_node.forced_on; },
          "Returns a map of assignments `site_index: atom_index` that are "
          "forced on.")
      .def(
          "forced_off",
          [](MappingNode const &m) { return m.assignment_node.forced_off; },
          "Returns a list of tuples of assignments `(site_index, atom_index)` "
          "that are forced off.")
      .def(
          "total_cost", [](MappingNode const &m) { return m.total_cost; },
          R"pbdoc(
          Returns the total mapping cost, as calculated by the `total_cost_f` \
          parameter of a :class:`~libcasm.mapping.mapsearch.MappingSearch`.
          )pbdoc")
      .def(
          "to_dict",
          [](MappingNode const &m) -> nlohmann::json {
            jsonParser json;
            auto pair_to_json = [&](std::pair<Index, Index> const &x,
                                    jsonParser &json) {
              jsonParser tmp;
              tmp.put_array();
              tmp.push_back(x.first);
              tmp.push_back(x.second);
              json.push_back(tmp);
            };
            json["forced_on"].put_array();
            for (auto const &x : m.assignment_node.forced_on) {
              pair_to_json(x, json["forced_on"]);
            }
            json["forced_off"].put_array();
            for (auto const &x : m.assignment_node.forced_off) {
              pair_to_json(x, json["forced_off"]);
            }
            json["atom_mapping"] = m.atom_mapping;
            json["atom_cost"] = m.atom_cost;
            json["lattice_mapping"] = m.lattice_mapping_data->lattice_mapping;
            json["lattice_cost"] = m.lattice_cost;
            json["total_cost"] = m.total_cost;
            return static_cast<nlohmann::json>(json);
          },
          "Represent a MappingNode as a Python dict.");

  pyMappingSearch
      .def(py::init<>(&make_MappingSearch), py::arg("min_cost") = 0.0,
           py::arg("max_cost") = 1e20, py::arg("k_best") = 1,
           py::arg("atom_cost_f") = std::nullopt,
           py::arg("total_cost_f") = std::nullopt,
           py::arg("atom_to_site_cost_f") = std::nullopt,
           py::arg("enable_remove_mean_displacement") = true,
           py::arg("infinity") = 1e20, py::arg("cost_tol") = 1e-5,
           py::arg("atom_to_site_cost_future_f") = std::nullopt,
           R"pbdoc(
          .. rubric:: Constructor

          Parameters
          ----------
          min_cost : float = 0.0
              Keep mappings with total cost >= min_cost. Nodes that have
              a lower cost will be added to the search queue to enable searching
              for less-optimal solutions but not included in the final results.
          max_cost : float = 1e20
              Keep mappings with total cost <= max_cost. Nodes that have
              a higher cost will not be included in the final results, nor will
              they be added to the search queue. During the course of a search,
              once `k_best` results have been found, the `max_cost` will be
              shrunk to match the `k_best`-ranked solution.
          k_best : int = 1
              Keep the k_best mappings with lowest total cost that also
              satisfy the min/max cost criteria. Approximate ties with the
              current `k_best`-ranked result are also kept.
          atom_cost_f : Optional[Callable] = None

              The function used to calculate the atom mapping cost. Expected
              to match the same signature as
              :class:`~libcasm.mapping.mapsearch.IsotropicAtomCost.cost`.
              Possible atom mapping cost functions include:

              - :class:`~libcasm.mapping.mapsearch.IsotropicAtomCost`
              - :class:`~libcasm.mapping.mapsearch.SymmetryBreakingAtomCost`

              If None, the default value is ``IsotropicAtomCost()``.

          total_cost_f : Optional[Callable] = None
              The function used to calculate the total mapping cost. Expected
              to match the same signature as
              :class:`~libcasm.mapping.mapsearch.WeightedTotalCost.cost`.
              If None, the default value is ``WeightedTotalCost(0.5)``.

          atom_to_site_cost_f : Optional[Callable] = None
              A function used to calculate the cost of mapping an atom to a
              particular site. Expected to match the same signature as
              :func:`~libcasm.mapping.mapsearch.make_atom_to_site_cost`, which
              is the default method.

              .. deprecated:: 2.3.0
                  The signature of
                  :func:`~libcasm.mapping.mapsearch.make_atom_to_site_cost`
                  will change in libcasm-mapping>=3.0.0 to accept the lattice
                  used for finding the displacements under periodic boundary
                  conditions. The function
                  :func:`~libcasm.mapping.mapsearch.make_atom_to_site_cost_future`
                  will be used as the default.

          enable_remove_mean_displacement : bool = True
              If true, the translation and displacements of an atom
              mapping are adjusted consistently so that the mean displacment
              is zero.
          infinity : float = 1e20
              The value to use in the assignment problem cost matrix for
              unallowed atom-to-site mappings.
          cost_tol : float = 1e-5
              Tolerance for checking if mapping costs are approximately equal.
          atom_to_site_cost_future_f : Optional[Callable] = None
              A function used to calculate the cost of mapping an atom to a
              particular site. Expected to match the same signature as
              :func:`~libcasm.mapping.mapsearch.make_atom_to_site_cost_future`.
              If provided, this function will be used with priority over
              `atom_to_site_cost_f`.

              .. deprecated:: 2.3.0
                  This argument will be removed in libcasm-mapping>=3.0.0.

          )pbdoc")
      .def_readonly("min_cost", &MappingSearch::min_cost,
                    "float: Keep mappings with total cost >= min_cost.")
      .def_readonly("max_cost", &MappingSearch::max_cost, R"pbdoc(
           float: Keep mappings with total cost <= max_cost.

            Notes
            -----
            - This parameter does not control the queue of MappingNode,
              it only controls which solutions are stored in `results`.
            - The `max_cost` is modified to shrink to the current
              `k_best`-ranked cost once `k_best` results are found

           )pbdoc")
      .def_readonly("k_best", &MappingSearch::k_best, R"pbdoc(
            int: Maximum number of results to keep (approximate ties are also \
            kept).
            )pbdoc")
      .def("front", &MappingSearch::front,
           "Returns a reference to the lowest cost MappingNode in the queue.")
      .def("back", &MappingSearch::back,
           "Returns a reference to the highest cost MappingNode in the queue.")
      .def("pop_front", &MappingSearch::pop_front,
           "Erase the lowest cost MappingNode in the queue.")
      .def("pop_back", &MappingSearch::pop_back,
           "Erase the highest cost MappingNode in the queue.")
      .def("size", &MappingSearch::size, "Returns the current queue size.")
      .def(
          "make_and_insert_mapping_node",
          [](MappingSearch &self, double lattice_cost,
             std::shared_ptr<LatticeMappingSearchData const>
                 lattice_mapping_data,
             Eigen::Vector3d const &trial_translation_cart,
             std::map<Index, Index> forced_on = {},
             std::vector<std::pair<Index, Index>> forced_off = {}) {
            auto it = self.make_and_insert_mapping_node(
                lattice_cost, lattice_mapping_data, trial_translation_cart,
                forced_on, forced_off);
          },
          py::arg("lattice_cost"), py::arg("lattice_mapping_data"),
          py::arg("trial_translation_cart"), py::arg("forced_on"),
          py::arg("forced_off"),
          R"pbdoc(
          Make and insert a mapping solution

          The (constrained) assignment problem is solved in context of
          a particular lattice mapping and trial translation, and
          the resulting AtomMapping, atom mapping cost, and total cost
          are stored in a MappingNode. The MappingNode is inserted into
          the MappingSearch queue if it satisfies the `max_cost` criteria.
          It is also inserted into the MappingSearch results, if it satisifies
          the cost range and k-best criteria.

          Parameters
          ----------
          lattice_cost : float
              The cost of the lattice mapping that forms the context
              in which atom mappings are solved.
          lattice_mapping_data : ~libcasm.mapping.mapsearch.LatticeMappingSearchData
              Holds the lattice mapping and related data that forms the context
              in which atom mappings are solved.
          trial_translation_cart : array_like, shape=(3,)
              A Cartesian translation applied to atom coordinates in the
              ideal superstructure setting (i.e.
              atom_coordinate_cart_in_supercell) to bring the atoms and sites
              into alignment.
          forced_on : Dict[int, int]
              A map of assignments `site_index: atom_index` that are forced
              on.
          forced_off : List[Tuple[int, int]]
              A list of tuples of assignments `(site_index, atom_index)` that
              are forced off.
          )pbdoc")
      .def(
          "partition", [](MappingSearch &self) { auto it = self.partition(); },
          R"pbdoc(
          Make and insert sub-optimal mapping solutions

          The Murty algorithm is used to generate sub-optimal assignments
          from the current lowest cost solution in the queue (available as
          :func:`~libcasm.mapping.mapsearch.MappingSearch.front`). The
          resulting MappingNode are inserted into the MappingSearch queue if
          they satisify the `max_cost` criteria. They are also inserted into
          the MappingSearch results, if they satisify the cost range and k-best
          criteria. Finally, the node that was partitioned is removed from the
          queue.
          )pbdoc")
      .def("results", &combined_results,
           R"pbdoc(
          Return the best structure mapping results found

          Returns the best results found which satisfy
          the acceptance criteria:

          - min_cost: Keep mappings with total cost >= min_cost
          - max_cost: Keep mappings with total cost <= max_cost
          - k_best: Keep the k_best mappings with lowest total cost
            that also satisfy the min/max cost criteria. Approximate
            ties with the current k_best result are also kept.

          Returns
          -------
          structure_mappings : ~libcasm.mapping.info.StructureMappingResults
              A :class:`~libcasm.mapping.info.StructureMappingResults` object,
              giving possible structure mappings, sorted by total cost.
          )pbdoc");

  py::class_<QueueConstraints>(m, "QueueConstraints", R"pbdoc(
      Used to constrain the structure mapping search queue

      The QueueConstraints functor implements a basic approach
      for managing the mapping search queue. It operates on
      a :class:`~libcasm.mapping.mapsearch.MappingSearch` and
      modifies the search queue to enforce the following
      optional constraints which can reduce the overall search
      time:

      - `min_queue_cost`: Skip searching for sub-optimal atom assignments if
        the total cost is less than `min_queue_cost` by repeatedly calling
        :func:`~libcasm.mapping.mapsearch.MappingSearch.pop_front`.
      - `max_queue_cost`: Skip searching for sub-optimal atom assignments if
        the total cost is greater than `max_queue_cost` by repeatedly calling
        :func:`~libcasm.mapping.mapsearch.MappingSearch.pop_back`.
      - `max_queue_size`: Reduce the queue size to `max_queue_size`, no matter
        what the total cost of queued mappings is, by repeatedly calling
        :func:`~libcasm.mapping.mapsearch.MappingSearch.pop_back`.

      Notes
      -----

      - The :class:`~libcasm.mapping.mapsearch.MappingSearch` constructor
        parameters `min_cost` and `max_cost` set bounds on mappings stored
        in the search :func:`~libcasm.mapping.mapsearch.MappingSearch.results`,
        but not on the :class:`~libcasm.mapping.mapsearch.MappingNode` (used to
        find next-best costing assignments) stored in the
        :class:`~libcasm.mapping.mapsearch.MappingSearch` queue.

      )pbdoc")
      .def(py::init<std::optional<double>, std::optional<double>,
                    std::optional<Index>>(),
           py::arg("min_queue_cost") = std::optional<double>(),
           py::arg("max_queue_cost") = std::optional<double>(),
           py::arg("max_queue_size") = std::optional<Index>(),
           R"pbdoc(
          .. rubric:: Constructor

          Parameters
          ----------
          min_queue_cost : Optional[float] = None
              Minimum cost mappings to keep in the search queue and use as the
              starting point for finding next-best cost atom mapping
              assignments.
          max_queue_cost : Optional[float] = None
              Maximum cost mappings to keep in the search queue and use as the
              starting point for finding next-best cost atom mapping
              assignments.
          max_queue_size : Optional[int] = None
              Maximum search queue size to allow.

          )pbdoc")
      .def("enforce", &QueueConstraints::operator(), py::arg("search"),
           R"pbdoc(
          Enforce constraints on a MappingSearch queue

          Parameters
          ----------
          search : ~libcasm.mapping.mapsearch.MappingSearch
              Search instance to manage.
          )pbdoc");

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif
}
