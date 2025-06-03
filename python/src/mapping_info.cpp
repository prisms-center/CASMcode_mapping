#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// nlohmann::json binding
#define JSON_USE_IMPLICIT_CONVERSIONS 0
#include "casm/casm_io/container/json_io.hh"
#include "casm/casm_io/json/jsonParser.hh"
#include "casm/crystallography/BasicStructure.hh"
#include "casm/mapping/AtomMapping.hh"
#include "casm/mapping/LatticeMapping.hh"
#include "casm/mapping/StructureMapping.hh"
#include "casm/mapping/io/json_io.hh"
#include "casm/mapping/lattice_cost.hh"
#include "pybind11_json/pybind11_json.hpp"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

/// CASM - Python binding code
namespace CASMpy {

using namespace CASM;
using namespace CASM::mapping;

StructureMapping make_structure_mapping(
    std::shared_ptr<xtal::BasicStructure const> const &shared_prim,
    LatticeMapping const &lattice_mapping, AtomMapping const &atom_mapping) {
  return StructureMapping(shared_prim, lattice_mapping, atom_mapping);
}

}  // namespace CASMpy

PYBIND11_MODULE(_mapping_info, m) {
  using namespace CASMpy;

  m.doc() = R"pbdoc(
        Data structures for representing mapping transformations

        libcasm.mapping.info
        --------------------

        The libcasm.mapping.info module contains data structures representing mapping
        transformations.

    )pbdoc";
  py::module_::import("libcasm.xtal");

  py::class_<LatticeMapping>(m, "LatticeMapping", R"pbdoc(
      A mapping between two lattices

      A lattice mapping has the form:

      .. math::

          F L_1 T N = L_2,

      where:

      - :math:`L_1` is a shape=(3,3) matrix with columns containing the
        reference "parent" lattice vectors
      - :math:`L_2` is a shape=(3,3) matrix with columns containing the
        "child" lattice vectors
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
        superlattice of :math:`L_1`
      - :math:`N` is a unimodular reorientation matrix that generates a
        lattice equivalent to :math:`L_1 T` with reoriented lattice
        vectors

      Notes
      -----
      Deformations can be validly defined as parent-to-child or
      child-to-parent. Be careful as to which convention is being used.
      )pbdoc")
      .def(py::init<Eigen::Matrix3d const &, Eigen::Matrix3d const &,
                    Eigen::Matrix3d const &>(),
           py::arg("deformation_gradient"),
           py::arg("transformation_matrix_to_super"), py::arg("reorientation"),
           R"pbdoc(
          Construct a lattice mapping

          Parameters
          ----------

          deformation_gradient : array_like, shape=(3,3)
              The parent-to-child deformation gradient tensor,:math:`F`, a shape=(3,3)
              matrix.
          transformation_matrix_to_super : array_like, shape=(3,3), dtype=int, optional
              The transformation matrix, :math:`T`, that generates a
              superlattice of the parent lattice, :math:`L_1`. The default
              value is np.eye(3).astype(int).
          reorientation : array_like, shape=(3,3), dtype=int, optional
              The unimodular matrix, :math:`N`, that generates a lattice
              equivalent to the parent superlattice, :math:`L_1 T`, with
              different lattice vectors. The default value is np.eye(3).astype(int).
          )pbdoc")
      .def(
          "deformation_gradient",
          [](LatticeMapping const &m) { return m.deformation_gradient; },
          "Returns the shape=(3,3) parent-to-child deformation gradient "
          "tensor.")
      .def(
          "transformation_matrix_to_super",
          [](LatticeMapping const &m) {
            return m.transformation_matrix_to_super;
          },
          "Returns the shape=(3,3) parent supercell transformation matrix, "
          ":math:`T`.")
      .def(
          "reorientation",
          [](LatticeMapping const &m) { return m.reorientation; },
          "Returns the shape=(3,3) unimodular matrix, :math:`N`.")
      .def(
          "isometry", [](LatticeMapping const &m) { return m.isometry; },
          "Returns the shape=(3,3) isometry matrix, :math:`Q`, of the "
          "parent-to-child deformation gradient tensor.")
      .def(
          "left_stretch",
          [](LatticeMapping const &m) { return m.left_stretch; },
          "Returns the shape=(3,3) left symmetric stretch tensor, :math:`V`, "
          "of the parent-to-child deformation gradient tensor.")
      .def(
          "right_stretch",
          [](LatticeMapping const &m) { return m.right_stretch; },
          "Returns the shape=(3,3) right symmetric stretch tensor, :math:`U`, "
          "of the parent-to-child deformation gradient tensor.")
      .def(
          "interpolated",
          [](mapping::LatticeMapping const &m, double f) {
            return interpolated_mapping(m, f);
          },
          py::arg("interpolation_factor"), R"pbdoc(
          Return a mapping along the transformation pathway from the
          parent to the mapped child lattice

          Interpolated lattices can be constructed with the function
          :func:`~libcasm.mapping.methods.make_mapped_lattice`:

          .. code-block:: Python

              from libcasm.mapping.methods import make_mapped_lattice

              interpolated_lattice = make_mapped_lattice(
                  parent_lattice,
                  structure_mapping.interpolated(interpolation_factor))

          Parameters
          ----------
          interpolation_factor : float
              Interpolation factor. The value 0.0 corresponds to the
              ideal parent lattice; and the value 1.0 corresponds to
              the mapped child lattice (the child lattice rigidly
              rotated to align with the ideal parent lattice).

          Returns
          -------
          interpolated_lattice_mapping : ~libcasm.mapping.info.LatticeMapping
              Interpolated lattice mapping
          )pbdoc")
      .def_static(
          "from_dict",
          [](const nlohmann::json &data) -> mapping::LatticeMapping {
            jsonParser json{data};
            return jsonConstructor<mapping::LatticeMapping>::from_json(json);
          },
          "Construct a LatticeMapping from a Python dict.", py::arg("data"))
      .def(
          "to_dict",
          [](mapping::LatticeMapping const &m) -> nlohmann::json {
            jsonParser json;
            to_json(m, json);
            return static_cast<nlohmann::json>(json);
          },
          "Represent the LatticeMapping as a Python dict.");

  py::class_<ScoredLatticeMapping, LatticeMapping>(m, "ScoredLatticeMapping",
                                                   R"pbdoc(
      A mapping between two lattices, plus the mapping cost.

      Notes
      -----
      Deformations can be validly defined as parent-to-child or
      child-to-parent. Be careful as to which convention is being used.
      )pbdoc")
      .def(py::init<double, LatticeMapping>(), py::arg("lattice_cost"),
           py::arg("lattice_mapping"),
           R"pbdoc(
          Construct a scored lattice mapping

          Parameters
          ----------

          lattice_cost : float
              The cost of lattice mapping. The value depends on the method used.
          lattice_mapping : ~libcasm.mapping.info.LatticeMapping
              A :class:`~libcasm.mapping.info.LatticeMapping`
          )pbdoc")
      .def(
          "lattice_cost",
          [](ScoredLatticeMapping const &m) { return m.lattice_cost; },
          "Returns the lattice mapping cost.")
      .def_static(
          "from_dict",
          [](const nlohmann::json &data) -> mapping::ScoredLatticeMapping {
            jsonParser json{data};
            return jsonConstructor<mapping::ScoredLatticeMapping>::from_json(
                json);
          },
          "Construct a ScoredLatticeMapping from a Python dict.",
          py::arg("data"))
      .def(
          "to_dict",
          [](mapping::ScoredLatticeMapping const &m) -> nlohmann::json {
            jsonParser json;
            to_json(m, json);
            return static_cast<nlohmann::json>(json);
          },
          "Represent the ScoredLatticeMapping as a Python dict.");

  py::class_<LatticeMappingResults>(m, "LatticeMappingResults", R"pbdoc(
      Holds a list of scored lattice mapping results.
      )pbdoc")
      .def(py::init<std::vector<ScoredLatticeMapping>>(),
           py::arg("data") = std::vector<ScoredLatticeMapping>(),
           R"pbdoc(
          Construct lattice mapping results data structure

          Parameters
          ----------

          data : List[:class:`~libcasm.mapping.info.ScoredLatticeMapping`]
              The list of scored lattice mappings.
          )pbdoc")
      .def("size", &LatticeMappingResults::size,
           "Returns the number of scored lattice mappings.")
      .def(
          "data", [](LatticeMappingResults const &m) { return m.data; },
          "Returns the list of scored lattice mappings.")
      .def("__len__", &LatticeMappingResults::size)
      .def("__getitem__",
           [](LatticeMappingResults const &m, Index i) { return m.data.at(i); })
      .def(
          "__iter__",
          [](LatticeMappingResults const &m) {
            return py::make_iterator(m.begin(), m.end());
          },
          py::keep_alive<
              0, 1>() /* Essential: keep object alive while iterator exists */)
      .def_static(
          "from_dict",
          [](const nlohmann::json &data) -> mapping::LatticeMappingResults {
            jsonParser json{data};
            return jsonConstructor<mapping::LatticeMappingResults>::from_json(
                json);
          },
          "Construct LatticeMappingResults from a Python dict.",
          py::arg("data"))
      .def(
          "to_dict",
          [](mapping::LatticeMappingResults const &m) -> nlohmann::json {
            jsonParser json;
            to_json(m, json);
            return static_cast<nlohmann::json>(json);
          },
          "Represent LatticeMappingResults as a Python dict.");

  py::class_<AtomMapping>(m, "AtomMapping", R"pbdoc(
     A mapping of atoms between two structures

     An atom mapping is defined in the context of an existing
     :class:`~libcasm.mapping.info.LatticeMapping`. An atom mapping has the form:

     .. math::

         F \left(\vec{r_1}(i) + \vec{d}(i) \right) = \vec{r_2}(p_i) + \vec{t}

     where:

     - :math:`\vec{r_1}(i)` is the Cartesian coordinates of the i-th atom in
       the parent superstructure. The parent superstructure can be constructed
       using `libcasm.xtal.make_superstructure`:

       .. code-block:: Python

           import libcasm.xtal as xtal
           parent_superstructure = xtal.make_superstructure(
               T * N, parent_structure)

       where :math:`T`, and :math:`N` are from a
       :class:`~libcasm.mapping.info.LatticeMapping`. Then the i-th atom coordinate,
       :math:`\vec{r_1}(i)`, is equal to:

       .. code-block:: Python

           parent_superstructure.atom_coordinate_cart()[:,i]

     - :math:`\vec{r_2}(i)` is the Cartesian coordinates of i-th atom in
       the child structure.
     - :math:`F` is the parent-to-child deformation gradient tensor from a
       :class:`~libcasm.mapping.info.LatticeMapping`.
     - :math:`p_i` is a permutation vector, specifying which atom in the
       child structure (:math:`p_i`) is mapped to the i-th site of the parent
       superstructure. Values of :math:`p_i` greater than the number of atoms
       in the child structure indicate inferred vacancies in mappings to
       :class:`~libcasm.xtal.Prim` with vacancies allowed.
     - :math:`\vec{t}` is a translation vector, in Cartesian coordinates, usually
       chosen so the average displacement is zero.
     - :math:`\vec{d}(i)`: The Cartesian displacement associated with the atom
        at the i-th site in the parent superstructure.

     Notes
     -----
     Displacements can be validly defined with different references. This
     method returns displacements that are consistent with the convention
     used by CASM for displacement DoFs: displacements are added to the
     ideal basis coordinates, then strain is applied.

     )pbdoc")
      .def(py::init<Eigen::MatrixXd const &, std::vector<Index> const &,
                    Eigen::Vector3d const &>(),
           py::arg("displacement"), py::arg("permutation"),
           py::arg("translation"), R"pbdoc(
          Construct an atom mapping

          Parameters
          ----------

          displacement : array_like, shape=(3,n)
             The shape=(3,n) matrix whose columns are the atom displacements
             :math:`\vec{d}(i)`.
          permutation : List[int], size=n
             The permutation vector, :math:`p_i`.
          translation : array_like, shape=(3,1),
             The translation vector, :math:`\vec{t}`.
          )pbdoc")
      .def(
          "displacement", [](AtomMapping const &m) { return m.displacement; },
          R"pbdoc(
            Returns the shape=(3,n) matrix whose columns are the Cartesian atom displacements :math:`\vec{d}(i)`.
          )pbdoc")
      .def(
          "permutation", [](AtomMapping const &m) { return m.permutation; },
          R"pbdoc(Returns the permutation vector, :math:`p_i`.)pbdoc")
      .def(
          "translation", [](AtomMapping const &m) { return m.translation; },
          R"pbdoc(Returns the translation vector, :math:`\vec{t}`.)pbdoc")
      .def(
          "interpolated",
          [](mapping::AtomMapping const &m, double f) {
            return interpolated_mapping(m, f);
          },
          py::arg("interpolation_factor"), R"pbdoc(
          Return a mapping along the transformation pathway from the
          ideal parent to the mapped child atom position

          Parameters
          ----------
          interpolation_factor : float
              Interpolation factor. The value 0.0 corresponds to the
              ideal parent sites; and the value 1.0 corresponds to
              the mapped child sites (the child sites rigidly
              rotated to align with the ideal parent lattice).

          Returns
          -------
          interpolated_atom_mapping : ~libcasm.mapping.info.AtomMapping
              Interpolated atom mapping
          )pbdoc")
      .def_static(
          "from_dict",
          [](const nlohmann::json &data) -> mapping::AtomMapping {
            jsonParser json{data};
            return jsonConstructor<mapping::AtomMapping>::from_json(json);
          },
          "Construct an AtomMapping from a Python dict.", py::arg("data"))
      .def(
          "to_dict",
          [](mapping::AtomMapping const &m) -> nlohmann::json {
            jsonParser json;
            to_json(m, json);
            return static_cast<nlohmann::json>(json);
          },
          "Represent the AtomMapping as a Python dict.");

  py::class_<ScoredAtomMapping, AtomMapping>(m, "ScoredAtomMapping", R"pbdoc(
      A mapping of atoms between two structures, plus the mapping cost.

      )pbdoc")
      .def(py::init<double, AtomMapping>(), py::arg("atom_cost"),
           py::arg("atom_mapping"),
           R"pbdoc(
          Construct a scored atom mapping

          Parameters
          ----------

          atom_cost : float
              The cost of atom mapping. The value depends on the method used.
          atom_mapping : ~libcasm.mapping.info.AtomMapping
              A :class:`~libcasm.mapping.info.AtomMapping`
          )pbdoc")
      .def(
          "atom_cost", [](ScoredAtomMapping const &m) { return m.atom_cost; },
          "Returns the lattice mapping cost.")
      .def_static(
          "from_dict",
          [](const nlohmann::json &data) -> mapping::ScoredAtomMapping {
            jsonParser json{data};
            return jsonConstructor<mapping::ScoredAtomMapping>::from_json(json);
          },
          "Construct a ScoredAtomMapping from a Python dict.", py::arg("data"))
      .def(
          "to_dict",
          [](mapping::ScoredAtomMapping const &m) -> nlohmann::json {
            jsonParser json;
            to_json(m, json);
            return static_cast<nlohmann::json>(json);
          },
          "Represent the ScoredAtomMapping as a Python dict.");

  py::class_<AtomMappingResults>(m, "AtomMappingResults", R"pbdoc(
      Holds a list of scored atom mapping results.
      )pbdoc")
      .def(py::init<std::vector<ScoredAtomMapping>>(),
           py::arg("data") = std::vector<ScoredAtomMapping>(),
           R"pbdoc(
          Construct atom mapping results data structure

          Parameters
          ----------

          data : List[:class:`~libcasm.mapping.info.ScoredAtomMapping`]
              The list of scored atom mappings.
          )pbdoc")
      .def("size", &AtomMappingResults::size,
           "Returns the number of scored atom mappings.")
      .def(
          "data", [](AtomMappingResults const &m) { return m.data; },
          "Returns the list of scored atom mappings.")
      .def("__len__", &AtomMappingResults::size)
      .def("__getitem__",
           [](AtomMappingResults const &m, Index i) { return m.data.at(i); })
      .def(
          "__iter__",
          [](AtomMappingResults const &m) {
            return py::make_iterator(m.begin(), m.end());
          },
          py::keep_alive<
              0, 1>() /* Essential: keep object alive while iterator exists */)
      .def_static(
          "from_dict",
          [](const nlohmann::json &data) -> mapping::AtomMappingResults {
            jsonParser json{data};
            return jsonConstructor<mapping::AtomMappingResults>::from_json(
                json);
          },
          "Construct AtomMappingResults from a Python dict.", py::arg("data"))
      .def(
          "to_dict",
          [](mapping::AtomMappingResults const &m) -> nlohmann::json {
            jsonParser json;
            to_json(m, json);
            return static_cast<nlohmann::json>(json);
          },
          "Represent the AtomMappingResults as a Python dict.");

  py::class_<StructureMapping>(m, "StructureMapping", R"pbdoc(
    A mapping between two structures

    A structure mapping is a combination of:

    - A :class:`~libcasm.xtal.Prim`
    - A :class:`~libcasm.mapping.info.LatticeMapping`
    - An :class:`~libcasm.mapping.info.AtomMapping`

    See those class descriptions for details of the mappings.

    )pbdoc")
      .def(py::init(&make_structure_mapping), py::arg("prim"),
           py::arg("lattice_mapping"), py::arg("atom_mapping"), R"pbdoc(
          Construct a structure mapping

          Parameters
          ----------

          prim : ~libcasm.xtal.Prim
              A :class:`~libcasm.xtal.Prim`
          lattice_mapping : ~libcasm.mapping.info.LatticeMapping
              A :class:`~libcasm.mapping.info.LatticeMapping`
          atom_mapping : ~libcasm.mapping.info.AtomMapping
              An :class:`~libcasm.mapping.info.AtomMapping`
          )pbdoc")
      .def(
          "prim", [](StructureMapping const &m) { return m.shared_prim; },
          "Returns the :class:`~libcasm.xtal.Prim`.")
      .def(
          "lattice_mapping",
          [](StructureMapping const &m) { return m.lattice_mapping; },
          "Returns the :class:`~libcasm.mapping.info.LatticeMapping`.")
      .def(
          "atom_mapping",
          [](StructureMapping const &m) { return m.atom_mapping; },
          "Returns the :class:`~libcasm.mapping.info.AtomMapping`.")
      .def(
          "interpolated",
          [](mapping::StructureMapping const &m, double f) {
            return interpolated_mapping(m, f);
          },
          py::arg("interpolation_factor"),
          R"pbdoc(
          Return a mapping along the transformation pathway from the
          ideal parent structure to the mapped child structure

          Interpolated structures can be constructed with the function
          :func:`~libcasm.mapping.methods.make_mapped_structure`:

          .. code-block:: Python

              from libcasm.mapping.methods import make_mapped_structure

              interpolated_structure = make_mapped_structure(
                  structure_mapping.interpolated(interpolation_factor),
                  unmapped_structure)


          Parameters
          ----------
          interpolation_factor : float
              Interpolation factor. The value 0.0 corresponds to the
              ideal parent structure; and the value 1.0 corresponds to
              the mapped child structure (the child structure
              rotated to align with the ideal parent structure).

          Returns
          -------
          interpolated_structure_mapping : ~libcasm.mapping.info.StructureMapping
              Interpolated structure mapping
          )pbdoc")
      .def_static(
          "from_dict",
          [](const nlohmann::json &data,
             std::shared_ptr<xtal::BasicStructure const> const &prim)
              -> mapping::StructureMapping {
            jsonParser json{data};
            return jsonConstructor<mapping::StructureMapping>::from_json(json,
                                                                         prim);
          },
          R"pbdoc(
          Construct a StructureMapping from a Python dict.

          Parameters
          ----------
          data : dict
              The StructureMapping as a Python dict
          prim : ~libcasm.xtal.Prim
              A :class:`~libcasm.xtal.Prim`

          Returns
          -------
          structure_mapping : ~libcasm.mapping.info.StructureMapping
              The structure mapping
          )pbdoc",
          py::arg("data"), py::arg("prim"))
      .def(
          "to_dict",
          [](mapping::StructureMapping const &m) -> nlohmann::json {
            jsonParser json;
            to_json(m, json);
            return static_cast<nlohmann::json>(json);
          },
          "Represent the StructureMapping as a Python dict.");

  py::class_<ScoredStructureMapping, StructureMapping>(
      m, "ScoredStructureMapping", R"pbdoc(
      A mapping between two structures, plus the mapping cost.

      )pbdoc")
      .def(py::init<double, double, double, StructureMapping>(),
           py::arg("lattice_cost"), py::arg("atom_cost"), py::arg("total_cost"),
           py::arg("structure_mapping"),
           R"pbdoc(
          Construct a scored atom mapping

          Parameters
          ----------

          lattice_cost : float
              The cost of lattice mapping. The value depends on the method used.
          atom_cost : float
              The cost of atom mapping. The value depends on the method used.
          total_cost : float
              The cost of structure mapping. The value depends on the method used.
          structure_mapping : ~libcasm.mapping.info.StructureMapping
              A :class:`~libcasm.mapping.info.StructureMapping`
          )pbdoc")
      .def(
          "atom_cost",
          [](ScoredStructureMapping const &m) { return m.atom_cost; },
          "Returns the atom mapping cost.")
      .def(
          "lattice_cost",
          [](ScoredStructureMapping const &m) { return m.lattice_cost; },
          "Returns the lattice mapping cost.")
      .def(
          "total_cost",
          [](ScoredStructureMapping const &m) { return m.total_cost; },
          "Returns the total mapping cost.")
      .def_static(
          "from_dict",
          [](const nlohmann::json &data,
             std::shared_ptr<xtal::BasicStructure const> const &prim)
              -> mapping::ScoredStructureMapping {
            jsonParser json{data};
            return jsonConstructor<mapping::ScoredStructureMapping>::from_json(
                json, prim);
          },
          R"pbdoc(
          Construct a ScoredStructureMapping from a Python dict.

          Parameters
          ----------
          data : dict
              The ScoredStructureMapping as a Python dict
          prim : ~libcasm.xtal.Prim
              A :class:`~libcasm.xtal.Prim`

          Returns
          -------
          scored_structure_mapping : ~libcasm.mapping.info.ScoredStructureMapping
              The scored structure mapping
          )pbdoc",
          py::arg("data"), py::arg("prim"))
      .def(
          "to_dict",
          [](mapping::ScoredStructureMapping const &m) -> nlohmann::json {
            jsonParser json;
            to_json(m, json);
            return static_cast<nlohmann::json>(json);
          },
          "Represent the ScoredStructureMapping as a Python dict.");

  py::class_<StructureMappingResults>(m, "StructureMappingResults", R"pbdoc(
      Holds a list of scored structure mapping results.
      )pbdoc")
      .def(py::init<std::vector<ScoredStructureMapping>>(),
           py::arg("data") = std::vector<ScoredStructureMapping>(),
           R"pbdoc(
          Construct structure mapping results data structure

          Parameters
          ----------

          data : List[:class:`~libcasm.mapping.info.ScoredStructureMapping`]
              The list of scored structure mappings.
          )pbdoc")
      .def("size", &StructureMappingResults::size,
           "Returns the number of scored structure mappings.")
      .def(
          "data", [](StructureMappingResults const &m) { return m.data; },
          "Returns the list of scored structure mappings.")
      .def("__len__", &StructureMappingResults::size)
      .def("__getitem__", [](StructureMappingResults const &m,
                             Index i) { return m.data.at(i); })
      .def(
          "__iter__",
          [](StructureMappingResults const &m) {
            return py::make_iterator(m.begin(), m.end());
          },
          py::keep_alive<
              0, 1>() /* Essential: keep object alive while iterator exists */)
      .def_static(
          "from_dict",
          [](const nlohmann::json &data,
             std::shared_ptr<xtal::BasicStructure const> const &prim)
              -> mapping::StructureMappingResults {
            StructureMappingResults results;
            jsonParser json{data};
            from_json(results, json, prim);
            return results;
          },
          R"pbdoc(
          Construct StructureMappingResults from a Python dict.

          Parameters
          ----------
          data : dict
              The StructureMappingResults as a Python dict
          prim : ~libcasm.xtal.Prim
              A :class:`~libcasm.xtal.Prim`

          Returns
          -------
          results : ~libcasm.mapping.info.StructureMappingResults
              The structure mapping results
          )pbdoc",
          py::arg("data"), py::arg("prim"))
      .def(
          "to_dict",
          [](mapping::StructureMappingResults const &m) -> nlohmann::json {
            jsonParser json;
            to_json(m, json);
            return static_cast<nlohmann::json>(json);
          },
          "Represent the StructureMappingResults as a Python dict.");

  m.def(
      "has_same_prim",
      [](StructureMapping const &first, StructureMapping const &second) {
        return first.shared_prim == second.shared_prim;
      },
      R"pbdoc(
      Check if StructureMapping share the same prim

      Notes
      -----
      Currently this check is not valid for StructureMapping manually constructed
      in Python, only StructureMapping output from a mapping method.


      Parameters
      ----------
      first : ~libcasm.mapping.info.StructureMapping
          The first :class:`~libcasm.mapping.info.StructureMapping`
      second : ~libcasm.mapping.info.StructureMapping
          The second StructureMapping

      Returns
      -------
      bool : True if first and second share the same prim
      )pbdoc",
      py::arg("first"), py::arg("second"));

  py::class_<StructureMappingCost>(m, "StructureMappingCost", R"pbdoc(
   Holds mapping cost between two structures

   The structure mapping cost is a combination of:

   - A lattice mapping cost, lattice_cost
   - An atom mapping cost, atom_cost

   The total structure mapping cost, total_cost, is lattice_cost_weight*lattice_cost + (1.0 - lattice_cost_weight)*atom_cost, where lattice_cost_weight is an input to the :func:`~libcasm.mapping.map_structures` method.

   See those :class:`~libcasm.mapping.info.LatticeMapping` and :class:`~libcasm.mapping.info.AtomMapping` for details of the lattice and atom mappings.

   )pbdoc")
      .def(py::init<double, double, double>(), py::arg("lattice_cost"),
           py::arg("atom_cost"), py::arg("total_cost"), R"pbdoc(
         Construct a structure mapping

         Parameters
         ----------

         lattice_cost : float
             The lattice mapping cost
         atom_cost : float
             The atom mapping cost
         total_cost : float
             The total mapping cost
         )pbdoc")
      .def(
          "lattice_cost",
          [](StructureMappingCost const &s) { return s.lattice_cost; },
          "Returns the lattice mapping cost.")
      .def(
          "atom_cost",
          [](StructureMappingCost const &s) { return s.atom_cost; },
          "Returns the atom mapping cost.")
      .def(
          "total_cost",
          [](StructureMappingCost const &s) { return s.total_cost; },
          "Returns the total mapping cost.")
      .def_static(
          "from_dict",
          [](const nlohmann::json &data) -> mapping::StructureMappingCost {
            jsonParser json{data};
            return jsonConstructor<mapping::StructureMappingCost>::from_json(
                json);
          },
          "Construct a StructureMappingCost from a Python dict.",
          py::arg("data"))
      .def(
          "to_dict",
          [](mapping::StructureMappingCost const &m) -> nlohmann::json {
            jsonParser json;
            to_json(m, json);
            return static_cast<nlohmann::json>(json);
          },
          "Represent the StructureMappingCost as a Python dict.");

  m.def(
      "pretty_json",
      [](const nlohmann::json &data) -> std::string {
        jsonParser json{data};
        std::stringstream ss;
        ss << json << std::endl;
        return ss.str();
      },
      "Pretty-print JSON to string.", py::arg("data"));

  m.def("isotropic_strain_cost", &isotropic_strain_cost, R"pbdoc(
      Return the isotropic strain cost for a lattice deformation.

      Parameters
      ----------
      deformation_gradient : array_like, shape=(3,3)
          The parent-to-child deformation gradient tensor, :math:`F`, a shape=(3,3)
          matrix.

      Returns
      -------
      cost: float
          The isotropic strain cost for the lattice deformation given by :math:`F`. See
          :func:`~libcasm.mapping.methods.map_atoms` for the definition.
      )pbdoc",
        py::arg("deformation_gradient"));

  m.def("symmetry_breaking_strain_cost", &symmetry_breaking_strain_cost,
        R"pbdoc(
      Return the symmetry-breaking strain cost for a lattice deformation.

      Parameters
      ----------
      deformation_gradient : array_like, shape=(3,3)
          The parent-to-child deformation gradient tensor, :math:`F`, a shape=(3,3)
          matrix.
      lattice1_point_group : list[libcasm.xtal.SymOp]
          The point group of the parent.

      Returns
      -------
      cost: float
          The symmetry-breaking strain cost for the lattice deformation given by
          :math:`F`. See :func:`~libcasm.mapping.methods.map_atoms` for the definition.
      )pbdoc",
        py::arg("deformation_gradient"), py::arg("lattice1_point_group"));

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif
}
