#ifndef CASM_mapping_SearchData
#define CASM_mapping_SearchData

#include <memory>
#include <optional>

#include "casm/crystallography/Lattice.hh"
#include "casm/crystallography/LinearIndexConverter.hh"
#include "casm/crystallography/SymType.hh"
#include "casm/global/eigen.hh"
#include "casm/mapping/LatticeMapping.hh"

namespace CASM {

namespace xtal {
class BasicStructure;
class SimpleStructure;
}  // namespace xtal

namespace mapping {

/// Mapping an atomic structure to another structure is a
/// heirarchical search problem:
/// - Try a prim
/// - Try lattice mappings to supercells of the prim
/// - Try translations of the lattice-mapped structures
/// - Try assignments of the translated, lattice-mapped structures
///
/// The data structures here hold information specific to
/// the various levels of this search:
/// - StructureSearchData: Data related to the structure
///   being mapped
/// - PrimSearchData: Data related to a prim the structure
///   might be mapped to
/// - LatticeMappingSearchData: Data related to a particular
///   lattice mapping of a structure to a superstructure prim
/// - TrialTranslationSearchData: Data related to a
///   particular trial translation of a lattice mapping

/// \brief Holds data on the structure being mapped
struct StructureSearchData {
  /// \brief Constructor
  StructureSearchData(xtal::Lattice const &_lattice,
                      Eigen::MatrixXd const &_atom_coordinate_cart,
                      std::vector<std::string> _atom_type,
                      std::optional<std::vector<xtal::SymOp>>
                          override_structure_factor_group = std::nullopt);

  /// \brief Constructor - For superstructures
  StructureSearchData(
      std::shared_ptr<StructureSearchData const> _prim_structure_data,
      Eigen::Matrix3l const &_transformation_matrix_to_super);

  /// \brief The structure's lattice
  xtal::Lattice const lattice;

  /// \brief The number of atoms (includes explicitly included vacancies)
  Index const N_atom;

  /// \brief Shape=(3,N_atom), with Cartesian coordinates of atoms as columns
  Eigen::MatrixXd const atom_coordinate_cart;

  /// \brief Size=N_atom, with the name of the atom at each site
  std::vector<std::string> const atom_type;

  /// \brief Symmetry operations that may be used to skip symmetrically
  ///     equivalent structure mappings
  std::vector<xtal::SymOp> const structure_factor_group;

  /// \brief Symmetry operations that may be used to skip symmetrically
  ///     equivalent lattice mappings
  std::vector<xtal::SymOp> const structure_crystal_point_group;

  /// \brief Pointer to primitive structure this represents a superstructure of
  ///
  /// Will be equal to nullptr if this is the primitive structure
  std::shared_ptr<StructureSearchData const> const prim_structure_data;

  /// \brief Transformation matrix from `prim_structure_data->lattice()`
  ///     to this->lattice, if not nullptr, else identity.
  Eigen::Matrix3l const transformation_matrix_to_super;

 private:
  /// \brief Private constructor
  StructureSearchData(
      std::shared_ptr<StructureSearchData const> _prim_structure_data,
      Eigen::Matrix3l const &_transformation_matrix_to_super,
      xtal::UnitCellCoordIndexConverter const &_unitcellcoord_index_converter);
};

/// \brief Construct a xtal::SimpleStructure corresponding to
/// StructureSearchData
xtal::SimpleStructure make_structure(StructureSearchData const &structure_data);

/// \brief Holds prim-related data used for mapping searches
struct PrimSearchData {
  /// \brief Constructor
  PrimSearchData(std::shared_ptr<xtal::BasicStructure const> _prim,
                 std::optional<std::vector<xtal::SymOp>>
                     override_prim_factor_group = std::nullopt,
                 bool enable_symmetry_breaking_atom_cost = true);

  /// \brief The prim
  std::shared_ptr<xtal::BasicStructure const> const prim;

  /// \brief The prim lattice
  xtal::Lattice const prim_lattice;

  /// \brief The number of prim sites
  Index const N_prim_site;

  /// \brief Shape=(3,N_prim_site), with prim site Cartesian
  ///     coordinates as columns
  Eigen::MatrixXd const prim_site_coordinate_cart;

  /// \brief Size=N_prim_site, with names of atoms allowed on
  ///     each prim site
  std::vector<std::vector<std::string>> const prim_allowed_atom_types;

  /// \brief Symmetry operations that may be used to skip symmetrically
  ///     equivalent structure mappings
  std::vector<xtal::SymOp> const prim_factor_group;

  /// \brief Symmetry operations that may be used to skip symmetrically
  ///     equivalent lattice mappings
  std::vector<xtal::SymOp> const prim_crystal_point_group;

  /// \brief Size=N_mode, with shape=(3,N_prim_site) matrices, giving the
  ///     symmetry invariant displacement modes, with columns
  ///     giving the displacement associated with each site
  ///     for a given mode.
  ///
  /// For example:
  ///
  ///     Eigen::Vector3d site_displacement =
  ///         sym_invariant_displacement_modes[mode_index].col(site_index)
  ///
  /// This member may be expensive so it is optional to construct, though
  /// it is required for calculating the symmetry_breaking_atom_cost.
  ///
  std::optional<std::vector<Eigen::MatrixXd>> const
      prim_sym_invariant_displacement_modes;
};

/// \brief Holds prim and lattice mapping-specific data used
///     for mapping searches
struct LatticeMappingSearchData {
  /// \brief Constructor
  LatticeMappingSearchData(
      std::shared_ptr<PrimSearchData const> _prim_data,
      std::shared_ptr<StructureSearchData const> _structure_data,
      LatticeMapping _lattice_mapping);

  /// \brief Holds prim-related data used for mapping searches
  std::shared_ptr<PrimSearchData const> const prim_data;

  /// \brief Holds structure-related data
  std::shared_ptr<StructureSearchData const> const structure_data;

  /// \brief A specific lattice mapping
  LatticeMapping const lattice_mapping;

  /// \brief The transformation matrix that gives the ideal
  ///     superstructure lattice, for this lattice mapping,
  ///     from the prim lattice (i.e. T*N of the lattice
  ///     mapping)
  ///
  /// This is equivalent to:
  ///
  ///     lround(lattice_mapping.transformation_matrix_to_super *
  ///         lattice_mapping.reorientation)
  ///
  Eigen::Matrix3l const transformation_matrix_to_super;

  /// \brief The lattice of the ideal supercell.
  xtal::Lattice const supercell_lattice;

  /// \brief Generates ideal superstructure sites
  xtal::UnitCellCoordIndexConverter const unitcellcoord_index_converter;

  /// \brief Number of sites in the ideal superstructure specified
  ///     by the lattice_mapping
  Index const N_supercell_site;

  /// \brief Matrix of shape=(3,N_atom) containing
  ///     the Cartesian coordinates of the child structure atoms,
  ///     in the state after the inverse lattice mapping deformation
  ///     is applied (\f$F^{-1}\vec{r_2}\f$). The supercell refers
  ///     to the ideal superlattice, S1 = L1 * T * N.
  Eigen::MatrixXd const atom_coordinate_cart_in_supercell;

  /// \brief A shape=(3,N_supercell_site) matrix of Cartesian
  ///     coordinates of the sites in the ideal supercell.
  Eigen::MatrixXd const supercell_site_coordinate_cart;

  /// \brief Size=N_supercell_site, with names of atoms allowed on
  ///     each supercell site
  std::vector<std::vector<std::string>> const supercell_allowed_atom_types;
};

/// \brief Make possible atom -> site translations to bring atoms into
///     registry with the sites.
std::vector<Eigen::Vector3d> make_trial_translations(
    LatticeMappingSearchData const &lattice_mapping_data);

/// \brief A function, such as `make_atom_to_site_cost`,
///     which calculates the atom-to-site mapping cost given the
///     lattice in which the displacements are calculated under
///     periodic boundary conditions, the site-to-atom
///     displacement, atom_type, allowed_atom_types,
///     and value to use for unallowed mappings (infinity).
using AtomToSiteCostFunction = std::function<double(
    xtal::Lattice const &lattice, Eigen::Vector3d const &displacement,
    std::string const &atom_type,
    std::vector<std::string> const &allowed_atom_types, double infinity)>;

/// \brief A function, such as `make_atom_to_site_cost`,
///     which calculates the atom-to-site mapping cost given the
///     lattice in which the displacements are calculated under
///     periodic boundary conditions, the site-to-atom
///     displacement, atom_type, allowed_atom_types,
///     and value to use for unallowed mappings (infinity).
using DispOnlyAtomToSiteCostFunction = std::function<double(
    Eigen::Vector3d const &displacement, std::string const &atom_type,
    std::vector<std::string> const &allowed_atom_types, double infinity)>;

/// \brief Make the mapping cost for a particular atom
///     to a particular structure site
double make_atom_to_site_cost(
    Eigen::Vector3d const &displacement, std::string const &atom_type,
    std::vector<std::string> const &allowed_atom_types, double infinity);

/// \brief Make the mapping cost for a particular atom
///     to a particular structure site
double make_atom_to_site_cost_future(
    xtal::Lattice const &lattice, Eigen::Vector3d const &displacement,
    std::string const &atom_type,
    std::vector<std::string> const &allowed_atom_types, double infinity);

/// \brief Holds data shared amongst all potential atom-to-site
///     assignment problems making use of the same trial
///     translation
struct AtomMappingSearchData {
  /// \brief Constructor
  AtomMappingSearchData(
      std::shared_ptr<LatticeMappingSearchData const> _lattice_mapping_data,
      Eigen::Vector3d const &_trial_translation_cart,
      AtomToSiteCostFunction _atom_to_site_cost_f =
          make_atom_to_site_cost_future,
      double _infinity = 1e20);

  /// \brief Holds lattice mapping-specific data used
  ///     for mapping searches
  std::shared_ptr<LatticeMappingSearchData const> const lattice_mapping_data;

  /// \brief A Cartesian translation applied to atom
  ///     coordinates in the ideal superstructure setting
  ///     (i.e. atom_coordinate_cart_in_supercell) to
  ///     bring the atoms into alignment with ideal
  ///     superstructure sites.
  Eigen::Vector3d const trial_translation_cart;

  /// \brief Shape=(N_supercell_site, N_atom), the
  ///     site-to-atom displacements, of minimum length under
  ///     periodic boundary conditions of the ideal
  ///     superstructure.
  std::vector<std::vector<Eigen::Vector3d>> const site_displacements;

  /// \brief Shape=(N_supercell_site, N_supercell_site) cost
  ///     matrix used in atom to site assignment problem. The
  ///     element `cost_matrix(i, j)` is set to the cost of
  ///     mapping the j-th atom to the i-th site. If there
  ///     are more sites than atoms, vacancies are added.
  Eigen::MatrixXd const cost_matrix;
};

}  // namespace mapping
}  // namespace CASM

#endif
