#include "casm/mapping/SearchData.hh"

#include "casm/crystallography/BasicStructure.hh"
#include "casm/crystallography/BasicStructureTools.hh"
#include "casm/crystallography/SimpleStructure.hh"
#include "casm/crystallography/SimpleStructureTools.hh"
#include "casm/crystallography/SymTools.hh"
#include "casm/crystallography/SymTypeComparator.hh"
#include "casm/misc/UnaryCompare.hh"

namespace CASM {
namespace mapping {

namespace mapping_impl {

std::vector<xtal::SymOp> make_structure_factor_group(
    xtal::Lattice const &lattice, Eigen::MatrixXd const &atom_coordinate_cart,
    std::vector<std::string> atom_type) {
  if (atom_coordinate_cart.rows() != 3) {
    throw std::runtime_error(
        "Error in make_structure_factor_group: atom_coordinate_cart.rows() != "
        "3");
  }
  if (atom_coordinate_cart.cols() != atom_type.size()) {
    throw std::runtime_error(
        "Error in make_structure_factor_group: atom_type.size() != "
        "atom_coordinate_cart.cols()");
  }

  xtal::BasicStructure prim(lattice);
  for (Index b = 0; b < atom_coordinate_cart.cols(); ++b) {
    xtal::Coordinate coord{atom_coordinate_cart.col(b), prim.lattice(), CART};
    std::vector<xtal::Molecule> site_occ;
    site_occ.push_back(xtal::Molecule{atom_type[b]});
    xtal::Site site{coord, site_occ};
    prim.push_back(site, CART);
  }
  return xtal::make_factor_group(prim);
}

std::vector<xtal::SymOp> make_superstructure_factor_group(
    xtal::Lattice const &prim_lattice,
    std::vector<xtal::SymOp> const &prim_factor_group,
    xtal::Lattice const &super_lattice) {
  double tol = prim_lattice.tol();

  auto all_lattice_points =
      make_lattice_points(prim_lattice, super_lattice, tol);

  std::vector<xtal::SymOp> point_group = make_point_group(super_lattice);
  std::vector<xtal::SymOp> factor_group;

  for (xtal::SymOp const &prim_op : prim_factor_group) {
    // If the primitive factor group operation with translations removed can't
    // map the original structure's lattice onto itself, then ditch that
    // operation.
    UnaryCompare_f<xtal::SymOpMatrixCompare_f> equals_prim_op_ignoring_trans(
        prim_op, tol);
    if (std::find_if(point_group.begin(), point_group.end(),
                     equals_prim_op_ignoring_trans) == point_group.end()) {
      continue;
    }

    // Otherwise take that factor operation, and expand it by adding additional
    // translations within the structure
    for (xtal::UnitCell const &lattice_point : all_lattice_points) {
      xtal::Coordinate lattice_point_coordinate = make_superlattice_coordinate(
          lattice_point, prim_lattice, super_lattice);
      factor_group.emplace_back(
          xtal::SymOp::translation_operation(lattice_point_coordinate.cart()) *
          prim_op);
    }
  }
  sort_factor_group(factor_group, super_lattice);
  return factor_group;
}

/// \brief Return prim site coordinates as columns of
///     a shape=(3, N_prim_site) matrix
Eigen::MatrixXd make_site_coordinate_cart(xtal::BasicStructure const &prim) {
  Eigen::MatrixXd prim_site_coordinate_cart(3, prim.basis().size());
  Index l = 0;
  for (xtal::Site const &site : prim.basis()) {
    prim_site_coordinate_cart.col(l) = site.const_cart();
    ++l;
  }
  return prim_site_coordinate_cart;
}

/// \brief Return supercell site coordinates as columns of
///     a shape=(3, N_supercell_site) matrix
Eigen::MatrixXd make_supercell_site_coordinate_cart(
    xtal::UnitCellCoordIndexConverter const &unitcellcoord_index_converter,
    Eigen::MatrixXd const &prim_site_coordinate_cart,
    xtal::Lattice const &prim_lattice) {
  Index N_supercell_site = unitcellcoord_index_converter.total_sites();
  Eigen::MatrixXd supercell_site_coordinate_cart(3, N_supercell_site);
  for (Index l = 0; l < N_supercell_site; ++l) {
    xtal::UnitCellCoord const &uccoord = unitcellcoord_index_converter(l);
    Index b = uccoord.sublattice();
    xtal::UnitCell const &ijk = uccoord.unitcell();
    supercell_site_coordinate_cart.col(l) =
        prim_site_coordinate_cart.col(b) +
        prim_lattice.lat_column_mat() * ijk.cast<double>();
  }
  return supercell_site_coordinate_cart;
}

/// \brief Return allowed atom types on each supercell site
std::vector<std::vector<std::string>> make_supercell_allowed_atom_types(
    xtal::UnitCellCoordIndexConverter const &unitcellcoord_index_converter,
    std::vector<std::vector<std::string>> const &prim_allowed_atom_types) {
  Index N_supercell_site = unitcellcoord_index_converter.total_sites();
  std::vector<std::vector<std::string>> supercell_allowed_atom_types;
  for (Index l = 0; l < N_supercell_site; ++l) {
    xtal::UnitCellCoord const &uccoord = unitcellcoord_index_converter(l);
    Index b = uccoord.sublattice();
    supercell_allowed_atom_types.push_back(prim_allowed_atom_types[b]);
  }
  return supercell_allowed_atom_types;
}

/// \brief Return atom types on each supercell site
std::vector<std::string> make_supercell_atom_types(
    xtal::UnitCellCoordIndexConverter const &unitcellcoord_index_converter,
    std::vector<std::string> const &prim_atom_types) {
  Index N_supercell_site = unitcellcoord_index_converter.total_sites();
  std::vector<std::string> supercell_atom_types;
  for (Index l = 0; l < N_supercell_site; ++l) {
    xtal::UnitCellCoord const &uccoord = unitcellcoord_index_converter(l);
    Index b = uccoord.sublattice();
    supercell_atom_types.push_back(prim_atom_types[b]);
  }
  return supercell_atom_types;
}

/// Can generate equivalent translations with:
///
///     test_translation + any internal translation
///         + any prim lattice vector translations
///
/// This method checks if a test_translation is unique by:
/// - Adding internal translations, one by one
/// - Checking if test_translation + internal_translation is
///   equivalent up to a prim lattice vector translation to
///   an existing unique translation
///
/// \param prim_lattice The prim's lattice
/// \param test_translation_cart Translation to check if it is unique
/// \param internal_translations_cart Prim internal translations
/// \param current_unique_translations_cart The current list of
///     found unique translations
bool is_new_unique_translation(
    xtal::Lattice const &prim_lattice,
    Eigen::Vector3d const &test_translation_cart,
    std::vector<Eigen::Vector3d> const &internal_translations_cart,
    std::vector<Eigen::Vector3d> const &current_unique_translations_cart) {
  if (internal_translations_cart.size() == 1) {
    return true;
  }

  // cart = L * frac
  //
  // we'll check if the difference between any
  //     x = test_translation_cart + internal translation cart,
  //     and y = current unique translation cart,
  // is a prim lattice vector translation, by checking if
  // the difference as fractional coordinates of the prim
  // lattice vectors,
  //     L.inverse() * (x - y),
  // is integer
  auto const &L_inv = prim_lattice.inv_lat_column_mat();
  double xtal_tol = prim_lattice.tol();

  for (auto const &internal_translation : internal_translations_cart) {
    Eigen::Vector3d x = test_translation_cart + internal_translation;
    for (auto const &y : current_unique_translations_cart) {
      if (is_integer(L_inv * (x - y), xtal_tol)) {
        return false;
      }
    }
  }
  return true;
};

/// \brief Make possible atom -> site translations to bring atoms into
/// registry with the sites.
///
/// The displacements calculated are the minimum length displacements
/// that satisfy:
///
///     site_coordinate_cart[i] + site_displacements[i][j] =
///         F^{-1}*atom_coordinate_cart[j] + trial_translation
///
/// under periodic boundary conditions.
///
/// Trial translations are found by choosing the fewest valid
/// atom type -> allowed site translations.
///
/// \param atom_coordinate_cart_in_supercell Matrix of
///     shape=(3,N_atom) containing the Cartesian coordinates of the
///     child structure atoms, in the state after the inverse
///     lattice mapping deformation is applied
///     (\f$F^{-1}\vec{r_2}\f$). S1 refers to the ideal
///     superlattice, S1 = L1 * T * N.
/// \param atom_type Vector of size=N_atom containing the
///     types of the atoms being mapped. May include vacancies.
///     Any vacancies included in the child atoms **must** be
///     mapped. Other vacancies may be added if there are more
///     sites than atoms.
/// \param prim_lattice Prim's lattice.
/// \param prim_site_coordinate_cart A shape=(3,N_prim_site)
///     matrix of Cartesian coordinates of the sites in the prim.
/// \param prim_allowed_atom_types The atom types allowed on each
///     site in the prim.
///
/// \returns Vector of possible trial_translation
///
std::vector<Eigen::Vector3d> make_trial_translations(
    Eigen::MatrixXd const &atom_coordinate_cart_in_supercell,
    std::vector<std::string> const &atom_type,
    xtal::Lattice const &prim_lattice,
    Eigen::MatrixXd const &prim_site_coordinate_cart,
    std::vector<std::vector<std::string>> const &prim_allowed_atom_types,
    std::vector<xtal::SymOp> const &prim_factor_group) {
  std::vector<Eigen::Vector3d> trial_translations;

  std::vector<Eigen::Vector3d> internal_translations =
      xtal::make_internal_translations(prim_factor_group, prim_lattice.tol());

  // choose atom (best_atom_index) with fewest allowed sites in prim
  // if any atom is not allowed on any site?:
  // -> no allowed translations / assignments so return
  Index N_atom = atom_type.size();
  Index best_atom_index = 0;
  Index min_N_allowed_sites = prim_allowed_atom_types.size() + 1;
  for (Index atom_index = 0; atom_index != N_atom; ++atom_index) {
    Index N_allowed_sites = 0;
    for (auto const &allowed_atom_types : prim_allowed_atom_types) {
      auto begin = allowed_atom_types.begin();
      auto end = allowed_atom_types.end();
      if (std::find(begin, end, atom_type[atom_index]) != end) {
        ++N_allowed_sites;
      }
    }
    if (N_allowed_sites == 0) {
      return trial_translations;
    }
    if (N_allowed_sites < min_N_allowed_sites) {
      best_atom_index = atom_index;
      min_N_allowed_sites = N_allowed_sites;
    }
  }

  // collect unique translations from chosen atom (best_atom_index)
  // to allowed prim sites
  Index N_prim_site = prim_allowed_atom_types.size();
  Eigen::Vector3d test_translation;
  for (Index prim_site_index = 0; prim_site_index < N_prim_site;
       ++prim_site_index) {
    auto const &allowed_atom_types = prim_allowed_atom_types[prim_site_index];
    auto begin = allowed_atom_types.begin();
    auto end = allowed_atom_types.end();
    if (std::find(begin, end, atom_type[best_atom_index]) == end) {
      continue;
    }
    test_translation = prim_site_coordinate_cart.col(prim_site_index) -
                       atom_coordinate_cart_in_supercell.col(best_atom_index);

    if (is_new_unique_translation(prim_lattice, test_translation,
                                  internal_translations, trial_translations)) {
      trial_translations.push_back(test_translation);
    }
  }

  return trial_translations;
}

/// \brief Make container of displacement vectors
///
/// The displacements calculated are the minimum length displacements
/// that satisfy:
///
///     site_coordinate_cart[i] + site_displacements[i][j] =
///         F^{-1}*atom_coordinate_cart[j] + trial_translation
///
/// under periodic boundary conditions.
///
/// \param site_displacements Container to be populated with
///     displacements. The values satify the relation
///     `site[i] + displacements[i][j] == atom[j] + translation`
///     using the minimum length displacement possible under periodic
///     boundary conditions with the given lattice.
/// \param lattice Lattice in which the displacements are calculated
///     under periodic boundary conditions.
/// \param site_coordinate_cart A shape=(3,N_site) matrix of Cartesian
///     coordinates of the sites.
/// \param atom_coordinate_cart_in_supercell Matrix of shape=(3,N_atom)
///     containing the Cartesian coordinates of the child structure atoms,
///     in the state after the inverse lattice mapping deformation is
///     applied (\f$F^{-1}\vec{r_2}\f$). S1 refers to the ideal
///     superlattice, S1 = L1 * T * N.
/// \param trial_translation A translation applied to atom coordinates to
///     bring the atoms and sites into alignment.
///
std::vector<std::vector<Eigen::Vector3d>> make_site_displacements(
    xtal::Lattice const &lattice,
    Eigen::MatrixXd const &supercell_site_coordinate_cart,
    Eigen::MatrixXd const &atom_coordinate_cart_in_supercell,
    Eigen::Vector3d const &trial_translation) {
  if (atom_coordinate_cart_in_supercell.cols() >
      supercell_site_coordinate_cart.cols()) {
    std::cout << "supercell_site_coordinate_cart.T:" << std::endl;
    std::cout << supercell_site_coordinate_cart.transpose() << std::endl;
    std::cout << "atom_coordinate_cart_in_supercell.T:" << std::endl;
    std::cout << atom_coordinate_cart_in_supercell.transpose() << std::endl;
    std::cout << "trial_translation.T:" << trial_translation.transpose()
              << std::endl;
    throw std::runtime_error(
        "Error in make_site_displacements: "
        "atom_coordinate_cart_in_supercell.cols() > "
        "supercell_site_coordinate_cart.cols()");
  }

  Index N_atom = atom_coordinate_cart_in_supercell.cols();
  Index N_site = supercell_site_coordinate_cart.cols();

  std::vector<std::vector<Eigen::Vector3d>> site_displacements;
  site_displacements.resize(N_site);
  for (auto &v : site_displacements) {
    v.resize(N_atom);
  }

  for (Index atom_index = 0; atom_index < N_atom; ++atom_index) {
    for (Index site_index = 0; site_index < N_site; ++site_index) {
      site_displacements[site_index][atom_index] = robust_pbc_displacement_cart(
          lattice, supercell_site_coordinate_cart.col(site_index),
          atom_coordinate_cart_in_supercell.col(atom_index) +
              trial_translation);
    }
  }
  return site_displacements;
}

/// \brief Calculate elements of the cost matrix
///
/// The assignment problem cost matrix is calculated from
/// site-to-atom displacements, atom types, and the
/// types allowed on each site.
///
/// The site displacements are the minimum length displacements
/// that satisfy:
///
///     site_coordinate_cart[i] + site_displacements[i][j] =
///         F^{-1}*atom_coordinate_cart[j] + trial_translation
///
/// under periodic boundary conditions.
///
/// \param cost_matrix Cost matrix to construct. Is resized to
///     shape=(N_site, N_site) if necessary. The element
///     `cost_matrix(i, j)` is set to the cost of mapping the
///     j-th atom to the i-th site. If there are more sites
///     than atoms, vacancies are added.
/// \param f A function used to calculate the atom mapping cost
///      to a particular site. Follows the signature of
///     `make_atom_mapping_cost`.
/// \param lattice Lattice in which the displacements are calculated
///     under periodic boundary conditions.
/// \param site_displacements The site-to-atom displacements,
///     of minimum length under periodic boundary conditions.
/// \param atom_type Vector of size=N_atom containing the
///     types of the atoms being mapped. May include vacancies.
///     Any vacancies included in the child atoms **must** be
///     mapped. Other vacancies may be added if there are more
///     sites than atoms.
/// \param allowed_atom_types The atom types allowed on each site
/// \param infinity The value to use for unallowed mappings
///
Eigen::MatrixXd make_cost_matrix(
    AtomToSiteCostFunction f, xtal::Lattice const &lattice,
    std::vector<std::vector<Eigen::Vector3d>> const &site_displacements,
    std::vector<std::string> const &atom_type,
    std::vector<std::vector<std::string>> const &allowed_atom_types,
    double infinity) {
  if (!f) {
    throw std::runtime_error(
        "Error in make_cost_matrix: atom mapping cost function is empty");
  }
  if (site_displacements.size() != allowed_atom_types.size()) {
    throw std::runtime_error(
        "Error in make_cost_matrix: site_displacements.size() != "
        "allowed_atom_types.size()");
  }

  for (auto const &site_displacements_i : site_displacements) {
    if (site_displacements_i.size() != atom_type.size()) {
      throw std::runtime_error(
          "Error in make_cost_matrix: an element of site_displacements != "
          "atom_type.size()");
    }
  }

  Index N_site = allowed_atom_types.size();
  Index N_atom = atom_type.size();

  Eigen::MatrixXd cost_matrix(N_site, N_site);

  // make cost matrix: use cost_matrix(site_index, atom_index)
  // to match AtomMapping permutation convention
  for (Index atom_index = 0; atom_index < N_atom; ++atom_index) {
    for (Index site_index = 0; site_index < N_site; ++site_index) {
      cost_matrix(site_index, atom_index) =
          f(lattice, site_displacements[site_index][atom_index],
            atom_type[atom_index], allowed_atom_types[site_index], infinity);
    }
  }
  // If N_atom < N_site, treat as additional vacancies to map
  for (Index atom_index = N_atom; atom_index < N_site; ++atom_index) {
    for (Index site_index = 0; site_index < N_site; ++site_index) {
      cost_matrix(site_index, atom_index) =
          f(lattice, Eigen::Vector3d::Zero(), "Va",
            allowed_atom_types[site_index], infinity);
    }
  }

  return cost_matrix;
}

}  // namespace mapping_impl

/// \brief Constructor
///
/// \param _lattice The child structure's lattice
/// \param _atom_coordinate_cart Matrix of shape=(3,N) containing
///     the Cartesian coordinates of the child structure atoms,
///     in their original state without the inverse lattice mapping
///     deformation applied (\vec{r_2}).
/// \param _atom_type Vector of size=N_atom containing the
///     types of the child structure atoms. May include vacancies.
///     Any vacancies included in the child atoms **must** be mapped.
///     Other vacancies may be added if there are more parent
///     superstructure sites than child atoms & vacancies.
/// \param override_structure_factor_group Optional, allows explicitly
///     setting the symmetry operations used to skip symmetrically
///     equivalent structure mappings. The default (std::nullopt),
///     uses the structure factor group as generated by
///     `xtal::make_factor_group`. The first symmetry operation should
///     always be the identity operation. Will throw if an empty
///     vector is provided.
StructureSearchData::StructureSearchData(
    xtal::Lattice const &_lattice, Eigen::MatrixXd const &_atom_coordinate_cart,
    std::vector<std::string> _atom_type,
    std::optional<std::vector<xtal::SymOp>> override_structure_factor_group)
    : lattice(_lattice),
      N_atom(_atom_coordinate_cart.cols()),
      atom_coordinate_cart(_atom_coordinate_cart),
      atom_type(std::move(_atom_type)),
      structure_factor_group(override_structure_factor_group == std::nullopt
                                 ? mapping_impl::make_structure_factor_group(
                                       lattice, atom_coordinate_cart, atom_type)
                                 : std::move(*override_structure_factor_group)),
      structure_crystal_point_group(xtal::make_crystal_point_group(
          structure_factor_group, lattice.tol())),
      prim_structure_data(nullptr),
      transformation_matrix_to_super(Eigen::Matrix3l::Identity()) {
  if (atom_type.size() != atom_coordinate_cart.cols()) {
    throw std::runtime_error(
        "Error in StructureSearchData: atom_type.size() != "
        "atom_coordinate_cart.cols()");
  }
  if (structure_factor_group.empty()) {
    throw std::runtime_error(
        "Error in StructureSearchData: Constructed with empty "
        "structure_factor_group.");
  }
}

/// \brief Constructor - For superstructures
///
/// \param _prim_structure_data The search data representing a
///     primitive structure
/// \param _transformation_matrix_to_super The transformation matrix
///     that gives this structure's lattice in terms of the
///     primitive structure's lattice,
///     `_prim_structure_data->lattice()`.
StructureSearchData::StructureSearchData(
    std::shared_ptr<StructureSearchData const> _prim_structure_data,
    Eigen::Matrix3l const &_transformation_matrix_to_super)
    : StructureSearchData(
          _prim_structure_data, _transformation_matrix_to_super,
          xtal::UnitCellCoordIndexConverter(_transformation_matrix_to_super,
                                            _prim_structure_data->N_atom)) {}

/// \brief Private constructor
///
/// To make use of temporary UnitCellCoordIndexConverter
StructureSearchData::StructureSearchData(
    std::shared_ptr<StructureSearchData const> _prim_structure_data,
    Eigen::Matrix3l const &_transformation_matrix_to_super,
    xtal::UnitCellCoordIndexConverter const &_unitcellcoord_index_converter)
    : lattice(xtal::make_superlattice(_prim_structure_data->lattice,
                                      _transformation_matrix_to_super)),
      N_atom(_unitcellcoord_index_converter.total_sites()),
      atom_coordinate_cart(mapping_impl::make_supercell_site_coordinate_cart(
          _unitcellcoord_index_converter,
          _prim_structure_data->atom_coordinate_cart,
          _prim_structure_data->lattice)),
      atom_type(mapping_impl::make_supercell_atom_types(
          _unitcellcoord_index_converter, _prim_structure_data->atom_type)),
      structure_factor_group(mapping_impl::make_superstructure_factor_group(
          _prim_structure_data->lattice,
          _prim_structure_data->structure_factor_group, lattice)),
      structure_crystal_point_group(xtal::make_crystal_point_group(
          structure_factor_group, lattice.tol())),
      prim_structure_data(_prim_structure_data),
      transformation_matrix_to_super(_transformation_matrix_to_super) {}

/// \brief Construct a xtal::SimpleStructure corresponding to
/// StructureSearchData
xtal::SimpleStructure make_structure(
    StructureSearchData const &structure_data) {
  xtal::SimpleStructure structure;
  structure.lat_column_mat = structure_data.lattice.lat_column_mat();
  structure.atom_info.resize(structure_data.N_atom);
  structure.atom_info.coords = structure_data.atom_coordinate_cart;
  structure.atom_info.names = structure_data.atom_type;
  return structure;
}

/// \brief Constructor
///
/// \param _prim The primitive reference "parent"
///     structure, where `prim.lattice()` is \f$L_1\f$ of the
///     lattice mapping. Site occupants must be atomic (single
///     atoms only, not molecular occupants).
/// \param override_prim_factor_group Optional, allows explicitly
///     setting the symmetry operations used to skip symmetrically
///     equivalent structure mappings. The default (std::nullopt),
///     uses the prim factor group as generated by
///     `xtal::make_factor_group(*_prim)`. The first
///     symmetry operation should always be the identity operation.
///     Will throw if an empty vector is provided.
/// \param enable_symmetry_breaking_atom_cost If
///     symmetry_breaking_atom_cost is intended to be used,
///     setting this to true will generate the symmetry-invariant
///     displacement modes required for the calculation using
///     this->prim_factor_group.
PrimSearchData::PrimSearchData(
    std::shared_ptr<xtal::BasicStructure const> _prim,
    std::optional<std::vector<xtal::SymOp>> override_prim_factor_group,
    bool enable_symmetry_breaking_atom_cost)
    : prim(std::move(_prim)),
      prim_lattice(prim->lattice()),
      N_prim_site(prim->basis().size()),
      prim_site_coordinate_cart(mapping_impl::make_site_coordinate_cart(*prim)),
      prim_allowed_atom_types(xtal::allowed_molecule_names(*prim)),
      prim_factor_group(override_prim_factor_group == std::nullopt
                            ? xtal::make_factor_group(*prim, prim_lattice.tol())
                            : std::move(*override_prim_factor_group)),
      prim_crystal_point_group(xtal::make_crystal_point_group(
          prim_factor_group, prim_lattice.tol())),
      prim_sym_invariant_displacement_modes(
          enable_symmetry_breaking_atom_cost
              ? std::optional<std::vector<Eigen::MatrixXd>>(
                    xtal::generate_invariant_shuffle_modes(
                        prim_factor_group,
                        xtal::make_permutation_representation(
                            *prim, prim_factor_group)))
              : std::optional<std::vector<Eigen::MatrixXd>>()) {
  // Validation
  for (xtal::Site const &site : prim->basis()) {
    for (xtal::Molecule const &mol : site.occupant_dof()) {
      if (mol.atoms().size() > 1) {
        throw std::runtime_error(
            "Error in PrimSearchData: only prim with atomic occupants are "
            "supported");
      }
    }
  }
  if (prim_factor_group.empty()) {
    throw std::runtime_error(
        "Error in PrimSearchData: Constructed with empty prim_factor_group.");
  }
}

/// \brief Constructor
///
/// \param _prim_data Data for the prim a structure
///     is being mapped to
/// \param _structure_data Data for the structure
///     being mapped
/// \param _lattice_mapping A lattice mapping relating
///     the lattice of a superstructure of the prim
///     to the lattice of the structure being mapped
LatticeMappingSearchData::LatticeMappingSearchData(
    std::shared_ptr<PrimSearchData const> _prim_data,
    std::shared_ptr<StructureSearchData const> _structure_data,
    LatticeMapping _lattice_mapping)
    : prim_data(std::move(_prim_data)),
      structure_data(std::move(_structure_data)),
      lattice_mapping(std::move(_lattice_mapping)),
      transformation_matrix_to_super(
          lround(lattice_mapping.transformation_matrix_to_super *
                 lattice_mapping.reorientation)),
      supercell_lattice(xtal::make_superlattice(
          prim_data->prim_lattice, transformation_matrix_to_super)),
      unitcellcoord_index_converter(transformation_matrix_to_super,
                                    prim_data->N_prim_site),
      N_supercell_site(unitcellcoord_index_converter.total_sites()),
      atom_coordinate_cart_in_supercell(
          lattice_mapping.deformation_gradient.inverse() *
          structure_data->atom_coordinate_cart),
      supercell_site_coordinate_cart(
          mapping_impl::make_supercell_site_coordinate_cart(
              unitcellcoord_index_converter,
              prim_data->prim_site_coordinate_cart, prim_data->prim_lattice)),
      supercell_allowed_atom_types(
          mapping_impl::make_supercell_allowed_atom_types(
              unitcellcoord_index_converter,
              prim_data->prim_allowed_atom_types)) {}

/// \brief Make possible atom -> site translations to bring atoms into
///     registry with the sites.
///
/// The displacements calculated are the minimum length displacements
/// that satisfy:
///
///     site_coordinate_cart[i] + site_displacements[i][j] =
///         F^{-1}*atom_coordinate_cart[j] + trial_translation
///
/// under periodic boundary conditions.
///
/// Trial translations are found by choosing the fewest valid
/// atom type -> allowed site translations.
///
/// \param lattice_mapping_data Data describing a
///     lattice mapping between a prim and a structure
///
/// \returns Vector of possible trial_translation
///
std::vector<Eigen::Vector3d> make_trial_translations(
    LatticeMappingSearchData const &lattice_mapping_data) {
  return mapping_impl::make_trial_translations(
      lattice_mapping_data.atom_coordinate_cart_in_supercell,
      lattice_mapping_data.structure_data->atom_type,
      lattice_mapping_data.prim_data->prim_lattice,
      lattice_mapping_data.prim_data->prim_site_coordinate_cart,
      lattice_mapping_data.prim_data->prim_allowed_atom_types,
      lattice_mapping_data.prim_data->prim_factor_group);
}

/// \brief Make the atom mapping cost for a particular atom
///     to a particular structure site
///
/// Note:
/// - Deprecated in v2.3.0. Will be replaced by `atom_to_site_cost_future`,
///   in >=v3.0.0.
///
/// Mapping cost:
/// - of a vacancy (xtal::is_vacancy is used to check
/// the atom_type) to any site that allows vacancies is set to 0.0.
/// - to a site that does not allow the atom type is infinity
/// - otherwise, equal to displacement length squared
///
/// \param displacement The minimum length displacement, accounting
///     for periodic boundaries, from the site to the atom
/// \param atom_type The atom type.
/// \param allowed_atom_types The atom types allowed on the site
/// \param infinity The value to use for unallowed mappings
double make_atom_to_site_cost(
    Eigen::Vector3d const &displacement, std::string const &atom_type,
    std::vector<std::string> const &allowed_atom_types, double infinity) {
  // if vacancy is allowed on site, return 0.0; else return infinity
  if (xtal::is_vacancy(atom_type)) {
    for (auto const &allowed_type : allowed_atom_types) {
      if (xtal::is_vacancy(allowed_type)) {
        return 0.0;
      }
    }
    return infinity;
  }

  // if non-vacancy is not allowed on site, return infinity
  auto begin = allowed_atom_types.begin();
  auto end = allowed_atom_types.end();
  if (std::find(begin, end, atom_type) == end) {
    return infinity;
  }

  // otherwise, return distance squared
  return displacement.dot(displacement);
}

/// \brief Make the atom mapping cost for a particular atom
///     to a particular structure site
///
/// Mapping cost:
/// - of a vacancy (xtal::is_vacancy is used to check
/// the atom_type) to any site that allows vacancies is set to 0.0.
/// - to a site that does not allow the atom type is infinity
/// - of a displacement on the boundary of the Voronoi cell is set to infinity,
/// - otherwise, equal to displacement length squared
///
/// \param lattice Lattice in which the displacements are calculated
///     under periodic boundary conditions.
/// \param displacement The minimum length displacement, accounting
///     for periodic boundaries, from the site to the atom
/// \param atom_type The atom type.
/// \param allowed_atom_types The atom types allowed on the site
/// \param infinity The value to use for unallowed mappings
double make_atom_to_site_cost_future(
    xtal::Lattice const &lattice, Eigen::Vector3d const &displacement,
    std::string const &atom_type,
    std::vector<std::string> const &allowed_atom_types, double infinity) {
  // if vacancy is allowed on site, return 0.0; else return infinity
  if (xtal::is_vacancy(atom_type)) {
    for (auto const &allowed_type : allowed_atom_types) {
      if (xtal::is_vacancy(allowed_type)) {
        return 0.0;
      }
    }
    return infinity;
  }

  // if non-vacancy is not allowed on site, return infinity
  auto begin = allowed_atom_types.begin();
  auto end = allowed_atom_types.end();
  if (std::find(begin, end, atom_type) == end) {
    return infinity;
  }

  Eigen::Vector3d lattice_trans;
  double v_dist = lattice.max_voronoi_measure(displacement, lattice_trans);
  if (v_dist > 1. + lattice.tol()) {
    throw std::runtime_error(
        "Error in make_atom_to_site_cost: "
        "displacement is larger than the maximum Voronoi measure");
  } else if (v_dist > (1. - lattice.tol())) {
    // If displacement is on the boundary of the Voronoi cell, then there
    // are non-equivalent displacement choices. To ensure
    // that the cost is the same for all equivalent displacements,
    // we return infinity and force the use of a larger supercell to
    // resolve the ambiguity.
    return infinity;
  }

  // otherwise, return distance squared
  return displacement.dot(displacement);
}

/// \brief Constructor
///
/// \param _lattice_mapping_data Lattice mapping-specific data
/// \param _trial_translation_cart A Cartesian translation applied
///     to atom coordinates in the ideal superstructure setting
///     (i.e. atom_coordinate_cart_in_supercell) to bring the
///     atoms and sites into alignment.
/// \param _atom_to_site_cost_f A function used to calculate
///      the atom mapping cost to a particular site. Follows
///     the signature of `make_atom_to_site_cost`,
///     which is the default method.
/// \param _infinity The value used in the assignment problem
///     cost matrix when a particular assignment is not
///     allowed.
AtomMappingSearchData::AtomMappingSearchData(
    std::shared_ptr<LatticeMappingSearchData const> _lattice_mapping_data,
    Eigen::Vector3d const &_trial_translation_cart,
    AtomToSiteCostFunction _atom_to_site_cost_f, double _infinity)
    : lattice_mapping_data(std::move(_lattice_mapping_data)),
      trial_translation_cart(_trial_translation_cart),
      site_displacements(mapping_impl::make_site_displacements(
          lattice_mapping_data->supercell_lattice,
          lattice_mapping_data->supercell_site_coordinate_cart,
          lattice_mapping_data->atom_coordinate_cart_in_supercell,
          trial_translation_cart)),
      cost_matrix(mapping_impl::make_cost_matrix(
          _atom_to_site_cost_f, lattice_mapping_data->supercell_lattice,
          site_displacements, lattice_mapping_data->structure_data->atom_type,
          lattice_mapping_data->supercell_allowed_atom_types, _infinity)) {}

}  // namespace mapping
}  // namespace CASM
