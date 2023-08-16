#include "casm/mapping/map_atoms.hh"

#include "casm/crystallography/BasicStructure.hh"
#include "casm/crystallography/Lattice.hh"
#include "casm/crystallography/SimpleStructure.hh"
#include "casm/crystallography/SymType.hh"
#include "casm/mapping/AtomMapping.hh"
#include "casm/mapping/LatticeMapping.hh"
#include "casm/mapping/SearchData.hh"
#include "casm/mapping/impl/SimpleStrucMapCalculator.hh"
#include "casm/mapping/impl/StrucMapping.hh"

namespace CASM {
namespace mapping_impl {

/// \brief Convert LatticeMapping to mapping_impl::LatticeNode
///
/// LatticeMapping convention is F * L1 * T * N = L2
/// LatticeNode convention is L1 * T * N = stretch * isometry * L2
///
/// \param parent_lattice The L1 lattice
/// \param lattice_mapping A LatticeMapping transformation
///
/// \returns A mapping_impl::LatticeNode equivalent to lattice_mapping
///
LatticeNode make_lattice_node(xtal::Lattice const &parent_lattice,
                              mapping::LatticeMapping const &lattice_mapping) {
  Eigen::Matrix3d L1 = parent_lattice.lat_column_mat();
  Eigen::Matrix3d T = lattice_mapping.transformation_matrix_to_super;
  Eigen::Matrix3d N = lattice_mapping.reorientation;
  Eigen::Matrix3d F = lattice_mapping.deformation_gradient;

  xtal::Lattice parent_scel(L1 * T * N, parent_lattice.tol());
  xtal::Lattice unmapped_child_prim(F * L1 * T * N, parent_lattice.tol());
  xtal::Lattice unmapped_child_scel = unmapped_child_prim;
  return make_lattice_node(parent_lattice, parent_scel, unmapped_child_prim,
                           unmapped_child_scel);
}

/// \brief Convert mapping_impl::LatticeNode to LatticeMapping
///
/// LatticeMapping convention is F * L1 * T * N = L2
/// LatticeNode convention is L1 * T * N = stretch * isometry * L2
/// LatticeNode convention is T * N =
/// lattice_node.parent.transformation_matrix_to_super()
mapping::LatticeMapping make_lattice_mapping(LatticeNode const &lattice_node) {
  Eigen::Matrix3d F = (lattice_node.stretch * lattice_node.isometry).inverse();
  Eigen::Matrix3d T =
      lattice_node.parent.transformation_matrix_to_super().cast<double>();
  Eigen::Matrix3d N = Eigen::Matrix3d::Identity();
  return mapping::LatticeMapping{F, T, N};
}

/// \brief Convert mapping_impl::MappingNode to AtomMapping
///
/// AtomMapping conventions is: F*(r1[i] + d[i]) = r2[perm[i]] + trans
/// MappingNode convention is: (r1[i] + d[i]) = F.inv*r2[perm[i]] +
/// trans_mapping_node
/// -> trans = F * trans_mapping_node
///
/// \param deformation_gradient The parent-to-child deformation
///     gradient tensor, F. This is passed separately because in
///     current use cases it is already calculated. It can
///     be calculated from mapping_node.lattice_node as:
///     `(lattice_node.stretch * lattice_node.isometry).inverse()`.
/// \param mapping_node A mapping_impl::MappingNode
///
/// \returns An AtomMapping equivalent to mapping_node
///
mapping::AtomMapping make_atom_mapping(
    Eigen::Matrix3d const &deformation_gradient,
    MappingNode const &mapping_node) {
  auto const &atomic_node = mapping_node.atomic_node;
  Eigen::MatrixXd disp = mapping_node.atom_displacement;
  std::vector<Index> perm = mapping_node.atom_permutation;
  Eigen::Vector3d trans = deformation_gradient * atomic_node.translation;
  return mapping::AtomMapping{disp, perm, trans};
}

bool is_symmetry_breaking_atom_cost(std::string atom_cost_method) {
  if (atom_cost_method == "isotropic_disp_cost") {
    return false;
  } else if (atom_cost_method == "symmetry_breaking_disp_cost") {
    return true;
  } else {
    throw std::runtime_error("Error: atom_cost_method not recognized");
  }
}

}  // namespace mapping_impl

namespace mapping {

// Note: See source file for full documentation

/// \brief Find atom mappings
///
/// \param prim The reference "parent" structure
/// \param structure2 The "child" structure
/// \param lattice_mapping The lattice mapping to use
/// \param prim_factor_group Used to skip mappings that are
///     symmetrically equivalent mappings. The default (empty), is
///     equivalent to only including the identity operation.
/// \param min_cost Keep atom mappings with cost >= min_cost
/// \param max_cost Keep atom mappings with cost <= max_cost
/// \param atom_cost_method One of "isotropic_disp_cost" or
///     "symmetry_breaking_disp_cost"
/// \param k_best Keep the k_best results satisfying the min_cost and
///     max_cost constraints. If there are approximate ties, those
///     will also be kept.
/// \param cost_tol Tolerance for checking if lattice mapping costs are
///     approximately equal
AtomMappingResults map_atoms(xtal::BasicStructure const &prim,
                             xtal::SimpleStructure const &structure2,
                             LatticeMapping const &lattice_mapping,
                             std::vector<xtal::SymOp> prim_factor_group,
                             double min_cost, double max_cost,
                             std::string atom_cost_method, int k_best,
                             double cost_tol) {
  bool symmetrize_atom_cost =
      mapping_impl::is_symmetry_breaking_atom_cost(atom_cost_method);

  if (k_best < 1) {
    throw std::runtime_error("Error in map_atoms: k_best < 1 is not allowed");
  }
  if (prim_factor_group.empty()) {
    prim_factor_group.push_back(xtal::SymOp::identity());
  }

  /// For the StrucMapper::map_deformed_struc_impose_lattice_node method:
  /// - Parameters `min_va_frac`, `max_va_frac`, `max_volume_change`,
  ///   and `soft_va_limit` have no effect.
  /// - `robust` search is used anyway if k_best > 1

  mapping_impl::SimpleStrucMapCalculator calculator(
      xtal::make_simple_structure(prim), prim_factor_group,
      CASM::xtal::SimpleStructure::SpeciesMode::ATOM,
      xtal::allowed_molecule_names(prim));
  double lattice_cost_weight = 0.0;  // atom mapping cost only
  double _max_volume_change = 0.5;   // no effect
  bool _robust = true;               // no effect if k_best > 1
  bool _soft_va_limit = false;       // no effect
  double _min_va_frac = 0.;          // no effect
  double _max_va_frac = 1.;          // no effect
  mapping_impl::StrucMapper strucmap(
      calculator, lattice_cost_weight, _max_volume_change, _robust,
      _soft_va_limit, cost_tol, _min_va_frac, _max_va_frac);

  if (symmetrize_atom_cost) {
    auto prim_permute_group =
        xtal::make_permutation_representation(prim, prim_factor_group);
    strucmap.set_symmetrize_atomic_cost(true, prim_factor_group,
                                        prim_permute_group);
  }

  bool keep_invalid = false;

  // Convert LatticeMapping -> LatticeNode
  mapping_impl::LatticeNode imposed_node =
      mapping_impl::make_lattice_node(prim.lattice(), lattice_mapping);

  std::set<mapping_impl::MappingNode> mappings =
      strucmap.map_deformed_struc_impose_lattice_node(
          structure2, imposed_node, k_best, max_cost, min_cost, keep_invalid);

  AtomMappingResults results;
  for (auto const &mapping_node : mappings) {
    if (!(mapping_node.cost > (min_cost - cost_tol) &&
          mapping_node.cost < (max_cost + cost_tol))) {
      continue;
    }
    results.data.emplace_back(
        mapping_node.atomic_node.cost,
        mapping_impl::make_atom_mapping(lattice_mapping.deformation_gradient,
                                        mapping_node));
  }

  return results;
}

}  // namespace mapping
}  // namespace CASM
