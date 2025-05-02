#include "casm/mapping/StructureMapping.hh"

#include "casm/casm_io/Log.hh"
#include "casm/crystallography/AnisoValTraits.hh"
#include "casm/crystallography/BasicStructure.hh"
#include "casm/crystallography/LinearIndexConverter.hh"
#include "casm/crystallography/SimpleStructure.hh"
#include "casm/crystallography/SimpleStructureTools.hh"
#include "casm/crystallography/Superlattice.hh"
#include "casm/crystallography/SymTools.hh"
#include "casm/crystallography/UnitCellCoordRep.hh"
#include "casm/mapping/LatticeMapping.hh"
#include "casm/misc/CASM_Eigen_math.hh"

// debug
#include "casm/casm_io/container/stream_io.hh"

namespace CASM {
namespace mapping {

/// \brief Return mappings that result in structures along the
///     transformation pathway from the parent to the aligned child
///     structure
///
/// \param structure_mapping Original structure mapping
/// \param interpolation_factor Interpolation factor. The value 0.0
///     corresponds to the child structure mapped to the ideal parent
///     lattice and sites; and the value 1.0 corresponds to the child
///     structure, transformed by isometry and translation to align
///     with the ideal parent structure.
StructureMapping interpolated_mapping(StructureMapping const &structure_mapping,
                                      double interpolation_factor) {
  double f = interpolation_factor;
  return StructureMapping(
      structure_mapping.shared_prim,
      interpolated_mapping(structure_mapping.lattice_mapping, f),
      interpolated_mapping(structure_mapping.atom_mapping, f));
}

/// \brief Return the mapped structure, with implied vacancies,
///     strain, and atomic displacement
///
/// The "mapped structure" lattice and site coordinates are constructed
/// from the parent structure by applying the lattice and atom mappings
/// without isometry. Atom names and atom properties are
/// determined from the unmapped structure, permutation, and
/// inverse isometry. Global properties are determined
/// from the unmapped global properties and inverse isometry. Strain,
/// using the right stretch tensor as the strain metric, is stored as
/// the global property "Ustrain". Displacement is stored as
/// atom properties.
///
/// Notes:
/// - This method is only implemented for atomic structures, not
///   molecular structures
/// - Implicit vacancies are added as "Va", with value 0.0 any atom
//    properties present in the unmapped_structure (i.e. force)
/// - Throws if unmapped_structure already has a strain or disp
///   property
///
/// \returns mapped_structure, including vacancies
///
xtal::SimpleStructure make_mapped_structure(
    StructureMapping const &structure_mapping,
    xtal::SimpleStructure const &unmapped_structure) {
  if (unmapped_structure.mol_info.size()) {
    throw std::runtime_error(
        "Error: CASM::mapping::make_mapped_structure is only implemented for "
        "atomic structures");
  }
  if (unmapped_structure.atom_info.coords.cols() !=
      unmapped_structure.atom_info.names.size()) {
    throw std::runtime_error(
        "Error in CASM::mapping::make_mapped_structure: unmapped_structure has "
        "inconsistent atom coords and names");
  }
  for (auto const &value : unmapped_structure.properties) {
    if (value.first.find("strain") != std::string::npos) {
      throw std::runtime_error(
          "Error in CASM::mapping::make_mapped_structure: unmapped_structure "
          "already has a strain property");
    }
  }
  for (auto const &value : unmapped_structure.atom_info.properties) {
    if (AnisoValTraits::name_suffix(value.first) == "disp") {
      throw std::runtime_error(
          "Error in CASM::mapping::make_mapped_structure: unmapped_structure "
          "already has a disp property");
    }
  }

  xtal::BasicStructure const &prim = *structure_mapping.shared_prim;

  // make the parent structure from the prim for the ideal lattice
  // and site coordinates - occupants are set to "Va"
  xtal::SimpleStructure parent_structure;
  parent_structure.lat_column_mat = prim.lattice().lat_column_mat();
  parent_structure.atom_info.resize(prim.basis().size());
  int b = 0;
  for (auto const &site : prim.basis()) {
    parent_structure.atom_info.coords.col(b) = site.const_cart();
    ++b;
  }

  xtal::SimpleStructure mapped_structure;

  // make mapped lattice
  auto const &lattice_mapping = structure_mapping.lattice_mapping;
  auto const &Q = lattice_mapping.isometry;
  auto const &U = lattice_mapping.right_stretch;
  auto const &L1 = parent_structure.lat_column_mat;
  auto const &L2 = unmapped_structure.lat_column_mat;
  auto const &T = lattice_mapping.transformation_matrix_to_super;
  auto const &N = lattice_mapping.reorientation;

  mapped_structure.lat_column_mat = U * L1 * T * N;

  // atomic coordinate relations:
  //
  ///    F \left(\vec{r_1}(i) + \vec{d}(i) \right) = \vec{r_2}(p_i) + \vec{t}
  //     U \left(\vec{r_1}(i) + \vec{d}(i) \right) = \vec{r_3}(i)
  //
  //     F: lattice_mapping.deformation_gradient
  //     U: lattice_mapping.right_stretch
  //     \vec{r_1}(i): parent_superstructure.atom_info.coords.col(i)
  //     \vec{d}(i): atom_mapping.displacement.col(i)
  //     \vec{r_2}(p_i): unmapped_structure.atom_info.coords.col(p_i)
  //     \vec{r_3}(i): mapped_struture.atom_info.coords.col(i)
  //     p_i: atom_mapping.permutation[i]
  //     \vec{t}: atom_mapping.translation
  //
  //     remember: due to mapping within periodic boundaries, coordinates
  //     comparisons must be made checking for equality up to a lattice
  //     translation.

  xtal::SimpleStructure parent_superstructure =
      make_superstructure(lround(T * N).cast<int>(), parent_structure);

  // Note, n_atoms includes implied vacancies
  auto const &atom_mapping = structure_mapping.atom_mapping;
  Index n_sites = parent_superstructure.atom_info.size();
  Index n_atoms = unmapped_structure.atom_info.size();
  mapped_structure.atom_info.resize(n_sites);
  mapped_structure.atom_info.coords =
      U * (parent_superstructure.atom_info.coords + atom_mapping.displacement);

  // atomic property relations (includes atom type/name):
  //
  //     q_3(i) = M(Q^{-1}) q_2(p_i)
  //
  //     q_2(p_i): Property at site p_i of the unmapped structure
  //     q_3(i): Property at site i of the mapped structure
  //     Q: lattice_mapping.isometry
  //     M(Q^{-1}): representation matrix for transforming property q
  //                according to inverse of lattice_mapping.isometry
  //

  // Q.transpose() == Q^{-1}
  xtal::SymOp Q_inv = xtal::SymOp::point_operation(Q.transpose());

  // Create transformed atom properties, with columns of zeros for implied
  // vacancies
  std::map<std::string, Eigen::MatrixXd> transformed_atom_properties;
  for (auto const &pair : unmapped_structure.atom_info.properties) {
    std::string key = pair.first;
    Eigen::MatrixXd const &q2 = pair.second;
    Eigen::MatrixXd q2_with_Va = Eigen::MatrixXd::Zero(q2.rows(), n_sites);
    q2_with_Va.block(0, 0, q2.rows(), q2.cols()) = q2;
    try {
      AnisoValTraits traits(key);
      Eigen::MatrixXd M = traits.symop_to_matrix(
          get_matrix(Q_inv), get_translation(Q_inv), get_time_reversal(Q_inv));
      transformed_atom_properties.emplace(key, M * q2_with_Va);
    } catch (std::exception &e) {
      std::stringstream msg;
      msg << "CASM does not know how to transform the local property '" << key
          << "'. The property name suffix must be the name of a local property "
             "that CASM can transform.";
      CASM::err_log().paragraph(msg.str());
      throw std::runtime_error(
          std::string("Cannot transform local property '") + key + "'");
    }
  }

  // Set zeroed atom properties
  for (auto const &pair : transformed_atom_properties) {
    mapped_structure.atom_info.properties.emplace(pair.first,
                                                  0.0 * pair.second);
  }

  // Set mapped atom names and properties
  for (Index i = 0; i < n_sites; ++i) {
    Index p_i = atom_mapping.permutation[i];
    if (p_i >= n_atoms) {
      // implied vacancy - has name "Va"
      mapped_structure.atom_info.names[i] = "Va";
    } else {
      // atoms or explicit vacancies
      mapped_structure.atom_info.names[i] =
          unmapped_structure.atom_info.names[p_i];
    }

    for (auto const &pair : transformed_atom_properties) {
      mapped_structure.atom_info.properties.at(pair.first).col(i) =
          pair.second.col(p_i);
    }
  }

  // Add displacement
  mapped_structure.atom_info.properties.emplace("disp",
                                                atom_mapping.displacement);

  // global property relations:
  //
  //     q_3 = M(Q^{-1}) q_2
  //
  //     q_2: Global property of the unmapped structure
  //     q_3: Global property of the mapped structure
  //     Q: lattice_mapping.isometry
  //     M(Q^{-1}): representation matrix for transforming property q
  //                according to inverse of lattice_mapping.isometry
  //

  // Set mapped global properties
  for (auto const &pair : unmapped_structure.properties) {
    std::string key = pair.first;
    Eigen::MatrixXd const &q2 = pair.second;
    try {
      AnisoValTraits traits(key);
      Eigen::MatrixXd M = traits.symop_to_matrix(
          get_matrix(Q_inv), get_translation(Q_inv), get_time_reversal(Q_inv));
      mapped_structure.properties.emplace(key, M * q2);
    } catch (std::exception &e) {
      std::stringstream msg;
      msg << "CASM does not know how to transform the global property '" << key
          << "'. The property name suffix must be the name of a global "
             "property that CASM can transform.";
      CASM::err_log().paragraph(msg.str());
      throw std::runtime_error(
          std::string("Cannot transform global property '") + key + "'");
    }
  }

  // Add Ustrain (vector in standard basis)
  Eigen::Matrix3d const &e = U;
  Eigen::VectorXd E_vector = Eigen::VectorXd::Zero(6);
  double w = std::sqrt(2.);
  E_vector << e(0, 0), e(1, 1), e(2, 2), w * e(1, 2), w * e(0, 2), w * e(0, 1);
  mapped_structure.properties.emplace("Ustrain", E_vector);

  return mapped_structure;
}

/// \brief Make the structure mapping to a different, but equivalent
///     structure
///
/// \param op A symmetry operation that generates the equivalent mapped
///     structure from the current mapped structure.
/// \param target The superlattice of the equivalent mapped structure.
/// \param structure_mapping A structure mapping.
///
/// \returns A structure mapping to an equivalent, but transformed,
///     mapped structure.
///
/// The following result in equivalent `transformed_mapped` structures
/// (assuming `target` is an equivalent superlattice and `op` is a prim
/// factor group operation):
/// \code
/// xtal::SimpleStructure unmapped;
/// xtal::SimpleStructure mapped = make_mapped_structure(
///     structure_mapping, unmapped);
/// xtal::SimpleStructure transformed_mapped =
///     copy_apply(op, mapped);
/// transformed_mapped.lat_column_mat = target.lat_column_mat();
/// transformed_mapped.within();
/// \endcode
///
/// \code
/// xtal::SimpleStructure unmapped;
/// StructureMapping transformed_structure_mapping =
///     make_mapping_to_equivalent_structure(op, target, structure_mapping);
/// xtal::SimpleStructure transformed_mapped =
///     make_mapped_structure(transformed_structure_mapping, unmapped);
/// \endcode
///
StructureMapping make_mapping_to_equivalent_structure(
    xtal::SymOp const &op, xtal::Lattice const &target,
    StructureMapping const &structure_mapping) {
  auto const &smap = structure_mapping;

  auto const &lmap = smap.lattice_mapping;
  auto const &amap = smap.atom_mapping;
  auto const &prim = *smap.shared_prim;
  Index n_basis = prim.basis().size();

  // note: '=' indicates "is equivalent to", not necessarily exact equality
  // F_a * L1 * T * N = L2
  // F_a * R.inv * R * L1 * T * N = L2
  // F_a * (r1_a[i] + disp_a[i]) = r2[perm_a[i]] + trans_a
  // F_b * (r1_b[j] + disp_b[j]) = r2[perm_b[j]] + trans_b

  // apply R=op.matrix to rotate mapped structure and determine new mapping:
  // L1 * T_b * N_b = R * L1 * T_a * N_a
  // use: T_b = L1.inv * R * L1 * T_a * N_a, N_b = I

  // F_a * R.inv * R * (r1_a[i] + disp_a[i]) = r2[perm_a[i]] + trans_a
  // use: F_b = F_a * R.inv; disp_b = R * disp_a, trans_b = trans_a,
  // but need to determine perm_b. To do so, make new superlattice, L1 * T_b,
  // then apply op to sites l_a to find transformed sites, l_b

  // the input mapping, 'a'
  Eigen::Matrix3d L1 = prim.lattice().lat_column_mat();
  Eigen::Matrix3d F_a = lmap.deformation_gradient;
  Eigen::Matrix3d T_a = lmap.transformation_matrix_to_super;
  Eigen::Matrix3d N_a = lmap.reorientation;

  xtal::Superlattice superlattice_a(prim.lattice(),
                                    xtal::Lattice(L1 * T_a * N_a));
  xtal::UnitCellCoordIndexConverter converter_a(
      superlattice_a.transformation_matrix_to_super(), n_basis);
  Eigen::MatrixXd const &disp_a = amap.displacement;
  std::vector<Index> const &perm_a = amap.permutation;
  Eigen::Vector3d const &trans_a = amap.translation;

  // the equivalent mapping, 'b'
  Eigen::Matrix3d F_b = F_a * op.matrix.transpose();
  // Eigen::Matrix3l T_b = L1.inverse() * op.matrix * L1 * T_a * N_a;
  // xtal::Superlattice superlattice_b (prim.lattice(), xtal::Lattice(L1 * T_b *
  // N_b));
  xtal::Superlattice superlattice_b(prim.lattice(), target);
  Eigen::Matrix3l T_b = superlattice_b.transformation_matrix_to_super();
  Eigen::Matrix3l N_b = Eigen::Matrix3l::Identity();

  // determine the equivalent disp_b and perm_b
  xtal::UnitCellCoordIndexConverter converter_b(T_b, n_basis);
  Eigen::MatrixXd disp_b(3, converter_b.total_sites());
  std::vector<Index> perm_b(converter_b.total_sites(), -1);
  Eigen::Vector3d trans_b = trans_a;

  /// Make the UnitCellCoordRep for op
  xtal::UnitCellCoordRep rep =
      make_unitcellcoord_rep(op, prim.lattice(), symop_site_map(op, prim));

  for (Index l_a = 0; l_a < converter_a.total_sites(); ++l_a) {
    xtal::UnitCellCoord site = converter_a(l_a);
    xtal::UnitCellCoord transformed_site = copy_apply(rep, site);
    Index l_b = converter_b(transformed_site);
    disp_b.col(l_b) = op.matrix * disp_a.col(l_a);
    perm_b[l_b] = perm_a[l_a];
  }

  return StructureMapping(
      smap.shared_prim,
      LatticeMapping(F_b, T_b.cast<double>(), N_b.cast<double>()),
      AtomMapping(disp_b, perm_b, trans_b));
}

/// \brief Make an equivalent mapping to a different, but symmetrically
///     equivalent superlattice of the prim
///
/// \param target The superlattice of the prim that the equivalent
///     mapping maps to.
/// \param structure_mapping A structure mapping
/// \param group The prim factor group (or a subset) used to generate
///     equivalent superlattices
///
/// \returns An equivalent structure mapping, for which the mapped
///     structure (excluding strain) has the target superlattice of
///     the prim.
StructureMapping make_mapping_to_equivalent_superlattice(
    xtal::Lattice const &target, StructureMapping const &structure_mapping,
    std::vector<xtal::SymOp> const &group) {
  auto const &smap = structure_mapping;

  Eigen::Matrix3d L1 = smap.shared_prim->lattice().lat_column_mat();
  Eigen::Matrix3d T_a = smap.lattice_mapping.transformation_matrix_to_super;
  Eigen::Matrix3d N_a = smap.lattice_mapping.reorientation;
  xtal::Lattice current(L1 * T_a * N_a);

  auto res = is_equivalent_superlattice(target, current, group.begin(),
                                        group.end(), target.tol());

  return make_mapping_to_equivalent_structure(*res.first, target, smap);
}

}  // namespace mapping
}  // namespace CASM
