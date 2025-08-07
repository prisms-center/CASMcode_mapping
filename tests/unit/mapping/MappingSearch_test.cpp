#include "casm/mapping/MappingSearch.hh"

#include "SearchTestData.hh"
#include "casm/crystallography/SimpleStructure.hh"
#include "casm/crystallography/StrainConverter.hh"
#include "casm/crystallography/io/SimpleStructureIO.hh"
#include "casm/mapping/io/json_io.hh"
#include "casm/mapping/lattice_cost.hh"
#include "casm/misc/CASM_Eigen_math.hh"
#include "gtest/gtest.h"
#include "teststructures.hh"

// debug
#include "casm/casm_io/container/json_io.hh"
#include "casm/casm_io/container/stream_io.hh"
#include "casm/casm_io/json/jsonParser.hh"

using namespace CASM;
using namespace CASM::xtal;
using namespace CASM::mapping;

namespace test {

jsonParser &_to_json(std::pair<Index, Index> const &value, jsonParser &json) {
  json.put_array();
  json.push_back(value.first);
  json.push_back(value.second);
  return json;
}

jsonParser &_to_json(std::map<Index, Index> const &value, jsonParser &json) {
  json.put_array();
  for (auto const &pair : value) {
    jsonParser tjson;
    json.push_back(_to_json(pair, tjson));
  }
  return json;
}

jsonParser &_to_json(std::vector<std::pair<Index, Index>> const &value,
                     jsonParser &json) {
  json.put_array();
  for (auto const &pair : value) {
    jsonParser tjson;
    json.push_back(_to_json(pair, tjson));
  }
  return json;
}

// Example:
// jsonParser json;
// test::_to_json(search.front().assignment_node, json);
// std::cout << json << std::endl;
jsonParser &_to_json(murty::Node const &assignment_node, jsonParser &json) {
  _to_json(assignment_node.forced_on, json["forced_on"]);
  _to_json(assignment_node.forced_off, json["forced_off"]);
  to_json(assignment_node.unassigned_rows, json["unassigned_rows"]);
  to_json(assignment_node.unassigned_cols, json["unassigned_cols"]);
  _to_json(assignment_node.sub_assignment, json["sub_assignment"]);
  to_json(assignment_node.cost, json["cost"]);
  return json;
}

}  // namespace test

// Test perfect BCC mapping to BCC
TEST(MappingSearchTest, Test1) {
  // Make a perfect BCC structure
  double latparam_a = 4.0;

  // F * L1 * T * N = L2
  Eigen::Matrix3d F = Eigen::Matrix3d::Identity();
  Eigen::Matrix3d T = Eigen::Matrix3d::Identity();
  Eigen::Matrix3d N = Eigen::Matrix3d::Identity();

  // F (r1_supercell[i] + disp) = r2[perm[i]] + trans
  // structure1_supercell_atom_type[i] = structure2_atom_type[perm[i]]
  Eigen::MatrixXd disp;
  disp.resize(3, 1);
  disp.col(0) << 0., 0., 0.;
  std::vector<std::string> structure1_supercell_atom_type({"A"});
  std::vector<Index> perm({0});
  Eigen::Vector3d trans(0., 0., 0.);

  test::SearchTestData d(test::make_search_prim_binary_BCC(latparam_a), F, T, N,
                         disp, structure1_supercell_atom_type, perm, trans);

  auto lattice_mapping_data = std::make_shared<LatticeMappingSearchData const>(
      d.prim_data, d.structure_data, d.lattice_mapping);
  EXPECT_EQ(d.prim_data->prim_factor_group.size(), 48);
  EXPECT_EQ(d.structure_data->structure_factor_group.size(), 48);

  // MappingSearch parameters
  double min_cost = 0.0;
  double max_cost = 1e20;
  int k_best = 1;
  AtomCostFunction atom_cost_f = IsotropicAtomCost();
  double lattice_cost_weight = 0.5;
  TotalCostFunction total_cost_f = WeightedTotalCost(lattice_cost_weight);
  AtomToSiteCostFunction atom_to_site_cost_f = make_atom_to_site_cost_future;
  bool enable_remove_mean_displacement = true;
  double infinity = 1e20;
  double cost_tol = 1e-5;

  MappingSearch search(min_cost, max_cost, k_best, atom_cost_f, total_cost_f,
                       atom_to_site_cost_f, enable_remove_mean_displacement,
                       infinity, cost_tol);

  // Make and insert MappingNode, given:
  double lattice_cost = isotropic_strain_cost(F);
  Eigen::Vector3d trial_translation_cart(0., 0., 0.);
  std::map<Index, Index> forced_on({});
  std::vector<std::pair<Index, Index>> forced_off({});
  search.make_and_insert_mapping_node(lattice_cost, lattice_mapping_data,
                                      trial_translation_cart, forced_on,
                                      forced_off);
  EXPECT_EQ(search.size(), 1);
  EXPECT_EQ(search.results.size(), 1);
  EXPECT_EQ(search.overflow.size(), 0);
  EXPECT_TRUE(almost_equal(search.front().total_cost, 0.))
      << "search.front().total_cost=" << search.front().total_cost;

  // No sub-assignments should be possible
  auto subnode_ptrs = search.partition();
  EXPECT_EQ(subnode_ptrs.size(), 0);
  EXPECT_EQ(search.size(), 0);
  EXPECT_EQ(search.results.size(), 1);
  EXPECT_EQ(search.overflow.size(), 0);

  // Nothing should happen if queue is empty
  subnode_ptrs = search.partition();
  EXPECT_EQ(subnode_ptrs.size(), 0);
  EXPECT_EQ(search.size(), 0);
  EXPECT_EQ(search.results.size(), 1);
  EXPECT_EQ(search.overflow.size(), 0);
}

// Test rotated, strained BCC mapping to BCC
TEST(MappingSearchTest, Test2) {
  // Make a perfect BCC structure
  double latparam_a = 4.0;

  // F * L1 * T * N = L2
  Eigen::Matrix3d Q;
  Q = Eigen::AngleAxis<double>(M_PI / 8., Eigen::Vector3d::UnitZ());
  Eigen::Matrix3d U;
  U << 1., 0., 0.,     //
      0., 1., 0.01,    //
      0., 0.01, 1.01;  //
  Eigen::Matrix3d F = Q * U;
  Eigen::Matrix3d T = Eigen::Matrix3d::Identity();
  Eigen::Matrix3d N = Eigen::Matrix3d::Identity();

  // F (r1_supercell[i] + disp) = r2[perm[i]] + trans
  // structure1_supercell_atom_type[i] = structure2_atom_type[perm[i]]
  Eigen::MatrixXd disp;
  disp.resize(3, 1);
  disp.col(0) << 0., 0., 0.;
  std::vector<std::string> structure1_supercell_atom_type({"A"});
  std::vector<Index> perm({0});
  Eigen::Vector3d trans(0., 0., 0.);

  test::SearchTestData d(test::make_search_prim_binary_BCC(latparam_a), F, T, N,
                         disp, structure1_supercell_atom_type, perm, trans);

  auto lattice_mapping_data = std::make_shared<LatticeMappingSearchData const>(
      d.prim_data, d.structure_data, d.lattice_mapping);
  EXPECT_EQ(d.prim_data->prim_factor_group.size(), 48);
  EXPECT_EQ(d.structure_data->structure_factor_group.size(), 4);

  // MappingSearch parameters
  double min_cost = 0.0;
  double max_cost = 1e20;
  int k_best = 1;
  AtomCostFunction atom_cost_f = IsotropicAtomCost();
  double lattice_cost_weight = 0.5;
  TotalCostFunction total_cost_f = WeightedTotalCost(lattice_cost_weight);
  AtomToSiteCostFunction atom_to_site_cost_f = make_atom_to_site_cost_future;
  bool enable_remove_mean_displacement = true;
  double infinity = 1e20;
  double cost_tol = 1e-5;

  MappingSearch search(min_cost, max_cost, k_best, atom_cost_f, total_cost_f,
                       atom_to_site_cost_f, enable_remove_mean_displacement,
                       infinity, cost_tol);

  // Make and insert MappingNode, given:
  double lattice_cost = isotropic_strain_cost(F);
  Eigen::Vector3d trial_translation_cart(0., 0., 0.);
  std::map<Index, Index> forced_on({});
  std::vector<std::pair<Index, Index>> forced_off({});
  search.make_and_insert_mapping_node(lattice_cost, lattice_mapping_data,
                                      trial_translation_cart, forced_on,
                                      forced_off);
  EXPECT_EQ(search.size(), 1);
  EXPECT_EQ(search.results.size(), 1);
  EXPECT_EQ(search.overflow.size(), 0);
  EXPECT_TRUE(almost_equal(search.front().lattice_cost, lattice_cost));
  EXPECT_TRUE(almost_equal(search.front().atom_cost, 0.));
  EXPECT_TRUE(almost_equal(search.front().total_cost,
                           lattice_cost * lattice_cost_weight));

  // No sub-assignments should be possible
  auto subnode_ptrs = search.partition();
  EXPECT_EQ(subnode_ptrs.size(), 0);
  EXPECT_EQ(search.size(), 0);
  EXPECT_EQ(search.results.size(), 1);
  EXPECT_EQ(search.overflow.size(), 0);

  // Nothing should happen if queue is empty
  subnode_ptrs = search.partition();
  EXPECT_EQ(subnode_ptrs.size(), 0);
  EXPECT_EQ(search.size(), 0);
  EXPECT_EQ(search.results.size(), 1);
  EXPECT_EQ(search.overflow.size(), 0);
}

// Test permuted BCC mapping to BCC
TEST(MappingSearchTest, Test3) {
  // Make a perfect BCC structure
  double latparam_a = 4.0;

  // F * L1 * T * N = L2
  Eigen::Matrix3d F = Eigen::Matrix3d::Identity();
  Eigen::Matrix3d T;
  T << 1., 0., 0.,  //
      0., 1., 0.,   //
      0., 0., 2.;   //
  Eigen::Matrix3d N = Eigen::Matrix3d::Identity();

  // F (r1_supercell[i] + disp) = r2[perm[i]] + trans
  // structure1_supercell_atom_type[i] = structure2_atom_type[perm[i]]
  Eigen::MatrixXd disp;
  disp.resize(3, 2);
  disp.col(0) << 0., 0., 0.;
  disp.col(1) << 0., 0., 0.;
  std::vector<std::string> structure1_supercell_atom_type({"A", "A"});
  std::vector<Index> perm({1, 0});  // <--- permute sites
  Eigen::Vector3d trans(0., 0., 0.);

  test::SearchTestData d(test::make_search_prim_binary_BCC(latparam_a), F, T, N,
                         disp, structure1_supercell_atom_type, perm, trans);

  auto lattice_mapping_data = std::make_shared<LatticeMappingSearchData const>(
      d.prim_data, d.structure_data, d.lattice_mapping);
  EXPECT_EQ(d.prim_data->prim_factor_group.size(), 48);
  EXPECT_EQ(d.structure_data->structure_factor_group.size(), 16);

  // MappingSearch parameters
  double min_cost = 0.0;
  double max_cost = 1e20;
  int k_best = 1;
  AtomCostFunction atom_cost_f = IsotropicAtomCost();
  double lattice_cost_weight = 0.5;
  TotalCostFunction total_cost_f = WeightedTotalCost(lattice_cost_weight);
  AtomToSiteCostFunction atom_to_site_cost_f = make_atom_to_site_cost_future;
  bool enable_remove_mean_displacement = true;
  double infinity = 1e20;
  double cost_tol = 1e-5;

  MappingSearch search(min_cost, max_cost, k_best, atom_cost_f, total_cost_f,
                       atom_to_site_cost_f, enable_remove_mean_displacement,
                       infinity, cost_tol);

  // Make and insert MappingNode, given:
  double lattice_cost = isotropic_strain_cost(F);
  Eigen::Vector3d trial_translation_cart(0., 0., 0.);
  std::map<Index, Index> forced_on({});
  std::vector<std::pair<Index, Index>> forced_off({});
  search.make_and_insert_mapping_node(lattice_cost, lattice_mapping_data,
                                      trial_translation_cart, forced_on,
                                      forced_off);
  EXPECT_EQ(search.size(), 1);
  EXPECT_EQ(search.results.size(), 1);
  EXPECT_EQ(search.overflow.size(), 0);
  EXPECT_TRUE(almost_equal(search.front().lattice_cost, 0.));
  EXPECT_TRUE(almost_equal(search.front().atom_cost, 0.));
  EXPECT_TRUE(almost_equal(search.front().total_cost, 0.));

  // check permutation
  EXPECT_EQ(search.front().atom_mapping.permutation, perm);
}

// Test rotated, strained BCC with vacancy mapping to BCC
// 7 atoms instead of 8
TEST(MappingSearchTest, Test4) {
  // Make a perfect BCC structure
  double latparam_a = 4.0;

  // F * L1 * T * N = L2
  Eigen::Matrix3d Q;
  // Q = Eigen::Matrix3d::Identity();
  Q = Eigen::AngleAxis<double>(M_PI / 8., Eigen::Vector3d::UnitZ());
  Eigen::Matrix3d U;
  U << 1.01, 0., 0.,  //
      0., 1., 0.,     //
      0., 0., 1.;     //
  Eigen::Matrix3d F = Q * U;
  Eigen::Matrix3d T = Eigen::Matrix3d::Identity() * 2;
  Eigen::Matrix3d N = Eigen::Matrix3d::Identity();

  // F (r1_supercell[i] + disp) = r2[perm[i]] + trans
  // structure1_supercell_atom_type[i] = structure2_atom_type[perm[i]]
  Eigen::MatrixXd disp = Eigen::MatrixXd::Zero(3, 7);
  disp.col(0) << 0.01, -0.01, 0.01;
  disp.col(1) << 0.00, 0.01, -0.01;
  disp.col(2) << 0.01, 0.00, -0.01;
  disp.col(3) << -0.01, 0.01, 0.0;
  disp.col(4) << -0.01, 0.00, 0.01;
  disp.col(5) << 0.0, 0.00, -0.01;
  disp.col(6) << 0.01, 0.00, 0.0;
  std::vector<Index> perm;
  std::vector<std::string> structure1_supercell_atom_type;
  for (Index i = 0; i < 7; ++i) {
    perm.push_back(i);
    structure1_supercell_atom_type.push_back("A");
  }
  Eigen::Vector3d trans(0., 0., 0.);

  test::SearchTestData d(test::make_search_prim_binary_vacancy_BCC(latparam_a),
                         F, T, N, disp, structure1_supercell_atom_type, perm,
                         trans);

  auto lattice_mapping_data = std::make_shared<LatticeMappingSearchData const>(
      d.prim_data, d.structure_data, d.lattice_mapping);
  EXPECT_EQ(d.prim_data->prim_factor_group.size(), 48);
  EXPECT_EQ(d.structure_data->structure_factor_group.size(), 1);

  // MappingSearch parameters
  double min_cost = 0.0;
  double max_cost = 1e20;
  int k_best = 1;
  AtomCostFunction atom_cost_f = IsotropicAtomCost();
  double lattice_cost_weight = 0.5;
  TotalCostFunction total_cost_f = WeightedTotalCost(lattice_cost_weight);
  AtomToSiteCostFunction atom_to_site_cost_f = make_atom_to_site_cost_future;
  bool enable_remove_mean_displacement = true;
  double infinity = 1e20;
  double cost_tol = 1e-5;

  MappingSearch search(min_cost, max_cost, k_best, atom_cost_f, total_cost_f,
                       atom_to_site_cost_f, enable_remove_mean_displacement,
                       infinity, cost_tol);

  // Make and insert MappingNode, given:
  double lattice_cost = isotropic_strain_cost(F);
  Eigen::Vector3d trial_translation_cart(0., 0., 0.);
  std::map<Index, Index> forced_on({});
  std::vector<std::pair<Index, Index>> forced_off({});
  search.make_and_insert_mapping_node(lattice_cost, lattice_mapping_data,
                                      trial_translation_cart, forced_on,
                                      forced_off);
  {
    EXPECT_EQ(search.size(), 1);
    EXPECT_EQ(search.results.size(), 1);
    EXPECT_EQ(search.overflow.size(), 0);

    // -- mapping_node --
    MappingNode const &mapping_node = search.front();
    EXPECT_TRUE(almost_equal(mapping_node.lattice_cost, lattice_cost));
    EXPECT_TRUE(!almost_equal(mapping_node.atom_cost, 0.0));
    EXPECT_TRUE(
        almost_equal(mapping_node.total_cost,
                     lattice_cost * lattice_cost_weight +
                         mapping_node.atom_cost * (1.0 - lattice_cost_weight)));

    // -- structure_data --
    StructureSearchData const &structure_data =
        *mapping_node.lattice_mapping_data->structure_data;
    EXPECT_EQ(structure_data.N_atom, 7);
    EXPECT_EQ(structure_data.atom_coordinate_cart.cols(), 7);
    EXPECT_EQ(structure_data.atom_type.size(), 7);

    // -- lattice_mapping_data --
    LatticeMappingSearchData const &lattice_mapping_data =
        *mapping_node.lattice_mapping_data;
    EXPECT_EQ(lattice_mapping_data.N_supercell_site, 8);
    EXPECT_EQ(lattice_mapping_data.atom_coordinate_cart_in_supercell.cols(), 7);
    EXPECT_EQ(lattice_mapping_data.supercell_site_coordinate_cart.cols(), 8);
    EXPECT_EQ(lattice_mapping_data.supercell_allowed_atom_types.size(), 8);

    // -- atom_mapping_data --
    AtomMappingSearchData const &atom_mapping_data =
        *mapping_node.atom_mapping_data;
    EXPECT_EQ(atom_mapping_data.site_displacements.size(), 8);
    for (auto const &disp : atom_mapping_data.site_displacements) {
      EXPECT_EQ(disp.size(), 7);
    }
    EXPECT_EQ(atom_mapping_data.cost_matrix.rows(), 8);
    EXPECT_EQ(atom_mapping_data.cost_matrix.cols(), 8);

    // -- results --
    auto results = combined_results(search);
    EXPECT_EQ(results.size(), 1);

    auto const &lattice_mapping = results.begin()->lattice_mapping;
    auto const &atom_mapping = results.begin()->atom_mapping;
    EXPECT_EQ(atom_mapping.displacement.cols(), 8);
    EXPECT_EQ(atom_mapping.permutation.size(), 8);
  }

  // test making mapped structure from unmapped structure and mapping results
  {
    jsonParser json;
    xtal::SimpleStructure unmapped_structure =
        make_structure(*d.structure_data);

    std::set<std::string> excluded_species = {};  // do not exclude vacancies
    to_json(unmapped_structure, json["unmapped_structure"], excluded_species,
            CART);

    auto results = combined_results(search);
    ScoredStructureMapping const &structure_mapping = *results.begin();
    to_json(structure_mapping, json["mapping"]);

    xtal::SimpleStructure mapped_structure =
        make_mapped_structure(structure_mapping, unmapped_structure);

    EXPECT_EQ(mapped_structure.atom_info.names.size(), 8);
    EXPECT_EQ(
        mapped_structure.atom_info.names,
        std::vector<std::string>({"A", "A", "A", "A", "A", "A", "A", "Va"}));

    xtal::StrainConverter strain_converter("Ustrain",
                                           Eigen::MatrixXd::Identity(6, 6));
    Eigen::VectorXd U_unrolled = strain_converter.from_E_matrix(U);
    EXPECT_TRUE(mapped_structure.properties.count("Ustrain"));
    EXPECT_TRUE(
        almost_equal(mapped_structure.properties.at("Ustrain"), U_unrolled));

    to_json(mapped_structure, json["mapped_structure"], excluded_species, CART);
    std::cout << json << std::endl;
  }
}
