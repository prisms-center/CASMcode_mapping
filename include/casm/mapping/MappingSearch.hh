#ifndef CASM_mapping_MappingSearch
#define CASM_mapping_MappingSearch

#include <map>
#include <memory>
#include <optional>
#include <set>

#include "casm/mapping/AtomMapping.hh"
#include "casm/mapping/LatticeMapping.hh"
#include "casm/mapping/SearchData.hh"
#include "casm/mapping/StructureMapping.hh"
#include "casm/mapping/murty.hh"

namespace CASM {
namespace mapping {

// --- Atom cost calculation ---

/// \brief Function type for calculating atom mapping cost
using AtomCostFunction =
    std::function<double(LatticeMappingSearchData const &lattice_mapping_data,
                         AtomMappingSearchData const &atom_mapping_data,
                         AtomMapping const &atom_mapping)>;

/// \brief Functor for isotropic atom cost
struct IsotropicAtomCost {
  double operator()(LatticeMappingSearchData const &lattice_mapping_data,
                    AtomMappingSearchData const &atom_mapping_data,
                    AtomMapping const &atom_mapping) const;
};

/// \brief Functor for symmetry breaking atom cost
struct SymmetryBreakingAtomCost {
  double operator()(LatticeMappingSearchData const &lattice_mapping_data,
                    AtomMappingSearchData const &atom_mapping_data,
                    AtomMapping const &atom_mapping) const;
};

// --- Total cost calculation ---

/// \brief Function type for calculating total mapping cost
using TotalCostFunction = std::function<double(
    double lattice_cost, LatticeMappingSearchData const &lattice_mapping_data,
    double atom_cost, AtomMappingSearchData const &atom_mapping_data,
    AtomMapping const &atom_mapping)>;

struct WeightedTotalCost {
  WeightedTotalCost(double _lattice_cost_weight);

  double lattice_cost_weight;

  double operator()(double lattice_cost,
                    LatticeMappingSearchData const &lattice_mapping_data,
                    double atom_cost,
                    AtomMappingSearchData const &atom_mapping_data,
                    AtomMapping const &atom_mapping) const;
};

// --- MappingSearch queue management ---

struct MappingSearch;

/// \brief Functor for enforcing MappingSearch queue constraints
struct QueueConstraints {
  /// \brief Constructor
  QueueConstraints(std::optional<double> _min_queue_cost = std::nullopt,
                   std::optional<double> _max_queue_cost = std::nullopt,
                   std::optional<Index> _max_queue_size = std::nullopt);

  /// \brief Optional, minimum cost node to store in queue
  std::optional<double> min_queue_cost;

  /// \brief Optional, maximum cost node to store in queue
  std::optional<double> max_queue_cost;

  /// \brief Optional, maximum size of queue
  ///
  /// If queue size exceeds max_queue_cost, erase the highest cost node
  std::optional<Index> max_queue_size;

  /// \brief Enforce MappingSearch queue constraints
  void operator()(MappingSearch &search) const;
};

// --- Structure mapping search data structures ---

/// \brief A "node" in the search for optimal structure mappings
///
/// This encodes a particular prim, lattice, and atom mapping,
/// and includes the information needed to continue searching
/// for suboptimal assignments.
struct MappingNode {
  /// \brief Constructor
  MappingNode(
      double _lattice_cost,
      std::shared_ptr<LatticeMappingSearchData const> _lattice_mapping_data,
      double _atom_cost,
      std::shared_ptr<AtomMappingSearchData const> _atom_mapping_data,
      murty::Node _assignment_node, AtomMapping _atom_mapping,
      double _total_cost);

  /// \brief The lattice mapping cost
  double const lattice_cost;

  /// \brief Holds lattice mapping-specific data used
  ///     for mapping searches
  std::shared_ptr<LatticeMappingSearchData const> const lattice_mapping_data;

  /// \brief The atom mapping cost
  double const atom_cost;

  /// \brief Data that can be used for all atom mappings with the
  ///     same lattice mapping and trial translation
  std::shared_ptr<AtomMappingSearchData const> const atom_mapping_data;

  /// \brief Encodes a constrained solution to the atom-to-site
  ///     assignment problem
  ///
  /// This includes the information necessary (assignments forced
  /// on and forced off) to continue searching for suboptimal
  /// assignments. When solved, the sub-assignment problem is
  /// stored in assignment_node.sub_assignment.
  murty::Node const assignment_node;

  /// \brief AtomMapping solution obtained from assignment_node
  ///
  /// This includes the atom mapping permutation, displacements,
  /// and translation determined from:
  /// - the lattice mapping, stored in lattice_mapping_data
  /// - the trial translation and site displacements, stored in
  ///   atom_mapping_data
  /// - the constrained assignment problem solution stored in
  ///   assignment_node
  AtomMapping const atom_mapping;

  /// \brief The total mapping cost
  double const total_cost;

  /// \brief Compare by total_cost only
  bool operator<(MappingNode const &rhs) const {
    return this->total_cost < rhs.total_cost;
  }
};

/// \brief Make mapping node
MappingNode make_mapping_node(
    MappingSearch const &search, double lattice_cost,
    std::shared_ptr<LatticeMappingSearchData const> lattice_mapping_data,
    Eigen::Vector3d const &trial_translation_cart,
    std::map<Index, Index> forced_on,
    std::vector<std::pair<Index, Index>> forced_off);

/// \brief Performs structure mapping searches
struct MappingSearch {
  /// \brief Constructor
  MappingSearch(double _min_cost = 0.0, double _max_cost = 1e20,
                int _k_best = 1,
                AtomCostFunction _atom_cost_f = IsotropicAtomCost(),
                TotalCostFunction _total_cost_f = WeightedTotalCost(0.5),
                AtomToSiteCostFunction _atom_to_site_cost_f =
                    make_atom_to_site_cost_future,
                bool _enable_remove_mean_displacement = true,
                double _infinity = 1e20, double _cost_tol = 1e-5);

  /// \brief A queue of structure mappings, sorted by total
  ///     cost only
  ///
  /// This stores mappings with exactly repeated total cost
  /// in order of insertion.
  std::multiset<MappingNode> queue;

  /// \brief Results, sorted by total cost, satisifying the
  ///     min/max cost and k-best criteria
  ///
  /// This stores mappings with exactly repeated total cost
  /// in order of insertion.
  std::multimap<StructureMappingCost, StructureMapping> results;

  /// \brief Holds results that are approximately tied with
  ///     the k-best result in results
  ///
  /// This stores mappings with exactly repeated total cost
  /// in order of insertion.
  std::multimap<StructureMappingCost, StructureMapping> overflow;

  /// \brief Minimum cost result to keep
  double min_cost;

  /// \brief Maximum cost result to keep
  double max_cost;

  /// \brief Maximum number of results to keep (approximate ties are also kept)
  int k_best;

  /// \brief Function to calculate the atom mapping cost
  AtomCostFunction atom_cost_f;

  /// \brief Function to calculate the total mapping cost
  TotalCostFunction total_cost_f;

  /// \brief Function used to calculate the atom-to-site mapping cost
  AtomToSiteCostFunction atom_to_site_cost_f;

  /// \brief If true, the AtomMapping translation and displacements
  ///     are adjusted consistently so that the mean displacment
  ///     (includes explicit vacancies) is zero.
  bool enable_remove_mean_displacement;

  /// \brief Cost used to prevent assignments that are not allowed
  double infinity;

  /// \brief Tolerance used for comparing costs
  double cost_tol;

  /// \brief Return lowest total cost MappingNode in the queue
  MappingNode const &front() const;

  /// \brief Return highest total cost MappingNode in the queue
  MappingNode const &back() const;

  /// \brief Erase lowest total cost MappingNode in the queue
  void pop_front();

  /// \brief Erase highest total cost MappingNode in the queue
  void pop_back();

  /// \brief Return the size of the queue
  Index size() const;

  /// \brief Make assignment and insert mapping node
  ///     into this->queue & this->results, maintaining k-best results
  std::multiset<MappingNode>::iterator make_and_insert_mapping_node(
      double lattice_cost,
      std::shared_ptr<LatticeMappingSearchData const> lattice_mapping_data,
      Eigen::Vector3d const &trial_translation_cart,
      std::map<Index, Index> forced_on = {},
      std::vector<std::pair<Index, Index>> forced_off = {});

  /// \brief Make the next level of sub-optimal assignments and
  ///     inserts them into this->queue & this->results, maintaining
  ///     k-best results
  std::vector<std::multiset<MappingNode>::iterator> partition();
};

/// \brief Return MappingSearch results combined with overflow
StructureMappingResults combined_results(MappingSearch const &search);

/// --- Inline implementation ---

/// \brief Return lowest total cost MappingNode in the queue
///
/// Invalid if !size()
inline MappingNode const &MappingSearch::front() const {
  return *queue.begin();
}

/// \brief Return highest total cost MappingNode in the queue
///
/// Invalid if !size()
inline MappingNode const &MappingSearch::back() const {
  return *queue.rbegin();
}

/// \brief Erase lowest total cost MappingNode in the queue
///
/// Invalid if !size()
inline void MappingSearch::pop_front() { queue.erase(queue.begin()); }

/// \brief Erase highest total cost MappingNode in the queue
///
/// Invalid if !size()
inline void MappingSearch::pop_back() {
  queue.erase(std::next(queue.rbegin()).base());
}

/// \brief Return the size of the queue
inline Index MappingSearch::size() const { return queue.size(); }

}  // namespace mapping
}  // namespace CASM

#endif
