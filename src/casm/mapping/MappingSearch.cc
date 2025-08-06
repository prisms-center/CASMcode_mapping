#include "casm/mapping/MappingSearch.hh"

#include "casm/mapping/atom_cost.hh"
#include "casm/mapping/hungarian.hh"
#include "casm/mapping/impl/LatticeMap.hh"
#include "casm/mapping/misc.hh"

namespace CASM {
namespace mapping {

namespace mapping_impl {

/// \brief Construct an AtomMapping from an assignment solution
///
/// The site displacements are the minimum length displacements
/// that satisfy:
///
///     site_coordinate_cart[i] + site_displacements[i][j] =
///         F^{-1}*atom_coordinate_cart[j] + trial_translation
///
/// under periodic boundary conditions, and in the context of
/// a particular lattice mapping. The AtomMapping transformation
/// is:
///
///     F * (r1[i] + d[i]) = r2[perm[i]] + translation
///
/// in the context of a LatticeMapping solution such that:
///
///     F * L1 * T * N = L2
///
/// Therefore, the AtomMapping translation is:
///
///     translation = F * trial_translation
///
///
/// \param lattice Lattice in which the displacements are calculated
///     under periodic boundary conditions.
/// \param assignment Assignment solution, with the convention
///     `atom_index = assignment[site_index]`. With the cost
///     matrix constructed according to the convention
///     `cost_matrix(site_index, atom_index)` this is equal
///     to the AtomMapping permutation.
/// \param site_displacements The site-to-atom displacements,
///     of minimum length under periodic boundary conditions
///     as used in the assignment problem.
/// \param trial_translation A translation applied to atom
///     coordinates to bring the atoms and sites into alignment.
/// \param deformation_gradient The deformation gradient of
///     the lattice mapping that this atom mapping is solved
///     in the context of.
/// \param enable_remove_mean_displacement If true, the
///     AtomMapping translation and displacements are adjusted
///     consistently so that the mean displacment is zero.
///
/// \returns An AtomMapping constructed from the inputs, with
///     mean displacement removed.
AtomMapping make_atom_mapping_from_assignment(
    xtal::Lattice const &lattice, std::vector<Index> const &assignment,
    std::vector<std::vector<Eigen::Vector3d>> const &site_displacements,
    Eigen::VectorXd trial_translation,
    Eigen::Matrix3d const &deformation_gradient,
    bool enable_remove_mean_displacement) {
  Index N_site = assignment.size();

  // atom[perm[i]] -> is assigned to -> site[i]
  auto const &perm = assignment;

  Eigen::VectorXd zero_disp = Eigen::Vector3d::Zero();
  Eigen::Vector3d mean_disp = Eigen::Vector3d::Zero();
  if (enable_remove_mean_displacement) {
    // calculate mean displacement
    double n = 0.0;
    for (Index site_index = 0; site_index < N_site; ++site_index) {
      Index atom_index = perm[site_index];
      if (atom_index >= site_displacements[site_index].size()) {
        // implied vacancies - do not include in mean_disp
        continue;
      }
      mean_disp += site_displacements[site_index][atom_index];
      n += 1.0;
    }
    mean_disp /= n;
  }
  mean_disp = robust_pbc_displacement_cart(lattice, zero_disp, mean_disp);

  // get displacements
  Eigen::MatrixXd disp = Eigen::MatrixXd::Zero(3, N_site);
  for (Index site_index = 0; site_index < N_site; ++site_index) {
    Index atom_index = perm[site_index];
    if (atom_index >= site_displacements[site_index].size()) {
      // implied vacancies - keep disp == 0
      continue;
    }
    // re-calculate the displacements under periodic boundary conditions after
    // removing the mean displacement
    disp.col(site_index) = robust_pbc_displacement_cart(
        lattice, zero_disp,
        site_displacements[site_index][atom_index] - mean_disp);
  }

  // adjust trial_translation
  trial_translation -= mean_disp;

  return AtomMapping(disp, perm, deformation_gradient * trial_translation);
}

/// \brief Make a new MappingNode from an assignment problem node
///     with a solved sub_assignment
///
/// When constructing the AtomMapping component of a MappingNode,
/// this removes mean displacements (if enabled) and makes the proper
/// AtomMapping displacements and translation. It calculates
/// the atom_cost and total_cost using the parameters specified
/// at MappingSearch construction time.
MappingNode make_mapping_node_from_assignment_node(
    MappingSearch const &search, murty::Node assignment_node,
    double lattice_cost,
    std::shared_ptr<LatticeMappingSearchData const> lattice_mapping_data,
    std::shared_ptr<AtomMappingSearchData const> atom_mapping_data) {
  if (!assignment_node.all_col_assigned()) {
    // Assignment failed: set atom_cost and total_cost to infinity
    double atom_cost = search.infinity;
    double total_cost = search.infinity;
    AtomMapping atom_mapping(Eigen::MatrixXd::Zero(0, 0),  // empty displacement
                             std::vector<Index>{},         // empty permutation
                             atom_mapping_data->trial_translation_cart);
    return MappingNode(lattice_cost, std::move(lattice_mapping_data), atom_cost,
                       std::move(atom_mapping_data), std::move(assignment_node),
                       atom_mapping, total_cost);
  }

  AtomMapping atom_mapping = make_atom_mapping_from_assignment(
      lattice_mapping_data->supercell_lattice,
      murty::make_assignment(assignment_node),
      atom_mapping_data->site_displacements,
      atom_mapping_data->trial_translation_cart,
      lattice_mapping_data->lattice_mapping.deformation_gradient,
      search.enable_remove_mean_displacement);
  double atom_cost = search.atom_cost_f(*lattice_mapping_data,
                                        *atom_mapping_data, atom_mapping);
  double total_cost =
      search.total_cost_f(lattice_cost, *lattice_mapping_data, atom_cost,
                          *atom_mapping_data, atom_mapping);
  return MappingNode(lattice_cost, std::move(lattice_mapping_data), atom_cost,
                     std::move(atom_mapping_data), std::move(assignment_node),
                     std::move(atom_mapping), total_cost);
}

/// \brief Insert mapping node into MappingSearch queue & results,
///     maintaining k-best results
///
/// A MappingNode is inserted into queue (always!), and then
/// inserted into results if it satisfies the min/max cost
/// and k-best criteria. Approximate ties with the k-best cost
/// are kept in overflow.
///
/// \param search MappingSearch structure where node is inserted
/// \param mapping_node MappingNode to insert
///
/// \returns Iterator to MappingNode in queue
///
std::multiset<MappingNode>::iterator insert(MappingSearch &search,
                                            MappingNode mapping_node) {
  MappingNode const &n = mapping_node;
  // --- maintain k_best results, keeping ties in overflow ---
  // note: max_cost is modified to shrink
  // to the current k_best-th cost once k_best results are found
  if (n.total_cost > (search.min_cost - search.cost_tol)) {
    if (n.total_cost < search.max_cost + search.cost_tol) {
      search.results.emplace(
          StructureMappingCost(n.lattice_cost, n.atom_cost, n.total_cost),
          StructureMapping(n.lattice_mapping_data->prim_data->prim,
                           n.lattice_mapping_data->lattice_mapping,
                           n.atom_mapping));
      mapping::maintain_k_best_results(
          search.k_best, search.cost_tol, search.results, search.overflow,
          [](StructureMappingCost const &key) { return key.total_cost; });
      if (search.results.size() == search.k_best) {
        search.max_cost = search.results.rbegin()->first.total_cost;
      }
    }
  }
  if (n.total_cost < search.max_cost + search.cost_tol) {
    return search.queue.insert(std::move(mapping_node));
  } else {
    return search.queue.end();
  }
}

}  // namespace mapping_impl

double IsotropicAtomCost::operator()(
    LatticeMappingSearchData const &lattice_mapping_data,
    AtomMappingSearchData const &atom_mapping_data,
    AtomMapping const &atom_mapping) const {
  auto const &prim_data = *(lattice_mapping_data.prim_data);
  auto const &L1 = prim_data.prim_lattice.lat_column_mat();
  auto const &lattice_mapping = lattice_mapping_data.lattice_mapping;
  return make_isotropic_atom_cost(L1, lattice_mapping,
                                  atom_mapping.displacement);
}

double SymmetryBreakingAtomCost::operator()(
    LatticeMappingSearchData const &lattice_mapping_data,
    AtomMappingSearchData const &atom_mapping_data,
    AtomMapping const &atom_mapping) const {
  auto const &prim_data = *(lattice_mapping_data.prim_data);
  if (!prim_data.prim_sym_invariant_displacement_modes.has_value()) {
    throw std::runtime_error(
        "Error in SymmetryBreakingAtomCost: prim symmetry-invariant "
        "displacement modes are not available. Use "
        "enable_symmetry_breaking_atom_cost when constructing PrimSearchData.");
  }
  auto const &L1 = prim_data.prim_lattice.lat_column_mat();
  auto const &lattice_mapping = lattice_mapping_data.lattice_mapping;
  if (prim_data.prim_sym_invariant_displacement_modes->size() == 0) {
    return make_isotropic_atom_cost(L1, lattice_mapping,
                                    atom_mapping.displacement);
  }
  return make_symmetry_breaking_atom_cost(
      L1, lattice_mapping, atom_mapping.displacement,
      lattice_mapping_data.unitcellcoord_index_converter,
      *prim_data.prim_sym_invariant_displacement_modes);
}

WeightedTotalCost::WeightedTotalCost(double _lattice_cost_weight)
    : lattice_cost_weight(_lattice_cost_weight) {}

double WeightedTotalCost::operator()(
    double lattice_cost, LatticeMappingSearchData const &lattice_mapping_data,
    double atom_cost, AtomMappingSearchData const &atom_mapping_data,
    AtomMapping const &atom_mapping) const {
  return this->lattice_cost_weight * lattice_cost +
         (1. - this->lattice_cost_weight) * atom_cost;
}

/// \brief Constructor
QueueConstraints::QueueConstraints(std::optional<double> _min_queue_cost,
                                   std::optional<double> _max_queue_cost,
                                   std::optional<Index> _max_queue_size)
    : min_queue_cost(_min_queue_cost),
      max_queue_cost(_max_queue_cost),
      max_queue_size(_max_queue_size) {}

/// \brief Enforce queue constraints
///
/// Enforce min_queue_cost, max_queue_cost, and max_queue_size by erasing
/// queue elements from the front or back if necessary.
///
void QueueConstraints::operator()(MappingSearch &search) const {
  // enforce min_queue_cost -- erase head if cost exceeds max_queue_cost
  if (this->min_queue_cost.has_value() && search.size()) {
    while (search.size() && search.front().total_cost <=
                                *this->min_queue_cost - search.cost_tol) {
      search.pop_front();
    }
  }
  // enforce max_queue_cost -- erase tail if cost exceeds max_queue_cost
  if (this->max_queue_cost.has_value() && search.size()) {
    while (search.size() && search.back().total_cost >=
                                *this->max_queue_cost + search.cost_tol) {
      search.pop_back();
    }
  }
  // enforce max_queue_size -- erase tail if size exceeds max_queue_size
  if (this->max_queue_size.has_value()) {
    while (search.size() > *this->max_queue_size) {
      search.pop_back();
    }
  }
}

/// \brief Constructor
MappingNode::MappingNode(
    double _lattice_cost,
    std::shared_ptr<LatticeMappingSearchData const> _lattice_mapping_data,
    double _atom_cost,
    std::shared_ptr<AtomMappingSearchData const> _atom_mapping_data,
    murty::Node _assignment_node, AtomMapping _atom_mapping, double _total_cost)
    : lattice_cost(_lattice_cost),
      lattice_mapping_data(std::move(_lattice_mapping_data)),
      atom_cost(_atom_cost),
      atom_mapping_data(std::move(_atom_mapping_data)),
      assignment_node(std::move(_assignment_node)),
      atom_mapping(std::move(_atom_mapping)),
      total_cost(_total_cost) {}

/// \brief Make mapping node
///
/// The (constrained) assignment problem is solved in context of
/// a particular lattice mapping and trial translation, and
/// the resulting AtomMapping, atom mapping cost, and total cost
/// are stored in a MappingNode, along with the (constrained)
/// assignment_node which allows continuing the search for
/// suboptimal solutions. The MappingNode is inserted in
/// this->queue. It is also inserted in this->results, if it
/// satisifies the cost range and k-best criteria.
///
///
/// \param search MappingSearch structure with parameters used
///     for constructing the mapping costs
/// \param lattice_cost The lattice mapping cost
/// \param lattice_mapping_data Data associated with the lattice
///     mapping that is the context in which the atom mapping is done
/// \param trial_translation_cart A Cartesian translation applied to
///     atom coordinates in the ideal superstructure setting
///     (i.e. atom_coordinate_cart_in_supercell) to bring the atoms
///     into alignment with ideal superstructure sites.
/// \param forced_on A map of {site_index, atom_index} of
///     assignments that are forced on
/// \param forced_off A vector of {site_index, atom_index} of
///     assignments that are forced off (given infinity cost)
///
/// \returns Resulting MappingNode
MappingNode make_mapping_node(
    MappingSearch const &search, double lattice_cost,
    std::shared_ptr<LatticeMappingSearchData const> lattice_mapping_data,
    Eigen::Vector3d const &trial_translation_cart,
    std::map<Index, Index> forced_on,
    std::vector<std::pair<Index, Index>> forced_off) {
  auto atom_mapping_data = std::make_shared<AtomMappingSearchData const>(
      lattice_mapping_data, trial_translation_cart, search.atom_to_site_cost_f,
      search.infinity);

  // --- Find optimal assignment ---
  murty::Node assignment_node =
      murty::make_node(atom_mapping_data->cost_matrix, std::move(forced_on),
                       std::move(forced_off));
  std::tie(assignment_node.cost, assignment_node.sub_assignment) =
      murty::make_sub_assignment(
          hungarian::solve, atom_mapping_data->cost_matrix,
          assignment_node.unassigned_rows, assignment_node.unassigned_cols,
          assignment_node.forced_off, search.infinity, search.cost_tol);

  // --- Make mapping node from assignment solution ---
  return mapping_impl::make_mapping_node_from_assignment_node(
      search, std::move(assignment_node), lattice_cost,
      std::move(lattice_mapping_data), std::move(atom_mapping_data));
}

/// \struct MappingSearch
/// \brief Performs structure mapping searches
///
/// The MappingSearch structure includes parameters, data,
/// and methods used to search for low cost structure mappings.
///
/// It holds a queue of MappingNode, which encode a particular
/// structure mapping, and the data necessary to start from
/// that structure mapping and find sub-optimal atom mappings
/// as part of a search using the Murty Algorithm for
/// sub-optimal assignments.
/// Parameters controlling queue insertion are:
/// - min_queue_cost: Queue sub-mappings with total
///   cost >= min_queue_cost to find sub-optimal mappings
/// - max_queue_cost: Queue sub-mappings with total
///   cost <= max_queue_cost to find sub-optimal mappings
/// - max_queue_size: Do not let the queue grow larger
///   than max_queue_size
///
/// It also holds a sorted container of the best results found
/// so far which satisfy some acceptance criteria:
/// - min_cost: Keep mappings with total cost >= min_cost
/// - max_cost: Keep mappings with total cost <= max_cost
/// - k_best: Keep the k_best mappings with lowest total cost
///   that also satisfy the min/max cost criteria.
///
/// It also holds an `overflow` container to keep approximate
/// ties with the k_best result.
///
/// Overview of methods:
/// - `MappingSearch::make_and_insert_mapping_node`:
///   - Given a lattice mapping, trial translation, and optionally assignments
///   to force on or off, this solves the assignment problem, constructs a
///   MappingNode holding the solution. The MappingNode is inserted in
///   this->queue (always) and this->results (if it satisfies the acceptance
///   criteria).
///   - This is typically used to initialize a search from one or more lattice
///   mappings and trial translations
/// - `MappingSearch::partition`:
///   - Given a MappingNode, this applies to Murty Algorithm to find the next
///   level of sub-optimal assignments, and for each constructs a MappingNode
///   holding the solution, and inserts the MappingNode in this->queue (always)
///   and this->results (if it satisfies the acceptance criteria).
///   - This is typically used to continue a search from the first element of
///   the queue. After partitioning, the element can be erased from the queue.
///
/// For more details, see J.C. Thomas, A.R. Natarajan, and
/// A.V. Van der Ven, npj Computational Materials (2021)7:164;
/// https://doi.org/10.1038/s41524-021-00627-0
///

/// \brief Constructor
///
/// \param _min_cost Keep mappings with total cost >= min_cost
/// \param _max_cost Keep mappings with total cost <= max_cost. Note that this
///     parameter does not control the queue of MappingNode, it only controls
///     which solutions are stored in `results`.
/// \param _lattice_cost_weight The fraction of the total cost due to
///     the lattice strain cost. The remaining fraction
///     (1.-lattice_cost_weight) is due to the atom cost. Default=0.5.
/// \param _atom_cost_f A function that implements the atom mapping
///     cost calculation.
/// \param _total_cost_f A function that implements the total mapping
///     cost calculation.
/// \param _k_best Keep the k_best results satisfying the min_cost and
///     max_cost constraints. If there are approximate ties, those
///     will also be kept.
/// \param enable_remove_mean_displacement If true, the
///     AtomMapping translation and displacements are adjusted
///     consistently so that the mean displacment is zero.
/// \param _infinity The value to use in the assignment problem cost
///     matrix for unallowed assignments
/// \param _cost_tol Tolerance for checking if mapping costs are
///     approximately equal
MappingSearch::MappingSearch(double _min_cost, double _max_cost, int _k_best,
                             AtomCostFunction _atom_cost_f,
                             TotalCostFunction _total_cost_f,
                             AtomToSiteCostFunction _atom_to_site_cost_f,
                             bool _enable_remove_mean_displacement,
                             double _infinity, double _cost_tol)
    : min_cost(_min_cost),
      max_cost(_max_cost),
      k_best(_k_best),
      atom_cost_f(_atom_cost_f),
      total_cost_f(_total_cost_f),
      atom_to_site_cost_f(_atom_to_site_cost_f),
      enable_remove_mean_displacement(_enable_remove_mean_displacement),
      infinity(_infinity),
      cost_tol(_cost_tol) {}

/// \brief Make assignment and insert mapping node
///     into this->queue & this->results, maintaining k-best results
///
/// The (constrained) assignment problem is solved in context of
/// a particular lattice mapping and trial translation, and
/// the resulting AtomMapping, atom mapping cost, and total cost
/// are stored in a MappingNode, along with the (constrained)
/// assignment_node which allows continuing the search for
/// suboptimal solutions. The MappingNode is inserted in
/// this->queue. It is also inserted in this->results, if it
/// satisifies the cost range and k-best criteria.
///
///
/// \param lattice_cost The lattice mapping cost
/// \param lattice_mapping_data Data associated with the lattice
///     mapping that is the context in which the atom mapping is done
/// \param trial_translation_cart A Cartesian translation applied to
///     atom coordinates in the ideal superstructure setting
///     (i.e. atom_coordinate_cart_in_supercell) to bring the atoms
///     into alignment with ideal superstructure sites.
/// \param forced_on A map of {site_index, atom_index} of
///     assignments that are forced on
/// \param forced_off A vector of {site_index, atom_index} of
///     assignments that are forced off (given infinity cost)
///
/// \returns Iterator to resulting MappingNode in queue
///     if inserted, else queue.end()
std::multiset<MappingNode>::iterator
MappingSearch::make_and_insert_mapping_node(
    double lattice_cost,
    std::shared_ptr<LatticeMappingSearchData const> lattice_mapping_data,
    Eigen::Vector3d const &trial_translation_cart,
    std::map<Index, Index> forced_on,
    std::vector<std::pair<Index, Index>> forced_off) {
  // --- Insert mapping node in queue and results, return queue iterator ---
  return mapping_impl::insert(
      *this,
      make_mapping_node(*this, lattice_cost, std::move(lattice_mapping_data),
                        trial_translation_cart, std::move(forced_on),
                        std::move(forced_off)));
}

/// \brief Make the next level of sub-optimal assignments and
///     inserts them into this->queue & this->results, maintaining
///     k-best results
///
/// The Murty algorithm is used to generate sub-optimal assignments
/// from the previous assignment solution stored in this->front().
/// The resulting MappingNode are inserted in this->queue. They are
/// also inserted in this->results, if they satisify the cost range
/// and k-best criteria. Finally, this->pop_front() is called.
///
/// Notes:
/// - Invalid if !size()
///
/// \param node A MappingNode to find sub-optimal assignments from
///     using the Murty algorithm
///
/// \returns A vector of iterators to the generated sub-nodes in queue,
///     or queue.end() if not inserted. Empty vector if queue is empty
///     or no sub-nodes are possible.
///
std::vector<std::multiset<MappingNode>::iterator> MappingSearch::partition() {
  // results are iterators to newly generated sub-nodes
  std::vector<std::multiset<MappingNode>::iterator> result;

  // if nothing in queue, nothing can be done
  if (!this->size()) {
    return result;
  }

  auto node_it = this->queue.begin();
  // -- Make the next level of sub-optimal assignment solutions ---
  std::multiset<murty::Node> s;
  murty::partition(s, hungarian::solve, node_it->atom_mapping_data->cost_matrix,
                   node_it->assignment_node, this->infinity, this->cost_tol);

  // The sub-optimal assignment solutions are in 's',
  // and we want them to all end up in MappingNode.
  // This loop extracts the values from 's', uses them to
  // construct MappingNode, and continues until they are
  // all extracted.
  while (s.size()) {
    // --- Make mapping node from sub-optimal assignment ---
    MappingNode mapping_node =
        mapping_impl::make_mapping_node_from_assignment_node(
            *this, std::move(s.extract(s.begin()).value()),
            node_it->lattice_cost, node_it->lattice_mapping_data,
            node_it->atom_mapping_data);

    // check assignment:
    for (auto const &forced_on : mapping_node.assignment_node.forced_on) {
      auto const &sub_assignment = mapping_node.assignment_node.sub_assignment;
      auto assignment_it = sub_assignment.find(forced_on.first);
      if (assignment_it == sub_assignment.end()) {
        continue;
      }
      if (assignment_it->second == forced_on.second) {
        throw std::runtime_error(
            "Error in partition: pair was supposed to be forced on, should not "
            "be included in sub_assignment");
      }
    }
    for (auto const &forced_off : mapping_node.assignment_node.forced_off) {
      auto const &sub_assignment = mapping_node.assignment_node.sub_assignment;
      auto assignment_it = sub_assignment.find(forced_off.first);
      if (assignment_it == sub_assignment.end()) {
        continue;
      }
      if (assignment_it->second == forced_off.second) {
        throw std::runtime_error(
            "Error in partition: pair was supposed to be forced off, should "
            "not be included in sub_assignment");
      }
    }

    // --- Insert mapping node in queue and results, return queue iterator ---
    result.emplace_back(mapping_impl::insert(*this, std::move(mapping_node)));
  }
  this->queue.erase(node_it);
  return result;
}

/// \brief Return MappingSearch results combined with overflow
StructureMappingResults combined_results(MappingSearch const &search) {
  StructureMappingResults results;
  for (auto const &pair : search.results) {
    results.data.emplace_back(pair);
  }
  for (auto const &pair : search.overflow) {
    results.data.emplace_back(pair);
  }
  return results;
}

}  // namespace mapping
}  // namespace CASM
