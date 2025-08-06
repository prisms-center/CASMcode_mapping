#ifndef CASM_mapping_murty
#define CASM_mapping_murty

#include <functional>
#include <map>
#include <optional>
#include <set>
#include <vector>

#include "casm/global/definitions.hh"
#include "casm/global/eigen.hh"

namespace CASM {
namespace mapping {
namespace murty {

/// \brief The assignment vector gives `j = assignment[i]`, where i is the
///     "worker" (row) and j is the assigned "task" (column).
typedef std::vector<Index> Assignment;

/// \brief The method used to solve for an optimal assignment
///
/// Expected parameters:
/// - Eigen::MatrixXd const &cost_matrix
/// - double infinity
/// - double tol
///
/// Expected results:
/// - {cost, assignment}, if optimal solution is found
/// - {infinity, {}}, if could not solve (no solution that doesn't
///   include a forced_off assignment)
typedef std::function<std::pair<double, Assignment>(Eigen::MatrixXd const &,
                                                    double, double)>
    AssignmentMethod;

/// \brief Find the k best solutions to the assignment problem
///     using the Murty algorithm
std::vector<std::pair<double, Assignment>> solve(
    AssignmentMethod assign_f, Eigen::MatrixXd const &cost_matrix, int k_best,
    std::optional<double> min_cost = std::nullopt,
    std::optional<double> max_cost = std::nullopt, double infinity = 1e20,
    double tol = 1e-5);

/// \brief Encodes a constrained solution to the assignment problem
///
/// The assignment problem is: minimize the cost of assigning m
/// "workers" to n "tasks", where the cost of assigning "worker" i
/// to "task" j is cost_matrix(i,j).
///
/// The Node forces some assignments (i,j) "on" (assignment is
/// made in all solutions considered) and some assignments (i,j)
/// "off" (cost is set to infinity so assignment is never made).
/// The unassigned rows and columns are stored for efficiency, and
/// the assignment of those rows and columns can be stored in
/// `sub_assignment` when the constrained problem is solved. The
/// final, full assignment can be constructed by combining
/// `forced_on` and `sub_assignment`.
struct Node {
  /// \brief Map of which row (the key) is assigned to
  /// which column (the value)
  std::map<Index, Index> forced_on;

  /// \brief The assignments which are forced off, as
  /// pairs of {row, column}
  std::vector<std::pair<Index, Index>> forced_off;

  /// \brief The unassigned rows - before assignment
  std::set<Index> unassigned_rows;

  /// \brief The unassigned columns - before assignment
  std::set<Index> unassigned_cols;

  /// \brief The optimal assignment {row, column} of the sub
  /// cost matrix formed from the unassigned_rows and unassigned_cols
  /// of the cost matrix
  std::map<Index, Index> sub_assignment;

  /// \brief Total cost, including forced_on and sub_assignment
  double cost;

  /// \brief Compare by cost only
  bool operator<(Node const &rhs) const { return this->cost < rhs.cost; }

  /// \brief Return true if assignment was made
  bool all_col_assigned() const {  // check if sub_assignment succeeded:
    return sub_assignment.size() == unassigned_cols.size();
  }
};

/// \brief Returns a Node representing the (constrained) assignment problem
Node make_node(Eigen::MatrixXd const &cost_matrix,
               std::map<Index, Index> forced_on = {},
               std::vector<std::pair<Index, Index>> forced_off = {});

/// \brief Returns the full assignment
Assignment make_assignment(Node const &node);

/// \brief Return the cost for a particular assignment
double make_cost(Eigen::MatrixXd const &cost_matrix,
                 Assignment const &assignment);

/// \brief Partition a Node, adding results to a multiset of Node
void partition(std::multiset<Node> &node_set, AssignmentMethod assign_f,
               Eigen::MatrixXd const &cost_matrix, Node const &node,
               double infinity, double tol);

/// \brief Solve the assignment problem given certain assignments
///    forced on and certain assignments forced off
std::pair<double, std::map<Index, Index>> make_sub_assignment(
    AssignmentMethod assign_f, Eigen::MatrixXd cost_matrix,
    std::set<Index> const &unassigned_rows,
    std::set<Index> const &unassigned_cols,
    std::vector<std::pair<Index, Index>> const &forced_off, double infinity,
    double tol);

}  // namespace murty
}  // namespace mapping
}  // namespace CASM

#endif
