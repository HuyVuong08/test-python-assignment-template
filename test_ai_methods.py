"""
Unit tests for AI Methods course starter code.

Run with: pytest test_ai_methods.py
"""

import pytest
from ai_methods import (
    Node, SearchProblem, GridNavigationProblem,
    breadth_first_search, depth_first_search, a_star_search,
    manhattan_distance, LinearRegression, reconstruct_path
)


# ============================================================================
# Test Node Class
# ============================================================================

def test_node_creation():
    """Test basic node creation."""
    node = Node(state=(0, 0), parent=None, action=None, path_cost=0)
    assert node.state == (0, 0)
    assert node.parent is None
    assert node.action is None
    assert node.path_cost == 0


def test_node_comparison():
    """Test node comparison for priority queue."""
    node1 = Node(state=(0, 0), path_cost=5)
    node2 = Node(state=(1, 1), path_cost=3)
    assert node2 < node1


# ============================================================================
# Test Grid Navigation Problem
# ============================================================================

def test_grid_problem_initialization():
    """Test grid problem initialization."""
    grid = [[0, 0], [0, 0]]
    problem = GridNavigationProblem(grid, (0, 0), (1, 1))
    assert problem.get_start_state() == (0, 0)
    assert problem.is_goal_state((1, 1))
    assert not problem.is_goal_state((0, 0))


def test_grid_successors():
    """Test successor generation in grid problem."""
    grid = [
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0]
    ]
    problem = GridNavigationProblem(grid, (1, 1), (2, 2))
    
    # From (0, 0), we should be able to move right and down
    successors = problem.get_successors((0, 0))
    states = [s[0] for s in successors]
    assert (0, 1) in states  # RIGHT
    assert (1, 0) in states  # DOWN
    assert len(successors) == 2


def test_grid_obstacle_blocking():
    """Test that obstacles block movement."""
    grid = [
        [0, 1, 0],
        [0, 1, 0],
        [0, 0, 0]
    ]
    problem = GridNavigationProblem(grid, (0, 0), (0, 2))
    
    # From (0, 0), we cannot move right due to obstacle
    successors = problem.get_successors((0, 0))
    states = [s[0] for s in successors]
    assert (0, 1) not in states
    assert (1, 0) in states  # Can only move down


# ============================================================================
# Test Search Algorithms
# ============================================================================

def test_bfs_simple_grid():
    """Test BFS on a simple grid without obstacles."""
    grid = [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ]
    problem = GridNavigationProblem(grid, (0, 0), (2, 2))
    solution = breadth_first_search(problem)
    
    assert solution is not None
    assert len(solution) == 4  # Optimal path length
    assert problem.get_cost_of_actions(solution) == 4


def test_dfs_simple_grid():
    """Test DFS on a simple grid."""
    grid = [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ]
    problem = GridNavigationProblem(grid, (0, 0), (2, 2))
    solution = depth_first_search(problem)
    
    assert solution is not None
    # DFS may not find optimal path, but should find a valid path
    assert problem.get_cost_of_actions(solution) >= 4


def test_astar_simple_grid():
    """Test A* on a simple grid with Manhattan distance heuristic."""
    grid = [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ]
    problem = GridNavigationProblem(grid, (0, 0), (2, 2))
    goal = (2, 2)
    heuristic = lambda state: manhattan_distance(state, goal)
    solution = a_star_search(problem, heuristic)
    
    assert solution is not None
    assert len(solution) == 4  # Optimal path length
    assert problem.get_cost_of_actions(solution) == 4


def test_search_with_obstacles():
    """Test search algorithms with obstacles."""
    grid = [
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0]
    ]
    problem = GridNavigationProblem(grid, (0, 0), (4, 4))
    
    # All algorithms should find a solution
    bfs_sol = breadth_first_search(problem)
    dfs_sol = depth_first_search(problem)
    goal = (4, 4)
    heuristic = lambda state: manhattan_distance(state, goal)
    astar_sol = a_star_search(problem, heuristic)
    
    assert bfs_sol is not None
    assert dfs_sol is not None
    assert astar_sol is not None


def test_no_solution():
    """Test search when no solution exists."""
    grid = [
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0]
    ]
    problem = GridNavigationProblem(grid, (0, 0), (2, 2))
    
    solution = breadth_first_search(problem)
    assert solution is None


def test_start_is_goal():
    """Test when start state is the goal state."""
    grid = [[0, 0], [0, 0]]
    problem = GridNavigationProblem(grid, (0, 0), (0, 0))
    
    solution = breadth_first_search(problem)
    assert solution == []


# ============================================================================
# Test Heuristics
# ============================================================================

def test_manhattan_distance():
    """Test Manhattan distance heuristic."""
    assert manhattan_distance((0, 0), (3, 4)) == 7
    assert manhattan_distance((2, 2), (2, 2)) == 0
    assert manhattan_distance((5, 3), (1, 1)) == 6


# ============================================================================
# Test Linear Regression
# ============================================================================

def test_linear_regression_simple():
    """Test linear regression on a simple dataset."""
    # y = 2x + 1
    X_train = [[1], [2], [3], [4], [5]]
    y_train = [3, 5, 7, 9, 11]
    
    model = LinearRegression(learning_rate=0.01, iterations=1000)
    model.fit(X_train, y_train)
    
    # Check that learned parameters are close to true parameters
    assert abs(model.weights[0] - 2.0) < 0.1  # weight should be close to 2
    assert abs(model.bias - 1.0) < 0.1  # bias should be close to 1


def test_linear_regression_prediction():
    """Test predictions from linear regression."""
    X_train = [[1], [2], [3]]
    y_train = [2, 4, 6]
    
    model = LinearRegression(learning_rate=0.01, iterations=1000)
    model.fit(X_train, y_train)
    
    predictions = model.predict([[4], [5]])
    
    # Predictions should be reasonable
    assert 7 < predictions[0] < 9  # Around 8
    assert 9 < predictions[1] < 11  # Around 10


def test_linear_regression_cost_decreases():
    """Test that cost decreases during training."""
    X_train = [[1], [2], [3], [4]]
    y_train = [2, 4, 6, 8]
    
    model = LinearRegression(learning_rate=0.01, iterations=100)
    model.fit(X_train, y_train)
    
    # Cost should decrease over time
    assert model.cost_history[-1] < model.cost_history[0]


# ============================================================================
# Test Utilities
# ============================================================================

def test_reconstruct_path():
    """Test path reconstruction from nodes."""
    # Create a simple path: root -> child1 -> child2
    root = Node(state='start', parent=None, action=None)
    child1 = Node(state='middle', parent=root, action='action1')
    child2 = Node(state='goal', parent=child1, action='action2')
    
    path = reconstruct_path(child2)
    assert path == ['action1', 'action2']


def test_reconstruct_path_single_node():
    """Test path reconstruction from root node."""
    root = Node(state='start', parent=None, action=None)
    path = reconstruct_path(root)
    assert path == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
