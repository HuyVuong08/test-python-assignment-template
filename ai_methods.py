"""
AI Methods Course - Starter Code
==================================

This module provides starter implementations for common AI algorithms
including search algorithms and basic machine learning concepts.

Author: Course Instructor
Date: 2026
"""

from collections import deque
from typing import List, Set, Tuple, Optional, Callable
import heapq


class Node:
    """
    A node in a search tree for graph/tree search algorithms.
    
    Attributes:
        state: The state represented by this node
        parent: The parent node (None for root)
        action: The action taken to reach this node from parent
        path_cost: The cost to reach this node from the start
    """
    
    def __init__(self, state, parent=None, action=None, path_cost=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
    
    def __lt__(self, other):
        """For priority queue comparisons."""
        return self.path_cost < other.path_cost
    
    def __repr__(self):
        return f"Node(state={self.state}, path_cost={self.path_cost})"


class SearchProblem:
    """
    Abstract class for a search problem. You should subclass this
    and implement the methods for your specific problem.
    """
    
    def get_start_state(self):
        """Return the start state for the search problem."""
        raise NotImplementedError
    
    def is_goal_state(self, state) -> bool:
        """Return True if state is a goal state."""
        raise NotImplementedError
    
    def get_successors(self, state) -> List[Tuple]:
        """
        Return a list of tuples (successor, action, step_cost) where:
        - successor is a successor state to the current state
        - action is the action required to get there
        - step_cost is the incremental cost of the action
        """
        raise NotImplementedError
    
    def get_cost_of_actions(self, actions: List) -> float:
        """
        Return the total cost of a particular sequence of actions.
        """
        raise NotImplementedError


def breadth_first_search(problem: SearchProblem) -> List:
    """
    Breadth-First Search (BFS) algorithm.
    
    Search the shallowest nodes in the search tree first.
    
    Args:
        problem: A SearchProblem instance
        
    Returns:
        A list of actions that reaches the goal, or None if no solution exists
    """
    start_node = Node(problem.get_start_state())
    
    if problem.is_goal_state(start_node.state):
        return []
    
    frontier = deque([start_node])  # FIFO queue for BFS
    frontier_states = {start_node.state}  # Track states in frontier
    explored = set()
    
    while frontier:
        node = frontier.popleft()
        frontier_states.discard(node.state)
        explored.add(node.state)
        
        for successor, action, step_cost in problem.get_successors(node.state):
            child = Node(successor, node, action, node.path_cost + step_cost)
            
            if child.state not in explored and child.state not in frontier_states:
                if problem.is_goal_state(child.state):
                    # Reconstruct path
                    return reconstruct_path(child)
                frontier.append(child)
                frontier_states.add(child.state)
    
    return None  # No solution found


def depth_first_search(problem: SearchProblem) -> List:
    """
    Depth-First Search (DFS) algorithm.
    
    Search the deepest nodes in the search tree first.
    
    Args:
        problem: A SearchProblem instance
        
    Returns:
        A list of actions that reaches the goal, or None if no solution exists
    """
    start_node = Node(problem.get_start_state())
    
    if problem.is_goal_state(start_node.state):
        return []
    
    frontier = [start_node]  # Stack for DFS
    frontier_states = {start_node.state}  # Track states in frontier
    explored = set()
    
    while frontier:
        node = frontier.pop()  # LIFO - pop from end
        frontier_states.discard(node.state)
        
        if problem.is_goal_state(node.state):
            return reconstruct_path(node)
        
        explored.add(node.state)
        
        for successor, action, step_cost in problem.get_successors(node.state):
            if successor not in explored and successor not in frontier_states:
                child = Node(successor, node, action, node.path_cost + step_cost)
                frontier.append(child)
                frontier_states.add(child.state)
    
    return None  # No solution found


def a_star_search(problem: SearchProblem, heuristic: Callable = None) -> List:
    """
    A* Search algorithm.
    
    Search the node with the lowest f(n) = g(n) + h(n) first.
    
    Args:
        problem: A SearchProblem instance
        heuristic: A function that takes a state and returns an estimate
                   of the cost to reach the goal. If None, defaults to 0
                   (making this Uniform Cost Search)
        
    Returns:
        A list of actions that reaches the goal, or None if no solution exists
    """
    if heuristic is None:
        heuristic = lambda state: 0
    
    start_node = Node(problem.get_start_state())
    
    if problem.is_goal_state(start_node.state):
        return []
    
    # Priority queue: (priority, counter, node)
    counter = 0  # Tie-breaker for nodes with same priority
    frontier = [(heuristic(start_node.state), counter, start_node)]
    explored = set()
    frontier_states = {start_node.state: start_node.path_cost}
    
    while frontier:
        _, _, node = heapq.heappop(frontier)
        
        # Skip if we've already explored a better path to this state
        if node.state in explored:
            continue
        
        if problem.is_goal_state(node.state):
            return reconstruct_path(node)
        
        explored.add(node.state)
        frontier_states.pop(node.state, None)  # Remove from frontier tracking
        
        for successor, action, step_cost in problem.get_successors(node.state):
            child = Node(successor, node, action, node.path_cost + step_cost)
            
            if child.state not in explored:
                priority = child.path_cost + heuristic(child.state)
                
                # Add to frontier if not there or if we found a better path
                if child.state not in frontier_states or child.path_cost < frontier_states[child.state]:
                    counter += 1
                    heapq.heappush(frontier, (priority, counter, child))
                    frontier_states[child.state] = child.path_cost
    
    return None  # No solution found


def reconstruct_path(node: Node) -> List:
    """
    Reconstruct the path from start to the given node.
    
    Args:
        node: The goal node
        
    Returns:
        A list of actions from start to goal
    """
    path = []
    while node.parent is not None:
        path.append(node.action)
        node = node.parent
    path.reverse()
    return path


# ============================================================================
# Example Problem: Grid Navigation
# ============================================================================

class GridNavigationProblem(SearchProblem):
    """
    A simple grid navigation problem where an agent needs to navigate
    from a start position to a goal position on a 2D grid.
    
    The grid may contain obstacles that cannot be traversed.
    """
    
    def __init__(self, grid: List[List[int]], start: Tuple[int, int], goal: Tuple[int, int]):
        """
        Initialize the grid navigation problem.
        
        Args:
            grid: 2D list where 0 is free space and 1 is obstacle
            start: Starting position (row, col)
            goal: Goal position (row, col)
        """
        self.grid = grid
        self.start = start
        self.goal = goal
        self.rows = len(grid)
        self.cols = len(grid[0]) if grid else 0
    
    def get_start_state(self):
        return self.start
    
    def is_goal_state(self, state):
        return state == self.goal
    
    def get_successors(self, state):
        """
        Returns valid moves from current state.
        Actions: 'UP', 'DOWN', 'LEFT', 'RIGHT'
        """
        successors = []
        row, col = state
        
        # Define possible moves
        moves = [
            ((-1, 0), 'UP'),
            ((1, 0), 'DOWN'),
            ((0, -1), 'LEFT'),
            ((0, 1), 'RIGHT')
        ]
        
        for (dr, dc), action in moves:
            new_row, new_col = row + dr, col + dc
            
            # Check if move is valid
            if (0 <= new_row < self.rows and 
                0 <= new_col < self.cols and 
                self.grid[new_row][new_col] == 0):
                
                successors.append(((new_row, new_col), action, 1))
        
        return successors
    
    def get_cost_of_actions(self, actions):
        """Each action costs 1."""
        return len(actions)


def manhattan_distance(state: Tuple[int, int], goal: Tuple[int, int]) -> float:
    """
    Calculate the Manhattan distance heuristic.
    
    This is an admissible heuristic for grid navigation with 4-way movement.
    """
    return abs(state[0] - goal[0]) + abs(state[1] - goal[1])


# ============================================================================
# Machine Learning Utilities
# ============================================================================

class LinearRegression:
    """
    Simple Linear Regression implementation using gradient descent.
    
    This is for educational purposes to understand the fundamentals.
    For production use, consider scikit-learn.
    """
    
    def __init__(self, learning_rate: float = 0.01, iterations: int = 1000):
        """
        Initialize the linear regression model.
        
        Args:
            learning_rate: Learning rate for gradient descent
            iterations: Number of iterations for training
        """
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None
        self.cost_history = []
    
    def fit(self, X: List[List[float]], y: List[float]):
        """
        Train the model using gradient descent.
        
        Args:
            X: Training features (n_samples x n_features)
            y: Training labels (n_samples,)
        """
        n_samples, n_features = len(X), len(X[0])
        
        # Initialize parameters
        self.weights = [0.0] * n_features
        self.bias = 0.0
        
        # Gradient descent
        for _ in range(self.iterations):
            # Predictions
            y_pred = [self._predict_single(x) for x in X]
            
            # Calculate gradients
            dw = [0.0] * n_features
            db = 0.0
            
            for i in range(n_samples):
                error = y_pred[i] - y[i]
                db += error
                for j in range(n_features):
                    dw[j] += error * X[i][j]
            
            # Update parameters
            for j in range(n_features):
                self.weights[j] -= (self.learning_rate / n_samples) * dw[j]
            self.bias -= (self.learning_rate / n_samples) * db
            
            # Track cost
            cost = sum((y_pred[i] - y[i]) ** 2 for i in range(n_samples)) / (2 * n_samples)
            self.cost_history.append(cost)
    
    def predict(self, X: List[List[float]]) -> List[float]:
        """
        Make predictions for given features.
        
        Args:
            X: Features (n_samples x n_features)
            
        Returns:
            Predictions
        """
        return [self._predict_single(x) for x in X]
    
    def _predict_single(self, x: List[float]) -> float:
        """Make prediction for a single sample."""
        return sum(w * xi for w, xi in zip(self.weights, x)) + self.bias


# ============================================================================
# Main / Demo
# ============================================================================

def demo_search_algorithms():
    """Demonstrate the search algorithms on a grid problem."""
    print("=" * 60)
    print("AI Methods Course - Search Algorithms Demo")
    print("=" * 60)
    
    # Define a simple grid (0 = free, 1 = obstacle)
    grid = [
        [0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0]
    ]
    
    start = (0, 0)
    goal = (4, 4)
    
    print("\nGrid (S=start, G=goal, X=obstacle, .=free):")
    for i, row in enumerate(grid):
        for j, cell in enumerate(row):
            if (i, j) == start:
                print('S', end=' ')
            elif (i, j) == goal:
                print('G', end=' ')
            elif cell == 1:
                print('X', end=' ')
            else:
                print('.', end=' ')
        print()
    
    problem = GridNavigationProblem(grid, start, goal)
    
    # BFS
    print("\n1. Breadth-First Search:")
    solution = breadth_first_search(problem)
    print(f"   Solution: {solution}")
    print(f"   Path length: {len(solution)}")
    
    # DFS
    print("\n2. Depth-First Search:")
    solution = depth_first_search(problem)
    print(f"   Solution: {solution}")
    print(f"   Path length: {len(solution)}")
    
    # A* with Manhattan distance heuristic
    print("\n3. A* Search (with Manhattan distance heuristic):")
    heuristic = lambda state: manhattan_distance(state, goal)
    solution = a_star_search(problem, heuristic)
    print(f"   Solution: {solution}")
    print(f"   Path length: {len(solution)}")
    
    print("\n" + "=" * 60)


def demo_linear_regression():
    """Demonstrate simple linear regression."""
    print("=" * 60)
    print("Linear Regression Demo")
    print("=" * 60)
    
    # Simple dataset: y = 2x + 1
    X_train = [[1], [2], [3], [4], [5]]
    y_train = [3, 5, 7, 9, 11]
    
    print("\nTraining data:")
    for x, y in zip(X_train, y_train):
        print(f"   X={x[0]}, y={y}")
    
    # Train model
    model = LinearRegression(learning_rate=0.01, iterations=1000)
    model.fit(X_train, y_train)
    
    print(f"\nLearned parameters:")
    print(f"   Weight: {model.weights[0]:.4f}")
    print(f"   Bias: {model.bias:.4f}")
    print(f"   (Expected: weight=2.0, bias=1.0)")
    
    # Make predictions
    X_test = [[6], [7], [8]]
    predictions = model.predict(X_test)
    
    print(f"\nPredictions:")
    for x, pred in zip(X_test, predictions):
        expected = 2 * x[0] + 1
        print(f"   X={x[0]}: predicted={pred:.2f}, expected={expected}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    demo_search_algorithms()
    print("\n")
    demo_linear_regression()
