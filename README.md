# AI Methods Course - Python Assignment Template

A comprehensive starter template for AI methods course assignments using Python. This repository provides foundational implementations of classic AI algorithms including search algorithms (BFS, DFS, A*) and basic machine learning concepts.

## 📚 Overview

This template is designed for students learning artificial intelligence methods. It includes:

- **Search Algorithms**: Breadth-First Search (BFS), Depth-First Search (DFS), and A* Search
- **Problem Solving Framework**: Abstract classes for defining search problems
- **Example Applications**: Grid navigation problem with obstacles
- **Machine Learning**: Simple linear regression implementation from scratch
- **Comprehensive Tests**: Unit tests to verify implementations

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. **Clone the repository** (if using GitHub Classroom, accept the assignment):
   ```bash
   git clone <your-repository-url>
   cd test-python-assignment-template
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 📖 Usage

### Running the Demo

The main file includes demonstrations of the implemented algorithms:

```bash
python ai_methods.py
```

This will run:
- Search algorithm demonstrations on a grid navigation problem
- Linear regression training and prediction example

### Running Tests

Execute the test suite to verify implementations:

```bash
# Run all tests
pytest test_ai_methods.py

# Run with verbose output
pytest test_ai_methods.py -v

# Run with coverage report
pytest test_ai_methods.py --cov=ai_methods --cov-report=html
```

## 📁 Project Structure

```
test-python-assignment-template/
├── ai_methods.py          # Main implementation file
├── test_ai_methods.py     # Unit tests
├── requirements.txt       # Python dependencies
├── .gitignore            # Git ignore rules
└── README.md             # This file
```

## 🧩 Core Components

### Search Algorithms

#### 1. Breadth-First Search (BFS)
- **Completeness**: Yes (if branching factor is finite)
- **Optimality**: Yes (for unweighted graphs)
- **Time Complexity**: O(b^d)
- **Space Complexity**: O(b^d)

```python
from ai_methods import breadth_first_search, GridNavigationProblem

grid = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
problem = GridNavigationProblem(grid, start=(0, 0), goal=(2, 2))
solution = breadth_first_search(problem)
print(f"Path: {solution}")
```

#### 2. Depth-First Search (DFS)
- **Completeness**: No (can get stuck in infinite loops)
- **Optimality**: No
- **Time Complexity**: O(b^m)
- **Space Complexity**: O(bm)

```python
from ai_methods import depth_first_search

solution = depth_first_search(problem)
print(f"Path: {solution}")
```

#### 3. A* Search
- **Completeness**: Yes
- **Optimality**: Yes (with admissible heuristic)
- **Time Complexity**: Depends on heuristic
- **Space Complexity**: O(b^d)

```python
from ai_methods import a_star_search, manhattan_distance

heuristic = lambda state: manhattan_distance(state, goal=(2, 2))
solution = a_star_search(problem, heuristic)
print(f"Optimal path: {solution}")
```

### Creating Custom Problems

To define your own search problem, subclass `SearchProblem`:

```python
from ai_methods import SearchProblem

class MyProblem(SearchProblem):
    def get_start_state(self):
        return your_start_state
    
    def is_goal_state(self, state):
        return state == your_goal_state
    
    def get_successors(self, state):
        # Return list of (successor_state, action, cost) tuples
        return successors_list
    
    def get_cost_of_actions(self, actions):
        return total_cost
```

### Machine Learning

#### Linear Regression

A simple implementation using gradient descent:

```python
from ai_methods import LinearRegression

# Training data
X_train = [[1], [2], [3], [4], [5]]
y_train = [3, 5, 7, 9, 11]

# Create and train model
model = LinearRegression(learning_rate=0.01, iterations=1000)
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict([[6], [7]])
print(f"Predictions: {predictions}")
```

## 🎓 Assignment Guidelines

### For Students

1. **Study the existing code**: Understand how the algorithms work
2. **Run the demos**: Execute the code and observe the outputs
3. **Experiment**: Modify parameters and observe changes
4. **Extend the code**: Add new features or algorithms as assigned
5. **Test your work**: Write tests for any new functionality
6. **Document your changes**: Add comments and update the README

### Suggested Exercises

1. **Implement Uniform Cost Search (UCS)**
2. **Add new heuristics** (e.g., Euclidean distance)
3. **Create a new problem domain** (e.g., 8-puzzle, maze solver)
4. **Optimize the algorithms** (e.g., bidirectional search)
5. **Implement additional ML algorithms** (e.g., logistic regression, k-NN)
6. **Visualize the search process** (e.g., using matplotlib)

## 🔧 Development

### Code Style

- Follow PEP 8 guidelines
- Use type hints where appropriate
- Write docstrings for all functions and classes
- Keep functions focused and modular

### Adding New Algorithms

1. Implement the algorithm in `ai_methods.py`
2. Add corresponding tests in `test_ai_methods.py`
3. Update the README with usage examples
4. Run the test suite to ensure nothing breaks

## 📝 Common Issues

### Import Errors

If you get import errors, make sure:
- You've activated your virtual environment
- All dependencies are installed: `pip install -r requirements.txt`
- You're running commands from the project root directory

### Tests Failing

If tests fail:
- Check that you haven't modified existing functionality
- Ensure your changes don't break backward compatibility
- Review the test output for specific failure details

## 🤝 Contributing

If you find bugs or have suggestions:
1. Create an issue describing the problem
2. Submit a pull request with your fix
3. Ensure all tests pass before submitting

## 📚 Additional Resources

- [Artificial Intelligence: A Modern Approach](http://aima.cs.berkeley.edu/) by Russell & Norvig
- [Python Documentation](https://docs.python.org/3/)
- [NumPy Documentation](https://numpy.org/doc/)
- [scikit-learn Tutorials](https://scikit-learn.org/stable/tutorial/index.html)

## 📄 License

This is an educational template for course assignments. Please follow your institution's academic integrity guidelines.

## 👨‍🏫 Course Information

- **Course**: AI Methods
- **Template Version**: 1.0
- **Last Updated**: 2026

## 🆘 Getting Help

If you need help:
1. Review the code comments and docstrings
2. Check the test files for usage examples
3. Consult the course materials and textbook
4. Ask questions during office hours or on the course forum

---

**Happy Coding! 🚀**
