# BO-AUGMECON

This project demonstrates how to use Bayesian Optimization (BO) to solve multi-objective optimization problems using the Augmented Epsilon-Constraint (AUGMECON) method. The example problem used is the Multi-Objective Knapsack Problem (MOKP).

The core idea is to use BO to intelligently select the epsilon values for the AUGMECON method, to efficiently explore the Pareto front.

## Project Components

-   `src/instance_generator.py`: A simple script to generate random instances of the MOKP. It creates profits and weights for a given number of items and objectives.

-   `src/solver.py`: This module contains the core optimization logic. It uses Gurobi to solve the knapsack problems.
    -   `get_knapsack_model`: Creates the Gurobi model for the MOKP.
    -   `solve_augmecon_subproblem`: Solves a subproblem of the AUGMECON method for a given set of epsilon values.
    -   `solve_single_objective_knapsack`: Solves the MOKP for a single objective, which can be used to find the bounds of the Pareto front.

-   `src/run.py`: This is the main script that ties everything together. It performs the following steps:
    1.  Generates a MOKP instance.
    2.  Defines a grid of epsilon values.
    3.  Selects an initial random subset of epsilon values to evaluate, creating an initial dataset.
    4.  Fits a Gaussian Process (GP) model on the initial data, mapping epsilon values to objective values.
    5.  Uses the Expected Hypervolume Improvement (EHVI) acquisition function to select the next best epsilon value to evaluate from the grid.

## Mathematical Formulations

The following formulations are implemented in `src/solver.py`.

### Multi-Objective Knapsack Problem (MOKP)

Given a set of $n$ items, each with a weight $w_i$ and $k$ different profit values $p_{ij}$ for each objective $j$, the goal is to select a subset of items that maximizes the total profit for each objective, without exceeding the knapsack's capacity $C$.

Let $x_i$ be a binary variable, where $x_i=1$ if item $i$ is selected, and $x_i=0$ otherwise.

$$ 
\begin{align*}
\text{maximize} \quad & f_j(\mathbf{x}) = \sum_{i=1}^n p_{ij} x_i, \quad \forall j \in \{1, ..., k \} \\
\text{subject to} \quad & \sum_{i=1}^n w_i x_i \le C \\
& x_i \in \{0, 1\}, \quad \forall i \in \{1, ..., n \}
\end{align*} 
$$

### Augmented Epsilon-Constraint (AUGMECON) Method

The AUGMECON method transforms a multi-objective problem into a single-objective one by keeping one objective and converting the others into constraints. The formulation in `solve_augmecon_subproblem` is as follows:

$$ 
\begin{align*}
\text{maximize} \quad & f_1(\mathbf{x}) + \rho \sum_{j=2}^k \frac{s_j}{r_j} \\
\text{subject to} \quad & \sum_{i=1}^n w_i x_i \le C \\
& f_j(\mathbf{x}) - s_j = \epsilon_j, \quad \forall j \in \{2, ..., k \} \\
& x_i \in \{0, 1\}, \quad \forall i \in \{1, ..., n \} \\
& s_j \ge 0, \quad \forall j \in \{2, ..., k \}
\end{align*} 
$$

Where:
-   $\epsilon_j$ is the desired minimum value for objective $j$.
-   $s_j$ are surplus variables, representing the amount by which objective $j$ exceeds $\epsilon_j$.
-   $\rho$ is a large penalty parameter to maximize the surplus.
-   $r_j$ is the range of objective $j$, used for normalization.

### Single-Objective Knapsack Problem

This is the standard knapsack problem, solved for a single objective $j$.

$$ 
\begin{align*}
\text{maximize} \quad & f_j(\mathbf{x}) = \sum_{i=1}^n p_{ij} x_i \\
\text{subject to} \quad & \sum_{i=1}^n w_i x_i \le C \\
& x_i \in \{0, 1\}, \quad \forall i \in \{1, ..., n \}
\end{align*} 
$$

## How to Run

To run the example, you need to have Python with Gurobi, NumPy, and BoTorch installed.

```bash
python src/run.py
```

This will execute the Bayesian Optimization loop for one iteration and print the next suggested epsilon grid point to evaluate.
