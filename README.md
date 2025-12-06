
-----

# tggl: Tree-Guided Group Lasso

**Efficient implementation of Tree-Guided Group Lasso using Block Coordinate Descent.**

`tggl` is an R package designed for high-dimensional multi-task regression problems where the features (e.g., SNPs) and tasks (e.g., tissues/cell types) follow a hierarchical tree structure. It enforces structural sparsity, allowing for the selection of variables that are shared across groups of tasks or specific to individual tasks.

## Key Features

  * **High Performance**: Implemented in C++ via `Rcpp` and `RcppArmadillo`.
  * **Scalable Algorithm**: Uses **Block Coordinate Descent (BCD)** with **Active Set** strategies and **Strong Rules**, achieving linear complexity relative to the number of features ($O(p)$).
  * **Flexible Tree Structure**: Supports arbitrary hierarchical structures (Root $\to$ Intermediate Groups $\to$ Leaves).
  * **Smart Tuning**: Implements a **"Coarse-to-Fine"** grid search strategy for Cross-Validation (CV) to find the optimal penalty parameter fast and accurately.
  * **Visualization**: Built-in support for visualizing coefficient patterns (heatmaps).

To address your request, here is a detailed **Problem Formulation** section designed to be placed at the very beginning of your README, just after the main title and brief description but before "Key Features".

This formulation clarifies the mathematical optimization problem `tggl` solves, defining the objective function and the tree-guided penalty term.

***

## Problem Formulation

Consider a multi-task regression problem with $n$ samples, $p$ features (predictors), and $q$ tasks (responses). Let $Y \in \mathbb{R}^{n \times q}$ be the response matrix, $X \in \mathbb{R}^{n \times p}$ be the design matrix, and $B \in \mathbb{R}^{p \times q}$ be the matrix of regression coefficients.

The **Tree-Guided Group Lasso (TGGL)** estimates the coefficient matrix $B$ by minimizing the following convex objective function:

$$
\min_{B} \underbrace{\frac{1}{2} \|Y - XB\|_F^2}_{\text{Loss Function}} + \underbrace{\lambda \sum_{j=1}^p \Omega_{\text{tree}}(B_{j\cdot})}_{\text{Structured Penalty}}
$$

### The Tree-Guided Penalty

The penalty term $\Omega_{\text{tree}}$ is defined based on a hierarchical tree $\mathcal{T}$ over the tasks. For a specific feature (row) $j$, the penalty is a weighted sum of $L_2$ norms over groups defined by the tree nodes:

$$
\Omega_{\text{tree}}(B_{j\cdot}) = \sum_{v \in \mathcal{V}} w_v \|B_{j, \mathcal{G}_v}\|_2
$$

Where:
* $\| \cdot \|_F$ is the Frobenius norm (sum of squared errors).
* $\mathcal{V}$ is the set of nodes in the hierarchy tree (including leaves, internal nodes, and root).
* $\mathcal{G}_v$ is the set of tasks (column indices of $B$) associated with the subtree rooted at node $v$.
* $w_v$ is the penalty weight associated with node $v$.
* $\lambda$ is the global regularization parameter.

### Hierarchical Sparsity

This overlapping group structure imposes **hierarchical sparsity**. If the coefficient group for a parent node is shrunk to zero, the coefficients for all its descendant nodes are automatically forced to zero. This allows the model to select features that are effective at different granularities—for example, selecting a variant that affects all tasks (global signal) versus one that affects only a specific subgroup of tasks (local signal).


## Installation

You can install the development version of `tggl` from GitHub:

```r
# install.packages("devtools")
devtools::install_github("your_username/tggl")
```

*(Note: Replace `your_username` with your actual GitHub username)*

## Quick Start

Here is a complete example simulating a "Group-Dominant" signal structure and recovering it using `tggl`.

### 1\. Simulate Data

```r
library(tggl)

set.seed(2025)
n <- 100; p <- 200; q <- 5

# Generate Genotypes (X)
X <- matrix(rnorm(n * p), n, p)

# Generate True Coefficients (B) with Tree Structure
B_true <- matrix(0, p, q)
# - Root Signal (Affects Task 1-5)
B_true[1:5, ] <- rnorm(25)
# - Group A Signal (Affects Task 1-2 only)
B_true[6:25, 1:2] <- rnorm(40)
# - Group B Signal (Affects Task 3-5 only)
B_true[26:45, 3:5] <- rnorm(60)

# Generate Phenotypes (Y)
Y <- X %*% B_true + matrix(rnorm(n * q), n, q) * 0.5
```

### 2\. Define Tree Structure

The tree is defined as a simple list of nodes. Each node specifies which **tasks** (columns of Y) it covers and its penalty **weight**.

```r
tree <- list(
  # --- Root Node ---
  list(index = 1:5, weight = 0.2),
  
  # --- Intermediate Groups ---
  list(index = 1:2, weight = 0.5), # Group A
  list(index = 3:5, weight = 0.5), # Group B
  
  # --- Leaf Nodes (Optional but recommended for refinement) ---
  list(index = 1, weight = 0.3), list(index = 2, weight = 0.3),
  list(index = 3, weight = 0.3), list(index = 4, weight = 0.3), list(index = 5, weight = 0.3)
)
```

### 3\. Fit Model with Cross-Validation

Use `cv.tggl` to automatically select the best lambda.

```r
# Fits the model using a Coarse-to-Fine grid search
cv_fit <- cv.tggl(X, Y, tree, nfolds = 5, verbose = TRUE)

print(paste("Best Lambda:", cv_fit$lambda.min))
```

### 4\. Predict and Visualize

Extract coefficients and visualize the recovery of the sparsity pattern.

```r
# Extract coefficients at the best lambda
beta_est <- coef(cv_fit, s = "lambda.min")

# Visualization (Requires pheatmap)
if (requireNamespace("pheatmap", quietly = TRUE)) {
  library(pheatmap)
  
  # Plot Estimated Coefficients
  pheatmap(beta_est, cluster_rows = FALSE, cluster_cols = FALSE,
           main = "Estimated Coefficients (Pattern Recovered)",
           labels_col = paste0("Task ", 1:q), show_rownames = FALSE)
}
```

## Advanced Usage

### Customizing Cross-Validation

You can control the precision and speed of the cross-validation using `nlambda_coarse` and `nlambda_fine`.

```r
cv_fit <- cv.tggl(
  X, Y, tree,
  nfolds = 5,
  nlambda_coarse = 30, # Initial broad search
  nlambda_fine = 50,   # Detailed search around the best candidate
  fine_factor = 3.0,   # Zoom range for the fine search
  parallel = TRUE      # Use parallel computing (via future.apply)
)
```

### Defining the Tree List

The `tree` argument is a `list` where each element represents a node in the hierarchy (a group of tasks).

  * `index`: An integer vector indicating the column indices of `Y` (tasks) that belong to this group.
  * `weight`: A positive scalar penalty weight.
      * **Higher weight**: Stronger penalty, encourages the group to be zero (sparser).
      * **Lower weight**: Weaker penalty, allows variables to be active in this group more easily.

The package automatically handles the hierarchical order (leaves to root) internally.

This is the final section of your README, focusing on the **Algorithms and Methodology**.

It contrasts the original Variational Method (Kim & Xing) with your efficient BCD implementation, highlighting the "Exact Zeros" capability and "Linear Complexity" of `tggl`.

---

## Algorithms and Methodology

`tggl` implements a direct optimization strategy that differs significantly from the variational approximations often used in early literature. Below, we compare the two approaches to highlight the advantages of our implementation.

### 1. The Variational Approach (Kim & Xing, 2010)

The seminal work by Kim & Xing (2010, 2012) addresses the non-smooth nature of the tree-guided penalty by using a **Variational Upper Bound**. They approximate the squared penalty term using auxiliary variables $d_{j,v}$:

$$
\left(\sum_{j=1}^p \sum_{v \in \mathcal{V}} w_v \|\beta_{\mathcal{G}_v}^j\|_2\right)^2 \le \sum_{j=1}^p \sum_{v \in \mathcal{V}} \frac{w_v^2 \|\beta_{\mathcal{G}_v}^j\|_2^2}{d_{j,v}}
$$

subject to $\sum_{j,v} d_{j,v} = 1$ and $d_{j,v} \ge 0$. This formulation transforms the original non-smooth problem into a smooth **Ridge Regression (Re-weighted Least Squares)** problem, which is solved iteratively.

* **Limitation 1: No Exact Sparsity**
    The variational reformulation leads to a weighted $L_2$ (ridge-type) subproblem, which typically produces dense coefficients. Achieving exact zeros usually requires ad-hoc post-hoc thresholding or an exact non-smooth solver, as the smooth approximation asymptotically approaches zero but rarely reaches it.

* **Limitation 2: High Per-Iteration Cost**
    Each iteration requires solving large linear systems involving $X^\top X + \lambda D$ (e.g., via Cholesky factorization). This operation can be prohibitively expensive in high-dimensional genomic settings where the number of features $p$ is very large (e.g., $p > 10^5$).

---

### 2. Our Approach: BCD + Tree Proximal Operator

`tggl` solves the **original non-smooth** convex optimization problem directly, ensuring theoretical exactness and high efficiency.

#### A. Block Coordinate Descent (BCD)
We optimize the coefficient matrix $B$ using Block Coordinate Descent. The algorithm updates the coefficients row-by-row (feature-by-feature). For a fixed feature $j$, the optimization sub-problem reduces to computing the **Proximal Operator** for the tree-structured norm.

#### B. The Tree-Structured Proximal Operator
This is the core engine of `tggl`. We utilize an efficient **Dual-Path Algorithm** (Jenatton et al., 2011) to compute the exact proximal map.

* **Mechanism**: The algorithm exploits the hierarchical structure of the groups. It computes the exact projection by traversing the tree in a topological order (typically **post-order**, from leaves to the root).
* **Sequential Shrinkage**: At each node $v$, a generalized soft-thresholding operation is applied to the associated vector of coefficients. The threshold is determined by the node's weight $w_v$ and the regularization parameter $\lambda$.
* **Exact Sparsity**: A crucial advantage of this operator is its ability to set coefficients to **exactly zero**. If the group norm falls below the threshold $\lambda w_v$ at any stage of the traversal, the entire group of coefficients is zeroed out, strictly enforcing the hierarchical sparsity constraint.

#### Complexity Analysis
Per epoch, the BCD algorithm with residual updates costs $O(\text{nnz}(X) \cdot q)$ (or $O(npq)$ for dense $X$), plus $O(p_{\text{active}} \cdot |\mathcal{V}|)$ for the tree proximal operator. In high-dimensional regimes where $p \gg n, q$, and treating $n, q$ as constants, the runtime grows **approximately linearly** with the number of active features $p_{\text{active}}$. This represents a drastic improvement over the cubic complexity of solver-based variational methods.

---


**Conclusion**: By combining Block Coordinate Descent with an active set strategy and the analytical tree proximal operator, `tggl` achieves high computational efficiency while guaranteeing the theoretical properties of the Tree-Guided Group Lasso.

## License

This project is licensed under the MIT License.

-----


    


