
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

## Methodology

The solver transforms the Tree-Guided Group Lasso problem into a summation of group norms and solves it using **Block Coordinate Descent**.

  * **Variable Update**: Each row of coefficients $B_{j \cdot}$ is updated analytically via a proximal operator for tree-structured norms.
  * **Active Set**: The algorithm iterates only over the non-zero (active) set of variables for efficiency.
  * **Strong Rules**: Safe screening rules are applied to discard predictors that are likely to be zero before fitting.

## License

This project is licensed under the MIT License.

-----

