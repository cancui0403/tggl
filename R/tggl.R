#' Fit Tree-Guided Group Lasso Path
#'
#' Fits a regularization path for the multi-task regression model.
#'
#' @param X Input matrix (n x p).
#' @param Y Response matrix (n x q).
#' @param tree A list structure describing the tree.
#' @param lambdas User-supplied lambda sequence.
#' @param nlambda Number of lambdas (default 50).
#' @param standardize Logical, whether to standardize data.
#' @param omega Optional q x q precision matrix.
#' @param verbose Print progress.
#' @export
tggl <- function(X, Y, tree,
                 lambdas = NULL,
                 nlambda = 50,
                 lambda.min.ratio = ifelse(nrow(X) < ncol(X), 0.01, 0.001),
                 standardize = TRUE,
                 omega = NULL,
                 verbose = FALSE,
                 max_iter = 100,
                 tol = 1e-5) {
  
  if (!is.matrix(X)) X <- as.matrix(X)
  if (!is.matrix(Y)) Y <- as.matrix(Y)
  n <- nrow(X); p <- ncol(X)
  
  # 标准化逻辑
  xm <- rep(0, p); xs <- rep(1, p); ym <- rep(0, ncol(Y))
  if (standardize) {
    xm <- colMeans(X)
    X_c <- sweep(X, 2, xm, "-")
    xs <- apply(X_c, 2, sd)
    xs[xs == 0] <- 1
    X <- sweep(X_c, 2, xs, "/")
    ym <- colMeans(Y)
    Y <- sweep(Y, 2, ym, "-")
  }
  
  # 准备树参数
  indices_list <- lapply(tree, function(x) x$index)
  weight_vec   <- vapply(tree, function(x) x$weight, numeric(1))
  node_order   <- get_postorder_indices(tree)
  
  # 生成 Lambda
  if (is.null(lambdas)) {
    Rchol <- .tggl_chol_omega(omega)
    lam_max <- .tggl_lambda_max_from(X, Y, tree, Rchol)
    lam_min <- lam_max * lambda.min.ratio
    lambdas <- exp(seq(log(lam_max), log(lam_min), length.out = nlambda))
  }
  
  # 调用 C++
  fit <- bcd_tggl_fit_cpp(
    X = X, Y = Y, lambdas = lambdas,
    indices_list = indices_list, weight_vec = weight_vec,
    node_order = node_order, max_iter = max_iter, tol = tol,
    verbose = verbose, omega = omega
  )
  
  structure(list(
    beta = fit$beta,
    lambda = fit$lambda,
    xm = xm, xs = xs, ym = ym,
    standardize = standardize,
    tree = tree
  ), class = "tggl")
}