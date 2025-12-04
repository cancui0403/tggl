.tggl_std_train_val <- function(X, Y, tr, va) {
  xm <- colMeans(X[tr, , drop = FALSE])
  xc <- sweep(X[tr, , drop = FALSE], 2, xm, "-")
  xs <- sqrt(colSums(xc^2) / length(tr)); xs[xs == 0] <- 1
  ym <- colMeans(Y[tr, , drop = FALSE])

  Xtr_std <- sweep(sweep(X[tr, , drop = FALSE], 2, xm, "-"), 2, xs, "/")
  Ytr_ctr <- sweep(Y[tr, , drop = FALSE], 2, ym, "-")
  Xva_std <- sweep(sweep(X[va, , drop = FALSE], 2, xm, "-"), 2, xs, "/")

  list(xm = xm, xs = xs, ym = ym,
       Xtr_std = Xtr_std, Ytr_ctr = Ytr_ctr, Xva_std = Xva_std)
}

.tggl_chol_omega <- function(omega) {
  if (is.null(omega)) return(NULL)
  Om <- 0.5 * (omega + t(omega))
  chol(Om)
}

.tggl_lambda_max_from <- function(Xs, Yc, tree, Rchol = NULL, tol_w = 1e-12) {
  Ytil <- if (is.null(Rchol)) Yc else Yc %*% Rchol
  G <- crossprod(Xs, Ytil)

  w <- vapply(tree, function(nd) nd$weight, numeric(1))
  pen_idx <- which(w > tol_w)
  if (length(pen_idx) == 0L) return(1.0)

  mx <- -Inf
  for (vid in pen_idx) {
    idx <- tree[[vid]]$index
    s   <- sqrt(rowSums(G[, idx, drop = FALSE]^2))
    mx  <- max(mx, max(s / w[vid]))
  }
  as.numeric(mx)
}

#' Get post-order indices based on group size
#'
#' For laminar families (tree-structured groups), the proximal operator
#' must be applied from the smallest groups (leaves) to the largest groups (root).
#' Instead of requiring the user to specify 'children' pointers, we simply
#' sort the groups by the number of indices they contain.
#'
#' @param tree The tree list structure.
#' @return Integer vector of indices in visitation order (Small -> Large).
#' @noRd
get_postorder_indices <- function(tree) {
  sizes <- vapply(tree, function(node) length(node$index), integer(1))

  node_order <- order(sizes, decreasing = FALSE)

  return(node_order)
}
