#' Cross-validation for TGGL
#'
#' Performs k-fold cross-validation using a coarse-to-fine grid search strategy.
#'
#' @param X Input matrix (n x p).
#' @param Y Response matrix (n x q).
#' @param tree Tree structure list.
#' @param lambdas User-supplied lambda sequence (optional).
#' @param nfolds Number of folds (default 5).
#' @param score_by Metric to select best lambda ("mse" or "omega").
#' @param nlambda_coarse Number of lambdas for coarse grid (default 15).
#' @param nlambda_fine Number of lambdas for fine grid (default 20).
#' @param fine_factor Range expansion factor for fine grid (default 2.0).
#' @param parallel Logical, whether to use parallel processing (future.apply).
#' @param omega Optional q x q precision matrix.
#' @param verbose Print progress.
#' @param ... Additional arguments passed to the C++ solver.
#' @importFrom stats sd predict
#' @export
cv.tggl <- function(X, Y, tree,
                    nfolds = 5,
                    lambdas = NULL,
                    score_by = c("mse", "omega"),
                    nlambda_coarse = 20,
                    nlambda_fine = 20,
                    fine_factor = 2.0,
                    omega = NULL,
                    parallel = TRUE,
                    verbose = FALSE,
                    ...) {

  score_by <- match.arg(score_by)
  n <- nrow(X); q <- ncol(Y)

  folds <- sample(rep(seq_len(nfolds), length.out = n))

  if (parallel) {
    if (!requireNamespace("future.apply", quietly = TRUE)) {
      warning("Package 'future.apply' not installed. Using sequential mode.")
      parallel <- FALSE
    }
  }

  fold_data <- lapply(1:nfolds, function(k) {
    tr <- which(folds != k)
    va <- which(folds == k)

    d <- .tggl_std_train_val(X, Y, tr, va)
    ybar_va <- matrix(d$ym, nrow = length(va), ncol = q, byrow = TRUE)
    sst_k   <- colSums((Y[va, , drop = FALSE] - ybar_va)^2)
    list(d = d, va_idx = va, sst = sst_k)
  })

  sst_total <- Reduce(`+`, lapply(fold_data, `[[`, "sst"))
  n_total <- n

  eval_on_path <- function(lam_grid, init_list = NULL) {
    worker <- function(k) {
      fd <- fold_data[[k]]
      init_B <- if (!is.null(init_list)) init_list[[k]] else NULL

      indices_list <- lapply(tree, function(x) x$index)
      weight_vec   <- vapply(tree, function(x) x$weight, numeric(1))
      node_order   <- get_postorder_indices(tree)

      fit <- bcd_tggl_fit_cpp(
        X = fd$d$Xtr_std, Y = fd$d$Ytr_ctr,
        lambdas = lam_grid,
        indices_list = indices_list,
        weight_vec = weight_vec,
        node_order = node_order,
        init = init_B,
        omega = omega,
        ...
      )

      B_list <- fit$beta
      n_lam <- length(lam_grid)
      sse_mat <- matrix(0, n_lam, q)
      score_vec <- numeric(n_lam)

      for (i in seq_len(n_lam)) {
        pred_std <- fd$d$Xva_std %*% B_list[[i]]

        pred <- sweep(pred_std, 2, fd$d$ym, "+")
        resid <- Y[fd$va_idx, , drop = FALSE] - pred

        sse_mat[i, ] <- colSums(resid^2)

        if (score_by == "omega" && !is.null(omega)) {
          score_vec[i] <- sum(resid %*% omega * resid)
        } else {
          score_vec[i] <- sum(resid^2)
        }
      }

      list(sse = sse_mat, score = score_vec, last_B = B_list[[n_lam]])
    }

    if (parallel) {
      res <- future.apply::future_lapply(1:nfolds, worker, future.seed = TRUE)
    } else {
      res <- lapply(1:nfolds, worker)
    }

    total_sse <- Reduce(`+`, lapply(res, `[[`, "sse"))
    total_score <- Reduce(`+`, lapply(res, `[[`, "score"))
    last_Bs <- lapply(res, `[[`, "last_B")

    list(sse = total_sse, score = total_score, last_Bs = last_Bs)
  }

  if (verbose) message("Step 1: Coarse Grid Search...")

  if (is.null(lambdas)) {
    Rchol <- .tggl_chol_omega(omega)
    lam_max <- .tggl_lambda_max_from(fold_data[[1]]$d$Xtr_std, fold_data[[1]]$d$Ytr_ctr, tree, Rchol)
    lam_min_ratio <- if (nrow(X) < ncol(X)) 0.05 else 0.005
    lam_grid_coarse <- exp(seq(log(lam_max), log(lam_max * lam_min_ratio), length.out = nlambda_coarse))
  } else {
    lam_grid_coarse <- sort(lambdas, decreasing = TRUE)
  }

  res_coarse <- eval_on_path(lam_grid_coarse, init_list = NULL)

  mean_scores_c <- res_coarse$score / n_total
  best_idx_c <- which.min(mean_scores_c)
  best_lam_c <- lam_grid_coarse[best_idx_c]

  if (verbose) message(sprintf("Step 2: Fine Grid Search around lambda = %.4f...", best_lam_c))

  if (is.null(lambdas)) {
    l_low <- max(best_lam_c / fine_factor, min(lam_grid_coarse))
    l_high <- min(best_lam_c * fine_factor, max(lam_grid_coarse))
    lam_grid_fine <- exp(seq(log(l_high), log(l_low), length.out = nlambda_fine))
  } else {
    lam_grid_fine <- lam_grid_coarse
    res_fine <- res_coarse
  }

  if (is.null(lambdas)) {
    res_fine <- eval_on_path(lam_grid_fine, init_list = res_coarse$last_Bs)
  }


  cv_scores <- res_fine$score / n_total
  cv_sse <- res_fine$sse

  sst_mat <- matrix(sst_total, nrow = length(lam_grid_fine), ncol = q, byrow = TRUE)
  cv_r2 <- 1 - cv_sse / sst_mat

  best_idx <- which.min(cv_scores)
  best_lambda <- lam_grid_fine[best_idx]

  if (verbose) message(sprintf("Best Lambda: %.4f", best_lambda))

  final_fit <- tggl(X, Y, tree, lambdas = lam_grid_fine, standardize = TRUE, omega = omega, ...)

  structure(list(
    tggl.fit = final_fit,
    lambda = lam_grid_fine,
    cvm = cv_scores,       # Mean Cross-Validation Error
    cv_r2 = cv_r2,         # R-squared matrix (nlambda x q)
    lambda.min = best_lambda,
    min_index = best_idx,
    score_by = score_by
  ), class = "cv.tggl")
}
