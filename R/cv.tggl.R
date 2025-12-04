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

  # 1. 构造折 (Folds)
  folds <- sample(rep(seq_len(nfolds), length.out = n))

  # 2. 准备并行环境
  if (parallel) {
    if (!requireNamespace("future.apply", quietly = TRUE)) {
      warning("Package 'future.apply' not installed. Using sequential mode.")
      parallel <- FALSE
    }
  }

  # 3. 预处理：生成每个折的标准化数据 (避免在循环中重复计算)
  fold_data <- lapply(1:nfolds, function(k) {
    tr <- which(folds != k)
    va <- which(folds == k)
    # 调用 utilities.R 中的标准化函数
    d <- .tggl_std_train_val(X, Y, tr, va)
    # 计算验证集均值用于 R2 计算
    ybar_va <- matrix(d$ym, nrow = length(va), ncol = q, byrow = TRUE)
    sst_k   <- colSums((Y[va, , drop = FALSE] - ybar_va)^2)
    list(d = d, va_idx = va, sst = sst_k)
  })

  # 汇总总 SST (用于计算 R2)
  sst_total <- Reduce(`+`, lapply(fold_data, `[[`, "sst"))
  n_total <- n

  # --- 内部评估函数 ---
  eval_on_path <- function(lam_grid, init_list = NULL) {
    worker <- function(k) {
      fd <- fold_data[[k]]
      # 获取热启动初始值
      init_B <- if (!is.null(init_list)) init_list[[k]] else NULL

      # 调用 C++ 拟合路径 (确保 bcd_tggl_fit_cpp 已正确导出)
      # 注意：C++ 接口需要索引列表和权重向量
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

      # 预测并计算误差
      B_list <- fit$beta
      n_lam <- length(lam_grid)
      sse_mat <- matrix(0, n_lam, q)
      score_vec <- numeric(n_lam)

      for (i in seq_len(n_lam)) {
        # 预测：X_va (std) * B -> 加上 Y_mean -> 得到 Yhat
        # 注意：fit$beta 是在标准化 X 和中心化 Y 上训练的
        pred_std <- fd$d$Xva_std %*% B_list[[i]]
        # 还原到原始尺度比较残差
        pred <- sweep(pred_std, 2, fd$d$ym, "+")
        resid <- Y[fd$va_idx, , drop = FALSE] - pred

        # 基础 SSE
        sse_mat[i, ] <- colSums(resid^2)

        # 评分标准 (MSE 或 Omega-weighted MSE)
        if (score_by == "omega" && !is.null(omega)) {
          score_vec[i] <- sum(resid %*% omega * resid)
        } else {
          score_vec[i] <- sum(resid^2)
        }
      }

      list(sse = sse_mat, score = score_vec, last_B = B_list[[n_lam]])
    }

    # 执行并行或串行
    if (parallel) {
      res <- future.apply::future_lapply(1:nfolds, worker, future.seed = TRUE)
    } else {
      res <- lapply(1:nfolds, worker)
    }

    # 汇总结果
    total_sse <- Reduce(`+`, lapply(res, `[[`, "sse"))
    total_score <- Reduce(`+`, lapply(res, `[[`, "score"))
    last_Bs <- lapply(res, `[[`, "last_B") # 留作下一阶段热启动

    list(sse = total_sse, score = total_score, last_Bs = last_Bs)
  }

  # --- 阶段 1: 粗网格搜索 (Coarse Grid) ---
  if (verbose) message("Step 1: Coarse Grid Search...")

  if (is.null(lambdas)) {
    # 自动生成粗网格
    Rchol <- .tggl_chol_omega(omega)
    # 取第一折的数据估算 lambda_max 即可
    lam_max <- .tggl_lambda_max_from(fold_data[[1]]$d$Xtr_std, fold_data[[1]]$d$Ytr_ctr, tree, Rchol)
    # 粗网格通常范围较大
    lam_min_ratio <- if (nrow(X) < ncol(X)) 0.05 else 0.005
    lam_grid_coarse <- exp(seq(log(lam_max), log(lam_max * lam_min_ratio), length.out = nlambda_coarse))
  } else {
    lam_grid_coarse <- sort(lambdas, decreasing = TRUE)
  }

  res_coarse <- eval_on_path(lam_grid_coarse, init_list = NULL)

  # 找到最佳索引
  mean_scores_c <- res_coarse$score / n_total
  best_idx_c <- which.min(mean_scores_c)
  best_lam_c <- lam_grid_coarse[best_idx_c]

  # --- 阶段 2: 细网格搜索 (Fine Grid) ---
  if (verbose) message(sprintf("Step 2: Fine Grid Search around lambda = %.4f...", best_lam_c))

  if (is.null(lambdas)) {
    # 在最佳点附近生成细网格
    # 范围：[best / factor, best * factor]，并限制在原范围内
    l_low <- max(best_lam_c / fine_factor, min(lam_grid_coarse))
    l_high <- min(best_lam_c * fine_factor, max(lam_grid_coarse))
    lam_grid_fine <- exp(seq(log(l_high), log(l_low), length.out = nlambda_fine))
  } else {
    # 如果用户指定了 lambdas，则不进行细搜索，直接使用用户输入的
    lam_grid_fine <- lam_grid_coarse
    res_fine <- res_coarse # 直接复用
  }

  # 只有当用户没有指定 lambdas 时才跑细网格
  if (is.null(lambdas)) {
    # 使用粗网格的最后结果作为热启动
    res_fine <- eval_on_path(lam_grid_fine, init_list = res_coarse$last_Bs)
  }

  # --- 汇总最终结果 ---
  cv_scores <- res_fine$score / n_total
  cv_sse <- res_fine$sse

  # 计算 R2: 1 - SSE / SST
  # 避免除以 0
  sst_mat <- matrix(sst_total, nrow = length(lam_grid_fine), ncol = q, byrow = TRUE)
  cv_r2 <- 1 - cv_sse / sst_mat

  best_idx <- which.min(cv_scores)
  best_lambda <- lam_grid_fine[best_idx]

  if (verbose) message(sprintf("Best Lambda: %.4f", best_lambda))

  # --- 阶段 3: 全数据拟合 (Final Fit) ---
  # 使用 tggl 主函数，这会自动处理标准化和参数
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
