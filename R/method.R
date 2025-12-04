#' Extract coefficients from a tggl object
#'
#' @param object A tggl object.
#' @param s Value(s) of the penalty parameter lambda at which coefficients are required.
#' @param exact Logical. If TRUE, forces exact calculation (not implemented yet, uses interpolation).
#' @param ... Additional arguments.
#' @export
coef.tggl <- function(object, s = NULL, exact = FALSE, ...) {
  if (is.null(s)) {
    return(object$beta) # Return the entire list
  }

  # Linear interpolation logic
  lambdas <- object$lambda
  n_lam <- length(lambdas)
  beta_list <- object$beta

  # Result container
  res_coeffs <- list()

  for (val in s) {
    # 1. Check for exact match (avoid interpolation out of bounds)
    # Use a tiny tolerance for floating-point comparison
    exact_match_idx <- which(abs(lambdas - val) < 1e-10)

    if (length(exact_match_idx) > 0) {
      # If it hits a grid point exactly, return directly without interpolation
      res_coeffs[[as.character(val)]] <- beta_list[[exact_match_idx[1]]]
      next
    }

    # 2. Boundary handling and interpolation
    if (val > max(lambdas)) {
      # Larger than max lambda -> return first (usually all zeros or most sparse)
      res_coeffs[[as.character(val)]] <- beta_list[[1]]
    } else if (val < min(lambdas)) {
      # Smaller than min lambda -> return last model
      res_coeffs[[as.character(val)]] <- beta_list[[n_lam]]
    } else {
      # In the intermediate range -> linear interpolation
      # Lambdas are sorted in descending order
      # Find the nearest index to the left (larger than val)
      idx_left <- max(which(lambdas > val))
      idx_right <- idx_left + 1 # Guaranteed to exist because we excluded the < min case

      lam_l <- lambdas[idx_left]
      lam_r <- lambdas[idx_right]

      # Interpolation weight
      w <- (val - lam_r) / (lam_l - lam_r)

      beta_interp <- w * beta_list[[idx_left]] + (1 - w) * beta_list[[idx_right]]
      res_coeffs[[as.character(val)]] <- beta_interp
    }
  }

  if (length(s) == 1) return(res_coeffs[[1]])
  return(res_coeffs)
}

#' Predict method for tggl objects
#'
#' @param object A tggl object.
#' @param newx Matrix of new values for X.
#' @param s Value(s) of the penalty parameter lambda.
#' @param ... Additional arguments.
#' @export
predict.tggl <- function(object, newx, s = NULL, ...) {
  if (missing(newx)) stop("You need to supply a value for 'newx'")
  if (!is.matrix(newx)) newx <- as.matrix(newx)

  # 1. Standardize new data (using statistics from the training set)
  if (object$standardize) {
    newx <- sweep(newx, 2, object$xm, "-")
    newx <- sweep(newx, 2, object$xs, "/")
  }

  # 2. Retrieve coefficients
  if (is.null(s)) {
    betas <- object$beta
    preds <- lapply(betas, function(B) {
      p <- newx %*% B
      if (object$standardize) sweep(p, 2, object$ym, "+") else p
    })
    return(preds)
  }

  B <- coef(object, s = s)

  if (is.matrix(B)) {
    pred <- newx %*% B
    if (object$standardize) {
      pred <- sweep(pred, 2, object$ym, "+")
    }
    return(pred)
  } else if (is.list(B)) {
    preds <- lapply(B, function(b_mat) {
      p <- newx %*% b_mat
      if (object$standardize) sweep(p, 2, object$ym, "+") else p
    })
    return(preds)
  }
}

#' Predict method for cv.tggl objects
#'
#' @export
predict.cv.tggl <- function(object, newx, s = c("lambda.min"), ...) {
  if (identical(s, "lambda.min")) {
    lambda_val <- object$lambda.min
  } else if (is.numeric(s)) {
    lambda_val <- s
  } else {
    stop("Invalid 's' value.")
  }

  predict(object$tggl.fit, newx, s = lambda_val, ...)
}

#' Extract coefficients from a cv.tggl object
#'
#' @param object A cv.tggl object.
#' @param s Value(s) of the penalty parameter lambda. Default is "lambda.min".
#' @param ... Additional arguments passed to coef.tggl.
#' @export
coef.cv.tggl <- function(object, s = c("lambda.min"), ...) {
  if (identical(s, "lambda.min")) {
    lambda_val <- object$lambda.min
  } else if (is.numeric(s)) {
    lambda_val <- s
  } else {
    stop("Invalid 's' value. Must be 'lambda.min' or a numeric value.")
  }

  coef(object$tggl.fit, s = lambda_val, ...)
}
