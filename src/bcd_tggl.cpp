// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
using namespace Rcpp;

static inline void build_groups_0based(
    const Rcpp::List& indices_list,
    std::vector<arma::uvec>& groups
){
  const int m = indices_list.size();
  groups.resize(m);
  for (int k = 0; k < m; ++k) {
    Rcpp::NumericVector Gk = indices_list[k]; // 1-based
    arma::uvec idx(Gk.size());
    for (int t = 0; t < (int)Gk.size(); ++t) idx[t] = (arma::uword)(Gk[t] - 1);
    groups[k] = std::move(idx);
  }
}

static inline void prox_tree_l2_laminar_inplace(
    arma::rowvec& z,
    const std::vector<arma::uvec>& groups,
    const Rcpp::IntegerVector& node_order,
    const arma::vec& tau_base,
    const double scale
){
  const int K = node_order.size();
  for (int kk = 0; kk < K; ++kk) {
    const int vidx = node_order[kk] - 1; // 0-based
    const arma::uvec& idx = groups[vidx];

    double s2 = 0.0;
    for (arma::uword t = 0; t < idx.n_elem; ++t) {
      const double v = z[idx[t]];
      s2 += v * v;
    }
    const double s = std::sqrt(s2);
    const double tau = tau_base[vidx] * scale;

    if (s <= tau) {
      for (arma::uword t = 0; t < idx.n_elem; ++t) z[idx[t]] = 0.0;
    } else {
      const double a = 1.0 - tau / (s + 1e-32);
      for (arma::uword t = 0; t < idx.n_elem; ++t) z[idx[t]] *= a;
    }
  }
}

static inline arma::vec screening_stat_max_over_nodes(
    const arma::mat& G,                       // p x q
    const std::vector<arma::uvec>& groups,
    const arma::vec& w,
    const double tol_w = 1e-15
){
  const int p = G.n_rows;
  arma::vec smax(p, arma::fill::zeros);

  for (size_t v = 0; v < groups.size(); ++v) {
    const double wv = (double) w[v];
    if (!(wv > tol_w)) continue;

    const arma::uvec& idx = groups[v];
    arma::vec acc(p, arma::fill::zeros); // sum_{t in Gv} G_{j,t}^2 for each j

    for (arma::uword c = 0; c < idx.n_elem; ++c) {
      const arma::vec col = G.col(idx[c]);
      acc += col % col;
    }
    acc = arma::sqrt(acc); // Row-wise L2 norm

    const double invw = 1.0 / wv;
    for (int j = 0; j < p; ++j) {
      const double val = acc[j] * invw;
      if (val > smax[j]) smax[j] = val;
    }
  }
  return smax;
}


static inline double spectral_norm_upper(const arma::mat& Om, int iters = 20)
{
  if (Om.n_rows == 0) return 1.0;
  arma::vec x = arma::randu<arma::vec>(Om.n_cols);
  x /= arma::norm(x, 2) + 1e-32;
  double val = 1.0;
  for (int t = 0; t < iters; ++t) {
    arma::vec y = Om * x;
    double n = arma::norm(y, 2) + 1e-32;
    x = y / n;
    val = arma::as_scalar(x.t() * (Om * x));
  }
  return std::max(val, 1e-12);
}

static inline void bcd_single_lambda_B(
    const arma::mat& X,                    // n x p
    const arma::mat& Y,                    // n x q
    const double lambda,
    const std::vector<arma::uvec>& groups,
    const Rcpp::IntegerVector& node_order,
    const arma::vec& w,
    const arma::vec& s_j,                  // ||x_j||^2
    const arma::vec& inv_s_j,
    const arma::mat* Omega_ptr,
    arma::mat& B,
    arma::mat& Rmat,
    std::vector<uint8_t>& active,          // [in/out] Active set flags (p)
    std::vector<int>& active_idx,          // [in/out] Active set indices
    const arma::vec* sstat_prev,
    const double lambda_prev,
    const int max_epoch,
    const double tol,
    const bool verbose
){
  const int p = X.n_cols;
  const int q = Y.n_cols;
  (void)q;

  const arma::mat& Omega = (Omega_ptr ? *Omega_ptr : arma::mat()); // Potentially empty
  const bool use_omega = (Omega_ptr != nullptr);

  // *** Residual right-multiplied by Omega, used for gradient and KKT
  arma::mat Rtil = use_omega ? (Rmat * Omega) : Rmat;

  // *** Upper bound of spectral norm of Omega (for row step size)
  const double L_Omega = use_omega ? spectral_norm_upper(Omega) : 1.0;

  const double epsKKT = 1e-8;

  // tau_base[v] = lambda * w_v
  arma::vec tau_base(groups.size());
  for (size_t v = 0; v < groups.size(); ++v) tau_base[v] = lambda * w[v];

  // ===== Initialize Active Set =====
  if (active_idx.empty()) {
    for (int j = 0; j < p; ++j) {
      if (arma::norm(B.row(j), 2) > 0) {
        if (!active[j]) { active[j] = 1; active_idx.push_back(j); }
      }
    }
  }
  // Strong Rules
  if (lambda_prev > 0.0 && sstat_prev != nullptr) {
    double thr = std::max(0.0, 2.0*lambda - lambda_prev);
    const arma::vec& Sprev = *sstat_prev; // Length p
    for (int j = 0; j < p; ++j) {
      if (!active[j] && Sprev[j] > thr) {
        active[j] = 1; active_idx.push_back(j);
      }
    }
  }

  // ===== Outer epoch loop =====
  for (int it = 0; it < max_epoch; ++it) {
    double max_row_change = 0.0;

    // Scan active blocks
    for (size_t a = 0; a < active_idx.size(); ++a) {
      int j = active_idx[a];

      arma::rowvec bj = B.row(j);
      arma::rowvec g  = X.col(j).t() * Rtil;
      double alpha_j  = inv_s_j[j] / L_Omega;
      arma::rowvec v  = bj - alpha_j * g;

      // *** Prox acts directly on b_j, using step size as scale
      prox_tree_l2_laminar_inplace(v, groups, node_order, tau_base, /*scale=*/alpha_j);

      arma::rowvec delta = v - bj;
      double dn2 = arma::dot(delta, delta);
      if (dn2 > 0) {
        // Rank-1 update for residual and right-multiplied residual
        Rmat += X.col(j) * delta;                      // R <- R + x_j delta^T
        if (use_omega) {
          Rtil += X.col(j) * (delta * Omega);          // R Omega <- R Omega + x_j (delta^T Omega)
        } else {
          Rtil += X.col(j) * delta;
        }
        B.row(j) = v;
        if (dn2 > max_row_change) max_row_change = dn2;
      }
    }

    // Convergence check (relative max row change)
    const double Bn  = arma::norm(B, "fro") + 1e-16;
    const double rel = std::sqrt(max_row_change) / Bn;
    if (verbose) Rcpp::Rcout << " epoch " << (it+1) << " rel=" << rel << std::endl;

    // ===== KKT Check: Find violators not in active set and add them =====
    arma::mat G = X.t() * Rtil;
    arma::vec S = screening_stat_max_over_nodes(G, groups, w);

    bool added = false;
    for (int j = 0; j < p; ++j) {
      if (!active[j] && S[j] > lambda*(1.0 + epsKKT)) {
        active[j] = 1; active_idx.push_back(j);
        added = true;
      }
    }
    if (!added && rel < tol) break;
  }
}


// [[Rcpp::export]]
Rcpp::List bcd_tggl_fit_cpp(
    const arma::mat& X,
    const arma::mat& Y,
    const Rcpp::NumericVector& lambdas,
    const Rcpp::List& indices_list,
    const Rcpp::NumericVector& weight_vec,
    const Rcpp::IntegerVector& node_order,
    const int max_iter = 100,
    const double tol = 1e-6,
    const bool verbose = false,
    const Rcpp::Nullable<arma::mat> init = R_NilValue,
    const Rcpp::Nullable<arma::mat> omega = R_NilValue
){
  const int n = X.n_rows;
  const int p = X.n_cols;
  const int q = Y.n_cols;

  if ((int)Y.n_rows != n) Rcpp::stop("X and Y must have same nrow.");
  if (lambdas.size() < 1) Rcpp::stop("lambdas must have length >= 1");

  arma::vec lam = Rcpp::as<arma::vec>(lambdas);
  lam = arma::sort(lam, "descend");

  // Handle Omega
  arma::mat Omega;
  const arma::mat* Omega_ptr = nullptr;
  if (omega.isNotNull()) {
    Omega = Rcpp::as<arma::mat>(omega);
    if ((int)Omega.n_rows != q || ((int)Omega.n_cols != q)) Rcpp::stop("omega must be q x q.");
    Omega = 0.5*(Omega + Omega.t());
    Omega_ptr = &Omega;
  }

  // Initialize B
  arma::mat B;
  if (init.isNotNull()) {
    B = Rcpp::as<arma::mat>(init);
    if ((int)B.n_rows != p || (int)B.n_cols != q) Rcpp::stop("init has wrong dimension.");
  } else {
    B.zeros(p, q);
  }

  // Pre-computations
  arma::mat Rmat = X * B - Y;
  arma::vec s_j(p), inv_s_j(p);
  for (int j = 0; j < p; ++j) {
    double sj = arma::dot(X.col(j), X.col(j));
    if (!(sj > 0)) sj = 1.0;
    s_j[j]     = sj;
    inv_s_j[j] = 1.0 / sj;
  }

  // Build tree structure
  std::vector<arma::uvec> groups;
  build_groups_0based(indices_list, groups);

  arma::vec w(groups.size());
  for (size_t v = 0; v < groups.size(); ++v) w[v] = (double)weight_vec[v];

  // Active Set containers
  std::vector<uint8_t> active(p, 0);
  std::vector<int>     active_idx;
  arma::vec  S_prev;
  double     lambda_prev = -1.0;

  // Result container
  Rcpp::List B_path(lam.n_elem);

  // Path Loop
  for (arma::uword k = 0; k < lam.n_elem; ++k) {
    double lambda = lam[k];
    if (verbose) Rcpp::Rcout << "Processing lambda " << (k+1) << ": " << lambda << std::endl;

    // Call core solver
    bcd_single_lambda_B(
      X, Y, lambda, groups, node_order, w,
      s_j, inv_s_j, Omega_ptr, B, Rmat, active, active_idx,
      (lambda_prev > 0.0 ? &S_prev : nullptr), lambda_prev,
      max_iter, tol, verbose
    );

    // Store result
    B_path[k] = B;

    // Calculate Strong Rule statistic for next iteration
    arma::mat Rtil = (Omega_ptr ? (Rmat * (*Omega_ptr)) : Rmat);
    arma::mat G    = X.t() * Rtil;
    S_prev         = screening_stat_max_over_nodes(G, groups, w);
    lambda_prev    = lambda;
  }

  return Rcpp::List::create(
    Rcpp::Named("beta") = B_path,
    Rcpp::Named("lambda") = lam
  );
}
