
import os
from math import factorial

import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.inspection import permutation_importance

from Fico import bb, X_search, X_eval, y_search, y_eval

RUN_AXIOMS = True
RUN_ITEM2 = True
RUN_ITEM3 = True
RUN_ITEM4 = True
RUN_ITEM5 = True

# =============================================================================
# Setup: model check, feature selection, reduced game f
# =============================================================================
try:
    shap.TreeExplainer(bb)
    print("TreeExplainer OK")
except Exception as e:
    print("TreeExplainer failed:", e)

# Top 12 features by permutation importance (on the test set)
pi = permutation_importance(bb, X_eval, y_eval, scoring="roc_auc",
                            n_repeats=5, random_state=42, n_jobs=-1)
order = np.argsort(-pi.importances_mean)
FEATS = list(X_eval.columns[order[:12]])
print("Selected features:", FEATS)

# The other features are fixed at the training mean (NaNs skipped)
FIXED = X_search.mean()


def to_full(Z):
    """Expand a (n,12) array to all columns; non-selected features fixed at FIXED."""
    full = pd.DataFrame(np.tile(FIXED.values, (len(Z), 1)), columns=X_search.columns)
    full[FEATS] = Z
    return full


def f(Z):
    """Z: (n, 12) in FEATS order -> P(Bad) of the reduced game."""
    return bb.predict_proba(to_full(Z))[:, 1]


def f_raw(Z):
    """Same reduced game, log-odds output."""
    return bb.decision_function(to_full(Z))


# =============================================================================
# Item 1: exact interventional Shapley values (no library)
# =============================================================================
N = len(FEATS)


def all_masks(n):
    """(2^n, n) boolean matrix. Row r is the subset whose bit j is (r >> j) & 1."""
    idx = np.arange(2 ** n)
    return ((idx[:, None] >> np.arange(n)) & 1).astype(bool)


def value_table(f, x, B, chunk=64):
    """v[r] = mean over background rows b of f(z), z_j = x_j if j in subset r else b_j."""
    n = len(x)
    M = all_masks(n)
    v = np.empty(len(M))
    for s in range(0, len(M), chunk):
        m = M[s:s + chunk]
        Z = np.where(m[:, None, :], x[None, None, :], B[None, :, :])
        p = f(Z.reshape(-1, n)).reshape(len(m), len(B))
        v[s:s + chunk] = p.mean(axis=1)
    return v


def shapley_exact(v, n):
    """Exact Shapley values from the full table of v over all 2^n subsets."""
    M = all_masks(n)
    sizes = M.sum(1)
    w = np.array([factorial(k) * factorial(n - k - 1) / factorial(n) for k in range(n)])
    phi = np.zeros(n)
    for i in range(n):
        without = np.where(~M[:, i])[0]
        with_i = without + (1 << i)
        phi[i] = np.sum(w[sizes[without]] * (v[with_i] - v[without]))
    return phi


def shap_values(f, x, B):
    v = value_table(f, x, B)
    return shapley_exact(v, len(x)), v


# Demo on one person with a 100-row background
B = X_search[FEATS].sample(100, random_state=0).values.astype(float)
x = X_eval[FEATS].iloc[0].values.astype(float)
phi_demo, v_demo = shap_values(f, x, B)
print(pd.Series(phi_demo, index=FEATS).round(4))
print("base value  v(empty):", v_demo[0], "  f(x) = v(full):", v_demo[-1])

# =============================================================================
# Axiom tests
# =============================================================================
if RUN_AXIOMS:
    TOL = 1e-10
    B_t = B[:20]
    results = []

    # 1) Efficiency
    phi, v = shap_values(f, x, B_t)
    err = abs(phi.sum() - (v[-1] - v[0]))
    results.append(("Efficiency", err, err < TOL))

    # 2) Dummy: a game that ignores feature j gives phi_j = 0
    j = 3

    def f_dummy(Z):
        Z = Z.copy()
        Z[:, j] = 0.0
        return f(Z)

    phi, _ = shap_values(f_dummy, x, B_t)
    results.append(("Dummy", abs(phi[j]), abs(phi[j]) < TOL))

    # 3) Symmetry: g symmetric in features a,b; with x_a = x_b and equal background columns
    a, b = 0, 5

    def f_sym(Z):
        Zs = Z.copy()
        Zs[:, [a, b]] = Z[:, [b, a]]
        return 0.5 * (f(Z) + f(Zs))

    x_s, B_s = x.copy(), B_t.copy()
    x_s[a] = x_s[b]
    B_s[:, a] = B_s[:, b]
    phi, _ = shap_values(f_sym, x_s, B_s)
    err = abs(phi[a] - phi[b])
    results.append(("Symmetry", err, err < TOL))

    # 4) Additivity
    def f2(Z):
        return np.nan_to_num(Z[:, 0]) / 100 + np.nan_to_num(Z[:, 1]) * np.nan_to_num(Z[:, 2]) / 1e4

    phi_sum, _ = shap_values(lambda Z: f(Z) + f2(Z), x, B_t)
    phi_1, _ = shap_values(f, x, B_t)
    phi_2, _ = shap_values(f2, x, B_t)
    err = np.max(np.abs(phi_sum - (phi_1 + phi_2)))
    results.append(("Additivity", err, err < TOL))

    print(pd.DataFrame(results, columns=["Axiom", "Max abs error", "Pass"]))

# =============================================================================
# Item 2: comparison with shap (interventional TreeSHAP, same 100-row background)
# =============================================================================
if RUN_ITEM2:
    B_full = to_full(B)
    idx_cols = [X_search.columns.get_loc(c) for c in FEATS]

    # --- probability output ---
    expl = shap.TreeExplainer(bb, data=B_full, feature_perturbation="interventional",
                              model_output="probability")
    print("library base value:", np.ravel(expl.expected_value), " mine:", v_demo[0])

    rows = []
    for i in range(10):
        xi = X_eval[FEATS].iloc[i].values.astype(float)
        phi_mine, vi = shap_values(f, xi, B)
        sv = np.asarray(expl.shap_values(to_full(xi[None, :]), check_additivity=False))
        if sv.ndim == 3:
            sv = sv[..., 1]
        phi_lib = pd.Series(sv[0], index=X_search.columns)
        diff = np.abs(phi_mine - phi_lib[FEATS].values)
        rows.append({"person": i,
                     "n_nan_in_x": int(np.isnan(xi).sum()),
                     "max_abs_diff": diff.max(),
                     "worst_feature": FEATS[int(diff.argmax())],
                     "max_abs_other_feats": np.abs(phi_lib.drop(FEATS)).max(),
                     "mine_efficiency_gap": phi_mine.sum() - (vi[-1] - vi[0]),
                     "lib_efficiency_gap": sv.sum() - (f(xi[None, :])[0] - f(B).mean())})
    cmp_prob = pd.DataFrame(rows)
    print(cmp_prob)
    print("LARGEST DIFFERENCE (probability):", cmp_prob.max_abs_diff.max())

    # --- log-odds output (same game as shap's tree algorithm) ---
    expl_raw = shap.TreeExplainer(bb, data=B_full, feature_perturbation="interventional",
                                  model_output="raw")
    rows = []
    for i in range(10):
        xi = X_eval[FEATS].iloc[i].values.astype(float)
        phi_mine_raw, _ = shap_values(f_raw, xi, B)
        sv = np.asarray(expl_raw.shap_values(to_full(xi[None, :]), check_additivity=False))
        if sv.ndim == 3:
            sv = sv[..., 1]
        diff = np.abs(phi_mine_raw - sv[0][idx_cols])
        rows.append({"person": i, "max_abs_diff": diff.max(),
                     "worst_feature": FEATS[int(diff.argmax())]})
    cmp_raw = pd.DataFrame(rows)
    print(cmp_raw)
    print("LARGEST DIFFERENCE (log-odds):", cmp_raw.max_abs_diff.max())
    print("lib base:", np.ravel(expl_raw.expected_value), " mine:", f_raw(B).mean())

# =============================================================================
# The three people (used by items 3, 4, 5)
# =============================================================================
p_eval = bb.predict_proba(X_eval)[:, 1]
PEOPLE = {lab: int(np.argmin(np.abs(p_eval - t)))
          for lab, t in [("low_risk", 0.10), ("borderline", 0.50), ("high_risk", 0.90)]}
print({k: (i, round(float(p_eval[i]), 3)) for k, i in PEOPLE.items()})

# =============================================================================
# Item 3: background-size experiment
# =============================================================================
if RUN_ITEM3:
    B_ref = X_search[FEATS].values.astype(float)       # reference = full training set
    print("reference background size:", len(B_ref))
    ref = {}
    for lab, i in PEOPLE.items():
        path = f"ref_{lab}.npy"                        # cached; delete if FEATS changes
        if os.path.exists(path):
            ref[lab] = np.load(path)
        else:
            xi = X_eval[FEATS].iloc[i].values.astype(float)
            ref[lab], _ = shap_values(f, xi, B_ref)
            np.save(path, ref[lab])
        print("reference ready:", lab)

    SIZES = [1, 10, 100, 500]
    SEEDS = range(10)                                  # assignment asks for at least 5
    rows = []
    for lab, i in PEOPLE.items():
        xi = X_eval[FEATS].iloc[i].values.astype(float)
        for n_bg in SIZES:
            for seed in SEEDS:
                sel = np.random.default_rng(seed).choice(len(B_ref), n_bg, replace=False)
                phi, v = shap_values(f, xi, B_ref[sel])
                rel = np.linalg.norm(phi - ref[lab]) / np.linalg.norm(ref[lab])
                rows.append(dict(person=lab, bg_size=n_bg, seed=seed, rel_dev=rel, base=v[0]))
    res = pd.DataFrame(rows)
    res.to_csv("bg_size_experiment.csv", index=False)

    summary = res.groupby(["person", "bg_size"]).rel_dev.agg(["mean", "std", "min", "max"])
    print(summary.round(4))

    plt.figure(figsize=(6, 4))
    for lab in PEOPLE:
        s = summary.loc[lab]
        plt.errorbar(s.index, s["mean"], yerr=s["std"], marker="o", capsize=3, label=lab)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("random background size")
    plt.ylabel("relative deviation (L2) from reference")
    plt.title("Shapley estimate vs background size (10 seeds)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("bg_size_deviation.png", dpi=200)
    print("saved bg_size_deviation.png")

# =============================================================================
# Item 4: reference groups
# =============================================================================
if RUN_ITEM4:
    p_train = bb.predict_proba(X_search)[:, 1]

    def tercile_mask(col, value):
        q1, q2 = X_search[col].quantile([1 / 3, 2 / 3])
        s = X_search[col]
        if np.isnan(value):
            return s.isna().values
        if value <= q1:
            return (s <= q1).values
        if value > q2:
            return (s > q2).values
        return ((s > q1) & (s <= q2)).values

    def groups_for(xi):
        d = dict(zip(FEATS, xi))
        return {
            "all_train":        np.ones(len(X_search), bool),
            "model_approved":   p_train < 0.5,
            "model_denied":     p_train >= 0.5,
            "same_history_len": tercile_mask("AverageMInFile", d["AverageMInFile"]),
            "same_risk_score":  tercile_mask("ExternalRiskEstimate", d["ExternalRiskEstimate"]),
        }

    THRESH = 0.002     # ignore sign flips among |phi| below this
    for lab, i in PEOPLE.items():
        xi = X_eval[FEATS].iloc[i].values.astype(float)
        G = groups_for(xi)
        cols_phi, bases, sizes = {}, {}, {}
        for g, m in G.items():
            Bg = X_search[FEATS].values[m].astype(float)
            phi, v = shap_values(f, xi, Bg)
            cols_phi[g], bases[g], sizes[g] = phi, float(v[0]), int(m.sum())
        df = pd.DataFrame(cols_phi, index=FEATS)
        big = df.abs() >= THRESH
        pos = (df > 0) & big
        neg = (df < 0) & big
        df["sign_flip"] = (pos.any(axis=1) & neg.any(axis=1)).map({True: "FLIP", False: ""})
        print(f"\n=== {lab} (test row {i}, f(x) = {v[-1]:.4f}) ===")
        print("group sizes:", sizes)
        print("base values:", {g: round(b, 4) for g, b in bases.items()})
        print(df.round(4))
        df.assign(person=lab).to_csv(f"groups_{lab}.csv")

# =============================================================================
# Item 5: path-dependent vs interventional TreeSHAP (full model, log-odds)
# =============================================================================
if RUN_ITEM5:
    def raw_sv(expl, X):
        sv = np.asarray(expl.shap_values(X, check_additivity=False))
        return sv[..., 1] if sv.ndim == 3 else sv

    def interventional(X, bg, chunk=100):
        """Interventional TreeSHAP in log-odds with the FULL background bg.

        shap subsamples any background to 100 rows, so we average over chunks of <= 100
        rows. Interventional values are an average over background rows, so the
        size-weighted average of the chunk results is exactly the full-background value.
        Returns (shap values, base value)."""
        sv, base, n = 0.0, 0.0, len(bg)
        for s in range(0, n, chunk):
            part = bg.iloc[s:s + chunk]
            e = shap.TreeExplainer(bb, data=part, feature_perturbation="interventional",
                                   model_output="raw")
            w = len(part) / n
            sv = sv + w * raw_sv(e, X)
            base += w * float(np.ravel(e.expected_value)[0])
        return sv, base

    expl_pd = shap.TreeExplainer(bb)            # path-dependent, log-odds
    base_pd = float(np.ravel(expl_pd.expected_value)[0])

    _, base_iv = interventional(X_eval.iloc[:1], X_search)
    print("base (log-odds): path-dependent", base_pd, "| interventional", base_iv,
          "| check, mean decision_function over X_search:",
          float(bb.decision_function(X_search).mean()))

    for lab, i in PEOPLE.items():
        xrow = X_eval.iloc[[i]]
        fx = float(bb.decision_function(xrow)[0])
        phi_pd = raw_sv(expl_pd, xrow)[0]
        sv_iv, _ = interventional(xrow, X_search)
        phi_iv = sv_iv[0]
        t = pd.DataFrame({"path_dependent": phi_pd, "interventional": phi_iv},
                         index=X_search.columns)
        t["diff"] = t.path_dependent - t.interventional
        print(f"\n=== {lab} (test row {i}, full-model log-odds = {fx:.4f}) ===")
        print(t.loc[FEATS].round(4))
        print("sum of all other features:",
              t.drop(FEATS)[["path_dependent", "interventional"]].sum().round(4).to_dict())
        print("efficiency gap  PD:", phi_pd.sum() + base_pd - fx,
              " IV:", phi_iv.sum() + base_iv - fx)
        print("max |diff| (all features):", t["diff"].abs().max(), "at", t["diff"].abs().idxmax())
        print("Spearman rank corr:", spearmanr(phi_pd, phi_iv)[0])
        print("top-3 PD:", list(t.path_dependent.abs().nlargest(3).index))
        print("top-3 IV:", list(t.interventional.abs().nlargest(3).index))
        strong = t[["path_dependent", "interventional"]].abs().max(axis=1) >= 0.01
        print("sign disagreements (|phi| >= 0.01):",
              t[strong & (np.sign(t.path_dependent) != np.sign(t.interventional))].index.tolist())
        t.to_csv(f"treeshap_pd_vs_iv_{lab}.csv")

    # Average gap over 100 random test rows, 500-row interventional background
    rs = np.random.default_rng(0)
    Xs = X_eval.iloc[rs.choice(len(X_eval), 100, replace=False)]
    bg500 = X_search.sample(500, random_state=0)
    sv500, _ = interventional(Xs, bg500)
    D = np.abs(raw_sv(expl_pd, Xs) - sv500)
    print("100 rows (500-row background): mean |diff| =", D.mean(),
          " mean of per-row max |diff| =", D.max(axis=1).mean())