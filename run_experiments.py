"""CirrhoTrust experiments on the Mayo Clinic PBC cohort.
Endpoint: death (status==2); liver transplant treated as censoring.
"""
import json, warnings, sys
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sksurv.util import Surv
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
from sksurv.metrics import concordance_index_censored, concordance_index_ipcw, cumulative_dynamic_auc, integrated_brier_score
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
rng_global = 2027

d = pd.read_csv(sys.argv[1])
d["sex"] = (d["sex"] == "f").astype(float)
d["event"] = (d["status"] == 2)
LOGS = ["bili", "chol", "copper", "alk.phos", "ast", "trig", "protime"]
for c in LOGS:
    d[c] = np.log(d[c])
FULL = ["age", "sex", "ascites", "hepato", "spiders", "edema", "bili", "chol", "albumin", "copper",
        "alk.phos", "ast", "trig", "platelet", "protime", "stage"]
CORE = ["age", "sex", "edema", "bili", "albumin", "platelet", "protime", "stage"]
MAYO_VARS = {"age", "bili", "albumin", "protime", "edema"}
trial = d[d.trt.notna()].reset_index(drop=True)
nontrial = d[d.trt.isna()].reset_index(drop=True)

def mayo_score(df):  # Dickson et al. 1989 Mayo risk score; bili, protime already log-transformed
    return 0.871 * df["bili"] + (-2.53) * np.log(df["albumin"]) + 0.039 * df["age"] + 2.38 * df["protime"] + 0.859 * df["edema"]

def models():
    return {
        "Cox PH": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), CoxPHSurvivalAnalysis(alpha=0.1)),
        "Random survival forest": make_pipeline(SimpleImputer(strategy="median"),
            RandomSurvivalForest(n_estimators=300, min_samples_leaf=10, max_features="sqrt", random_state=rng_global, n_jobs=-1)),
        "Gradient boosting survival": make_pipeline(SimpleImputer(strategy="median"),
            GradientBoostingSurvivalAnalysis(n_estimators=200, learning_rate=0.05, max_depth=2, subsample=0.8, random_state=rng_global)),
    }

TIMES = np.array([730, 1095, 1460, 1825, 2190, 2555])  # 2-7 years

def evaluate(model_name, risk, surv_fn, y_tr, y_te):
    out = {}
    out["harrell_c"] = concordance_index_censored(y_te["event"], y_te["time"], risk)[0]
    tau = min(y_te["time"][y_te["event"]].max(), y_tr["time"].max()) - 1
    out["uno_c"] = concordance_index_ipcw(y_tr, y_te, risk, tau=min(tau, 3650))[0]
    times = TIMES[(TIMES > y_te["time"].min()) & (TIMES < y_te["time"].max())]
    auc, mean_auc = cumulative_dynamic_auc(y_tr, y_te, risk, times)
    out["auc_2y"] = float(auc[np.where(times == 730)[0][0]]) if 730 in times else np.nan
    out["auc_5y"] = float(auc[np.where(times == 1825)[0][0]]) if 1825 in times else np.nan
    if surv_fn is not None:
        S = np.vstack([fn(times) for fn in surv_fn])
        out["ibs"] = integrated_brier_score(y_tr, y_te, S, times)
    else:
        out["ibs"] = np.nan
    return out

# ---------------- Experiment 1: internal repeated CV on trial cohort ----------------
def exp_internal(feats, n_rep=10):
    X = trial[feats].values; y = Surv.from_arrays(trial["event"].values, trial["time"].values)
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_rep, random_state=rng_global)
    rows = []
    for k, (tr, te) in enumerate(rskf.split(X, trial["event"])):
        ytr, yte = y[tr], y[te]
        r = mayo_score(trial.iloc[te]).values
        rows.append({"model": "Mayo risk score (fixed)", "fold": k, **evaluate("mayo", r, None, ytr, yte)})
        for name, m in models().items():
            m.fit(X[tr], ytr)
            r = m.predict(X[te]); sf = m.predict_survival_function(X[te])
            rows.append({"model": name, "fold": k, **evaluate(name, r, sf, ytr, yte)})
    return pd.DataFrame(rows)

# ---------------- Experiment 2: shift — train trial (core feats), test non-trial ----------------
def exp_shift():
    Xtr = trial[CORE].values; ytr = Surv.from_arrays(trial["event"].values, trial["time"].values)
    Xte = nontrial[CORE].values; yte = Surv.from_arrays(nontrial["event"].values, nontrial["time"].values)
    rows = []
    r = mayo_score(nontrial.assign(protime=nontrial.protime.fillna(trial.protime.median()))).values
    rows.append({"model": "Mayo risk score (fixed)", **evaluate("mayo", r, None, ytr, yte)})
    B = 200; rng = np.random.default_rng(rng_global)
    boot = {}
    preds = {"Mayo risk score (fixed)": (r, None)}
    for name, m in models().items():
        m.fit(Xtr, ytr); rr = m.predict(Xte); sf = m.predict_survival_function(Xte)
        preds[name] = (rr, sf)
        rows.append({"model": name, **evaluate(name, rr, sf, ytr, yte)})
    # bootstrap CI for Harrell C on the external set
    for name, (rr, _) in preds.items():
        cs = []
        for _ in range(B):
            idx = rng.integers(0, len(Xte), len(Xte))
            if yte["event"][idx].sum() < 2: continue
            cs.append(concordance_index_censored(yte["event"][idx], yte["time"][idx], rr[idx])[0])
        boot[name] = (float(np.percentile(cs, 2.5)), float(np.percentile(cs, 97.5)))
    df = pd.DataFrame(rows); df["c_ci"] = df["model"].map(lambda n: boot[n])
    return df

# ---------------- Experiment 3: conformal 5-year mortality + selective prediction ----------------
def exp_conformal(feats, n_rep=100, alpha=0.10, H=1825):
    lab = trial[(trial.time >= H) | (trial.event & (trial.time < H))].reset_index(drop=True)
    yb = ((lab.event) & (lab.time < H)).astype(int).values
    X = lab[feats].values
    res = []
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    for rep in range(n_rep):
        rng = np.random.default_rng(rep)
        idx = rng.permutation(len(lab)); n = len(idx)
        tr, ca, te = idx[: int(.6 * n)], idx[int(.6 * n): int(.8 * n)], idx[int(.8 * n):]
        for mname, clf in {"Logistic regression": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=0.5, max_iter=2000)),
                           "Random forest": make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=rep, n_jobs=-1))}.items():
            clf.fit(X[tr], yb[tr])
            pca, pte = clf.predict_proba(X[ca]), clf.predict_proba(X[te])
            for mode in ["marginal", "class-conditional"]:
                qs = {}
                if mode == "marginal":
                    s = 1 - pca[np.arange(len(ca)), yb[ca]]; m_ = len(s)
                    q = np.quantile(s, min(1, np.ceil((m_ + 1) * (1 - alpha)) / m_), method="higher"); qs = {0: q, 1: q}
                else:
                    for c in [0, 1]:
                        s = 1 - pca[yb[ca] == c, c]; m_ = len(s)
                        qs[c] = np.quantile(s, min(1, np.ceil((m_ + 1) * (1 - alpha)) / m_), method="higher")
                sets = np.stack([(1 - pte[:, c]) <= qs[c] for c in [0, 1]], 1)
                cover = sets[np.arange(len(te)), yb[te]]
                size = sets.sum(1); single = size == 1
                pred = pte.argmax(1)
                acc_all = (pred == yb[te]).mean()
                acc_single = (sets[single].argmax(1) == yb[te][single]).mean() if single.any() else np.nan
                res.append({"rep": rep, "model": mname, "mode": mode, "coverage": cover.mean(),
                            "cov_dead": cover[yb[te] == 1].mean() if (yb[te] == 1).any() else np.nan,
                            "cov_alive": cover[yb[te] == 0].mean(), "avg_size": size.mean(),
                            "singleton_rate": single.mean(), "empty_rate": (size == 0).mean(),
                            "acc_all": acc_all, "acc_singleton": acc_single,
                            "auc": roc_auc_score(yb[te], pte[:, 1]) if len(set(yb[te])) > 1 else np.nan})
    return pd.DataFrame(res), len(lab), int(yb.sum())

# ---------------- Experiment 4: risk-coverage (selective prediction) curve ----------------
def exp_risk_coverage(feats, n_rep=100, H=1825):
    from sklearn.ensemble import RandomForestClassifier
    lab = trial[(trial.time >= H) | (trial.event & (trial.time < H))].reset_index(drop=True)
    yb = ((lab.event) & (lab.time < H)).astype(int).values; X = lab[feats].values
    covs = np.linspace(0.3, 1.0, 15); curves = []
    for rep in range(n_rep):
        rng = np.random.default_rng(1000 + rep); idx = rng.permutation(len(lab)); n = len(idx)
        tr, te = idx[: int(.8 * n)], idx[int(.8 * n):]
        clf = make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=rep, n_jobs=-1))
        clf.fit(X[tr], yb[tr]); p = clf.predict_proba(X[te])[:, 1]
        conf = np.abs(p - 0.5); order = np.argsort(-conf); err = ((p > 0.5).astype(int) != yb[te])[order]
        curves.append([err[: max(1, int(round(c * len(te))))].mean() for c in covs])
    curves = np.array(curves)
    return covs, curves.mean(0), curves.std(0)

# ---------------- Experiment 5: explainability (SHAP) + Mayo variable recovery + stability ----------------
def exp_shap(feats, n_folds=5, k=5):
    import shap
    X = trial[feats]; y = Surv.from_arrays(trial["event"].values, trial["time"].values)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=rng_global)
    tops = []; mean_abs = []
    for tr, te in skf.split(X, trial["event"]):
        imp = SimpleImputer(strategy="median").fit(X.iloc[tr])
        Xtr = pd.DataFrame(imp.transform(X.iloc[tr]), columns=feats); Xte = pd.DataFrame(imp.transform(X.iloc[te]), columns=feats)
        m = GradientBoostingSurvivalAnalysis(n_estimators=200, learning_rate=0.05, max_depth=2, subsample=0.8, random_state=rng_global).fit(Xtr.values, y[tr])
        bg = shap.sample(Xtr, 50, random_state=0)
        ex = shap.Explainer(lambda a: m.predict(np.asarray(a)), bg, algorithm="permutation", seed=0)
        sv = ex(Xte, max_evals=2 * len(feats) + 1).values
        ma = np.abs(sv).mean(0); mean_abs.append(ma)
        tops.append(set(np.array(feats)[np.argsort(-ma)[:k]]))
    mean_abs = np.mean(mean_abs, 0)
    rank = pd.Series(mean_abs, index=feats).sort_values(ascending=False)
    jac = [len(a & b) / len(a | b) for i, a in enumerate(tops) for b in tops[i + 1:]]
    recov = [len(t & MAYO_VARS) / len(MAYO_VARS) for t in tops]
    return rank, float(np.mean(jac)), float(np.mean(recov)), [sorted(t) for t in tops]

if __name__ == "__main__":
    out = {"n_total": len(d), "n_trial": len(trial), "n_nontrial": len(nontrial),
           "deaths_trial": int(trial.event.sum()), "deaths_nontrial": int(nontrial.event.sum()),
           "transplant_trial": int((trial.status == 1).sum()), "transplant_nontrial": int((nontrial.status == 1).sum()),
           "median_fu_trial_days": float(trial.time.median()), "median_fu_nontrial_days": float(nontrial.time.median())}
    print("internal full..."); r1 = exp_internal(FULL); r1.to_csv("res_internal_full.csv", index=False)
    print("internal core..."); r1c = exp_internal(CORE); r1c.to_csv("res_internal_core.csv", index=False)
    print("shift..."); r2 = exp_shift(); r2.to_csv("res_shift.csv", index=False)
    print("conformal..."); r3, nlab, npos = exp_conformal(FULL); r3.to_csv("res_conformal.csv", index=False)
    out["conformal_n"] = nlab; out["conformal_pos"] = npos
    print("risk-coverage..."); covs, mu, sd = exp_risk_coverage(FULL)
    pd.DataFrame({"coverage": covs, "error_mean": mu, "error_sd": sd}).to_csv("res_riskcov.csv", index=False)
    print("shap..."); rank, jac, recov, tops = exp_shap(FULL)
    rank.to_csv("res_shap_rank.csv"); out.update({"shap_jaccard_top5": jac, "shap_mayo_recovery_top5": recov, "shap_top5_per_fold": tops})
    json.dump(out, open("res_meta.json", "w"), indent=1, default=str)
    print("done")
