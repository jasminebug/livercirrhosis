import sys, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
d = pd.read_csv(sys.argv[1]); d["event"] = d.status == 2
t = d[d.trt.notna()]; n = d[d.trt.isna()]
def s(g):
    return {"N": len(g), "Age, median (IQR)": f"{g.age.median():.1f} ({g.age.quantile(.25):.1f}–{g.age.quantile(.75):.1f})",
            "Female, n (%)": f"{(g.sex=='f').sum()} ({100*(g.sex=='f').mean():.1f})",
            "Bilirubin mg/dL, median (IQR)": f"{g.bili.median():.1f} ({g.bili.quantile(.25):.1f}–{g.bili.quantile(.75):.1f})",
            "Albumin g/dL, median (IQR)": f"{g.albumin.median():.2f} ({g.albumin.quantile(.25):.2f}–{g.albumin.quantile(.75):.2f})",
            "Prothrombin time s, median": f"{g.protime.median():.1f}",
            "Edema present (0.5 or 1), n (%)": f"{(g.edema>0).sum()} ({100*(g.edema>0).mean():.1f})",
            "Histologic stage 4 (cirrhosis), n (%)": f"{(g.stage==4).sum()} ({100*(g.stage==4).mean():.1f})",
            "Deaths, n (%)": f"{g.event.sum()} ({100*g.event.mean():.1f})",
            "Liver transplants, n": f"{(g.status==1).sum()}",
            "Follow-up days, median": f"{g.time.median():.0f}"}
tab = pd.DataFrame({"Trial cohort (development)": s(t), "Non-trial cohort (shifted test)": s(n)})
tab.to_csv("table1.csv"); print(tab)
lr = logrank_test(t.time, n.time, t.event, n.event); print("logrank p", lr.p_value)
open("logrank.txt","w").write(str(lr.p_value))

# Fig KM
fig, ax = plt.subplots(figsize=(6, 3.6), dpi=200)
for g, lab, c in [(t, "Trial cohort (n=312)", "#2F5597"), (n, "Non-trial cohort (n=106)", "#ED7D31")]:
    KaplanMeierFitter().fit(g.time / 365.25, g.event, label=lab).plot_survival_function(ax=ax, color=c, ci_show=True)
ax.set_xlabel("Years since enrolment"); ax.set_ylabel("Survival probability (death)"); ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, fontsize=8); plt.tight_layout(); plt.savefig("fig_km.png"); plt.close()

# Fig internal performance
r = pd.read_csv("res_internal_full.csv")
order = ["Mayo risk score (fixed)", "Cox PH", "Random survival forest", "Gradient boosting survival"]
lbl = ["Mayo score\n(1989, fixed)", "Cox PH", "Random survival\nforest", "Gradient boosting\nsurvival"]
fig, axs = plt.subplots(1, 2, figsize=(8, 3.3), dpi=200)
cols = ["#7F7F7F", "#2F5597", "#70AD47", "#ED7D31"]
for ax, met, tt in [(axs[0], "harrell_c", "Harrell's C-index"), (axs[1], "auc_5y", "Time-dependent AUC at 5 years")]:
    m = r.groupby("model")[met].mean()[order]; sd = r.groupby("model")[met].std()[order]
    ax.bar(range(4), m, yerr=sd, color=cols, capsize=3); ax.set_xticks(range(4)); ax.set_xticklabels(lbl, fontsize=7)
    ax.set_ylim(0.7, 1.0); ax.set_title(tt, fontsize=9); ax.spines[["top", "right"]].set_visible(False)
    for i, v in enumerate(m): ax.text(i, 0.705, f"{v:.3f}", ha="center", fontsize=7, color="white", fontweight="bold")
plt.tight_layout(); plt.savefig("fig_internal.png"); plt.close()

# Fig risk-coverage
rc = pd.read_csv("res_riskcov.csv")
fig, ax = plt.subplots(figsize=(5.2, 3.3), dpi=200)
ax.plot(rc.coverage, rc.error_mean, "o-", color="#2F5597", ms=3)
ax.fill_between(rc.coverage, rc.error_mean - rc.error_sd, rc.error_mean + rc.error_sd, alpha=.2, color="#2F5597")
ax.set_xlabel("Coverage (fraction of patients the model answers)"); ax.set_ylabel("Error rate on answered patients")
ax.invert_xaxis(); ax.spines[["top", "right"]].set_visible(False); plt.tight_layout(); plt.savefig("fig_riskcov.png"); plt.close()

# Fig SHAP
sh = pd.read_csv("res_shap_rank.csv", index_col=0).iloc[:, 0]
mayo = {"age", "bili", "albumin", "protime", "edema"}
fig, ax = plt.subplots(figsize=(5.2, 3.8), dpi=200)
sh = sh[::-1]; ax.barh(range(len(sh)), sh.values, color=["#C00000" if k in mayo else "#A5A5A5" for k in sh.index])
ax.set_yticks(range(len(sh))); ax.set_yticklabels(sh.index, fontsize=8); ax.set_xlabel("Mean |SHAP value| (risk score units)")
ax.spines[["top", "right"]].set_visible(False)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color="#C00000", label="Variable in Mayo risk score"), Patch(color="#A5A5A5", label="Other variable")], frameon=False, fontsize=7, loc="lower right")
plt.tight_layout(); plt.savefig("fig_shap.png"); plt.close()

# Fig pipeline
fig, ax = plt.subplots(figsize=(10, 3.2), dpi=200); ax.set_xlim(0, 100); ax.set_ylim(0, 30); ax.axis("off")
def box(x, y, w, h, t, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1", fc=fc, ec="#333", lw=1))
    ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=7.5)
def ar(x1, x2, y=15): ax.annotate("", xy=(x2, y), xytext=(x1, y), arrowprops=dict(arrowstyle="-|>", color="#444"))
box(0.5, 7, 14, 16, "Mayo PBC cohort\n418 patients\n312 trial\n106 non-trial", "#E8EEF7")
box(18, 7, 15, 16, "Fold-internal\npreprocessing\nlog labs, median\nimputation, scaling", "#DCE8F5")
box(36.5, 7, 15, 16, "Survival models\nMayo score (fixed)\nCox PH, RSF,\nGB survival", "#DCE8F5")
box(55, 19, 21, 9, "Q1 Accuracy & transport\nrepeated CV + trial→non-trial", "#FFF2CC")
box(55, 10, 21, 8, "Q2 Uncertainty: conformal\n5-year mortality sets", "#FCEBD9")
box(55, 1, 21, 8, "Q3 Explanation: SHAP vs\nMayo variables, stability", "#E3F1E0")
box(80, 7, 19, 16, "Selective prediction\nanswer only confident\ncases (singleton sets,\nrisk–coverage curve)", "#F4E1EC")
ar(14.8, 18); ar(33.3, 36.5)
for yy in [23.5, 14, 5]: ax.annotate("", xy=(55, yy), xytext=(51.8, 15), arrowprops=dict(arrowstyle="-|>", color="#444"))
for yy in [23.5, 14, 5]: ax.annotate("", xy=(80, 15), xytext=(76.2, yy), arrowprops=dict(arrowstyle="-|>", color="#444"))
plt.tight_layout(); plt.savefig("fig_pipeline.png", bbox_inches="tight"); plt.close()
