"""Shared logic for mapping weighted mobility-profile shares by area (PC4 or municipality)."""
import matplotlib.pyplot as plt

from .io import save_figure
from .settings import MOBILITY_GROUPS

KEYS = {"Car users": "car_users", "Multimodal users": "multimodal_users", "PT users": "pt_users",
        "Infrequent car and PT users": "infrequent_car_pt_users"}
SHARE_COLS = {g: f"share_{k}_pct" for g, k in KEYS.items()}
WEIGHT_COLS = {g: f"weighted_{k}" for g, k in KEYS.items()}
N_COLS = {g: f"n_{k}" for g, k in KEYS.items()}
DOMINANT_COLOURS = {"Car users": "#440154", "Multimodal users": "#31688e", "PT users": "#35b779", "Infrequent car and PT users": "#fde725"}


def area_profile_shares(analysis, area, group="mobility_group_2", weight="FactorP"):
    """
    Per area: respondent counts and weighted totals by profile, weighted shares (summing to 100),
    the dominant profile (from raw shares) and an HHI of profile concentration.
    """
    totals = analysis.groupby(area, as_index=False).agg(n_classified_respondents=("OPID", "nunique"),
                                                        weighted_classified_population=(weight, "sum"))
    by_profile = analysis.groupby([area, group], as_index=False).agg(n_respondents=("OPID", "nunique"), weighted_respondents=(weight, "sum"))

    def wide(col, names):
        return (by_profile.pivot(index=area, columns=group, values=col).reindex(columns=MOBILITY_GROUPS)
                .fillna(0).rename(columns=names).reset_index())

    out = (totals.merge(wide("weighted_respondents", WEIGHT_COLS), on=area, how="left", validate="one_to_one")
           .merge(wide("n_respondents", N_COLS), on=area, how="left", validate="one_to_one"))

    shares = list(SHARE_COLS.values())
    for g in MOBILITY_GROUPS:
        out[SHARE_COLS[g]] = 100 * out[WEIGHT_COLS[g]] / out["weighted_classified_population"]
    out["profile_share_sum_pct"] = out[shares].sum(axis=1)
    deviation = (out["profile_share_sum_pct"] - 100).abs().max()
    print(f"Maximum deviation of profile shares from 100%: {deviation:.10f}")
    if deviation > 0.0001:
        raise ValueError("The four profile shares do not sum to 100% within an area.")

    out["dominant_profile_share_pct"] = out[shares].max(axis=1)
    out["dominant_mobility_profile"] = out[shares].idxmax(axis=1).map({v: k for k, v in SHARE_COLS.items()})
    out["mobility_profile_hhi"] = (out[shares] / 100).pow(2).sum(axis=1)
    return out, deviation


def national_summary(analysis, area_table, area_label, count_label, group="mobility_group_2", weight="FactorP"):
    """National weighted share per profile plus the distribution of area-level shares (area_label e.g. "pc4", count_label "pc4" / "municipalities")."""
    total = analysis[weight].sum()
    rows = []
    for g in MOBILITY_GROUPS:
        d, v = analysis[analysis[group].eq(g)], area_table[SHARE_COLS[g]].dropna()
        rows.append({
            "mobility_profile": g, "n_unique_respondents": d["OPID"].nunique(), "weighted_respondents": d[weight].sum(),
            "national_weighted_share_pct": 100 * d[weight].sum() / total, f"n_{count_label}_with_classified_respondents": len(v),
            **{f"{stat}_{area_label}_weighted_share_pct": val for stat, val in [
                ("mean", v.mean()), ("median", v.median()), ("p25", v.quantile(0.25)), ("p75", v.quantile(0.75)),
                ("p90", v.quantile(0.90)), ("maximum", v.max())]},
        })
    return rows


def plot_dominant_map(gdf, title, path, labels=None, edge=None):
    """Areas coloured by dominant profile (raw shares); grey where no classified respondents."""
    edge_kwargs = {"linewidth": 0.2, "edgecolor": "white"} if edge else {"linewidth": 0}
    labels = labels or {}
    fig, ax = plt.subplots(figsize=(10, 12))
    mapped = False
    for g in MOBILITY_GROUPS:
        subset = gdf[gdf["dominant_mobility_profile"].eq(g)]
        if not subset.empty:
            subset.plot(ax=ax, color=DOMINANT_COLOURS[g], label=labels.get(g, g), **edge_kwargs)
            mapped = True
    missing = gdf[gdf["dominant_mobility_profile"].isna()]
    if not missing.empty:
        missing.plot(ax=ax, color="lightgrey", label="No classified respondents", **edge_kwargs)
    ax.set_title(title, fontsize=14, pad=20)
    ax.set_axis_off()
    if mapped:
        ax.legend(title="Dominant mobility profile", loc="lower left", fontsize=9)
    save_figure(fig, path)
