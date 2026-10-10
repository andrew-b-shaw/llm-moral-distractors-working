import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# The 6 comparisons tested for every non-disaggregated option: 3 vs. baseline (one-sample,
# since every row already has its own matched baseline value to pair against) and 3 between
# distractor sentiments (two-sample, since there's no row-to-row pairing between e.g. a
# specific positive-distractor instance and a specific neutral-distractor instance).
_BASELINE_COMPARISONS = [('baseline', 'positive'), ('baseline', 'neutral'), ('baseline', 'negative')]
_BETWEEN_DISTRACTOR_COMPARISONS = [('positive', 'neutral'), ('negative', 'neutral'), ('positive', 'negative')]


def _vc_formula(long_df, distractor_disagg):
    """Crossed random effects for scenario_id and distractor_id (dropping distractor_id when
    it's degenerate, e.g. a single-distractor disaggregation slice). Disaggregation pools
    multiple models' data together, where scenario_id is shared across models in a way that
    would make distractor_id's crossed effect redundant with the model fixed effect, so it's
    left out there entirely (matching the pre-existing disaggregation-specific treatment)."""
    if distractor_disagg:
        return {"scenario_var": "0 + C(scenario_id)"}
    vc = {'scenario_var': "0 + C(scenario_id)"}
    if long_df['distractor_id'].nunique() > 1:
        vc['distractor_var'] = "0 + C(distractor_id)"
    return vc


def _fit_mixed_lm(long_df, formula, vc_formula, term):
    """Crossed-random-effects fit shared by every comparison type (one-sample or two-sample,
    net change or magnitude of change): builds the model, fits it, and extracts coef/se/stat/
    pvalue for `term`."""
    model = smf.mixedlm(
        formula=formula,
        data=long_df,
        groups='dummy_group',
        # dummy_group is constant, so its default random intercept would be entirely
        # redundant with the fixed intercept; suppress it explicitly and rely only on the
        # crossed variance components above. (Does not remove the occasional "random effects
        # covariance is singular" warning statsmodels prints for this dummy-group vc_formula
        # technique when a variance component's estimate lands very close to zero -- verified
        # this doesn't correspond to unstable coef/se estimates.)
        re_formula="0",
        vc_formula=vc_formula,
        use_sparse=True,
    ).fit(disp=False, method='lbfgs')
    return {
        'coef': model.params.get(term, np.nan),
        'se': model.bse.get(term, np.nan),
        'stat': model.tvalues.get(term, np.nan),
        'pvalue': model.pvalues.get(term, np.nan),
    }


def _one_sample_test(df, diff_col, use_abs, distractor_disagg):
    """baseline-vs-condition comparison: diff ~ 1 (net change) or |diff| ~ 1 (magnitude of
    change), using each row's own paired diff from its matched baseline value. This is more
    powerful than comparing two *unpaired* groups of raw scores (score ~ condition) would be,
    since pre-subtracting the exact observed baseline cancels the dominant between-scenario
    variance component before the model ever sees the data -- the same mechanism that makes a
    paired t-test more powerful than an unpaired one for repeated measurements on the same
    unit. Verified empirically (lower SE, same point estimate as the unpaired approach)."""
    cols = ['scenario_id', 'distractor_id', diff_col] + (['model'] if distractor_disagg else [])
    long_df = df[cols].copy().rename(columns={diff_col: 'outcome'})
    long_df[['scenario_id', 'distractor_id']] = long_df[['scenario_id', 'distractor_id']].fillna("none")
    long_df = long_df.astype({'scenario_id': str, 'distractor_id': str})
    if use_abs:
        long_df['outcome'] = long_df['outcome'].abs()
    if distractor_disagg:
        long_df['model'] = long_df['model'].astype(str)
    long_df['dummy_group'] = 1
    long_df = long_df.reset_index(drop=True)

    formula = "outcome ~ C(model)" if distractor_disagg else "outcome ~ 1"
    result = _fit_mixed_lm(long_df, formula, _vc_formula(long_df, distractor_disagg), 'Intercept')
    if use_abs:
        # one-sided: H0 mean(|diff|)=0 vs Ha mean(|diff|)>0 -- |diff| can't be negative, so a
        # negative-going stat only happens from numerical noise near a true value of 0, in
        # which case the effect clearly isn't significant either way.
        two_sided_p = result['pvalue']
        result['pvalue'] = two_sided_p / 2 if result['stat'] > 0 else 1 - two_sided_p / 2
    return result


def _two_sample_test(option_results, cond_a, cond_b, diff_col, use_abs, distractor_disagg):
    """distractor-vs-distractor comparison: diff ~ condition (net change) or |diff| ~
    condition (magnitude of change), stacking each condition's own paired diff-from-baseline.
    There's no row-to-row pairing between the two conditions (a scenario's specific positive-
    distractor instance was never the same trial as its neutral-distractor instance), but
    since both conditions draw from the same underlying scenario pool, pre-subtracting each
    row's own matched baseline still cancels most of the shared between-scenario variance --
    verified empirically to produce the identical point estimate as comparing raw scores
    directly (score ~ condition), with a smaller standard error.

    Two-sided throughout, for both net and magnitude: unlike the one-sample magnitude test
    (where |diff| can't be negative, so "greater than zero" is the only sensible direction),
    a magnitude *comparison* between two conditions can legitimately go either way."""
    cols = ['scenario_id', 'distractor_id', diff_col] + (['model'] if distractor_disagg else [])
    df_a = option_results[cond_a][cols].assign(condition=cond_a)
    df_b = option_results[cond_b][cols].assign(condition=cond_b)
    long_df = pd.concat([df_a, df_b], axis=0).rename(columns={diff_col: 'outcome'})
    long_df[['scenario_id', 'distractor_id']] = long_df[['scenario_id', 'distractor_id']].fillna("none")
    long_df = long_df.astype({'scenario_id': str, 'distractor_id': str})
    if use_abs:
        long_df['outcome'] = long_df['outcome'].abs()
    if distractor_disagg:
        long_df['model'] = long_df['model'].astype(str)
    long_df['dummy_group'] = 1
    long_df = long_df.reset_index(drop=True)

    levels = sorted([cond_a, cond_b])
    term = f"condition[T.{levels[1]}]"

    if distractor_disagg:
        formula = "outcome ~ condition + C(model)"
        vc_formula = {"scenario_var": "0 + C(scenario_id)"}
    else:
        formula = "outcome ~ condition"
        vc_formula = _vc_formula(long_df, distractor_disagg)

    result = _fit_mixed_lm(long_df, formula, vc_formula, term)
    result['ref'] = levels[0]
    result['test'] = levels[1]
    return result


def aggregate_and_test(option_results, score_col, diff_col, sig_testing=False, distractor_disagg=False):
    """Shared aggregation + significance-testing entry point used by every calculate_*_results
    / disaggregate_*_results function: given per-key dataframes (each already merged against
    baseline, holding a raw score column and a baseline-subtracted diff column), computes
    mean/std/standard-error summaries for every key, and -- if sig_testing -- runs every
    baseline-vs-key and (for non-disaggregated results) key-vs-key significance test, for both
    net change (signed diff, testing whether the mean shifts) and magnitude of change (|diff|,
    testing whether responses change at all even if individual shifts cancel out in the mean).

    distractor_disagg=False tests baseline against each of the three sentiment categories and
    all pairs among them (6 comparisons x 2 test types = 12 model fits); distractor_disagg=True
    instead tests baseline against every other key present (one specific distractor at a time)
    -- there's no "distractor vs. distractor" comparison in that context, so only the
    one-sample tests apply, with model included as a fixed-effect covariate since
    disaggregation pools multiple models' data together.
    """
    mean_scores, mean_diffs = {}, {}
    std_scores, std_diffs = {}, {}
    st_error_scores, st_error_diffs = {}, {}

    for key, df in option_results.items():
        mean_scores[key] = np.mean(df[score_col])
        mean_diffs[key] = np.mean(df[diff_col])
        std_scores[key] = np.std(df[score_col])
        std_diffs[key] = np.std(df[diff_col])
        st_error_scores[key] = np.std(df[score_col]) / np.sqrt(len(df[score_col]))
        st_error_diffs[key] = np.std(df[diff_col]) / np.sqrt(len(df[diff_col]))

    sig, sig_magnitude = {}, {}
    if sig_testing:
        if distractor_disagg:
            comparisons = [('baseline', key) for key in option_results if key != 'baseline']
        else:
            comparisons = _BASELINE_COMPARISONS + _BETWEEN_DISTRACTOR_COMPARISONS

        for cond_a, cond_b in comparisons:
            label = f'{cond_a}_vs_{cond_b}'
            if cond_a == 'baseline':
                net = _one_sample_test(option_results[cond_b], diff_col, False, distractor_disagg)
                mag = _one_sample_test(option_results[cond_b], diff_col, True, distractor_disagg)
                net['ref'], net['test'] = 'baseline', cond_b
                mag['ref'], mag['test'] = 'baseline', cond_b
            else:
                net = _two_sample_test(option_results, cond_a, cond_b, diff_col, False, distractor_disagg)
                mag = _two_sample_test(option_results, cond_a, cond_b, diff_col, True, distractor_disagg)
            sig[label] = net
            sig_magnitude[label] = mag

        for key in st_error_diffs:
            if key == 'baseline':
                continue
            se = sig.get(f'baseline_vs_{key}', {}).get('se')
            if se is not None and not np.isnan(se):
                st_error_diffs[key] = se

    return {
        'mean_scores': mean_scores,
        'mean_diffs': mean_diffs,
        'std_scores': std_scores,
        'std_diffs': std_diffs,
        'st_error_scores': st_error_scores,
        'st_error_diffs': st_error_diffs,
        'sig': sig,
        'sig_magnitude': sig_magnitude,
    }
