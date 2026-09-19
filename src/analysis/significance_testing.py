import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

def pairwise_mixed_lm(results_dict, cond_a, cond_b, score_col):
    type_map = {
        'scenario_id': str,
        'distractor_id': str,
        'score': float
    }
    df_a = results_dict[cond_a][['scenario_id', 'distractor_id', score_col]].assign(condition=cond_a)
    df_b = results_dict[cond_b][['scenario_id', 'distractor_id', score_col]].assign(condition=cond_b)
    long_df = pd.concat([df_a, df_b], axis=0).rename(columns={score_col: 'score'}).astype(type_map)
    long_df = long_df.fillna("none")
    long_df['dummy_group'] = 1
    # MixedLM's vc_formula processing indexes into `data` per group via
    # DataFrame.loc, so a duplicated index (inherited from the per-rule
    # concatenation upstream) causes it to multiply rows instead of
    # selecting them, silently blowing up the design matrices.
    long_df = long_df.reset_index(drop=True)
    vc_full = {
        'scenario_var': "0 + C(scenario_id)",
        'distractor_var': "0 + C(distractor_id)"
    }
    model = smf.mixedlm(
        formula="score ~ condition",
        data=long_df,
        groups='dummy_group',
        vc_formula=vc_full,
        use_sparse=True,
    ).fit(disp=False, method='lbfgs')
    levels = sorted([cond_a, cond_b])
    term = f"condition[T.{levels[1]}]"
    return {
        'coef':   model.params.get(term, np.nan),
        'stat':   model.tvalues.get(term, np.nan),
        'pvalue': model.pvalues.get(term, np.nan),
        'ref':    levels[0],
        'test':   levels[1],
    }