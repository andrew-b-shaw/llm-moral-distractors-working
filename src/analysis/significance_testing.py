import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

def pairwise_mixed_lm(results_dict, cond_a, cond_b, score_col):
    df_a = results_dict[cond_a][['baseline_id', score_col]].assign(condition=cond_a)
    df_b = results_dict[cond_b][['baseline_id', score_col]].assign(condition=cond_b)
    long_df = pd.concat([df_a, df_b], axis=0).rename(columns={score_col: 'score'})
    model = smf.mixedlm("score ~ condition", long_df, groups=long_df['baseline_id']).fit(disp=False)
    levels = sorted([cond_a, cond_b])
    term = f"condition[T.{levels[1]}]"
    return {
        'coef':   model.params.get(term, np.nan),
        'stat':   model.tvalues.get(term, np.nan),
        'pvalue': model.pvalues.get(term, np.nan),
        'ref':    levels[0],
        'test':   levels[1],
    }