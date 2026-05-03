import numpy as np
import pandas as pd

from src.analysis.significance_testing import pairwise_mixed_lm
from src.config import PATH_DISTRACTORS, PATH_CSV_RESULTS

def calculate_normbank_results(
        csv_result_filepath,
        sig_testing=False,
        distractor_filepath="distractors.csv",
        options_ordering=None
):
    if options_ordering is None:
        options_ordering = ['good', 'ok', 'bad']

    response_df = pd.read_csv(PATH_CSV_RESULTS / csv_result_filepath)
    distractor_df = pd.read_csv(PATH_DISTRACTORS / distractor_filepath)

    # drop rows with sum of probabilities == 0
    response_df['baseline_id'] = response_df['scenario_id'].astype(str)
    invalid_ids = response_df.loc[(response_df[[f'{option}_prob' for option in options_ordering]].sum(axis=1) == 0), 'baseline_id'].tolist()
    response_df = response_df.loc[~response_df['baseline_id'].isin(invalid_ids)]

    # join with distractor df
    response_df =  response_df.merge(distractor_df, left_on='distractor_id', right_on='id', how='left')

    distractor_dfs = {
        'baseline': response_df.loc[(pd.isna(response_df['sentiment']))],
        'positive': response_df.loc[(response_df['sentiment'] == 'positive')],
        'neutral': response_df.loc[(response_df['sentiment'] == 'neutral')],
        'negative': response_df.loc[(response_df['sentiment'] == 'negative')]
    }

    results = {}
    for option in options_ordering:
        option_results = {}
        for k, df in distractor_dfs.items():
            df_merge = df.merge(
                distractor_dfs['baseline'][(['scenario_id'] + [f'{option}_prob' for option in options_ordering])],
                on='scenario_id',
                how='left',
                suffixes=['_distractor', '_baseline']
            )

            # calculate marginal probabilities
            for condition in ['distractor', 'baseline']:
                df_merge[f'total_prob_{condition}'] = df_merge[[f'{option}_prob_{condition}' for option in options_ordering]].sum(axis=1)
                df_merge[f'mp_{option}_{condition}'] = df_merge[f'{option}_prob_{condition}'] / df_merge[f'total_prob_{condition}']
            # calculate differences in marginal probabilities
            df_merge[f'mp_diff_{option}'] = df_merge[f'mp_{option}_distractor'] - df_merge[f'mp_{option}_baseline']
            option_results[k] = df_merge

        mean_mps = {}
        mean_diffs = {}
        std_mps = {}
        std_diffs = {}
        st_error_mps = {}
        st_error_diffs = {}

        for distractor, df in option_results.items():
            (mean_mps[distractor], mean_diffs[distractor],
             std_mps[distractor], std_diffs[distractor],
             st_error_mps[distractor], st_error_diffs[distractor],
             sig_mps) = {}, {}, {}, {}, {}, {}, {}

            mean_mps[distractor] = np.mean(df[f'mp_{option}_distractor'])
            mean_diffs[distractor] = np.mean(df[f'mp_diff_{option}'])
            std_mps[distractor] = np.std(df[f'mp_{option}_distractor'])
            std_diffs[distractor] = np.std(df[f'mp_diff_{option}'])
            st_error_mps[distractor] = np.std(df[f'mp_{option}_distractor']) / np.sqrt(len(df[f'mp_{option}_distractor']))
            st_error_diffs[distractor] = np.std(df[f'mp_diff_{option}']) / np.sqrt(len(df[f'mp_diff_{option}']))

        # significance testing
        sig_mps = {}
        if sig_testing:
            for cond_a, cond_b in [
                ('baseline', 'positive'), ('baseline', 'neutral'), ('baseline', 'negative'),
                ('positive', 'neutral'), ('negative', 'neutral'), ('positive', 'negative')
            ]:
                sig_results = pairwise_mixed_lm(option_results, cond_a, cond_b, f'mp_{option}_distractor')
                sig_mps[f'{cond_a}_vs_{cond_b}'] = sig_results

        results[option] ={
            'results': results,
            'mean_scores': mean_mps,
            'mean_diffs': mean_diffs,
            'st_error_scores': st_error_mps,
            'st_error_diffs': st_error_diffs,
            'sig': sig_mps,
        }

    return results