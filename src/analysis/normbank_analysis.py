import pandas as pd

from src.analysis.significance_testing import aggregate_and_test
from src.config import PATH_DISTRACTORS, PATH_CSV_RESULTS

def _score_normbank_option(distractor_dfs, option, options_ordering):
    """For each key's dataframe, merge against baseline and compute the marginal probability
    of `option` (normalized over all of options_ordering) and its baseline-subtracted diff.
    Shared by calculate_normbank_results and disaggregate_normbank_results, which differ only
    in how distractor_dfs is built."""
    verdict_prob_cols = [f'{o}_prob' for o in options_ordering]
    option_results = {}
    for k, df in distractor_dfs.items():
        df_merge = df.merge(
            distractor_dfs['baseline'][(['scenario_id'] + verdict_prob_cols)],
            on='scenario_id',
            how='left',
            suffixes=['_distractor', '_baseline']
        )

        # calculate marginal probabilities
        for condition in ['distractor', 'baseline']:
            df_merge[f'total_prob_{condition}'] = df_merge[[f'{col}_{condition}' for col in verdict_prob_cols]].sum(axis=1)
            df_merge[f'mp_{option}_{condition}'] = df_merge[f'{option}_prob_{condition}'] / df_merge[f'total_prob_{condition}']
        # calculate differences in marginal probabilities
        df_merge[f'mp_diff_{option}'] = df_merge[f'mp_{option}_distractor'] - df_merge[f'mp_{option}_baseline']

        # fold question_ordering into scenario_id for significance_testing.py's crossed
        # random-effects grouping, where present (see moralchoice_analysis.py for rationale);
        # a no-op today since NormBank has no question_ordering column, kept for consistency
        # in case that ever changes.
        if 'question_ordering' in df_merge.columns:
            df_merge['scenario_id'] = (
                df_merge['scenario_id'].astype(str) + '_' + df_merge['question_ordering'].astype(str)
            )

        option_results[k] = df_merge

    return option_results

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
    invalid_ids = response_df.loc[(response_df[[f'{option}_prob' for option in options_ordering]].sum(axis=1) <= 0.01), 'baseline_id'].tolist()
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
        option_results = _score_normbank_option(distractor_dfs, option, options_ordering)
        aggregated = aggregate_and_test(option_results, f'mp_{option}_distractor', f'mp_diff_{option}', sig_testing, distractor_disagg=False)
        results[option] = {k: aggregated[k] for k in ('mean_scores', 'mean_diffs', 'st_error_scores', 'st_error_diffs', 'sig', 'sig_magnitude')}

    return results

def disaggregate_normbank_results(
        response_df,
        sig_testing=False,
        distractor_filepath="distractors.csv",
        options_ordering=None
):
    if options_ordering is None:
        options_ordering = ['good', 'ok', 'bad']

    distractor_df = pd.read_csv(PATH_DISTRACTORS / distractor_filepath)

    # drop rows with sum of probabilities == 0
    response_df["baseline_id"] = response_df['model'].astype(str) + response_df["scenario_id"].astype(str)
    invalid_ids = response_df.loc[(response_df[[f'{option}_prob' for option in options_ordering]].sum(axis=1) == 0), 'baseline_id'].tolist()
    response_df = response_df.loc[~response_df['baseline_id'].isin(invalid_ids)]

    # join with distractor df
    response_df =  response_df.merge(distractor_df, left_on='distractor_id', right_on='id', how='left')

    distractor_ids = response_df['distractor_id'].dropna().unique()
    distractor_dfs = {}
    for distractor_id in distractor_ids:
        distractor_dfs[distractor_id] = response_df.loc[response_df['distractor_id'] == distractor_id]
    distractor_dfs['baseline'] = response_df.loc[(pd.isna(response_df['sentiment']))]

    results = {}
    for option in options_ordering:
        option_results = _score_normbank_option(distractor_dfs, option, options_ordering)
        aggregated = aggregate_and_test(option_results, f'mp_{option}_distractor', f'mp_diff_{option}', sig_testing, distractor_disagg=True)
        results[option] = {k: aggregated[k] for k in ('mean_scores', 'mean_diffs', 'st_error_scores', 'st_error_diffs', 'sig')}

    return results

def disaggregate_normbank_results_from_files(
        csv_filenames,
        sig_testing=False,
        distractor_filepath="distractors.csv",
        options_ordering=None
):
    """Same as disaggregate_normbank_results, but builds the multi-model response_df
    from a list of csv filenames itself, so the call is cacheable by results_cache."""
    dfs = []
    for i, filename in enumerate(csv_filenames):
        df = pd.read_csv(PATH_CSV_RESULTS / filename)
        df['model'] = str(i)
        dfs.append(df)
    response_df = pd.concat(dfs)
    return disaggregate_normbank_results(response_df, sig_testing, distractor_filepath, options_ordering)
