import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import matplotlib.patches as mpatches


def _lighten_color(color, amount=0.55):
    """Blend a color toward white by `amount` (0 = unchanged, 1 = white)."""
    r, g, b = mcolors.to_rgb(color)
    return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount)


def fmt_pct(v):
    return f"{v*100:.2f}\\%"

def fmt_sci(v):
    return f"{v:.2e}"

def color_min_max(val_dict):
    """Given {col: float}, return {col: latex_string} with red/green for min/max."""
    mn = min(val_dict.values())
    mx = max(val_dict.values())
    out = {}
    for k, v in val_dict.items():
        s = fmt_pct(v)
        if v == mx:
            s = f"\\textcolor{{mygreen}}{{{s}}}"
        elif v == mn:
            s = f"\\textcolor{{red}}{{{s}}}"
        out[k] = s
    return out

def build_latex_table(rows, col_headers, caption, label):
    """
    rows: list of lists of strings (each inner list = one table row)
    col_headers: list of column header strings
    """
    n_cols = len(col_headers)
    col_fmt = "l" + "c" * (n_cols - 1)  # assume all cols except 1st are numeric
    header = " & ".join(f"\\textbf{{{h}}}" for h in col_headers) + " \\\\"
    body = "\n".join(" & ".join(r) + " \\\\" for r in rows)
    return "\n".join([
        "\\begin{table*}[!h]",
        "\\centering",
        "\\small",
        f"\\begin{{tabular}}{{{col_fmt}}}",
        "\\toprule",
        header,
        "\\midrule",
        body,
        "\\bottomrule",
        "\\end{tabular}",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\end{table*}",
    ])

def _render_header_rows(col_headers):
    """col_headers is either a flat list of strings (one header row) or a list of lists
    (several stacked header rows, e.g. "Positive vs." above "Baseline" -- an empty string
    renders as a blank cell rather than an empty \\textbf{})."""
    if col_headers and isinstance(col_headers[0], (list, tuple)):
        rows = col_headers
    else:
        rows = [col_headers]
    return "\n".join(
        " & ".join(f"\\textbf{{{h}}}" if h else "" for h in row) + " \\\\"
        for row in rows
    )

def build_sectioned_latex_table(sections, col_headers, caption, label, col_fmt=None):
    """Like build_latex_table, but groups rows into labeled sections (e.g. High-/Low-Ambiguity,
    or per-verdict) separated by a \\multicolumn divider row, matching the appendix table
    style where several related tables (e.g. across ambiguity levels) are combined into one.

    sections: list of (section_title, rows); section_title=None omits that section's divider
    (use for a single unsectioned table). col_headers: see _render_header_rows.
    """
    n_cols = len(col_headers[0]) if col_headers and isinstance(col_headers[0], (list, tuple)) else len(col_headers)
    if col_fmt is None:
        col_fmt = "l" + "c" * (n_cols - 1)
    header = _render_header_rows(col_headers)
    body_parts = []
    for i, (section_title, rows) in enumerate(sections):
        if section_title is not None:
            if i > 0:
                body_parts.append("\\midrule")
            body_parts.append(f"\\multicolumn{{{n_cols}}}{{c}}{{{section_title}}} \\\\")
            body_parts.append("\\midrule")
        body_parts.append("\n".join(" & ".join(r) + " \\\\" for r in rows))
    body = "\n".join(body_parts)
    return "\n".join([
        "\\begin{table*}[b]",
        "\\centering",
        "\\small",
        f"\\begin{{tabular}}{{{col_fmt}}}",
        "\\toprule",
        header,
        "\\midrule",
        body,
        "\\bottomrule",
        "\\end{tabular}",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\end{table*}",
    ])

def build_grouped_latex_table(groups, col_headers, caption, label, col_fmt=None):
    """Like build_sectioned_latex_table, but for the appendix's ablations-table style: a
    leading group-label column (e.g. "Size", "Instruction-Tuning") shown only on the first
    row of its group and blank on the rest, rather than a \\multicolumn divider row.

    groups: list of (group_label, rows), where each row already starts with its own model
    name (i.e. does NOT include the group-label column -- that's prepended here).
    """
    n_cols = len(col_headers[0]) if col_headers and isinstance(col_headers[0], (list, tuple)) else len(col_headers)
    if col_fmt is None:
        col_fmt = "ll" + "c" * (n_cols - 2)
    header = _render_header_rows(col_headers)
    body_lines = []
    for gi, (group_label, rows) in enumerate(groups):
        if gi > 0:
            body_lines.append("\\midrule")
        for ri, row in enumerate(rows):
            label_cell = group_label if ri == 0 else ""
            body_lines.append(" & ".join([label_cell] + row) + " \\\\")
    body = "\n".join(body_lines)
    return "\n".join([
        "\\begin{table*}[b]",
        "\\centering",
        "\\small",
        f"\\begin{{tabular}}{{{col_fmt}}}",
        "\\toprule",
        header,
        "\\midrule",
        body,
        "\\bottomrule",
        "\\end{tabular}",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\end{table*}",
    ])

def plot_multi_bar_chart(
        results,
        result_keys,
        plot_labels,
        output_filename,
        output_dir,
        figsize,
        ylabel,
        xlabel,  # global x-axis label
        verdict_keys,  # array of x-axis labels within bar groups
        verdict_labels,
        absolute,
        distractor_keys=None,
        color_mapping=None,
        width=0.2,  # width of bars
        capsize=3,  # cap width for error bars
        capthick=1,  # cap thickness for error bars
):
    # plot settings
    if distractor_keys is None:
        distractor_keys = ['baseline', 'positive', 'neutral', 'negative']
    if color_mapping is None:
        color_mapping = {
            'baseline': 'gray',
            'positive': 'green',
            'neutral': 'orange',
            'negative': 'red'
        }
    if not absolute:
        distractor_keys = distractor_keys[1:]

    # generate plot
    plt.style.use('default')
    fig, axs = plt.subplots(nrows=1, ncols=len(result_keys), figsize=figsize)
    xs = np.arange(len(verdict_keys))
    offsets = np.linspace(-width * (len(distractor_keys) - 1) / 2, width * (len(distractor_keys) - 1) / 2, len(distractor_keys))

    for i, key in enumerate(result_keys):
        ax = axs if len(result_keys) == 1 else axs[i]
        plot_label = plot_labels[i]

        scores = results[key]['mean_scores'] if absolute else results[key]['mean_diffs']
        st_errors = results[key]['st_error_scores'] if absolute else results[key]['st_error_diffs']
        ys = np.array([[scores[verdict][distractor] for distractor in distractor_keys] for verdict in verdict_keys]).T
        errors = np.array([[st_errors[verdict][distractor] for distractor in distractor_keys] for verdict in verdict_keys]).T

        for j, distractor in enumerate(distractor_keys):
            ax.bar(xs + offsets[j], ys[j], width, color=color_mapping[distractor], label=distractor.capitalize())
            ax.errorbar(xs + offsets[j], ys[j], yerr=errors[j], fmt='none', color='black', capsize=capsize, capthick=capthick)

        ax.set_xticks(xs, verdict_labels)
        ax.set_title(plot_label)
        ax.axhline(linestyle=":", color="black")

    fig.supylabel(ylabel, x=0)
    fig.supxlabel(xlabel)
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside right upper', bbox_to_anchor=(1.07, 1))
    plt.tight_layout()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(output_dir / f"{output_filename}.svg", bbox_inches="tight")
    plt.show()

def plot_multi_net_and_magnitude_bar_chart(
        results,
        result_keys,
        plot_labels,
        output_filename,
        output_dir,
        figsize,
        ylabel,
        xlabel,  # global x-axis label
        verdict_keys,  # array of x-axis labels within bar groups
        verdict_labels,
        distractor_keys=None,
        color_mapping=None,
        width=0.2,  # width of bars
        capsize=3,  # cap width for error bars
        capthick=1,  # cap thickness for error bars
):
    """Multi-subplot (one per result_key) counterpart to plot_net_and_magnitude_bar_chart,
    for results shaped like NormBank/Reddit's verdict-keyed mean_diffs/sig_magnitude (each
    subplot groups bars by verdict on the x-axis instead of showing a single group). See
    plot_net_and_magnitude_bar_chart for the magnitude-vs-net overlay rationale."""
    if distractor_keys is None:
        distractor_keys = ['positive', 'neutral', 'negative']
    if color_mapping is None:
        color_mapping = {'positive': 'green', 'neutral': 'orange', 'negative': 'red'}

    plt.style.use('default')
    fig, axs = plt.subplots(nrows=1, ncols=len(result_keys), figsize=figsize)
    xs = np.arange(len(verdict_keys))
    offsets = np.linspace(-width * (len(distractor_keys) - 1) / 2, width * (len(distractor_keys) - 1) / 2, len(distractor_keys))

    for i, key in enumerate(result_keys):
        ax = axs if len(result_keys) == 1 else axs[i]
        plot_label = plot_labels[i]

        net_ys = np.array([[results[key]['mean_diffs'][verdict][d] for d in distractor_keys] for verdict in verdict_keys]).T
        net_errs = np.array([[results[key]['st_error_diffs'][verdict][d] for d in distractor_keys] for verdict in verdict_keys]).T
        mag_ys = np.array([[results[key]['sig_magnitude'][verdict][f'baseline_vs_{d}']['coef'] for d in distractor_keys] for verdict in verdict_keys]).T
        mag_errs = np.array([[results[key]['sig_magnitude'][verdict][f'baseline_vs_{d}']['se'] for d in distractor_keys] for verdict in verdict_keys]).T
        # display each magnitude bar on the same side of zero as its net bar (magnitude is
        # always >= 0, but flipping it for display keeps each distractor's pair of bars
        # together on one side instead of splitting across zero -- see
        # plot_net_and_magnitude_bar_chart for the full rationale)
        signed_mag_ys = mag_ys * np.where(net_ys >= 0, 1, -1)

        for j, distractor in enumerate(distractor_keys):
            base_color = color_mapping[distractor]
            light_color = _lighten_color(base_color)
            ax.bar(xs + offsets[j], signed_mag_ys[j], width, color=light_color, zorder=1,
                   label=f"{distractor.capitalize()} (magnitude)")
            ax.errorbar(xs + offsets[j], signed_mag_ys[j], yerr=mag_errs[j], fmt='none', color='0.85',
                        capsize=capsize, capthick=capthick, zorder=2)
            ax.bar(xs + offsets[j], net_ys[j], width, color=base_color, zorder=3,
                   label=f"{distractor.capitalize()} (net)")
            ax.errorbar(xs + offsets[j], net_ys[j], yerr=net_errs[j], fmt='none', color='black',
                        capsize=capsize, capthick=capthick, zorder=4)

        ax.set_xticks(xs, verdict_labels)
        ax.set_title(plot_label)
        ax.axhline(linestyle=":", color="black")

    fig.supylabel(ylabel, x=0)
    fig.supxlabel(xlabel)
    handles, labels = (axs if len(result_keys) == 1 else axs[0]).get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=3)
    # reserve room at the top for the legend so it doesn't overlap the subplot titles
    plt.tight_layout(rect=(0, 0, 1, 0.88))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(output_dir / f"{output_filename}.svg", bbox_inches="tight")
    plt.show()

def plot_single_bar_chart(
        results,
        result_keys,
        output_filename,
        output_dir,
        figsize,
        ylabel,
        xlabel,  # global x-axis label
        x_labels,  # array of x-axis labels within bar groups
        absolute,
        distractor_keys=None,
        color_mapping=None,
        width=0.2,  # width of bars
        capsize=3,  # cap width for error bars
        capthick=1,  # cap thickness for error bars
        xtick_rotation=0,  # rotation (degrees) for x-axis tick labels, useful for long model names
        legend_loc=None,  # explicit legend location, useful when 'best' picks a spot that overlaps bars
):
    # plot settings
    if distractor_keys is None:
        distractor_keys = ['baseline', 'positive', 'neutral', 'negative']
    if color_mapping is None:
        color_mapping = {
            'baseline': 'gray',
            'positive': 'green',
            'neutral': 'orange',
            'negative': 'red'
        }
    if not absolute:
        distractor_keys = distractor_keys[1:]

    # generate plot
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    xs = np.arange(len(x_labels))
    offsets = np.linspace(-width * (len(distractor_keys) - 1) / 2, width * (len(distractor_keys) - 1) / 2, len(distractor_keys))

    score_key = 'mean_scores' if absolute else 'mean_diffs'
    error_key = 'st_error_scores' if absolute else 'st_error_diffs'
    ys = np.array([[results[key][score_key]['all'][distractor] for distractor in distractor_keys] for key in result_keys]).T
    errors = np.array([[results[key][error_key]['all'][distractor] for distractor in distractor_keys] for key in result_keys]).T

    for j, distractor in enumerate(distractor_keys):
        ax.bar(xs + offsets[j], ys[j], width, color=color_mapping[distractor], label=distractor.capitalize())
        ax.errorbar(xs + offsets[j], ys[j], yerr=errors[j], fmt='none', color='black', capsize=capsize, capthick=capthick)

    if xtick_rotation:
        ha = 'center' if xtick_rotation == 90 else 'right'
        ax.set_xticks(xs, x_labels, rotation=xtick_rotation, ha=ha)
    else:
        ax.set_xticks(xs, x_labels)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.axhline(linestyle=":", color="black")
    ax.legend(loc=legend_loc)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(output_dir / f"{output_filename}.svg", bbox_inches="tight")
    plt.show()

def plot_net_and_magnitude_bar_chart(
        results,
        result_keys,
        output_filename,
        output_dir,
        figsize,
        ylabel,
        xlabel,  # global x-axis label
        x_labels,  # array of x-axis labels within bar groups
        distractor_keys=None,
        color_mapping=None,
        width=0.2,  # width of bars
        capsize=3,  # cap width for error bars
        capthick=1,  # cap thickness for error bars
        xtick_rotation=0,  # rotation (degrees) for x-axis tick labels, useful for long model names
):
    """For each distractor, overlays the net (signed) mean change from baseline -- a
    full-color bar -- on top of the magnitude of that change (mean |diff|, a lighter-shade
    bar at the same position/width). mean(|diff|) >= |mean(diff)| always (Jensen's
    inequality), so the lighter magnitude bar is always at least as tall as the darker
    net-diff bar, making how much of a distractor's total effect survives as a net
    (non-canceling) shift visible at a glance, rather than requiring two separate charts.
    The magnitude bar (always >= 0 by construction) is displayed pointing the same
    direction as its net bar (flipped negative when the net diff is negative) purely for
    display, so a distractor's pair of bars always sits on one side of zero instead of
    splitting across it, which otherwise made charts with many groups look noisy/dense."""
    if distractor_keys is None:
        distractor_keys = ['positive', 'neutral', 'negative']
    if color_mapping is None:
        color_mapping = {'positive': 'green', 'neutral': 'orange', 'negative': 'red'}

    plt.style.use('default')
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    xs = np.arange(len(x_labels))
    offsets = np.linspace(-width * (len(distractor_keys) - 1) / 2, width * (len(distractor_keys) - 1) / 2, len(distractor_keys))

    net_ys = np.array([[results[key]['mean_diffs']['all'][d] for d in distractor_keys] for key in result_keys]).T
    net_errs = np.array([[results[key]['st_error_diffs']['all'][d] for d in distractor_keys] for key in result_keys]).T
    mag_ys = np.array([[results[key]['sig_magnitude']['all'][f'baseline_vs_{d}']['coef'] for d in distractor_keys] for key in result_keys]).T
    mag_errs = np.array([[results[key]['sig_magnitude']['all'][f'baseline_vs_{d}']['se'] for d in distractor_keys] for key in result_keys]).T
    signed_mag_ys = mag_ys * np.where(net_ys >= 0, 1, -1)

    for j, distractor in enumerate(distractor_keys):
        base_color = color_mapping[distractor]
        light_color = _lighten_color(base_color)
        ax.bar(xs + offsets[j], signed_mag_ys[j], width, color=light_color, zorder=1,
               label=f"{distractor.capitalize()} (magnitude)")
        # lighter gray (vs. the net error bar's black) so the two remain distinguishable
        # wherever a magnitude and net error bar overlap
        ax.errorbar(xs + offsets[j], signed_mag_ys[j], yerr=mag_errs[j], fmt='none', color='0.65',
                    capsize=capsize, capthick=capthick, zorder=2)
        ax.bar(xs + offsets[j], net_ys[j], width, color=base_color, zorder=3,
               label=f"{distractor.capitalize()} (net)")
        ax.errorbar(xs + offsets[j], net_ys[j], yerr=net_errs[j], fmt='none', color='black',
                    capsize=capsize, capthick=capthick, zorder=4)

    if xtick_rotation:
        ha = 'center' if xtick_rotation == 90 else 'right'
        ax.set_xticks(xs, x_labels, rotation=xtick_rotation, ha=ha)
    else:
        ax.set_xticks(xs, x_labels)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.axhline(linestyle=":", color="black")

    # matplotlib's legend fills column-major (top-to-bottom within a column, then the next
    # column), so handles in their natural per-distractor creation order -- [pos-magnitude,
    # pos-net, neu-magnitude, neu-net, neg-magnitude, neg-net] -- already lays out as
    # column 1 = positive, column 2 = neutral, column 3 = negative, with magnitude on top
    # and net on the bottom in every column. Placed above the axes so it can't overlap bars.
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=3)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(output_dir / f"{output_filename}.svg", bbox_inches="tight")
    plt.show()

def plot_single_horizontal_bar_chart(
        results,
        output_filename,
        output_dir,
        figsize,
        ylabel,
        xlabel,  # global x-axis label
        absolute,
        capsize=3,  # cap width for error bars
        capthick=1,  # cap thickness for error bars
):
    # plot settings
    plt.style.use('default')
    bar_labels = list(results['all']['mean_scores'].keys())
    bar_labels.sort()
    if not absolute:
        bar_labels = bar_labels[1:]
    bar_colors = ['red'] * 10 + ['orange'] * 10 + ['green'] * 10

    # generate plot
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    score_key = 'mean_scores' if absolute else 'mean_diffs'
    error_key = 'st_error_scores' if absolute else 'st_error_diffs'
    ys = np.array([results['all'][score_key][distractor] for distractor in bar_labels])
    errors = np.array([results['all'][error_key][distractor] for distractor in bar_labels])

    ax.bar(bar_labels, ys, yerr=errors, align='center', color=bar_colors, error_kw={'capsize': capsize, 'capthick': capthick})
    # ax.xaxis.set_inverted(True)  # arrange data from top to bottom
    ax.set_ylabel(xlabel)
    ax.set_xlabel(ylabel)
    ax.tick_params(axis='x', rotation=90)

    ax.axhline(linestyle=":", color="black")
    patches = [
        mpatches.Patch(color='green', label='Positive'),
        mpatches.Patch(color='orange', label='Neutral'),
        mpatches.Patch(color='red', label='Negative'),
    ]
    ax.legend(handles=patches)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(output_dir / f"{output_filename}.svg", bbox_inches="tight")
    plt.show()

def plot_multi_horizontal_bar_chart(
        results,
        output_filename,
        output_dir,
        figsize,
        ylabel,
        xlabel,  # global x-axis label
        absolute,
        capsize=3,  # cap width for error bars
        capthick=1,  # cap thickness for error bars
):
    # plot settings
    plt.style.use('default')

    for key in results.keys():
        bar_labels = list(results[key]['mean_scores'].keys())
        bar_labels.sort()
        if not absolute:
            bar_labels = bar_labels[1:]
        bar_colors = ['red'] * 10 + ['orange'] * 10 + ['green'] * 10

        # generate plot
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

        score_key = 'mean_scores' if absolute else 'mean_diffs'
        error_key = 'st_error_scores' if absolute else 'st_error_diffs'
        ys = np.array([results[key][score_key][distractor] for distractor in bar_labels])
        errors = np.array([results[key][error_key][distractor] for distractor in bar_labels])

        ax.bar(bar_labels, ys, yerr=errors, align='center', color=bar_colors, error_kw={'capsize': capsize, 'capthick': capthick})
        # ax.xaxis.set_inverted(True)  # arrange data from top to bottom
        ax.set_ylabel(xlabel)
        ax.set_xlabel(ylabel)
        ax.set_title(key)
        ax.tick_params(axis='x', rotation=90)

        ax.axhline(linestyle=":", color="black")
        patches = [
            mpatches.Patch(color='green', label='Positive'),
            mpatches.Patch(color='orange', label='Neutral'),
            mpatches.Patch(color='red', label='Negative'),
        ]
        ax.legend(handles=patches)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(output_dir / f"{output_filename}_{key}.svg", bbox_inches="tight")
        plt.show()

def generate_spider_plot(
        ax,
        scores,
        x_labels,
        title,
        color_mapping=None,
        distractors=None,
        ylim=None,
        linewidth=2,  # width of outline
        alpha=0.25  # transparency of fill
):
    if color_mapping is None:
        color_mapping = {
            'baseline': 'black',
            'positive': 'green',
            'neutral': 'orange',
            'negative': 'red'
        }
    if distractors is None:
        distractors = ['baseline', 'positive', 'neutral', 'negative']

    ys = dict([(k, list(v.values())) for k, v in scores.items()])
    num_vars = len(x_labels)

    # split the circle into even parts and save the angles
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

    # "complete the loop" by adding the start to the end
    angles += angles[:1]
    for k, v in ys.items():
        ys[k] += v[:1]

    # plot
    for distractor in distractors:
        ax.plot(angles, ys[distractor], color=color_mapping[distractor], linewidth=linewidth, label=distractor.capitalize())
        ax.fill(angles, ys[distractor], color=color_mapping[distractor], alpha=alpha)

    # set axis labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(x_labels)
    ax.tick_params(axis='x', pad=20)

    if ylim is not None:
        ax.set_ylim(**ylim)
    ax.set_title(title, y=1.1)
    plt.tight_layout()
