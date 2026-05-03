import os
import matplotlib.pyplot as plt
import numpy as np

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

def plot_multi_bar_chart(
        results,
        result_keys,
        plot_labels,
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
    xs = np.arange(len(x_labels))
    offsets = np.linspace(-width * (len(distractor_keys) - 1) / 2, width * (len(distractor_keys) - 1) / 2, len(distractor_keys))

    for i, key in enumerate(result_keys):
        ax = axs if len(result_keys) == 1 else axs[i]
        plot_label = plot_labels[i]

        scores = results[key]['mean_scores'] if absolute else results[key]['mean_diffs']
        st_errors = results[key]['st_error_scores'] if absolute else results[key]['st_error_diffs']
        ys = np.array([[v[distractor] for distractor in distractor_keys] for v in scores.values()]).T
        errors = np.array([[v[distractor] for distractor in distractor_keys] for v in st_errors.values()]).T

        for j, distractor in enumerate(distractor_keys):
            ax.bar(xs + offsets[j], ys[j], width, color=color_mapping[distractor], label=distractor.capitalize())
            ax.errorbar(xs + offsets[j], ys[j], yerr=errors[j], fmt='none', color='black', capsize=capsize, capthick=capthick)

        ax.set_xticks(xs, x_labels)
        ax.set_title(plot_label)
        ax.axhline(linestyle=":", color="black")

    fig.supylabel(ylabel, x=0)
    fig.supxlabel(xlabel)
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside right upper', bbox_to_anchor=(1.07, 1))
    plt.tight_layout()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(output_dir / f"{output_filename}.png", bbox_inches='tight')
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

    ax.set_xticks(xs, x_labels)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.axhline(linestyle=":", color="black")
    ax.legend()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(output_dir / f"{output_filename}.png", bbox_inches='tight')
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

