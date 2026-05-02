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