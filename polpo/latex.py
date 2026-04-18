def join_table_rows(rows):
    return "\\\\\n".join(rows)


def add_table(table_name):
    table_template = r"""\begin{{table}}[H]
\centering
\input{{{table_name}}}
\end{{table}}"""
    return table_template.format(table_name=table_name)


def add_figure(figure_name, width=1.0):
    per_subject_fig_template = r"""\begin{{figure}}[H]
\centering
\includegraphics[width={width}\linewidth]{{{figure_name}}}
\end{{figure}}"""
    return per_subject_fig_template.format(
        figure_name=figure_name,
        width=width,
    )


def add_subfigure(figure_name, width=1.0):
    template = r"""\begin{{subfigure}}{{{width}\textwidth}}\hfill
\centering
\includegraphics[width=1.\linewidth]{{{figure_name}}}
\end{{subfigure}}"""
    return template.format(
        figure_name=figure_name,
        width=width,
    )


def add_subfigures(figure_names, width=None, n_cols=2, max_rows=None):
    if width is None:
        width = round(1 / n_cols, 2)

    if max_rows is None:
        max_rows = len(figure_names) + 1

    template = r"""\begin{{figure}}[H]
\centering
{rows}
\end{{figure}}"""

    template_float = r"""\begin{{figure}}[H]\ContinuedFloat
\centering
{rows}
\end{{figure}}"""

    text = ""
    rows = []
    n_figs = 0
    for index, name in enumerate(figure_names):
        row = index // n_cols
        col = index % n_cols
        if row > 0 and col == 0 and row % max_rows == 0:
            rows_str = "".join(rows)
            if n_figs > 0:
                text += "\n\n"
                text += template_float.format(rows=rows_str)
            else:
                text += template.format(rows=rows_str)

            rows = []
            n_figs += 1

        elif index > 0 and index % n_cols == 0:
            rows.append("\n\n")
            rows.append(r"\vspace{0.5em}")
            rows.append("\n\n")

        rows.append(add_subfigure(name, width))

    rows_str = "".join(rows)
    if n_figs > 0:
        text += "\n\n"
        text += template_float.format(rows=rows_str)
    else:
        text += template.format(rows=rows_str)

    return text


def add_subsection(name):
    return r"""\subsection{{{name}}}""".format(name=name)
