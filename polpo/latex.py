def join_table_rows(rows):
    return "\\\\\n".join(rows)


def add_table(table_name):
    table_template = r"""\begin{{table}}[H]
\centering
\input{{{table_name}}}
\end{{table}}
"""
    return table_template.format(table_name=table_name)


def add_figure(figure_name, width=1.0):
    per_subject_fig_template = r"""\begin{{figure}}[H]
\centering
\includegraphics[width={width}\linewidth]{{{figure_name}}}
\end{{figure}}
"""
    return per_subject_fig_template.format(
        figure_name=figure_name,
        width=width,
    )


def add_subfigure(figure_name, width=1.0):
    template = r"""\begin{{subfigure}}{{{width}\textwidth}}\hfill
\centering
\includegraphics[width=1.\linewidth]{{{figure_name}}}
\end{{subfigure}}
"""
    return template.format(
        figure_name=figure_name,
        width=width,
    )


def add_subfigures(figure_names, width, n_cols):
    template = r"""\begin{{figure}}[H]
\centering
{rows}
\end{{figure}}
"""
    rows = []
    for index, name in enumerate(figure_names):
        if index > 0 and (index + 2) % n_cols == 0:
            rows.append(r"\medskip")
        rows.append(add_subfigure(name, width))

    rows_str = "".join(rows)

    return template.format(rows=rows_str)


def add_subsection(name):
    return r"""\subsection{{{name}}}""".format(name=name)
