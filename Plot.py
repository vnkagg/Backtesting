import plotly.graph_objects as go


def plot_df(df, *columns, fig = None, show = False):
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta']

    if fig is None:
        fig = go.Figure()

    if 'ix' not in df.columns:
        df['ix'] = range(1, len(df) + 1)

    for i, column in enumerate(columns):
        fig.add_trace(go.Scatter(
            x=df['ix'],
            y=df[column],
            mode='lines',
            name=f'{column}',
            line=dict(color=colors[i % len(colors)]),
            hovertext=df.index.strftime('%Y-%m-%d %H:%M:%S'),
            hoverinfo='text+y'
        ))

    fig.update_layout(
        title="Plot with Multiple Columns and Dark Background",
        xaxis_title="Index",
        yaxis_title="Values"
    )
    if show:
        fig.show()

    return fig

def draw_horizontal_line(fig, y, x0, x1, color='Red'):
    fig.add_shape(
        type="line",
        x0=x0, x1=x1,
        y0=y, y1=y,
        line=dict(color=color, width=2, dash="dash"),
        name=f"{y} Line"
    )
    return fig