import warnings
from typing import Union, Optional

import numpy as np

from dabry.flowfield import DiscreteFF
from dabry.obstacle import CircleObs
from dabry.solver_ef import SolverEFResampling, SolverEFTrimming
from dabry.trajectory import Trajectory


def display(solver: Union[SolverEFResampling | SolverEFTrimming],
            trajectories: Optional[list[Trajectory]] = None, isub=4, timeslider=False,
            no_trajectories=False, no_value_func=False, autoshow=False, theme_dark=False):
    try:
        import plotly.figure_factory as figfac
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError('"plotly" package requirement for solver display')
    trajectories = [] if trajectories is None else trajectories
    template = 'plotly_dark' if theme_dark else 'plotly'
    ff_disc = DiscreteFF.from_ff(solver.pb.model.ff, (solver.pb.bl, solver.pb.tr),
                                 nt=10, force_no_diff=True)

    values = ff_disc.values if ff_disc.values.ndim == 3 else ff_disc.values[0]
    fig = figfac.create_quiver(*np.meshgrid(
        np.linspace(ff_disc.bounds[-2, 0], ff_disc.bounds[-2, 1], values[..., ::isub, ::isub, :].shape[-3]),
        np.linspace(ff_disc.bounds[-1, 0], ff_disc.bounds[-1, 1], values[..., ::isub, ::isub, :].shape[-2]),
        indexing='ij'),
                               values[::isub, ::isub, 0], values[::isub, ::isub, 1],
                               scaleratio=1., scale=0.1, line=dict(color='white' if theme_dark else 'black'),
                               hoverinfo='none', name='Flow field')
    colors = ['blue', 'red', 'green', 'cyan', 'magenta', 'orange']
    nx, ny = 100, 100
    cost_map = None
    if not no_value_func:
        cost_map = solver.cost_map_triangle(nx, ny)
        fig.add_trace(go.Heatmap(z=cost_map.transpose(),
                                 x=np.linspace(solver.pb.bl[0], solver.pb.tr[0], cost_map.shape[0]),
                                 y=np.linspace(solver.pb.bl[1], solver.pb.tr[1], cost_map.shape[1]),
                                 name='Cost map', connectgaps=False, coloraxis='coloraxis'))
    # fig.add_trace(go.Contour(z=solver.cost_map_triangle(nx, ny).transpose()[1::-1, 1::-1],
    #                          x=np.linspace(pb.bl[0], pb.tr[0], nx)[1::-1],
    #                          y=np.linspace(pb.bl[1], pb.tr[1], ny)[1::-1],
    #                          name='Cost contour', connectgaps=False, coloraxis='coloraxis'))
    fig.update_coloraxes(showscale=False)
    fig.add_trace(go.Scatter(x=[solver.pb.x_init[0]], y=[solver.pb.x_init[1]], name='Start',
                             marker=dict(size=15, color='white' if theme_dark else 'black')))
    fig.add_trace(go.Scatter(x=[solver.pb.x_target[0]], y=[solver.pb.x_target[1]], name='Target',
                             marker=dict(size=20, color='white' if theme_dark else 'black', symbol='star')))
    fig.add_shape(type="circle", x0=solver.pb.x_target[0] - solver.target_radius,
                  y0=solver.pb.x_target[1] - solver.target_radius,
                  x1=solver.pb.x_target[0] + solver.target_radius, y1=solver.pb.x_target[1] + solver.target_radius,
                  xref="x", yref="y", fillcolor=None, line_color='white' if theme_dark else 'black')

    for obs in solver.pb.obstacles:
        # TODO: adapt to show wrapped obstacles
        if isinstance(obs, CircleObs):
            fig.add_shape(type="circle", x0=obs.center[0] - obs.radius,
                          y0=obs.center[1] - obs.radius,
                          x1=obs.center[0] + obs.radius,
                          y1=obs.center[1] + obs.radius,
                          xref="x", yref="y", fillcolor="Grey", opacity=0.5)

    fig.update_layout(template=template,
                      xaxis_range=[solver.pb.bl[0], solver.pb.tr[0]],
                      yaxis_range=[solver.pb.bl[1], solver.pb.tr[1]],
                      width=800, height=800)

    sites_by_depth = sorted(solver.sites.values(), key=lambda x: x.depth)

    if not no_trajectories:
        if not timeslider:
            fig.add_traces([go.Scatter(x=site.traj.states[:, 0], y=site.traj.states[:, 1],
                                       line=dict(color=colors[site.depth % len(colors)]), name=site.name, mode='lines')
                            for site in sites_by_depth])
            if solver.solution_site is not None:
                if solver.solution_site.traj_full is None:
                    warnings.warn('Solver solutions have not been reconstructed over the full time window')
                else:
                    fig.add_traces([go.Scatter(x=site.traj_full.states[:, 0], y=site.traj_full.states[:, 1],
                                               line=dict(color='lightgreen', width=3), name=site.name, mode='lines')
                                    for site in [solver.solution_site] if site is not None])
                    fig.add_traces([go.Scatter(x=site.traj_full.states[:, 0], y=site.traj_full.states[:, 1],
                                               line=dict(color='lightgreen', dash='dash'), name=site.name, mode='lines')
                                    for site in solver.suboptimal_sites])
            fig.add_traces([go.Scatter(x=traj.states[:, 0], y=traj.states[:, 1],
                                       line=dict(), mode='lines') for traj in trajectories])
        else:
            # Add traces, one for each slider step
            substep = 10
            fig_tsteps = list(range(solver.n_time))[::substep]
            for i_major in fig_tsteps:
                fig.add_traces(
                    [go.Scatter(
                        x=np.where((site.traj.times >= solver.times[i_major]) *
                                   (site.traj.times <= solver.times[min(i_major + substep, solver.n_time - 1)]),
                                   site.traj.states[:, 0], np.ones(site.traj.times.shape[0]) * np.nan),
                        y=np.where((site.traj.times >= solver.times[i_major]) * (
                                site.traj.times <= solver.times[min(i_major + substep, solver.n_time - 1)]),
                                   site.traj.states[:, 1], np.ones(site.traj.times.shape[0]) * np.nan),
                        line=dict(color=colors[site.depth % len(colors)]), name=site.name, mode='lines',
                        visible=True)
                        for site in sites_by_depth]
                )

            # Create and add slider
            steps = []
            for i in range(len(fig_tsteps)):
                step = dict(
                    method="update",
                    args=[
                        {"visible": [True] * i * len(solver.trajs) + [False] * (len(fig.data) - i * len(solver.trajs))}]
                )
                steps.append(step)

            sliders = [dict(
                active=len(steps) - 1,
                # currentvalue={"prefix": "Frequency: "},
                pad={"t": 50},
                steps=steps
            )]

            fig.update_layout(
                sliders=sliders
            )

    if autoshow:
        fig.show()

    # fig_cst = go.Figure()
    # for depth in range(n_steps):
    #     fig_cst.add_traces(
    #         [go.Scatter(x=traj.costates[:, 0], y=traj.costates[:, 1], line=dict(color=colors[depth % len(colors)]),
    #                     name=traj_name,
    #                     # legendgroup=depth, legendgrouptitle={'text': 'Depth %d' % depth},
    #                     mode='lines')
    #          for traj_name, traj in solver.get_trajs_by_depth(depth).items()])
    #
    # fig_cst.show()

    fig_cost = go.Figure()
    if not no_value_func:
        fig_cost.add_traces([go.Surface(z=cost_map[1:-1, 1:-1].transpose(),
                                        x=np.linspace(solver.pb.bl[0], solver.pb.tr[0], cost_map.shape[0])[1:-1],
                                        y=np.linspace(solver.pb.bl[1], solver.pb.tr[1], cost_map.shape[1])[1:-1],
                                        coloraxis='coloraxis')])
    fig_cost.update_coloraxes(showscale=False)
    if not no_trajectories:
        fig_cost.add_traces([go.Scatter3d(x=site.traj.states[:, 0], y=site.traj.states[:, 1], z=site.traj.cost,
                                          line=dict(color=colors[site.depth % len(colors)]),
                                          name=site.name,
                                          # legendgroup=depth, legendgrouptitle={'text': 'Depth %d' % depth},
                                          mode='lines')
                             for site in sites_by_depth])
    fig_cost.update_layout(template=template,
                           title='Value function', autosize=False, width=800, height=800,
                           margin=dict(l=65, r=50, b=65, t=90),
                           scene=dict(aspectmode='data'))
    if autoshow:
        fig_cost.show()
    return fig, fig_cost
