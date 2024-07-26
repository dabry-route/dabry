import itertools
import warnings
from typing import Union, Optional

import matplotlib
import numpy as np
from matplotlib import pyplot as plt, patches

from dabry.flowfield import DiscreteFF
from dabry.obstacle import CircleObs, FrameObs, is_frame_obstacle, is_circle_obstacle
from dabry.solver_ef import SolverEFResampling, SolverEFTrimming, SiteManager
from dabry.trajectory import Trajectory


class Style:
    colors = ['blue', 'red', 'green', 'cyan', 'magenta', 'orange']


def display(solver: Union[SolverEFResampling | SolverEFTrimming],
            trajectories: Optional[list[Trajectory]] = None, isub=4, timeslider=False,
            no_trajectories=False, no_value_func=False, autoshow=False, theme_dark=False, no_3d=False):
    try:
        import plotly.figure_factory as figfac
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError('"plotly" package requirement for solver display')
    trajectories = [] if trajectories is None else trajectories
    template = 'plotly_dark' if theme_dark else 'plotly'
    def_color = 'white' if theme_dark else 'black'
    ff_disc = DiscreteFF.from_ff(solver.pb.model.ff, (solver.pb.bl, solver.pb.tr),
                                 nt=10, no_diff=True)

    values = ff_disc.values if ff_disc.values.ndim == 3 else ff_disc.values[0]
    fig = figfac.create_quiver(*np.meshgrid(
        np.linspace(ff_disc.bounds[-2, 0], ff_disc.bounds[-2, 1], values[..., ::isub, ::isub, :].shape[-3]),
        np.linspace(ff_disc.bounds[-1, 0], ff_disc.bounds[-1, 1], values[..., ::isub, ::isub, :].shape[-2]),
        indexing='ij'),
                               values[::isub, ::isub, 0], values[::isub, ::isub, 1],
                               scaleratio=1., scale=0.1, line=dict(color=def_color),
                               hoverinfo='none', name='Flow field')
    cost_map = None
    if not no_value_func:
        cost_map = solver.get_cost_map()
        fig.add_trace(go.Contour(z=cost_map.transpose(),
                                 x=np.linspace(solver.pb.bl[0], solver.pb.tr[0], cost_map.shape[0]),
                                 y=np.linspace(solver.pb.bl[1], solver.pb.tr[1], cost_map.shape[1]),
                                 name='Cost map',
                                 contours=dict(start=0, end=1.2, size=0.1),
                                 coloraxis='coloraxis'))
    # fig.add_trace(go.Contour(z=solver.cost_map_triangle(nx, ny).transpose()[1::-1, 1::-1],
    #                          x=np.linspace(pb.bl[0], pb.tr[0], nx)[1::-1],
    #                          y=np.linspace(pb.bl[1], pb.tr[1], ny)[1::-1],
    #                          name='Cost contour', connectgaps=False, coloraxis='coloraxis'))
    fig.update_coloraxes(showscale=False, colorscale='Jet')
    fig.add_trace(go.Scatter(x=[solver.pb.x_init[0]], y=[solver.pb.x_init[1]], name='Start',
                             marker=dict(size=15, color=def_color)))
    fig.add_trace(go.Scatter(x=[solver.pb.x_target[0]], y=[solver.pb.x_target[1]], name='Target',
                             marker=dict(size=20, color=def_color, symbol='star')))
    fig.add_shape(type="circle", x0=solver.pb.x_target[0] - solver.target_radius,
                  y0=solver.pb.x_target[1] - solver.target_radius,
                  x1=solver.pb.x_target[0] + solver.target_radius, y1=solver.pb.x_target[1] + solver.target_radius,
                  xref="x", yref="y", fillcolor=None, line_color=def_color)

    for obs in solver.pb.obstacles:
        # TODO: adapt to show wrapped obstacles
        if isinstance(obs, CircleObs):
            fig.add_shape(type="circle", x0=obs.center[0] - obs.radius,
                          y0=obs.center[1] - obs.radius,
                          x1=obs.center[0] + obs.radius,
                          y1=obs.center[1] + obs.radius,
                          xref="x", yref="y", fillcolor="Grey", opacity=0.5)

        if isinstance(obs, FrameObs):
            fig.add_shape(type="rect", x0=obs.bl[0], y0=obs.bl[1],
                          x1=obs.tr[0], y1=obs.tr[1],
                          xref="x", yref="y", layer="below")

    fig.update_layout(template=template,
                      xaxis_range=[solver.pb.bl[0], solver.pb.tr[0]],
                      yaxis_range=[solver.pb.bl[1], solver.pb.tr[1]],
                      width=800, height=800)

    sites_by_depth = sorted(solver.sites.values(), key=lambda x: x.name)

    if not no_trajectories:
        if not timeslider:
            fig.add_traces([go.Scatter(x=site.traj.states[:, 0], y=site.traj.states[:, 1],
                                       line=dict(color=Style.colors[site.depth % len(Style.colors)]),
                                       name=site.name, mode='lines',
                                       hovertemplate='(%{x}, %{y})<br>%{text}',
                                       text=['({index}, {cost}){addinfo}'.format(
                                           index=index, cost=cost,
                                           addinfo=r'<br>CLS {clsr}'.format(clsr=site.closure_reason.value)
                                           if site.index_closed <= index else '')
                                             for index, cost in zip(np.arange(len(site.traj)), site.traj.cost)])
                            for site in sites_by_depth if site.traj is not None])
            fig.add_traces([go.Scatter(x=[site.traj.states[site.index_t_check_next, 0]],
                                       y=[site.traj.states[site.index_t_check_next, 1]],
                                       line=dict(color=def_color),
                                       name=f"{site.name} {site.index_t_check_next}", mode='markers',
                                       marker=dict(symbol='diamond-open', line_color=def_color),
                                       showlegend=False)
                            for site in sites_by_depth if site.traj is not None])
            if solver.solution_site is not None:
                if solver.solution_site.traj is None:
                    warnings.warn('Solver solutions have not been reconstructed over the full time window')
                else:
                    fig.add_traces([go.Scatter(x=site.traj.states[:, 0], y=site.traj.states[:, 1],
                                               line=dict(color='lightgreen', width=3), name=site.name, mode='lines')
                                    for site in [solver.solution_site] if site is not None])
                    fig.add_traces([go.Scatter(x=site.traj.states[:, 0], y=site.traj.states[:, 1],
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
                        line=dict(color=Style.colors[site.depth % len(Style.colors)]), name=site.name, mode='lines',
                        visible=True)
                        for site in sites_by_depth]
                )

            # Create and add slider
            steps = []
            for i in range(len(fig_tsteps)):
                step = dict(
                    method="update",
                    args=[
                        {"visible": [True] * i * len(solver.trajs) + [False] * (len(fig.data) - i * len(solver.sites))}]
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

    site_pb = solver.nemesis
    if site_pb is not None:
        fig.add_trace(go.Scatter(x=[site_pb.state_at_index(solver.validity_index)[0]],
                                 y=[site_pb.state_at_index(solver.validity_index)[1]],
                                 name=f'Pb: {site_pb.name} {solver.validity_index}',
                                 marker=dict(symbol='diamond', color=def_color)))

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
    if no_3d:
        fig_cost = None
    else:
        fig_cost = go.Figure()
        if not no_value_func:
            fig_cost.add_traces([go.Surface(z=cost_map[1:-1, 1:-1].transpose(),
                                            x=np.linspace(solver.pb.bl[0], solver.pb.tr[0], cost_map.shape[0])[1:-1],
                                            y=np.linspace(solver.pb.bl[1], solver.pb.tr[1], cost_map.shape[1])[1:-1],
                                            coloraxis='coloraxis')])
        fig_cost.update_coloraxes(showscale=False)
        if not no_trajectories:
            fig_cost.add_traces([go.Scatter3d(x=site.traj.states[:, 0], y=site.traj.states[:, 1], z=site.traj.cost,
                                              line=dict(color=Style.colors[site.depth % len(Style.colors)]),
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


def solver_structure(solver: SolverEFResampling):
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError('"plotly" package requirement for solver display')
    fig = go.Figure()
    # fig.add_traces(
    #     [go.Scatter(x=np.cos(2 * np.pi * index / (solver.n_costate_sectors * 2 ** 5)) * np.arange(0, solver.n_time),
    #                 y=np.sin(2 * np.pi * index / (solver.n_costate_sectors * 2 ** 5)) * np.arange(0, solver.n_time),
    #                 line=dict(color='grey'), hoverinfo='skip', mode='lines')
    #      for index in np.arange(solver.n_costate_sectors * 2 ** 5)])
    fig.add_traces([go.Scatter(x=-site.costate_init[0] * np.arange(0, solver.n_time) / np.linalg.norm(site.costate_init),
                               y=-site.costate_init[1] * np.arange(0, solver.n_time) / np.linalg.norm(site.costate_init),
                               line=dict(color=Style.colors[site.depth % len(Style.colors)]),
                               name=site.name, mode='lines')
                    for site in solver.sites.values()])

    fig.update_layout(width=800, height=800)
    return fig


def static_figure(solver, ff_on=False, no_quiver=False, ff_sub=1, t_min=None, t_max=None):
    plt.rc('font', size=18)
    plt.rc('axes', titlesize=18)
    plt.rc('axes', labelsize=18)
    plt.rc('xtick', labelsize=18)
    plt.rc('ytick', labelsize=18)
    plt.rc('legend', fontsize=10)
    plt.rc('mathtext', fontset='cm')
    plt.rc('text', usetex=True)
    if t_min is None:
        t_min = 0
    if t_max is None:
        t_max = solver.total_duration
    norm = matplotlib.colors.Normalize(vmin=0, vmax=t_max, clip=True)
    c_levels = np.arange(0, t_max + 0.01, 0.1)
    nrows, ncols = 2, 3
    fig, axs = plt.subplots(nrows, ncols, figsize=1.1 * np.array((12, 8)))
    times = np.linspace(t_min, t_max, 7)
    for k, (i, j) in enumerate(itertools.product(range(nrows), range(ncols))):
        t_cur_m1 = times[k]
        if k == 0:
            t_cur_m1 = 0
        t_cur = times[k + 1]
        levels = np.hstack((c_levels[c_levels < t_cur], t_cur))
        ax = axs[i, j]
        ax.axis('equal')
        ax.set_xlim(solver.pb.bl[0], solver.pb.tr[0])
        ax.set_ylim(solver.pb.bl[1], solver.pb.tr[1])
        points_scatter = np.zeros((len(solver.trajs), 2))
        for i_traj, traj in enumerate(solver.trajs):
            sl = np.logical_and(t_cur_m1 < traj.times, traj.times < t_cur)
            ax.plot(*traj.states[sl].T, color='grey', zorder=4)
            points_scatter[i_traj, :] = traj.states[sl][-1]
        ax.scatter(*points_scatter.T, color='black', s=10, zorder=5)
        for site in solver.solution_sites:
            ax.plot(*site.traj.states[site.traj.times < t_cur].T, color='black', zorder=8)
        if solver.solution_site is not None:
            ax.plot(*solver.solution_site.traj.states[solver.solution_site.traj.times < t_cur].T, color='red', zorder=8)
            ax.scatter(*solver.solution_site.traj.states[solver.solution_site.traj.times < t_cur][-1], color='red',
                       s=10, zorder=8)
        if not ff_on:
            if k == nrows * ncols - 1:
                c = ax.contourf(solver._cost_map.grid_vectors[1:-1, 1:-1, 0],
                                solver._cost_map.grid_vectors[1:-1, 1:-1, 1],
                                solver._cost_map.values[1:-1, 1:-1], levels=levels,
                                cmap='jet', norm=norm, zorder=4, alpha=0.5)
                c = ax.contour(solver._cost_map.grid_vectors[1:-1, 1:-1, 0],
                               solver._cost_map.grid_vectors[1:-1, 1:-1, 1],
                               solver._cost_map.values[1:-1, 1:-1], levels=c_levels[c_levels < t_cur],
                               alpha=1, colors=((0.2, 0.2, 0.2),) if not ff_on else 'black', zorder=5)
                ax.clabel(c, c.levels, inline=True, fontsize=15)
        else:
            ff = solver.pb.model.ff
            grid_vectors_ff = np.stack(
                np.meshgrid(
                    np.linspace(
                        ff.bounds[-2, 0],
                        ff.bounds[-2, 1],
                        ff.values.shape[-3]
                    ),
                    np.linspace(
                        ff.bounds[-1, 0],
                        ff.bounds[-1, 1],
                        ff.values.shape[-2]
                    ), indexing='ij'), -1)
            grid_vectors_ff = grid_vectors_ff[::ff_sub, ::ff_sub]
            t_virt_ff = (t_cur - ff.bounds[0, 0]) / (ff.bounds[0, 1] - ff.bounds[0, 0])
            it = np.floor(t_virt_ff * (ff.values.shape[0] - 1)).astype(np.int32)
            alpha = t_virt_ff * ff.values.shape[0] - it
            if it == ff.values.shape[0]:
                it = ff.values.shape[0] - 2
                alpha = 1
            ff_frame = (1 - alpha) * ff.values[it, ::ff_sub, ::ff_sub] + alpha * ff.values[it + 1, ::ff_sub, ::ff_sub]
            ff_norms = np.linalg.norm(ff_frame, axis=-1)
            ax.pcolormesh(grid_vectors_ff[..., 0], grid_vectors_ff[..., 1],
                          ff_norms, zorder=2,
                          shading='gouraud', cmap='turbo')
            rect = patches.Rectangle(solver.pb.bl, solver.pb.tr[0] - solver.pb.bl[0], solver.pb.tr[1] - solver.pb.bl[1],
                                     alpha=0.3, color='white', zorder=3)
            ax.add_patch(rect)
            if not no_quiver:
                ax.quiver(grid_vectors_ff[..., 0], grid_vectors_ff[..., 1], ff_frame[..., 0], ff_frame[..., 1],
                          zorder=4)

        obs = solver.pb.obstacles[0]
        grid_vectors_obs = np.stack(
            np.meshgrid(
                np.linspace(
                    obs.bounds[-2, 0],
                    obs.bounds[-2, 1],
                    obs.values.shape[-2]
                ),
                np.linspace(
                    obs.bounds[-1, 0],
                    obs.bounds[-1, 1],
                    obs.values.shape[-1]
                ), indexing='ij'), -1)
        t_virt_obs = (t_cur - obs.bounds[0, 0]) / (obs.bounds[0, 1] - obs.bounds[0, 0])
        it = np.floor(t_virt_obs * (obs.values.shape[0] - 1)).astype(np.int32)
        alpha = t_virt_obs * obs.values.shape[0] - it
        if it == obs.values.shape[0]:
            it = obs.values.shape[0] - 2
            alpha = 1
        obs_frame = (1 - alpha) * obs.values[it] + alpha * obs.values[it + 1]
        ax.contourf(grid_vectors_obs[..., 0], grid_vectors_obs[..., 1],
                    obs_frame, levels=[-100, 0],
                    colors='purple', extend='min', alpha=0.5, zorder=4)
        ax.contour(grid_vectors_obs[..., 0], grid_vectors_obs[..., 1], obs_frame, levels=0, colors='purple', zorder=4)
        ax.scatter(*solver.pb.x_init, color='black', edgecolor='white', s=100, zorder=10)
        ax.scatter(*solver.pb.x_target, color='black', edgecolor='white', s=200, marker='*', zorder=10)
        circ = patches.Circle(solver.pb.x_target, solver.target_radius, facecolor='none', edgecolor='black',
                              linewidth=1, zorder=8)
        ax.add_patch(circ)
        ax.set_title(rf'$t={t_cur:.3g}$')
        ax.set_xticks(np.arange(0, 1.1, 0.2))
        ax.grid(True)
    return fig, axs