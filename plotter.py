from collections import OrderedDict

import pandas as pd
import numpy as np
import scipy.stats
import scipy.interpolate
import statsmodels.api as sm
import statsmodels.stats.proportion

import matplotlib
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import matplotlib.pyplot
from IPython import embed

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

x_plotting_resolution = 200

grid_linewidth = 1.5
main_linewidth = 2
label_fontsize = 24
tick_fontsize = 22
defaultmarkersize = 130


def add_plotting_data(df, columns):
    for column in columns:
        df[column + '_ci'] = df[[column, column+'_dataset_size']].apply(get_ci, axis=1)
    return df


def get_ci(df_row):
    acc = df_row[[x for x in df_row.axes[0] if '_dataset_size' not in x][0]]
    dataset_size = df_row[[x for x in df_row.axes[0] if '_dataset_size' in x][0]]
    acc = acc / 100
    lo, hi = clopper_pearson(acc * dataset_size, dataset_size)
    #hi = min(1.0, hi)
    #lo = max(0.0, lo)
    low, high = acc - lo, hi - acc
    low, high = low * 100, high * 100
    return (low, high)


def clopper_pearson(k, n, alpha=0.005):
    lo = scipy.stats.beta.ppf(alpha/2, k, n-k+1)
    hi = scipy.stats.beta.ppf(1 - alpha/2, k+1, n-k)
    return lo, hi


def run_bootstrap_linreg(xs, ys, num_bootstrap_samples, x_eval_grid, seed):
    rng = np.random.RandomState(seed)
    num_samples = xs.shape[0]
    result_coeffs = []
    result_y_grid_vals = []
    x_eval_grid_padded = np.stack([np.ones(x_eval_grid.shape[0]), x_eval_grid], axis=1)
    for ii in range(num_bootstrap_samples):
        cur_indices = rng.choice(num_samples, num_samples)
        cur_x = np.stack([np.ones(num_samples), xs[cur_indices]], axis=1)
        cur_y = ys[cur_indices]
        cur_coeffs = np.linalg.lstsq(cur_x, cur_y, rcond=None)[0]
        result_coeffs.append(cur_coeffs)
        cur_y_grid_vals = np.dot(x_eval_grid_padded, cur_coeffs)
        result_y_grid_vals.append(cur_y_grid_vals)
    return np.vstack(result_coeffs), np.vstack(result_y_grid_vals)


def get_bootstrap_cis(xs, ys, num_bootstrap_samples, x_eval_grid, seed, significance_level_coeffs=95, significance_level_grid=95):
    coeffs, y_grid_vals = run_bootstrap_linreg(xs, ys, num_bootstrap_samples, x_eval_grid, seed)
    result_coeffs = []
    result_grid_lower = []
    result_grid_upper = []
    percentile_lower_coeffs = (100.0 - significance_level_coeffs) / 2
    percentile_upper_coeffs = 100.0 - percentile_lower_coeffs
    percentile_lower_grid = (100.0 - significance_level_grid) / 2
    percentile_upper_grid = 100.0 - percentile_lower_grid
    for ii in range(coeffs.shape[1]):
        cur_lower = np.percentile(coeffs[:, ii], percentile_lower_coeffs, interpolation='lower')
        cur_upper = np.percentile(coeffs[:, ii], percentile_upper_coeffs, interpolation='higher')
        result_coeffs.append((cur_lower, cur_upper))
    for ii in range(x_eval_grid.shape[0]):
        cur_lower = np.percentile(y_grid_vals[:, ii], percentile_lower_grid, interpolation='lower')
        cur_upper = np.percentile(y_grid_vals[:, ii], percentile_upper_grid, interpolation='higher')
        result_grid_lower.append(cur_lower)
        result_grid_upper.append(cur_upper)
    return result_coeffs, result_grid_lower, result_grid_upper


def transform_acc(acc, transform='linear'):
    if type(acc) is not np.ndarray:
        acc = np.array(acc)
    if transform == 'linear':
        return acc
    elif transform == 'probit':
        return scipy.stats.norm.ppf(acc / 100.0)
    elif transform == 'logit':
        return np.log(np.divide(acc / 100.0, 1.0 - acc / 100.0))


def tick_locs(low, hi, step):
    res = []
    assert step > 0
    cur = 0
    while cur <= hi:
        if cur >= low:
            res.append(cur)
        cur += step
    return res


def inv_logit(pred_logit):
    return (np.exp(pred_logit)/(1 + np.exp(pred_logit)))*100 


def model_scatter_plot_drop_eff_robustness(df, x_axis0, x_axis1, y_axis0, y_axis1, xlim, ylim, model_types, num_bootstrap_samples,
                       title, transform='linear', unit='top-1, %', include_legend=True,
                       tick_multiplier=10, extra_x_ticks=[], extra_y_ticks=[], set_aspect=False,
                       return_ordered_names=False, x_label=None, y_label=None, figsize=(10, 10), include_random_chance_bl=False, 
                       include_ideal=True, fit_color="red", bl_value=None, bl_name=None,
                       return_separate_legend=False, num_legend_columns=3):

    '''
    Fit line on x_axis, y_axis0, compute effective robustness with y_axis1
    '''
    assert transform == "logit"
    print('ylim range / xlim range aspect ratio: ', (ylim[1] - ylim[0]) / (xlim[1] - xlim[0]))

    fig, ax = matplotlib.pyplot.subplots(1, figsize=figsize)
    ax.grid(which='both', color='lightgray', linestyle='-', linewidth=grid_linewidth)

    #tick_loc_x = np.array(sorted(tick_locs(xlim[0], xlim[1], tick_multiplier) + extra_x_ticks))
    #ax.set_xticks(transform_acc(tick_loc_x, transform))
    #ax.set_xticklabels([str(int(loc)) for loc in tick_loc_x])
    #tick_loc_y = np.array(sorted(tick_locs(ylim[0], ylim[1], tick_multiplier) + extra_y_ticks))
    #ax.set_yticks(transform_acc(tick_loc_y, transform))
    #ax.set_yticklabels([str(int(loc)) for loc in tick_loc_y])
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)

    df_line = df[df.use_for_line_fit == True]

    df_plot = df[df.show_in_plot == True]
    xs_0 = df_plot[x_axis0].to_numpy()
    ys_0 = df_plot[y_axis0].to_numpy()
    xs_1 = df_plot[x_axis1].to_numpy()
    ys_1 = df_plot[y_axis1].to_numpy()
    if x_axis0 == x_axis1:
        xs_0 = xs_0[:, 0]
        xs_1 = xs_1[:, 0]


    if y_axis0 == y_axis1:
        ys_0 = ys_0[:, 0]
        ys_1 = ys_1[:, 0]

    x_acc_line_trans = transform_acc(xs_0, transform)
    y_acc_line_trans = transform_acc(ys_0, transform)

    xs = np.linspace(transform_acc(xlim[0], transform), transform_acc(xlim[1], transform), x_plotting_resolution)
    lin_fit = scipy.stats.linregress(x_acc_line_trans, y_acc_line_trans)

    x1s = df
    slope = lin_fit[0]
    intercept = lin_fit[1]
    lin_fit_ys = transform_acc(xs_0, transform) * slope + intercept
    eff_robustness = inv_logit(transform_acc(ys_0, transform)) - inv_logit(lin_fit_ys)
    print("Eff robustness", eff_robustness)
    drop_under_corruptions = ys_1 - xs_1
    print("Drop Under Corruptions", drop_under_corruptions)
    labels = df_plot.model_type.to_numpy()
    model_points = OrderedDict()
    model_names = []
    y_max = max(eff_robustness) + 1
    x_max = max(drop_under_corruptions) + 1
    
    y_min = min(eff_robustness) - 1
    x_min = min(drop_under_corruptions) - 1
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    # ax.set_ylim([-1.5, 2.5])
    
    for m in model_types:
        if len(m.value) > 2:
            n, c, s = m.value
        else:
            n, c = m.value
            s = defaultmarkersize
        #ax.errorbar(x_acc_plot_trans[labels == m], y_acc_plot_trans[labels == m], xerr=xerr[:, labels == m], yerr=yerr[:, labels == m],
        #            capsize=2, linewidth=0.5, ls='none', color=c, alpha=0.5, zorder=8)
        points = ax.scatter(drop_under_corruptions[labels == m], eff_robustness[labels == m],
                            zorder=9, color=c, s=s, label=n, alpha=0.5, linewidths=0)
        model_points[n] = points
        # Reverse models within groups, then reverse all models. This is the
        # path that gives us the order that matches matplotlib's artists.
        model_names.extend(list(reversed(df_plot[labels == m].index)))
    model_names = model_names[::-1]
    if set_aspect:
        ax.set_aspect('equal', adjustable='box')
    if include_legend:
            ax.legend(list(model_points.values()),
                       list(model_points.keys()),
                      fontsize=label_fontsize, markerscale=1.5, frameon=False)
    fig.tight_layout()

    if return_separate_legend:  
        fig_legend = matplotlib.pyplot.figure(figsize=(18, 2)) 
        fig_legend.legend(list(model_points.values()),
                          list(model_points.keys()),
                          fontsize=label_fontsize, ncol=num_legend_columns, markerscale=1.5, 
                          loc='center', frameon=False) 
        fig_legend.tight_layout(pad=1.0)    
        return fig, ax, fig_legend  

    elif return_ordered_names:
        return fig, ax, model_names

    else:
        return fig, ax


def model_scatter_plot(df, x_axis, y_axis, xlim, ylim, model_types, num_bootstrap_samples,
                       title, transform='logit', x_unit='top-1, %', y_unit='top-1, %', include_legend=True,
                       tick_multiplier=10, extra_x_ticks=[], extra_y_ticks=[], set_aspect=False,
                       return_ordered_names=False, x_label=None, y_label=None, figsize=(10, 10), 
                       include_random_chance_bl=False, include_ideal=True, fit_color="red", bl_value=None, 
                       bl_name=None, return_separate_legend=False, num_legend_columns=3, error_line_width=0.5, error_alpha=0.5):

    print("Include random chance: ", include_random_chance_bl)
    assert (df[df.show_in_plot == True][x_axis] <= xlim[1]).all()
    assert (df[df.show_in_plot == True][x_axis] >= xlim[0]).all()
    assert (df[df.show_in_plot == True][y_axis] <= ylim[1]).all()
    assert (df[df.show_in_plot == True][y_axis] >= ylim[0]).all()
    print('ylim range / xlim range aspect ratio: ', (ylim[1] - ylim[0]) / (xlim[1] - xlim[0]))

    fig, ax = matplotlib.pyplot.subplots(1, figsize=figsize)
    ax.grid(which='both', color='lightgray', linestyle='-', linewidth=grid_linewidth)

    tick_loc_x = np.array(sorted(tick_locs(xlim[0], xlim[1], tick_multiplier) + extra_x_ticks))
    ax.set_xticks(transform_acc(tick_loc_x, transform))
    ax.set_xticklabels([str(int(loc)) for loc in tick_loc_x])
    tick_loc_y = np.array(sorted(tick_locs(ylim[0], ylim[1], tick_multiplier) + extra_y_ticks))
    ax.set_yticks(transform_acc(tick_loc_y, transform))
    ax.set_yticklabels([str(int(loc)) for loc in tick_loc_y])
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)

    df_line = df[df.use_for_line_fit == True]
    x_acc_line_trans = transform_acc(df_line[x_axis], transform)
    y_acc_line_trans = transform_acc(df_line[y_axis], transform)

    xs = np.linspace(transform_acc(xlim[0], transform), transform_acc(xlim[1], transform), x_plotting_resolution)
    lin_fit = scipy.stats.linregress(x_acc_line_trans, y_acc_line_trans)
    slope = lin_fit[0]
    intercept = lin_fit[1]
    lin_fit_ys = xs * slope + intercept
    idx = np.argmax(y_acc_line_trans)
    #embed() 
    x_max = x_acc_line_trans[idx]
    y_max = y_acc_line_trans[idx]
    print("ymax",y_max)
    random_slope = y_max/x_max


    print(f'Slope {slope}, intercept {intercept}, r {lin_fit[2]}, pvalue {lin_fit[3]}, stderr {lin_fit[4]}')
    #coeffs_ci, fit_lower, fit_upper = get_bootstrap_cis(x_acc_line_trans, y_acc_line_trans, num_bootstrap_samples, xs, 720257663,
    #                                                    significance_level_coeffs=95, significance_level_grid=95)
    #print(f'Bootstrap CIs: {coeffs_ci}')
    #sm_model = sm.OLS(y_acc_line_trans, np.stack([np.ones(x_acc_line_trans.shape[0]), x_acc_line_trans], axis=1))
    #sm_results = sm_model.fit()
    #print(sm_results.summary())

    ax.set_xlim([transform_acc(xlim[0], transform), transform_acc(xlim[1], transform)])
    ax.set_ylim([transform_acc(ylim[0], transform), transform_acc(ylim[1], transform)])
    if include_ideal:
        ideal_repro_line = ax.plot(xs, xs, linestyle='dashed', color='black', linewidth=main_linewidth, label='y=x')
    if include_random_chance_bl:
        assert bl_value is not None
        assert bl_name is not None
        ((xb,yb), (x0,y0)) = bl_value
        print("B1 value is ", bl_value)
        random_slope = (yb - y0)/(xb - x0)

        intr = yb - random_slope*xb
        print("Interpolate slope is ", random_slope)
        print("Interpolate ntr is ", intr)
        assert np.isclose(random_slope*xb + intr,yb)
        assert np.isclose(random_slope*x0 + intr,y0)
        xs_random  = np.linspace(xlim[0], xlim[1])
        other_line = ax.plot(transform_acc(xs_random, transform), transform_acc(xs_random*random_slope + intr, transform), 
                             linestyle='dotted', color='black', linewidth=main_linewidth, label=f'Interpolate between {bl_name} and chance')
    #ax.fill_between(xs, fit_upper, fit_lower, color=f'tab:{fit_color}', alpha=0.4, zorder=6, edgecolor='none', linewidth=2.0)

    df_plot = df[df.show_in_plot == True]
    x_acc_plot = df_plot[x_axis]
    y_acc_plot = df_plot[y_axis]

    x_acc_ci = pd.DataFrame(df_plot[f'{x_axis}_ci'].tolist(), index=df_plot.index).to_numpy()
    y_acc_ci = pd.DataFrame(df_plot[f'{y_axis}_ci'].tolist(), index=df_plot.index).to_numpy()
    x_acc_plot_trans = transform_acc(x_acc_plot, transform)
    y_acc_plot_trans = transform_acc(y_acc_plot, transform)
    labels = df_plot.model_type.to_numpy()

    xerr_low = x_acc_plot_trans - transform_acc(x_acc_plot - x_acc_ci[:, 0], transform)
    xerr_high = transform_acc(x_acc_plot + x_acc_ci[:, 1], transform) - x_acc_plot_trans
    xerr = np.stack((xerr_low, xerr_high), axis=0)
    yerr_low = y_acc_plot_trans - transform_acc(y_acc_plot - y_acc_ci[:, 0], transform)
    yerr_high = transform_acc(y_acc_plot + y_acc_ci[:, 1], transform) - y_acc_plot_trans
    yerr = np.stack((yerr_low, yerr_high), axis=0)

    model_points = OrderedDict()
    model_names = []
    for m in model_types:
        if len(m.value) > 2:
            n, c, s = m.value
        else:
            n, c = m.value
            s = defaultmarkersize
        if not any(labels == m):
            continue
        ax.errorbar(x_acc_plot_trans[labels == m], y_acc_plot_trans[labels == m], xerr=xerr[:, labels == m], yerr=yerr[:, labels == m],
                    capsize=2, linewidth=error_line_width, ls='none', color=c, alpha=error_alpha, zorder=8)
        alpha = None if isinstance(c, (list, tuple)) and len(c) == 4 else 0.5
        points = ax.scatter(x_acc_plot_trans[labels == m], y_acc_plot_trans[labels == m],
                            zorder=9, color=c, s=s, label=n, alpha=alpha, linewidths=0)
        model_points[n] = points
        # Reverse models within groups, then reverse all models. This is the
        # path that gives us the order that matches matplotlib's artists.
        model_names.extend(list(reversed(df_plot[labels == m].index)))
    model_names = model_names[::-1]

    fit_line = ax.plot(xs, lin_fit_ys, color=f'tab:{fit_color}', zorder=7, linewidth=main_linewidth, label='Linear fit')
    if set_aspect:
        ax.set_aspect('equal', adjustable='box')
    if include_legend:
        if include_random_chance_bl:
            if not include_ideal:
                ax.legend([other_line[0]] + list(model_points.values()) + [fit_line[0]],
                      [f'Interpolate {bl_name} and random'] + list(model_points.keys()) + ['Linear fit'],
                       fontsize=label_fontsize, markerscale=1.5, frameon=False)
            else:
                ax.legend([ideal_repro_line[0]] + [other_line[0]] + list(model_points.values()) + [fit_line[0]],
                      ['y=x'] + [f'Interpolate {bl_name} and random'] + list(model_points.keys()) + ['Linear fit'],
                       fontsize=label_fontsize, markerscale=1.5, frameon=False)
        else:
            if not include_ideal:
                ax.legend(list(model_points.values()) + [fit_line[0]],
                        list(model_points.keys()) + ['Linear fit'],
                        fontsize=label_fontsize, markerscale=1.5, frameon=False)
            else:
                ax.legend([ideal_repro_line[0]] + list(model_points.values()) + [fit_line[0]],
                        ['y=x'] + list(model_points.keys()) + ['Linear fit'],
                        fontsize=label_fontsize, markerscale=1.5, frameon=False)
    fig.tight_layout()

    if return_separate_legend:  
        fig_legend = matplotlib.pyplot.figure(figsize=(25, 2)) 
        fig_legend.legend([ideal_repro_line[0]] + list(model_points.values()) + [fit_line[0]],  
                          ['y=x'] + list(model_points.keys()) + ['Linear fit'],   
                          fontsize=label_fontsize, ncol=num_legend_columns, markerscale=1.5, 
                          loc='center', frameon=False) 
        fig_legend.tight_layout(pad=1.0)    
        return fig, ax, fig_legend  

    elif return_ordered_names:
        return fig, ax, model_names

    else:
        return fig, ax


def get_confidence_interval(p, n, alpha=0.05, method='beta'):
    assert p >= 0.0
    assert p <= 1.0
    return statsmodels.stats.proportion.proportion_confint(p * n, n, alpha=alpha, method=method)


def add_confidence_interval_to_dataframe(df,
                                         accuracy_col_name,
                                         dataset_size_name,
                                         transform,
                                         upper_bound_name=None,
                                         lower_bound_name=None,
                                         alpha=0.05,
                                         method='beta'):
    assert accuracy_col_name in df.columns
    assert dataset_size_name in df.columns
    if upper_bound_name is None:
        upper_bound_name = accuracy_col_name + '_transformed_ci_upper_delta'
    assert upper_bound_name not in df.columns
    if lower_bound_name is None:
        lower_bound_name = accuracy_col_name + '_transformed_ci_lower_delta'
    assert lower_bound_name not in df.columns

    df2 = df.copy()
    for ii, row in df2.iterrows():
        cur_acc = row[accuracy_col_name]
        cur_n = row[dataset_size_name]
        cur_ci = get_confidence_interval(cur_acc / 100.0, cur_n, alpha=alpha)
        cur_upper_delta = transform_acc(cur_ci[1] * 100.0, transform) - transform_acc(cur_acc, transform)
        cur_lower_delta = transform_acc(cur_acc, transform) - transform_acc(cur_ci[0] * 100.0, transform)
        df2.loc[ii, upper_bound_name] = cur_upper_delta
        df2.loc[ii, lower_bound_name] = cur_lower_delta

    return df2
