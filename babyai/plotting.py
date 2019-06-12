"""Loading and plotting data from CSV logs.

Schematic example of usage

- load all `log.csv` files that can be found by recursing a root directory:
  `dfs = load_logs($BABYAI_STORAGE)`
- concatenate them in the master dataframe
  `df = pandas.concat(dfs, sort=True)`
- plot average performance for groups of runs using `plot_average(df, ...)`
- plot performance for each run in a group using `plot_all_runs(df, ...)`

Note:
- you can choose what to plot
- groups are defined by regular expressions over full paths to .csv files.
  For example, if your model is called "model1" and you trained it with multiple seeds,
  you can filter all the respective runs with the regular expression ".*model1.*"
- you may want to load your logs from multiple storage directories
  before concatening them into a master dataframe

"""

import os
import re
import numpy as np
from matplotlib import pyplot
import pandas
import random


def load_log(dir_):
    """Loads log from a directory and adds it to a list of dataframes."""
    #dir_ = dir_.decode("utf-8")
    df = pandas.read_csv(os.path.join(*[dir_, 'log.csv']),
                         error_bad_lines=False,
                         warn_bad_lines=True)
    if not len(df):
        print("empty df at {}".format(dir_))
        return
    df['model'] = dir_
    return df


def load_logs(root):
    dfs = []
    for root, dirs, files in os.walk(root, followlinks=True):
        for file_ in files:
            if file_ == 'log.csv':
                dfs.append(load_log(root))
    return dfs


def plot_average_impl(df, regexps, y_value='return_mean', window=1, agg='mean', 
                      x_value='frames'):
    """Plot averages over groups of runs  defined by regular expressions."""
    df = df.dropna(subset=[y_value])

    unique_models = df['model'].unique()
    model_groups = [[m for m in unique_models if re.match(regex, m)]
                     for regex in regexps]

    for regex, models in zip(regexps, model_groups):
        df_re = df[df['model'].isin(models)]
        # the average doesn't make sense if most models are not included,
        # so we only for the period of training that has been done by all models
        num_frames_per_model = [df_model[x_value].max()
                               for _, df_model in df_re.groupby('model')]
        median_progress = sorted(num_frames_per_model)[(len(num_frames_per_model) - 1) // 2]
        mean_duration = np.mean([
            df_model['duration'].max() for _, df_model in df_re.groupby('model')])
        df_re = df_re[df_re[x_value] <= median_progress]

        # smooth
        parts = []
        for _, df_model in df_re.groupby('model'):
            df_model = df_model.copy()
            df_model.loc[:, y_value] = df_model[y_value].rolling(window).mean()
            parts.append(df_model)
        df_re = pandas.concat(parts)

        df_agg = df_re.groupby([x_value]).agg([agg])
        values = df_agg[y_value][agg]
        pyplot.plot(df_agg.index, values, label=regex)
        #print(values)
        #print(regex, median_progress, mean_duration / 86400.0, values.iloc[-1])
        #pyplot.show()
    pyplot.show()

def plot_average(*args, **kwargs):
    """Plot averages over groups of runs  defined by regular expressions."""
    pyplot.figure(figsize=(15, 5))
    plot_average_impl(*args, **kwargs)
    pyplot.legend()


def plot_all_runs(df, regex, quantity='return_mean', x_axis='frames', window=1, color=None):
    """Plot a group of runs defined by a regex."""
    pyplot.figure(figsize=(15, 5))

    df = df.dropna(subset=[quantity])

    kwargs = {}
    if color:
        kwargs['color'] = color
    unique_models = df['model'].unique()
    models = [m for m in unique_models if re.match(regex, m)]
    df_re = df[df['model'].isin(models)]
    for model, df_model in df_re.groupby('model'):
        values = df_model[quantity]
        values = values.rolling(window).mean()
        pyplot.plot(df_model[x_axis],
                    values,
                    label=model,
                    **kwargs)
        #print(model, df_model[x_axis].max())

    pyplot.legend()

def plot_mean_std(dfs, color=None, y_value='validation_success_rate', show_std=False, label=None,plot_all=True, linestyle=None, compute_90val_suc_rate=False, ax=None, marker=None, plot_trend=False):
    if y_value == 'validation_success_rate':
        for df in dfs:
            # convert success rate to percentage
            df[y_value] *= 100

    # if y_value == 'val_cic':
    #     # scale for better readability
    #     for df in dfs:
    #         df[y_value] *= 1000

    df_concat = pandas.concat((dfs))
    by_row_index = df_concat.groupby(df_concat.index)
    df_means = by_row_index.mean()[y_value]
    df_stds = by_row_index.std()[y_value]

    num_points = len(df_means)
    x = np.arange(1, num_points + 1)

    if ax:
        ax.plot(x, df_means, label=label, color=color, linestyle=linestyle, marker=marker)
    else:
        fig, ax = pyplot.subplots()
        ax.plot(x, df_means, label=label, color=color, linestyle=linestyle, marker=marker)
        ax.xaxis.set_major_locator(pyplot.MaxNLocator(5))

    if compute_90val_suc_rate:
        print('90% val success rate reached at epoch:')
        threshold_idx = next(x[0] + 1 for x in enumerate(list(df_means)) if x[1] >= 90.0)
        print(threshold_idx)

    if plot_trend:
        z = np.polyfit(x, df_means, 1)
        p = np.poly1d(z)
        pyplot.plot(x, p(x), color="#fc8d62", linestyle="--")

    if plot_all:
        for df in dfs:
            x = np.arange(1, len(df) + 1)
            if ax:
                ax.plot(x, df[y_value], alpha=0.25, label='_nolegend_', color=color, linestyle=linestyle) #, linewidth=0.5) #, marker=marker)
            else:
                pyplot.plot(x, df[y_value], alpha=0.25, label='_nolegend_',color=color, linestyle=linestyle) #, linewidth=0.5) #, marker=marker)

    if show_std:
        pyplot.fill_between(x, df_means - df_stds, df_means + df_stds, facecolor=color, alpha=0.5)

def plot_compared_models(log_roots, nr_runs, title, legend_labels=None, max_update=35, show_std=False, plot_all=True, y_value='validation_success_rate', filename=None, compute_90val_suc_rate=False, plot_trend=False):
    pyplot.rc('font', size=18) #14-18

    #colors = ['#66c2a5', '#fc8d62', '#ffd92f', '#8da0cb', '#e78ac3', '#a6d854']
    colors = ['#fc8d62', '#66c2a5', '#8da0cb', '#e78ac3', '#a6d854', '#ffd92f']

    markers = ['o', 'v', '^', 's', '+', 'x']

    if legend_labels is None:
        legend_labels = [root.split('/')[-1] for root in log_roots]

    fig, ax = pyplot.subplots(figsize=(7,5)) #(12,5)

    for idx, root in enumerate(log_roots):
        dirs = [root + str(i) for i in range(1, nr_runs + 1)]
        dfs = []
        for dir in dirs:
            try:
                dfs += [load_log(dir).query('update <=' + str(max_update))]
            except:
                pass

        plot_mean_std(dfs, label=legend_labels[idx], show_std=show_std,plot_all=plot_all,color=colors[idx], marker=markers[idx], y_value=y_value, compute_90val_suc_rate=compute_90val_suc_rate, plot_trend=plot_trend, ax=ax)
    pyplot.xlabel('epochs')

    if y_value == 'validation_success_rate':
        pyplot.ylabel('validation success %')
        pyplot.ylim(0, 102)

    elif y_value == 'correction_weight_loss':
        pyplot.ylabel('guidance weight')
    elif y_value == 'val_cic':
        #pyplot.ylabel(r'CIC $\times 10^3$')
        pyplot.ylabel('CIC')

    #pyplot.title(title)

    pyplot.legend(loc=4)
    #pyplot.legend(loc='upper center', bbox_to_anchor=(0.5, -0.5))

    pyplot.tight_layout()

    #pyplot.show()
    pyplot.savefig(filename)
    pyplot.clf()

def plot_cic(cic_file,):
    with open(cic_file, 'r') as cf:
        values = [float(item) for item in cf.readlines()]

    pyplot.plot(np.arange(1, len(values) + 1), values, '-o')
    pyplot.xlabel('epoch')
    pyplot.ylabel('CIC metric')
    pyplot.title('CIC in Learner + pretrained Corrector, GoToObj')
    pyplot.show()

def plot_two_vars(log_root, nr_runs, vars, title, max_update=200, show_std=False, plot_all=True, filename=None):
    pyplot.rc('font', size=13)  # 18
    colors = ['#fc8d62', '#66c2a5', '#8da0cb', '#e78ac3', '#a6d854', '#ffd92f']
    linestyles = ['-', '--', '-.', (0, (5, 1)), (0, (3, 1, 1, 1)), (0, (1, 1))]

    fig, ax1 = pyplot.subplots()

    dirs = [log_root + str(i) for i in range(1, nr_runs + 1)]
    dfs = []
    for dir in dirs:
        # print(dir)
        try:
            dfs += [load_log(dir).query('update <=' + str(max_update))]
        except:
            pass

    ax1.set_xlabel('epochs')
    ax1.set_ylabel('validation success %', color=colors[0])

    plot_mean_std(dfs, label=vars[0], show_std=show_std, plot_all=plot_all, color=colors[0],
                  linestyle=linestyles[0], y_value=vars[0], ax = ax1)

    ax1.tick_params(axis='y', labelcolor=colors[0])
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('guidance weight', color=colors[1])  # we already handled the x-label with ax1
    plot_mean_std(dfs, label=vars[1], show_std=show_std, plot_all=plot_all, color=colors[1],
                  linestyle=linestyles[1], y_value=vars[1], ax=ax2)

    ax2.tick_params(axis='y', labelcolor=colors[1])
    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    pyplot.title(title)

    #pyplot.show()
    pyplot.savefig(filename)
    pyplot.clf()

# EXAMPLES
#
# roots = ['/Users/mathijs/Documents/Studie/AI/Thesis/code/babyai-repo/logs/pickuploc/pickuploc-nocor',
#          '/Users/mathijs/Documents/Studie/AI/Thesis/code/babyai-repo/logs/pickuploc/pickuploc-pretrainedcor-ownvocab',
#          '/Users/mathijs/Documents/Studie/AI/Thesis/code/babyai-repo/logs/pickuploc/pickuploc-pretrainedcor-gotolocal',
#          '/Users/mathijs/Documents/Studie/AI/Thesis/code/babyai-repo/logs/pickuploc/pickuploc-pretrainedcor-putnextlocal',
#          '/Users/mathijs/Documents/Studie/AI/Thesis/code/babyai-repo/logs/pickuploc/pickuploc-pretrainedlearner-gotolocal',
#          '/Users/mathijs/Documents/Studie/AI/Thesis/code/babyai-repo/logs/pickuploc/pickuploc-pretrainedlearner-putnextlocal',
#          # '/Users/mathijs/Documents/Studie/AI/Thesis/code/babyai-repo/logs/pickuploc/pickuploc-pretrainedcor-multicor'
#          ]
#
# labels = ['Learner',
#           'Learner + Guide, pretrained at same level',
#           'Learner + Guide, pretrained at GoToLocal',
#           'Learner + Guide, pretrained at PutNextLocal',
#           'Learner, pretrained at GoToLocal',
#           'Learner, pretrained at PutNextLocal'
#           #'Learner + Guide, \n pretrained at 3 levels'
#           ]
#
# plot_compared_models(log_roots=roots, nr_runs=3, title='PickupLoc', legend_labels=labels,
#                      max_update=25,
#                      plot_all=False,
#                      filename='pickuploc-interlevel.pdf')

# roots = ['/Users/mathijs/Documents/Studie/AI/Thesis/code/babyai-repo/logs/gotoobj/gotoobj-pretrainedcor-cic',
#          '/Users/mathijs/Documents/Studie/AI/Thesis/code/babyai-repo/logs/gotolocal/gotolocal-pretrainedcor-cic',
#          '/Users/mathijs/Documents/Studie/AI/Thesis/code/babyai-repo/logs/gotoobjmaze/gotoobjmaze-pretrainedcor-cic',
#          '/Users/mathijs/Documents/Studie/AI/Thesis/code/babyai-repo/logs/pickuploc/pickuploc-pretrainedcor-cic',
#          '/Users/mathijs/Documents/Studie/AI/Thesis/code/babyai-repo/logs/putnextlocal/putnextlocal-pretrainedcor-cic',
#          '/Users/mathijs/Documents/Studie/AI/Thesis/code/babyai-repo/logs/goto/goto-pretrainedcor-ownvocab',
#          ]
#
# labels = ['GoToObj',
#           'GoToLocal',
#           'GoToObjMaze',
#           'PickupLoc',
#           'PutNextLocal',
#           'GoTo'
#           ]
#
# epochs = [20,
#           25,
#           8,
#           25,
#           25,
#           25
#           ]
#
# for idx, root in enumerate(roots):
#     label = [labels[idx]]
#     plot_compared_models(log_roots=[root], nr_runs=3, title='CIC',
#                          legend_labels=label, max_update=epochs[idx],
#                          y_value='val_cic',
#                          show_std=True,
#                          plot_all=False,
#                          filename=str(label) + '-cic.pdf',
#                          plot_trend=True)