import numpy as np
import pandas as pd
import copy
import scipy as sp
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import learning_curve

# Citation: color paletes from http://colorbrewer2.org/#type=sequential&scheme=BuGn&n=3

def variance_explained_plot(pca, pca_null=None, title=None, max_comp=None,
                            leg_loc='best', figsize=None):
    """
    Plot variance explained and cumulative variance explained for PCA.
    Citation: code modified from:
    Sebastian Raschka. 2015. Python Machine Learning. Packt Publishing.

    Parameters
    ----------
    pca : sklearn PCA object
        A fitted PCA object from sklearn.
    pca_null : sklearn PCA object
        A PCA object fitted on an uncorrelated null dataset, optional.
    title : String
        A title for the plot, optional.
    max_comp : int
        Maximum number of components to plot, default no limit.
    leg_loc : String
        Location for legend on the plot using matplotlib arguments, default 'best'
    figsize : tuple
        Tuple of width and height for figure, optional.
    """

    colors = ['#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e', '#e6a602',
              '#a6761d', '#666666', '#000000']

    fig, ax = plt.subplots()
    if figsize is not None:
        fig.set_size_inches(figsize)

    if title is not None:
        fig.suptitle(title)

    if max_comp is None:
        max_comp = len(pca.explained_variance_ratio_)

    var_exp = pca.explained_variance_ratio_[:max_comp]
    cum_var_exp = np.cumsum(var_exp)
    component_indices = range(1,max_comp+1)

    if pca_null is not None:
            var_exp_null = pca_null.explained_variance_ratio_[:max_comp]
            cum_var_exp_null = np.cumsum(var_exp_null)
            component_indices_null = range(1,max_comp+1)

    ax.bar(component_indices, var_exp, align='edge', width=-0.4,
           label='comp.', color=colors[0])
    ax.step(component_indices, cum_var_exp, where='mid', label='cum.',
            color=colors[0])
    if pca_null is not None:
            ax.bar(component_indices_null, var_exp_null, align='edge',
                   width=0.4, label='comp, null',
                   color=colors[1])
            ax.step(component_indices_null, cum_var_exp_null, where='mid',
                    label='cum, null', color=colors[1])

    ax.set_xlabel('Principal component index')
    ax.set_xticks(component_indices)
    ax.set_ylabel('Explained variance ratio')
    ax.legend(loc=leg_loc)
    plt.show()

def heat_map(X, x_labels, y_labels, title=None, sort_f=True, print_num=True,
             figsize=None):
    """
    Plots a heat map showing the composition of components.
    Code is adapted from:
    https://matplotlib.org/gallery/images_contours_and_fields/image_annotated_heatmap.html

    Parameters
    ----------
    X : np.ndarray
        The ndarray of heatmap values.
    x_labels : List of strings
        Labels for the horizontal axis of the heat map.
    y_labels : List of strings
        Labels for the vertical axis of the heat map.
    title : String
        A title for the plot, optional.
    sort_f : Boolean
        Whether to sort the array columns by the values in the first row, default True.
    print_num : Boolean
        Whether to plot values within the heatmap, default True.
    figsize : tuple
        Tuple of width and height for figure, optional.
    """

    # Sort inputs for better visibility of patterns
    if sort_f == True:
        x_labels = np.array(x_labels)[X[0,:].argsort()]
        X = X[:,X[0,:].argsort()]

    fig, ax = plt.subplots()
    im = ax.imshow(X, cmap='RdBu')

    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_xticklabels(x_labels)
    ax.set_yticklabels(y_labels)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    if print_num==True:
        for i in range(len(y_labels)):
            for j in range(len(x_labels)):
                text = ax.text(j, i, np.round(X[i, j],2),
                               ha="center", va="center", color="black",
                               fontsize=8)

    if title is not None:
        ax.set_title(title)
    fig.tight_layout()
    if figsize is not None:
        fig.set_size_inches(figsize)
    plt.show()

def pca_2d_plot(pca_X, Y, pca=None, title=None, labels=['label=0', 'label=1'],
                xlim=None, ylim=None, features=None, plot_num_features=None,
                vectors=True, match_plot=False, label_swap=False,
                figsize=None):
    """
    Plot transformed dataset over first two principal components.
    The idea for this type of visualization with vectors representing original
    features is inspired by:
    James Et Al. 2013. An Introduction to Statistical Learning. Springer.

    This visualization is a prototype and is not fully documented yet. It also
    includes some temporary hacks to keep labels from overlapping.
    """

    sc = 5

    colors = ['blue', 'red']
    fig, ax = plt.subplots()
    if title is not None:
        fig.suptitle(title)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    if label_swap:
        Y = (Y*-1) + 1

    if match_plot:
        colors[3] = colors[1]
        colors[0] = colors[2]

    for y in np.unique(np.array(Y)):
        label_mask = np.where(Y==y)
        ax.scatter(pca_X[label_mask,0], pca_X[label_mask,1], color=colors[int(y)],
                   label=labels[int(y)], alpha=0.05, s=10)

    # Optionally, add feature vectors into R2 PCA space
    if pca is not None:
        pca1_fs = pca.components_[0]
        pca2_fs = pca.components_[1]
        feature_vectors = []
        for i in range(len(pca1_fs)):
            feature_vectors.append((features[i], pca1_fs[i], pca2_fs[i],
                                    math.sqrt(pca1_fs[i]**2 + pca2_fs[i]**2), i))
        # Sort feature vectors by magnitude
        feature_vectors.sort(key=lambda feature: -feature[3])

        offsets = [0,0,0,-0.25,0.25,0,0,0,0,0,0,0,0,0]

        # Plot first feature_plot_num features
        if plot_num_features is not None:
            for i in range(plot_num_features):
                if vectors:
                    ax.quiver(0,0, feature_vectors[i][1], feature_vectors[i][2],
                              scale_units='xy', scale=0.2)
                    ax.text(feature_vectors[i][1] * sc, feature_vectors[i][2] * sc + offsets[i],
                            feature_vectors[i][0], fontsize=10, fontweight='bold')
                else:
                    ax.text(feature_vectors[i][1] * sc, feature_vectors[i][2] * sc,
                            str(feature_vectors[i][4]), fontweight='bold')

    if figsize is not None:
        fig.set_size_inches(figsize)

    ax.set_xlabel('Principal component #1')
    ax.set_ylabel('Principal component #2')
    ax.legend(loc='best')
    ax.grid()
    ax.set_axisbelow(True)
    plt.show()

# http://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
# Some modifications have been made
def learning_curves(estimator, X, y, ylim=None, cv=10, title='', n_jobs=1,
                        train_sizes=np.linspace(.1, 1.0, 10), leg_loc='best',
                        scoring='accuracy', baseline=0, baseline_loc=1.0,
                        random_seed=None):
    """
    Generate a simple plot of the test and training learning curves.
    Citation: http://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    title : string, optional
        Title for the chart (default no title).

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).

    train_sizes : array-like, shape(user-defined, 1)
        Training size increments as proportion between 0.0 and 1.0.

    scoring : string
        Scoring method for learning curve method (default 'Accuracy').

    baseline : float, optional
        Baseline accuracy for plotting for comparison.

    baseline_loc : float, optional
        Multiple of times to move baseline label to the right from first point.

    leg_loc : string, optional
        Location for legend (default 'best').

    random_seed : int, optional
        Random seed (if any) to initialize before computing learning curves.
    """

    if random_seed is not None:
        np.random.seed(random_seed)

    colors = ['#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e']
    plt.figure()
    if len(title) > 0:
        plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel(scoring)
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes,
        scoring=scoring)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color=colors[-2])
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color=colors[-1])
    plt.plot(train_sizes, train_scores_mean, 'o-', color=colors[-2],
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color=colors[-1],
             label="Cross-validation score")

    # Plot a baseline accuracy if provided
    if baseline > 0:
        plt.axhline(baseline, color = 'black', linestyle='dashed')
        plt.text(baseline_loc * train_sizes[1], baseline, 'baseline',
                     bbox=dict(facecolor='white'))

    plt.legend(loc=leg_loc)
    plt.show()

def km_elbow_diagram(n_clusters_range, metric_values,
                     metrics=['Within Clusters Sample Variance'],
                     xlim=None, ylim=None, title=None, stdev=None,
                     leg_loc1='upper right', leg_loc2='lower right',
                     figsize=None):
    """
    Plot a metric or metrics as a function of number of clusters.

    Parameters
    ----------
    n_clusters_range : list
        Number of clusters corresponding to metric values.

    metric_values : list of lists
        Lists of metric values for each metric considered.

    metrics : list of strings, optional
        Labels for metrics, default 'Within Clusters SST'.

    xlim : tuple of floats, optional
        Bounds for x-axis, default None.

    ylim : tuple of floats, optional
        Bounds for y-axis, default None.

    title : string, optional
        Title for the chart.
    """
    colors = ['#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e', '#e6a602',
              '#a6761d', '#666666', '#000000']
    fig, ax1 = plt.subplots()

    if figsize is not None:
        fig.set_size_inches(figsize)

    if title is not None:
        fig.suptitle(title)
    if xlim is not None:
        ax1.set_xlim(xlim)
    if ylim is not None:
        ax1.set_ylim(ylim)

    ax1.set_xlabel('Number of clusters')

    y_max = 0
    x_max = 0

    # Plots results for each metric value
    for i, y_vals in enumerate(metric_values):
        label = metrics[i]
        if i == 0:
            if y_vals[0] == 'na':
                n_clusters_range = n_clusters_range[1:]
                y_vals = y_vals[1:]
            ax1.plot(n_clusters_range, y_vals, marker='o', markersize=4,
                     color = colors[i], label=label)
            ax1.set_ylabel(label)
            ax1.set_xticks(n_clusters_range)
            ax1.legend(loc=leg_loc1)
            if stdev is not None:
                ax1.fill_between(n_clusters_range, y_vals+stdev, y_vals-stdev,
                                 alpha=0.2, color = colors[i+1])
        elif i == 1:
            ax2 = ax1.twinx()
            if y_vals[0] == 'na':
                n_clusters_range = n_clusters_range[1:]
                y_vals = y_vals[1:]
            ax2.plot(n_clusters_range, y_vals, marker='o', markersize=4,
                     color = colors[i], label=label)
            ax2.set_ylabel(label)
            ax2.legend(loc=leg_loc2)

    ax1.grid(b=True)
    plt.show()

def complexity_curves(gs, primary_hp, log_scale=False, plot_stdev=False,
                      response='scores', title=None, show_err=False, ylim=None,
                      figsize=(5,5), leg_outside=True):
    """
    Plot model training and validation curves from results of a GridSearchCV
    object.

    Parameters
    ----------
    gs : dict
        A fitted gridsearch object.

    title : string, optional
        Title for the chart.
    """
    colors = ['#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e', '#e6a602',
              '#a6761d', '#666666']

    fig, ax = plt.subplots()
    if figsize is not None:
        fig.set_size_inches(figsize)

    if title is not None:
        ax.set_title(title)

    if ylim is not None:
        ax.set_ylim(0,ylim)

    gs_df = pd.DataFrame(copy.deepcopy(gs.cv_results_))

    # Separate parameters for neural network
    def count_layers(params_dict):
        params_dict['num_layers'] = len(params_dict['hidden_layer_sizes'])
        params_dict['hidden_layer_sizes'] = params_dict['hidden_layer_sizes'][0]
    if primary_hp == 'hidden_layer_sizes':
        gs_df['params'].apply(count_layers)
        gs_df['param_hidden_layer_sizes'] = gs_df['param_hidden_layer_sizes'].map(lambda tup : tup[0])

    # Handle case with only one varying hyperparameter
    if len(gs_df['params'][0]) == 1:
        x = gs_df['param_'+primary_hp]
        if log_scale == True:
            x = np.log(x.astype('float'))
        train = gs_df['mean_train_score']
        validation = gs_df['mean_test_score']

        if plot_stdev==True:
            train_err = gs_df['std_train_score']
            validation_err = gs_df['std_test_score']
            ax.errorbar(x, train, train_err, color=colors[0], marker='o',
                        label='Train', linestyle='--', ecolor='k', capsize=4)
            ax.errorbar(x, validation, validation_err, color=colors[0], marker='o',
                        label='Validation', linestyle='-', ecolor='k', capsize=4)
        else:
            ax.plot(x, train, color=colors[0], marker='o',
                        label='Train', linestyle='--')
            ax.plot(x, validation, color=colors[0], marker='o',
                        label='Validation', linestyle='-')

    # Handle case with two varying hyperparameters
    else:
        gs_df['params'].apply(lambda dict : dict.pop(primary_hp))
        gs_df['params'] = gs_df['params'].map(lambda dict : tuple(dict.items()))
        for i, params in enumerate(gs_df['params'].unique()):
            x = gs_df['param_'+primary_hp][gs_df['params'] == params]
            if log_scale == True:
                x = np.log(x.astype('float'))
            train = gs_df['mean_train_score'][gs_df['params'] == params]
            validation = gs_df['mean_test_score'][gs_df['params'] == params]

            if plot_stdev==True:
                train_err = gs_df['std_train_score'][gs_df['params'] == params]
                validation_err = gs_df['std_test_score'][gs_df['params'] == params]
                ax.errorbar(x, train, train_err, color=colors[i], marker='o',
                            label='Train, '+str(params), linestyle='--', ecolor='k',
                            capsize=4)
                ax.errorbar(x, validation, validation_err, color=colors[i],
                            marker='o', label='Validation, ' + str(params),
                            linestyle='-', ecolor='k', capsize=4)
            else:
                ax.plot(x, train, color=colors[i], marker='o',
                            label='Train, ' + str(params), linestyle='--')
                ax.plot(x, validation, color=colors[i], marker='o',
                            label='Validation, ' + str(params), linestyle='-')
    if log_scale == False:
        ax.set_xlabel(primary_hp)
    else:
        ax.set_xlabel('log(' + primary_hp + ')')
    ax.set_ylabel('Score')
    ax.grid()
    if leg_outside:
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    else:
        ax.legend(loc='best')
    plt.show()
