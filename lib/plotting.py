import numpy as np 
import pandas as pd

import matplotlib.pyplot as plt 
import seaborn as sns
import plotly.graph_objects as go


def missing_value_plot(df: pd.DataFrame, cmap='viridis'):
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.isnull(), cbar=False, cmap=cmap)
    plt.title('Missing Value Plot')
    plt.show()


# Compare Correlation 

def plot_correlation_heatmap(df: pd.DataFrame, cmap='coolwarm'):
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap=cmap)
    plt.title('Correlation Heatmap')
    plt.show()

def compare_correlation(df1: pd.DataFrame, df2: pd.DataFrame, df1_name='First DataFrame', df2_name='Second DataFrame', cmap='coolwarm'):
    fig, ax = plt.subplots(1, 2, figsize=(15, 7))
    sns.heatmap(df1.corr(numeric_only=True), annot=True, cmap=cmap, ax=ax[0]) 
    ax[0].set_title(f'Correlation Heatmap for {df1_name}')
    sns.heatmap(df2.corr(numeric_only=True), annot=True, cmap=cmap, ax=ax[1])
    ax[1].set_title(f'Correlation Heatmap for {df2_name}')
    plt.show()


# Column Distribution Comparison

def plot_feature_distribution_comparison(source_df, target_df, source_label='source', target_label='target'):
    num_columns = source_df.shape[1]
    column_names = source_df.columns

    rows = int(np.ceil(num_columns/2))
    cols = 2

    fig_height = 3*rows
    fig, axs = plt.subplots(rows, cols, figsize=(10, fig_height))
    for i, column in enumerate(column_names):
        row = i // cols
        col = i % cols
        if pd.api.types.is_numeric_dtype(source_df[column]):
            sns.kdeplot(source_df[column], ax=axs[row][col], color="red", linewidth=2, label=source_label)
            sns.kdeplot(target_df[column], ax=axs[row][col], color="black", linewidth=1.5, label=target_label)
        else:
            sns.countplot(x=source_df[column], ax=axs[row][col], color="red", label=source_label, alpha=0.6)
            sns.countplot(x=target_df[column], ax=axs[row][col], color="black", label=target_label, alpha=0.3)
        axs[row][col].legend()

    plt.tight_layout()
    plt.show()

def plot_feature_distribution_comparison_multiple(dfs: list[pd.DataFrame], labels: list[str], cat_features: list, save=False, save_path=None):
    num_columns = dfs[0].shape[1]
    column_names = dfs[0].columns

    rows = int(np.ceil(num_columns / 2))
    cols = 2

    fig_height = 3 * rows
    fig, axs = plt.subplots(rows, cols, figsize=(10, fig_height))
    for i, column in enumerate(column_names):
        row = i // cols
        col = i % cols
        if column not in cat_features:
            for df, label in zip(dfs, labels):
                sns.kdeplot(df[column], ax=axs[row][col], linewidth=2, label=label)
        else:
            plot_data = []
            for df, label in zip(dfs, labels):
                counts = df[column].value_counts(normalize=True).reset_index()
                counts.columns = [column, 'Proportion']
                counts['Dataset'] = label
                plot_data.append(counts)

            plot_df = pd.concat(plot_data)
            # print(plot_df.head())

            sns.barplot(x=column, y='Proportion', hue='Dataset', data=plot_df, ax=axs[row][col], alpha=0.9)

            axs[row][col].set_xticklabels(axs[row][col].get_xticks(), rotation=45, ha='right')
 
        axs[row][col].legend()

    plt.tight_layout()
    plt.show()

    if save:
        if save_path is None:
            raise ValueError("Please provide a save path to save the plot.")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)

def optuna_timeline_plot(trials_df):
        
    first_trial_start = trials_df['datetime_start'].min()
    trials_df['timedelta_start'] = (trials_df['datetime_start'] - first_trial_start).dt.total_seconds()
    trials_df['timedelta_end'] = (trials_df['datetime_complete'] - first_trial_start).dt.total_seconds()
    trials_df['timedelta_duration'] = trials_df['timedelta_end'] - trials_df['timedelta_start']
    min_x_data = trials_df['timedelta_start'].min()
    max_x_data = (trials_df['timedelta_start'] + trials_df['timedelta_duration']).max() 

    time_unit = 's'  # seconds
    if max_x_data >= 3600:
        time_unit = 'h' 
        max_x_data /= 3600
        trials_df['timedelta_duration'] /= 3600
        trials_df['timedelta_start'] /= 3600
    elif max_x_data >= 60:
        time_unit = 'm'
        max_x_data /= 60
        trials_df['timedelta_duration'] /= 60
        trials_df['timedelta_start'] /= 60

    timeline_plot  = go.Figure()
    timeline_plot.add_trace(go.Bar(
        y=trials_df['number'],
        x=trials_df['timedelta_duration'],
        base=trials_df['timedelta_start'],
        orientation='h',
        marker = {'color': 'blue'},
    ))
    timeline_plot.update_layout(
        title_text=None,
        yaxis_title="Trial",
        xaxis_title=f"Duration ({time_unit})",
        xaxis=dict(
            range=[min(0, min_x_data - (max_x_data * 0.05)), max_x_data + (max_x_data * 0.05)],
        ),
    )

    return timeline_plot
