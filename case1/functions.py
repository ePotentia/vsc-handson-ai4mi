import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def plot_train_oob_val(hyperparam, train, oob, val):
    fig, ax  = plt.subplots(1,1, figsize=(6, 4))
    sns.lineplot(x=hyperparam, y=train, label='Training')
    sns.lineplot(x=hyperparam, y=oob, label='OOB')
    sns.lineplot(x=hyperparam, y=val, label='Validation')
    plt.xlabel('Hyperparameter value', fontsize=14, weight='bold')
    plt.ylabel('Accuracy', fontsize=14, weight='bold')
    plt.xticks(fontsize=12, weight=800)
    plt.yticks(fontsize=12, weight=800)
    ax.set_ylim((0.75,1.))
    plt.show()


n_colors = 256 # Use 256 colors for the diverging color palette
palette = sns.diverging_palette(20, 220, n=n_colors) # Create the palette
#palette = sns.color_palette("colorblind")
color_min, color_max = [-1, 1]

def value_to_color(val):
    val_position = float((val - color_min)) / (color_max - color_min) # position of value in the input range, relative to the length of the input range
    ind = int(val_position * (n_colors - 1)) # target index in the color palette
    return palette[ind]

def heatmap(x, y, size,fsize, **kwargs):
    if 'color' in kwargs:
        color = kwargs['color']
    else:
        color = [1]*len(x)
    fig, ax = plt.subplots(figsize=(fsize,fsize))
    
    # Mapping from column names to integer coordinates
    x_labels = [v for v in sorted(x.unique())]
    y_labels = [v for v in sorted(y.unique())]
    x_to_num = {p[1]:p[0] for p in enumerate(x_labels)} 
    y_to_num = {p[1]:p[0] for p in enumerate(y_labels)} 
    size_scale = 20*fsize
    ax.scatter(
        x=x.map(x_to_num), # Use mapping for x
        y=y.map(y_to_num), # Use mapping for y
        s=size.abs() * size_scale, # Vector of square sizes, proportional to size parameter
        c=[value_to_color(float(c)) for c in size],
        marker='o' # Use square as scatterplot marker
    )
    
    # Show column labels on the axes
    ax.set_xticks([x_to_num[v] for v in x_labels])
    ax.set_xticklabels(x_labels, rotation=45, horizontalalignment='right')
    ax.set_yticks([y_to_num[v] for v in y_labels])
    ax.set_yticklabels(y_labels)

def plot_correlation(df, input=None, method='spearman'):
    if not input:
        inp = df.columns
    else:
        inp = input
    # inp = ['X_Size', 'Y_Size', 'X_Center', 'Y_Center', 'Pixels_Areas',
    #     'X_Perimeter', 'Y_Perimeter','Sum_of_Luminosity',
    #     'Minimum_of_Luminosity', 'Maximum_of_Luminosity', 'Length_of_Conveyer',
    #     'TypeOfSteel_A300', 'Steel_Plate_Thickness',
    #     'Edges_Index', 'Empty_Index', 'Square_Index', 'Outside_X_Index',
    #     'Edges_X_Index', 'Edges_Y_Index', 'Outside_Global_Index',
    #         'Orientation_Index', 'Luminosity_Index', 'Bumps', 'Other_Faults']
    corr = df[inp].corr(method=method)
    corr = pd.melt(corr.reset_index(), id_vars='index') # Unpivot the dataframe, so we can get pair of arrays for x and y
    corr.columns = ['x', 'y', 'value']
    heatmap(
        x=corr['x'],
        y=corr['y'],
        size=corr['value'],
        fsize=12
    )