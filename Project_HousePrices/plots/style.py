import seaborn as sns
import matplotlib.pyplot as plt

def set_style():
    """Apply global Seaborn + Matplotlib style for all plots."""
    sns.set_theme(
        style="whitegrid",   # use "darkgrid" if you want stronger grid lines
        palette="Dark2"
    )
    plt.rcParams["figure.figsize"] = (8, 5)  # consistent figure size
    plt.rcParams["axes.grid"] = True         # ensure gridlines are on
