import seaborn as sns
import matplotlib.pyplot as plt

def BoxPlot(x, y=None, hue=None, title=None):
    sns.boxplot(x=x, y=y, hue=hue)
    if title:
        plt.title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()

def ViolinPlot(x, y=None, hue=None, title=None):
    sns.violinplot(x=x, y=y, hue=hue)
    if title:
        plt.title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()

def HistogramPlot(x, y=None, hue=None, title=None):
    sns.histplot(x=x, y=y, hue=hue)
    if title:
        plt.title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()