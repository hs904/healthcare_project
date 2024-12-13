import matplotlib.pyplot as plt
import seaborn as sns

def plot_distributions(df, column):
    """
    Plot the boxplot and histogram for a given column.
    
    Parameters:
    df (pd.DataFrame): The dataset.
    column (str): The column to visualize.
    """
    plt.figure(figsize=(14, 6))

    # Box plot
    plt.subplot(1, 2, 1)
    sns.boxplot(data=df, x=column)
    plt.title(f"Boxplot of {column}")
    
    # Histogram
    plt.subplot(1, 2, 2)
    sns.histplot(data=df, x=column, kde=True, bins=30, color='blue')
    plt.title(f"Histogram of {column}")

    plt.tight_layout()
    plt.show()


def plot_target_distribution(df, target_column):
    """Plot the distribution of the target variable."""
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x=target_column)
    plt.title("Target Variable Distribution")
    plt.show()

def plot_correlation_matrix(df):
    """Plot a correlation matrix of the dataset."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix")
    plt.show()

def plot_feature_vs_target(df, feature, target):
    """Plot a feature against the target variable."""
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x=target, y=feature)
    plt.title(f"{feature} vs {target}")
    plt.show()

def plot_categorical_vs_target(df, categorical_feature, target):
    """Plot categorical features against the target variable."""
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x=categorical_feature, hue=target, palette="viridis")
    plt.title(f"{categorical_feature.capitalize()} vs {target.capitalize()}")
    plt.xlabel(categorical_feature.capitalize())
    plt.ylabel("Count")
    plt.legend(title=target.capitalize())
    plt.show()