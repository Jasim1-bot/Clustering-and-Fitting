import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression


# Relational Plot
def plot_relational_plot(df):
    """Creates a scatterplot comparing Anxiety_Score and Depression_Score."""
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df, x='Anxiety_Score', y='Depression_Score',
        hue='Stress_Level', palette='viridis'
    )
    plt.title('Relational Plot: Anxiety vs Depression Score')
    plt.savefig('relational_plot.png')
    plt.close()


# Additional Relational Plot: Line Plot
def plot_line_plot(df):
    """Creates a line plot showing Anxiety Score by Age."""
    plt.figure(figsize=(8, 6))
    sns.lineplot(
        data=df, x='Age', y='Anxiety_Score', hue='Gender', marker='o'
    )
    plt.title('Line Plot: Anxiety Score by Age')
    plt.savefig('line_plot.png')
    plt.close()


# Categorical Plot
def plot_categorical_plot(df):
    """Creates a bar plot showing average Stress_Level by Employment_Status."""
    plt.figure(figsize=(8, 6))
    sns.barplot(
        data=df, x='Employment_Status', y='Stress_Level',
        errorbar=None, palette='pastel'
    )
    plt.title('Categorical Plot: Average Stress Level by Employment Status')
    plt.xticks(rotation=45)
    plt.savefig('categorical_plot.png')
    plt.close()


# Additional Categorical Plot: Pie Chart
def plot_pie_chart(df):
    """Creates a pie chart showing distribution of Employment Status."""
    plt.figure(figsize=(8, 6))
    data = df['Employment_Status'].value_counts()
    plt.pie(
        data, labels=data.index, autopct='%1.1f%%',
        startangle=90, colors=sns.color_palette('pastel')
    )
    plt.title('Pie Chart: Employment Status Distribution')
    plt.savefig('pie_chart.png')
    plt.close()


# Statistical Plots
def plot_statistical_plot(df):
    """Creates a heatmap of the correlation matrix."""
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Statistical Plot: Correlation Heatmap of Numeric Features')
    plt.savefig('statistical_plot.png')
    plt.close()


# Statistical Analysis
def statistical_analysis(df, col: str):
    """Calculates the four main statistical moments."""
    return (
        df[col].mean(),
        df[col].std(),
        df[col].skew(),
        df[col].kurt()
    )


# Preprocessing
def preprocessing(df):
    """Preprocesses the data by handling missing values."""
    return df.dropna()


# Writing Statistical Moments
def writing(moments, col):
    """Prints the analysis of statistical moments for a given column."""
    print(f'For the attribute {col}:')
    print(
        f'Mean = {moments[0]:.2f}, Standard Deviation = {moments[1]:.2f}, '
        f'Skewness = {moments[2]:.2f}, Excess Kurtosis = {moments[3]:.2f}.'
    )
    skew_desc = (
        "not skewed" if abs(moments[2]) < 0.5
        else "right-skewed" if moments[2] > 0
        else "left-skewed"
    )
    kurtosis_desc = (
        "mesokurtic" if abs(moments[3]) < 0.5
        else "leptokurtic" if moments[3] > 0
        else "platykurtic"
    )
    print(f'The data was {skew_desc} and {kurtosis_desc}.')


# Clustering
def perform_clustering(df, col1, col2):
    """Performs K-Means clustering on two selected variables."""
    data = df[[col1, col2]].values
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    distortions = [
        KMeans(n_clusters=k, random_state=42).fit(scaled_data).inertia_
        for k in range(1, 10)
    ]
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, 10), distortions, marker='o')
    plt.title('Elbow Method')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Distortion')
    plt.savefig('elbow_plot.png')
    plt.close()
    kmeans = KMeans(n_clusters=3, random_state=42)
    labels = kmeans.fit_predict(scaled_data)
    return labels, scaled_data, kmeans.cluster_centers_


# Clustering Plot
def plot_clustered_data(labels, data, centers):
    """Plots clustered data with centroids."""
    plt.figure(figsize=(8, 6))
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', alpha=0.6)
    plt.scatter(
    centers[:, 0], centers[:, 1], s=200, c='red',
    label='Centroids'
)
    plt.title('Clustered Data')
    plt.legend()
    plt.savefig('clustering.png')
    plt.close()


# Linear Fitting
def perform_fitting(df, col1, col2):
    """Fits a linear regression model."""
    x = df[col1].values.reshape(-1, 1)
    y = df[col2].values
    model = LinearRegression()
    model.fit(x, y)
    return x, y, model.predict(x)


# Linear Regression Plot
def plot_fitted_data(x, y, predictions):
    """Plots original data and the regression line."""
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, label='Original Data', alpha=0.7)
    plt.plot(x, predictions, color='red', label='Regression Line')
    plt.title('Fitted Data')
    plt.legend()
    plt.savefig('fitting.png')
    plt.close()


# Main Function
def main():
    df = pd.read_csv('data.csv')
    df = preprocessing(df)
    col = 'Depression_Score'
    moments = statistical_analysis(df, col)
    writing(moments, col)
    plot_relational_plot(df)
    plot_line_plot(df)
    plot_statistical_plot(df)
    plot_categorical_plot(df)
    plot_pie_chart(df)
    clustering_results = perform_clustering(
    df, 'Anxiety_Score', 'Depression_Score'
)
    plot_clustered_data(*clustering_results)
    x, y, predictions = perform_fitting(df, 'Sleep_Hours', 'Depression_Score')
    plot_fitted_data(x, y, predictions)


if __name__ == '__main__':
    main()
