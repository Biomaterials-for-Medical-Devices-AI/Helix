# Data Visualisation
The **Data Visualisation** page allows you to explore and analyse your data using a variety of visual tools. This helps in understanding data distributions, correlations, and feature interactions.

![Data Visualisation Page](../_static/data-visualisation-page.png)

To begin, select your experiment from the dropdown menu where it says **"Select an experiment"**. This will load the dataset associated with your selected experiment.

## Dataset Overview
The dataset overview gives you a snapshot of how your data look. This is split into two tabs: one showing the **Raw Data** data (i.e. how it looked before preprocessing) and one showing the **Preprocessed Data** (i.e. how the data are after preprocessing). The overview also includes a normality test for each variable in the dataset (independent and dependent). Normality is determined by two statistical tests: the Shapiro-Wilk and the Kolmogorov-Smirnov test. If the p-value < 0.05, the data is likely not normally distributed. If the p-value ≥ 0.05, the data might be normally distributed.

![Dataset Overview](../_static/dataset-overview.png)
![Normality Tests](../_static/normality-tests.png)

## Target Variable Distribution
Visualise the distribution of your target variable using histograms and KDE (Kernel Density Estimation).

- **Show KDE**: Toggle to include a KDE curve.
- **Number of Bins**: Adjust the number of bins in the histogram.

![Target Variable Distribution](../_static/target-variable-distribution.png)

## Correlation Heatmap
Display a heatmap showing correlations between selected features.

- **Select All Descriptors**: Include all features in the heatmap.
- **Select Columns**: Manually select specific features for the heatmap.

![Correlation Heatmap](../_static/correlation-heatmap.png)

## Pairplot
Generate pairwise scatter plots for selected features to identify trends and interactions.

- **Select All Descriptors**: Include all features in the pairplot.
- **Select Columns**: Manually select specific features for pairplot generation.

![Pairplot](../_static/pairplot.png)

## t-SNE Plot
Visualise high-dimensional data in two dimensions using t-SNE.

![t-SNE Plot](../_static/tsne-plot.png)

## Volcano Plot
Compare statistical significance with fold-change for many labelled samples.

- **Enter the Fold Change (FC) Threshold**: Configure the location of the vertical lines. Represents significant magnitude of change.
- **Enter the p-value threshold**: Configure the location of the horizontal line. Represents the significance level
- **Select the logarithm base**: Choice of 2, 10 or e for calculating the logarithm of the fold change.
- **Check each feature for normality**: If this toggle is on, all features will be checked for normality. If a feature is normal, its p-value will be calculated using a t-test, otherwise, a Mann-Whitney U Test will be used. If this toggle is off, all features are assumed to be normally distributed, and all p-values are calculated using a t-test.
- **Use FDR correction**: Uses the Benjamini-Hochberg procedure to control the False Discovery Rate (limit number of false positives).

![Volcano Plot](../_static/volcano-plot.png)

![Volcano Plot Data Table](../_static/volcano-plot-table.png)

## Saving Visualisations
You can save generated plots and tables to disk for reporting or further analysis. To save a plot, click the **"Save Plot"** button beneath each plot. To save a table, in the case of a Volcano Plot, click the **"Save Table"** button beneath the table. You can also edit each plot individually by clicking the **"Edit Plot"** button beneath each plot and changing the settings in the panel.

## How to Start
1. Select an experiment.
2. Choose the visualisation type.
3. Adjust parameters as needed.
4. Click the buttons to generate and save visualisations.

Press **"Create and Save"** to save plots or tables for further use.
