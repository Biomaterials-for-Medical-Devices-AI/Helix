#Volcano_plot 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

volc_plot = pd.read_csv("Trap_vs_DUPI_Core.csv", index_col=0)
group_labels = volc_plot.iloc[0, 1:].values

feature_names = volc_plot.iloc[1:, 0].values

#extract intensity matrix 
intensity_data = volc_plot.iloc[1:,1:].astype(float)

# Basically, I have sliced the dataframe to separate text data from numbers
unique_groups= np.unique(group_labels)
print("Groups identified in dataset:", unique_groups)

g1_h = group_labels == unique_groups[1]
g1_d = (group_labels == unique_groups[0])


print("g1_d length:", len(g1_d))  # Will print 20
print("g1_h length:", len(g1_h))  # Will print 20

# make empty lists to store calculated results for each row

log2fc_list = []
pvalue_list = []

g1_d = np.array(g1_d).ravel()
g1_h = np.array(g1_h).ravel()

for i in range(len(intensity_data)):
    row_values = intensity_data.iloc[i].values.astype(float)

    g1_raw = row_values[g1_d]
    g2_raw = row_values[g1_h]
    
    mean_g1 = np.nanmean(g1_raw)
    mean_g2 = np.nanmean(g2_raw)
   
    # calculate log 2 fold change and P-value
   # first log 2 fold change
    log2fc = np.log2((mean_g2 + 1e-6)/ (mean_g1 + 1e-6))
    log2fc_list.append(log2fc)

    #calculate p-value 
    g1_c = g1_raw[~np.isnan(g1_raw)]
    g2_c = g2_raw[~np.isnan(g2_raw)]

    _, p_val = stats.ttest_ind(g1_c, g2_c, equal_var=False)
    pvalue_list.append(p_val)

print(f"Features: {len(feature_names)}")
print(f"Log2FCs: {len(log2fc_list)}")
print(f"P-values: {len(pvalue_list)}")
# Assemble results into a clean Pandas DataFrame
results_volc_plot = pd.DataFrame({
    "Feature": feature_names,
    "Log2FC": log2fc_list,
    "pvalue": pvalue_list,
    "neg_log10_p": -np.log10(pvalue_list),})
print(results_volc_plot.head())

# Define significance tresholds
foldchange_tresh = 1.0
p_tresh = -np.log10(0.05)
# categorise using boolean indexing??
results_volc_plot["Group"] = "Not significant"
results_volc_plot.loc[
    (results_volc_plot["Log2FC"] > foldchange_tresh) & (results_volc_plot["neg_log10_p"] > p_tresh), "Group",
] = "Upregulated"
results_volc_plot.loc[
    (results_volc_plot["Log2FC"] < -foldchange_tresh) & (results_volc_plot["neg_log10_p"] > p_tresh), "Group", 
] = "Downregulated"
results_volc_plot.head()
print(results_volc_plot.columns.tolist()) 

#Fix index alignment
results_volc_plot = results_volc_plot.reset_index(drop=True)

# generate the scatter plot 
plt.figure(figsize=(10, 9))
sns.set_style("whitegrid")

# color palette 
color_dict = {
    "Not significant": "grey",
    "Upregulated": "crimson",
    "Downregulated": "royalblue",
}
# scatterplot
scatter_plot  = sns.scatterplot(
    data = results_volc_plot,
    x="Log2FC",
    y="neg_log10_p",
    hue="Group",
    palette=color_dict,
    alpha=0.4,
    s=60,
    edgecolor="none",
)
# Add treshold cutoff lines
plt.axvline(x=foldchange_tresh, color="black", linestyle="--", linewidth=1, label="Foldchange Treshold")
plt.axvline(x=-foldchange_tresh, color="black", linestyle="--", linewidth=1)
plt.axhline(
    y=p_tresh,
    color="black",
    linestyle=":",
    linewidth=1,
    label="p = 0.05 Treshold",
)
# add title and axis labels 
plt.title("Volcano Plot", fontsize = 16, fontweight= "bold", pad = 15)
plt.xlabel("Log2(FC)", fontsize=12)
plt.ylabel("-10log10(p-value)", fontsize=12)

plt.xlim(-5, 5)
plt.ylim(0, 7)

#label top points
top_hits = (
    results_volc_plot[results_volc_plot["Group"] != "Not significant"]
    .sort_values(by="neg_log10_p", ascending=False)
    .head(4)
)
for _, row in top_hits.iterrows():
    plt.text(
        row["Log2FC"]  +0.05,
        row["neg_log10_p"],
        row["Feature"],
        fontsize= 8,
        alpha= 0.85,
    )
# adjust layout and display legend
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0)
plt.tight_layout()
plt.show()