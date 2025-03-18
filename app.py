import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from scipy import stats

# Load the first 1000 rows of the dataset
@st.cache_data
def load_data():
    file_path = "geneExpressionData.xlsx"  # Ensure the file is in the same directory
    df = pd.read_excel(file_path, nrows=1000)
    return df

df = load_data()

# Identify columns for each group
control_cols = [col for col in df.columns if col.startswith("Control")]
p250_cols = [col for col in df.columns if col.startswith("P250")]
p350_cols = [col for col in df.columns if col.startswith("P350")]

st.title("Organoids Gene Expression Filtering & Visualization App")

# ---------------------------
# Sidebar: Filter Options
# ---------------------------
st.sidebar.header("Filter Options")

# Gene ID filter
selected_gene = st.sidebar.selectbox("Select Gene ID", ["All"] + sorted(df["gene_id"].astype(str).unique()))

# Expression range filtering for Control group
st.sidebar.subheader("Control Group Filtering")
control_min, control_max = st.sidebar.slider(
    "Control Expression Range",
    int(df[control_cols].min().min()), int(df[control_cols].max().max()),
    (0, 5000)
)

# Expression range filtering for P250kPa group
st.sidebar.subheader("P250kPa Organoids Filtering")
p250_min, p250_max = st.sidebar.slider(
    "P250kPa Expression Range",
    int(df[p250_cols].min().min()), int(df[p250_cols].max().max()),
    (0, 5000)
)

# Expression range filtering for P350kPa group
st.sidebar.subheader("P350kPa Organoids Filtering")
p350_min, p350_max = st.sidebar.slider(
    "P350kPa Expression Range",
    int(df[p350_cols].min().min()), int(df[p350_cols].max().max()),
    (0, 5000)
)

# ---------------------------
# Sidebar: Advanced Filters
# ---------------------------
st.sidebar.header("Advanced Filters")

# 1. Low Expression & Variance Filters
min_mean_expr = st.sidebar.number_input("Minimum Mean Expression", min_value=0, value=10)
min_variance = st.sidebar.number_input("Minimum Variance", min_value=0.0, value=5.0)

# 2. Differential Expression Filtering (Fold Change)
st.sidebar.subheader("Differential Expression Filtering")
# List available conditions (excluding 'gene_id')
all_conditions = df.columns[1:]
condition1 = st.sidebar.selectbox("Condition 1 (numerator)", all_conditions, 
                                  index=all_conditions.get_loc("P250kPa_Hi1") if "P250kPa_Hi1" in all_conditions else 0)
condition2 = st.sidebar.selectbox("Condition 2 (denom)", all_conditions, 
                                  index=all_conditions.get_loc("Control_1") if "Control_1" in all_conditions else 0)
fc_threshold = st.sidebar.number_input("Log₂ Fold Change Threshold", min_value=0.0, value=1.0)

# 3. Outlier Removal
st.sidebar.header("Outlier Removal")
remove_outliers = st.sidebar.checkbox("Remove Outliers (Z-score filtering)", value=False)
z_threshold = st.sidebar.number_input("Z-score Threshold", min_value=0.0, value=3.0)

# ---------------------------
# Data Filtering Pipeline
# ---------------------------
filtered_df = df.copy()

# (a) Filter by Gene ID if a specific gene is selected
if selected_gene != "All":
    filtered_df = filtered_df[filtered_df["gene_id"].astype(str) == selected_gene]

# (b) Apply expression range filters per group using masks
control_mask = filtered_df[control_cols].applymap(lambda x: control_min <= x <= control_max)
p250_mask = filtered_df[p250_cols].applymap(lambda x: p250_min <= x <= p250_max)
p350_mask = filtered_df[p350_cols].applymap(lambda x: p350_min <= x <= p350_max)
filtered_df = filtered_df[control_mask.any(axis=1) | p250_mask.any(axis=1) | p350_mask.any(axis=1)]

# (c) Remove lowly expressed genes based on mean expression (across all expression columns)
mean_expr = filtered_df.iloc[:, 1:].mean(axis=1)
filtered_df = filtered_df[mean_expr >= min_mean_expr]

# (d) Remove genes with low variance
variance_expr = filtered_df.iloc[:, 1:].var(axis=1)
filtered_df = filtered_df[variance_expr >= min_variance]

# (e) Differential Expression Filtering:
# Compute log₂ fold change between two selected conditions (add a small constant to avoid log(0))
epsilon = 1e-6
filtered_df["log2FC"] = np.log2(filtered_df[condition1] + epsilon) - np.log2(filtered_df[condition2] + epsilon)
filtered_df = filtered_df[abs(filtered_df["log2FC"]) >= fc_threshold]

# (f) Outlier Removal: Remove genes with extreme z-scores across expression columns
if remove_outliers:
    expr_cols = filtered_df.columns.difference(["gene_id", "log2FC"])
    z_scores = np.abs(stats.zscore(filtered_df[expr_cols], nan_policy='omit'))
    filtered_df = filtered_df[(z_scores < z_threshold).all(axis=1)]

st.write(f"### Filtered Data ({len(filtered_df)} results)")
st.dataframe(filtered_df)

# ---------------------------
# Visualization Section
# ---------------------------
st.write("## Visualization")

# If a specific gene is selected, show its expression as a bar chart
if selected_gene != "All" and not filtered_df.empty:
    gene_row = filtered_df.iloc[0]  # Only one row should match
    gene_data = gene_row.drop(["gene_id", "log2FC"])
    gene_df = gene_data.reset_index()
    gene_df.columns = ["Condition", "Expression"]
    
    st.write(f"### Expression for Gene {selected_gene}")
    bar_chart = alt.Chart(gene_df).mark_bar().encode(
        x=alt.X('Condition:N', sort=None),
        y=alt.Y('Expression:Q'),
        tooltip=['Condition', 'Expression']
    ).properties(width=800, height=400)
    st.altair_chart(bar_chart)

else:
    # Overall visualization for all filtered genes
    if not filtered_df.empty:
        # Convert data to long format for visualization
        melted_df = filtered_df.melt(id_vars=["gene_id", "log2FC"], var_name="Condition", value_name="Expression")
        
        st.write("### Overall Expression Distribution")
        histogram = alt.Chart(melted_df).mark_bar().encode(
            alt.X("Expression:Q", bin=alt.Bin(maxbins=50)),
            y='count()',
            tooltip=['count()']
        ).properties(width=800, height=400)
        st.altair_chart(histogram)
        
        st.write("### Box Plot by Condition")
        boxplot = alt.Chart(melted_df).mark_boxplot().encode(
            x='Condition:N',
            y='Expression:Q',
            tooltip=['Condition', 'Expression']
        ).properties(width=800, height=400)
        st.altair_chart(boxplot)
    else:
        st.write("No data available after filtering. Adjust your filters.")
