import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Streamlit App Configuration ---
st.set_page_config(page_title="K-Means Clustering App", layout="centered")
st.title("K-Means Clustering Visualization")
st.write("Upload your CSV file and visualize K-Means clustering interactively.")

#__________________________________________________________________
# Sidebar UI
#__________________________________________________________________
st.sidebar.header("Customer Segmentation App")

# Select number of clusters
n_clusters = st.sidebar.slider("Number of Clusters (k)",min_value=2,max_value=10,value=3,step=1,help="Choose the number of clusters")
# File upload
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Read CSV
    df = pd.read_csv(uploaded_file)
    st.subheader("Original Data Preview")
    st.write(df.head())
    # Drop rows with missing values
    df = df.dropna()

    # Feature selection
    st.sidebar.markdown("---")
    st.sidebar.subheader("Select Features for Clustering")
    all_columns = df.columns.tolist()
    feature1 = st.sidebar.selectbox("Select Feature 1 (X-axis)", all_columns)
    feature2 = st.sidebar.selectbox("Select Feature 2 (Y-axis)", all_columns)
    # Ensure numeric features
    if pd.api.types.is_numeric_dtype(df[feature1]) and pd.api.types.is_numeric_dtype(df[feature2]):
        X = df[[feature1, feature2]].to_numpy()

        # --- Run KMeans ---
        from sklearn.cluster import KMeans

        # --- Elbow Method ---
        st.write("### Elbow Method (Optimal k Finder)")

        max_k = st.sidebar.slider("Max k for Elbow Method", min_value=5, max_value=15, value=10)
        inertias = []

        X_elbow = df[[feature1, feature2]].to_numpy()

        for k in range(1, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
            kmeans.fit(X_elbow)
            inertias.append(kmeans.inertia_)

        # Plot elbow curve
        fig, ax = plt.subplots()
        ax.plot(range(1, max_k + 1), inertias, marker="o", linestyle="--")
        ax.set_xlabel("Number of Clusters (k)")
        ax.set_ylabel("Inertia (WCSS)")
        ax.set_title("Elbow Method for Optimal k")
        st.pyplot(fig)


       # Move Cluster to the first column
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
        df["Cluster"] = kmeans.fit_predict(X)
        cluster_col = df.pop("Cluster")
        df.insert(0, "Cluster", cluster_col)

            # Cluster centers
        centers = kmeans.cluster_centers_

        C_X = centers[:,0]
        C_Y = centers[:,1]
        # Create a DataFrame with feature names
        centers_df = pd.DataFrame(centers, columns=[feature1, feature2])

        st.write("### Centers")
        st.write(centers_df)

        # Show cluster sizes
        st.write("### Cluster Sizes")
        st.write(df["Cluster"].value_counts().sort_index())

        st.write("### Cluster Centers")
        # Format with commas and round with 2 decimals
        C_X_str = ",".join(map(str, np.round(C_X, 2)))
        st.write(f"###### Cluster Centers of X-axis ({feature1}):", C_X_str)

        C_Y_str = ",".join(map(str, np.round(C_Y, 2)))
        st.write(f"###### Cluster Centers of Y-axis ({feature2}):", C_Y_str)




            # --- Visualization ---
        st.write("### Cluster Visualization")
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = ['gold',"navy", "gray", "orange", "purple", "brown", "cyan", "teal", "olive", "pink"]
        for cluster_id in range(n_clusters):
            cluster_points = df[df["Cluster"] == cluster_id]
            ax.scatter(
                cluster_points[feature1],
                cluster_points[feature2],
                s=50,
                c=colors[cluster_id % len(colors)],
                label=f"Cluster {cluster_id}"
            )

                # Plot cluster centers
        ax.scatter(C_X, C_Y, c="red", marker="o", s=50, label="Centers")
        ax.set_xlabel(feature1)
        ax.set_ylabel(feature2)
        ax.set_title("K-Means Clustering")
        ax.legend()
        st.pyplot(fig)

            # --- Final Data ---
        st.write("### Final Data with Cluster Labels")
        st.write(df.head())

        st.warning("⚠ Please select at least 2 features for clustering.")






