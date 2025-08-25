import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns



#_________________________________________________________________
# Load Dataset
#_________________________________________________________________
@st.cache_data
def load_data():
    return pd.read_csv("new.csv")

data = load_data()

#__________________________________________________________________
# Sidebar UI
#__________________________________________________________________
st.sidebar.header("Customer Segmentation App")

# Select number of clusters
n_clusters = st.sidebar.slider("Number of Clusters (k)",min_value=2,max_value=10,value=3,step=1,help="Choose the number of clusters")

#___________________________________________________________________
# Main UI
#___________________________________________________________________
st.title("Customer Segmentation using K-Means")
st.markdown("This app segments customers based on their **behavioural and spending patterns**.")

# Show dataset preview
st.write("### Original Data Preview")
st.dataframe(data.head())

#___________________________________________________________________
# Select Features
#___________________________________________________________________
numeric_features =["Income","Kidhome","Teenhome","Recency","MntWines","MntFruits","MntMeatProducts","MntFishProducts","MntSweetProducts","MntGoldProds","NumDealsPurchases","NumWebPurchases","NumCatalogPurchases","NumStorePurchases","NumWebVisitsMonth"]

selected_features = st.multiselect("Select features for clustering:",numeric_features,default=["Income","MntWines","MntFruits","MntMeatProducts","MntFishProducts","MntSweetProducts","MntGoldProds","NumDealsPurchases","NumWebPurchases","NumStorePurchases","NumWebVisitsMonth"])

if len(selected_features) >=2:
    X=data[selected_features].dropna()

    # Scale Data
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    #________________________________________________________________
    # Elbow Method
    #________________________________________________________________
    from sklearn.cluster import KMeans
    st.write("### Elbow Method (to find optimal k)")
    inertia = []
    K = range(2,11)
    for k in K:
        kmeans_model = KMeans(n_clusters=k, random_state=42)
        kmeans_model.fit(X_scaled)
        inertia.append(kmeans_model.inertia_)

    fig,ax =plt.subplots()
    ax.plot(K,inertia,marker="o")
    ax.set_xlabel("Number of Clusters(k)")
    ax.set_ylabel("Inertia (Within-cluster SSE)")
    ax.set_title("Elbow Method")
    st.pyplot(fig)

    #_____________________________________________________________________
    # K Means
    #_____________________________________________________________________

    st.subheader(f"K-Means Clustering with k={n_clusters} clusters")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)

    data["Cluster"] = -1
    data.loc[X.index,"Cluster"] = clusters

    # Show Cluster centers
    st.write("### Cluster Centers")
    centers = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_),columns=selected_features)

    st.dataframe(centers)

    # Show cluster sizes
    st.write("### Cluster Sizes")
    st.write(data["Cluster"].value_counts().sort_index())

    #________________________________________________________________________
    # Dropdowns for feature selection
    #________________________________________________________________________
    st.write("### Select Features for Visualization")

    x_feature = st.selectbox("Select feature for X-axis:",selected_features, index=0)
    y_feature = st.selectbox("Select feature for Y-axis:",selected_features, index=1)
    

    #________________________________________________________________________
    # Visualization
    #________________________________________________________________________
    import numpy as np
    C_X = centers[x_feature].to_numpy()
    C_Y = centers[y_feature].to_numpy()

# Format with commas and round with 2 decimals
    C_X_str = ",".join(map(str,np.round(C_X, 2)))
    st.write(f"###### Cluster Centers of X-axis ({x_feature}):" ,C_X_str)

    C_Y_str = ",".join(map(str, np.round(C_Y, 2)))
    st.write(f"###### Cluster Centers of Y-axis ({y_feature}):", C_Y_str)


    st.write("### Cluster Visualization ( With selected features)")

    if len(selected_features) >= 2:
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = ["gold", "navy", "gray", "orange", "purple", "brown", "cyan", "teal", "olive", "pink"]
        for cluster_id in range(n_clusters):
            cluster_points = data[data["Cluster"] == cluster_id]
            ax.scatter(
                cluster_points[selected_features[selected_features.index(x_feature)]],
                cluster_points[selected_features[selected_features.index(y_feature)]],
                c=colors[cluster_id % len(colors)],
                label=f"Cluster {cluster_id}",
                alpha=0.6
            )


        ax.scatter(C_X, C_Y,c="red",s=50,marker="o",label="Cluster Centers",edgecolors="k")

        ax.set_xlabel(x_feature)
        ax.set_ylabel(y_feature)
        ax.set_title(f"Customer Segments (k={n_clusters})")
        ax.legend()
        st.pyplot(fig)


#_____________________________________________________________
# Final Clustered Data
#_____________________________________________________________
cols =list(data.columns)

if "ID" in cols and "Cluster" in cols:
    id_index = cols.index("ID")
    cols.remove("Cluster")
    cols.insert(id_index +1,"Cluster")
    data =data[cols]

st.write("### Data with Cluster Labels")
st.dataframe(data.head(20))


st.warning("⚠ Please select at least 2 features for clustering.")

    



        
        



    
    
        


