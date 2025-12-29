"""
Apply each clustering algorithm and save separate CSV files.
"""

from pathlib import Path
import pandas as pd

from clustering import (
    kmeans_clustering,
    gmm_clustering,
    pca_kmeans_clustering,
    hdbscan_clustering,
)


def main():
    # Setup paths - assuming script is in src/ directory
    src_dir = Path(__file__).parent
    project_dir = src_dir.parent
    data_dir = project_dir / "data"
    
    # Load modeling data
    input_path = data_dir / "modeling_data.csv"
    
    if not input_path.exists():
        print(f"Error: {input_path} not found!")
        print("Please ensure modeling_data.csv exists in the data/ directory")
        return
    
    print("Loading modeling_data.csv...")
    data = pd.read_csv(input_path, parse_dates=["Date"])
    data = data.set_index(["Date", "Ticker"]).sort_index()
    
    print(f"✓ Loaded {len(data):,} observations")
    print(f"  Date range: {data.index.get_level_values('Date').min()} to {data.index.get_level_values('Date').max()}")
    print(f"  Unique tickers: {data.index.get_level_values('Ticker').nunique()}")
    
    # Feature columns for clustering
    feature_cols = [
        "gkv", "rsi", "bblow", "bbmid", "bbhigh", "atr", "macd", "dv_rank",
        "return_1m", "return_2m", "return_3m", "return_6m", "return_9m", "return_12m",
        "beta_mkt", "beta_smb", "beta_hml", "beta_rmw", "beta_cma",
    ]
    
    print(f"\nUsing {len(feature_cols)} features for clustering")
    print("\n" + "="*60)
    
    # 1. K-MEANS
    print("\n1. Applying K-Means clustering (5 clusters)...")
    kmeans_data = kmeans_clustering(
        data=data,
        feature_cols=feature_cols,
        n_clusters=5,
        random_state=42,
    )
    output_path = data_dir / "clustered_kmeans.csv"
    kmeans_data.to_csv(output_path)
    print(f"   ✓ Saved to: {output_path}")
    print(f"   Clusters found: {kmeans_data['cluster'].nunique()}")
    
    # 2. GMM
    print("\n2. Applying GMM clustering (5 components)...")
    gmm_data = gmm_clustering(
        data=data,
        feature_cols=feature_cols,
        n_components=5,
        random_state=42,
    )
    output_path = data_dir / "clustered_gmm.csv"
    gmm_data.to_csv(output_path)
    print(f"   ✓ Saved to: {output_path}")
    print(f"   Clusters found: {gmm_data['cluster'].nunique()}")
    
    # 3. PCA + K-MEANS
    print("\n3. Applying PCA + K-Means clustering (6 components → 5 clusters)...")
    pca_kmeans_data = pca_kmeans_clustering(
        data=data,
        feature_cols=feature_cols,
        n_clusters=5,
        n_components=6,
        random_state=42,
    )
    output_path = data_dir / "clustered_pca_kmeans.csv"
    pca_kmeans_data.to_csv(output_path)
    print(f"   ✓ Saved to: {output_path}")
    print(f"   Clusters found: {pca_kmeans_data['cluster'].nunique()}")
    
    # 4. HDBSCAN
    print("\n4. Applying HDBSCAN clustering (min_cluster_size=15)...")
    hdbscan_data = hdbscan_clustering(
        data=data,
        feature_cols=feature_cols,
        min_cluster_size=5,
        min_samples=3,
    )
    output_path = data_dir / "clustered_hdbscan.csv"
    hdbscan_data.to_csv(output_path)
    print(f"   ✓ Saved to: {output_path}")
    print(f"   Clusters found: {hdbscan_data['cluster'].nunique()}")
    
    # Count noise points for HDBSCAN
    n_noise = (hdbscan_data['cluster'] == -1).sum()
    print(f"   Noise points: {n_noise:,} ({n_noise/len(hdbscan_data)*100:.1f}%)")
    
    print("\n" + "="*60)
    print("\n✓ All clustering methods applied successfully!")
    print(f"\nGenerated files in {data_dir}:")
    print("  - clustered_kmeans.csv")
    print("  - clustered_gmm.csv")
    print("  - clustered_pca_kmeans.csv")
    print("  - clustered_hdbscan.csv")


if __name__ == "__main__":
    main()