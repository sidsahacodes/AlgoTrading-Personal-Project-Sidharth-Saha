"""
Momentum Within Clusters Strategy

Strategy Logic:
- Each month, within each cluster, rank stocks by momentum signal
- Long top N stocks per cluster
- Short bottom N stocks per cluster
- Equal-weight positions
- Monthly rebalancing
"""

from pathlib import Path
import pandas as pd
import numpy as np


def calculate_forward_returns(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate forward 1-month returns for each stock."""
    data = data.copy()
    
    # Sort by ticker and date to ensure proper shifting
    data = data.sort_index()
    
    # Forward return = next month's return
    data['forward_return_1m'] = (
        data.groupby('Ticker')['return_1m']
        .shift(-1)  # Shift backwards to get next month's return
    )
    
    return data


def generate_momentum_signals(
    data: pd.DataFrame,
    momentum_feature: str = 'return_3m',
    top_n: int = 3,
    bottom_n: int = 3,
) -> pd.DataFrame:
    """
    Generate long/short signals based on momentum within clusters.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Clustered data with Date and Ticker index
    momentum_feature : str
        Feature to rank on (e.g., 'return_3m', 'return_6m')
    top_n : int
        Number of top stocks to long per cluster
    bottom_n : int
        Number of bottom stocks to short per cluster
    
    Returns:
    --------
    pd.DataFrame with 'signal' column: 1 (long), -1 (short), 0 (no position)
    """
    
    data = data.copy()
    data['signal'] = 0
    
    # Process each month independently
    for date in data.index.get_level_values('Date').unique():
        month_data = data.xs(date, level='Date')
        
        # Process each cluster independently
        for cluster_id in month_data['cluster'].unique():
            cluster_mask = (
                (data.index.get_level_values('Date') == date) &
                (data['cluster'] == cluster_id)
            )
            
            cluster_data = month_data[month_data['cluster'] == cluster_id]
            
            # Skip if cluster too small
            if len(cluster_data) < (top_n + bottom_n):
                continue
            
            # Rank by momentum (higher is better)
            cluster_data = cluster_data.sort_values(momentum_feature, ascending=False)
            
            # Get top and bottom stocks
            top_stocks = cluster_data.head(top_n).index
            bottom_stocks = cluster_data.tail(bottom_n).index
            
            # Assign signals
            for ticker in top_stocks:
                data.loc[(date, ticker), 'signal'] = 1  # Long
            
            for ticker in bottom_stocks:
                data.loc[(date, ticker), 'signal'] = -1  # Short
    
    return data


def calculate_portfolio_returns(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate portfolio returns from signals and forward returns.
    
    Returns monthly portfolio returns.
    """
    
    # Get positions (only where signal != 0)
    positions = data[data['signal'] != 0].copy()
    
    # Calculate position returns
    positions['position_return'] = positions['signal'] * positions['forward_return_1m']
    
    # Group by date and calculate equal-weighted portfolio return
    portfolio_returns = (
        positions.groupby('Date')
        .agg({
            'position_return': 'mean',
            'signal': lambda x: (x == 1).sum(),  # Count longs
        })
        .rename(columns={'signal': 'n_long', 'position_return': 'portfolio_return'})
    )
    
    # Count shorts
    portfolio_returns['n_short'] = (
        positions[positions['signal'] == -1]
        .groupby('Date')
        .size()
    )
    
    portfolio_returns['n_total'] = portfolio_returns['n_long'] + portfolio_returns['n_short']
    
    return portfolio_returns


def backtest_strategy(
    data: pd.DataFrame,
    momentum_feature: str = 'return_3m',
    top_n: int = 3,
    bottom_n: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run full backtest of momentum within clusters strategy.
    
    Returns:
    --------
    results : pd.DataFrame
        Portfolio returns by month
    positions : pd.DataFrame
        All positions taken (for analysis)
    """
    
    print(f"\n{'='*80}")
    print(f"BACKTESTING MOMENTUM WITHIN CLUSTERS")
    print(f"{'='*80}")
    print(f"Momentum signal: {momentum_feature}")
    print(f"Long top {top_n} stocks per cluster")
    print(f"Short bottom {bottom_n} stocks per cluster")
    print(f"{'='*80}\n")
    
    # Calculate forward returns
    print("Calculating forward returns...")
    data = calculate_forward_returns(data)
    
    # Generate signals
    print("Generating trading signals...")
    data = generate_momentum_signals(data, momentum_feature, top_n, bottom_n)
    
    # Calculate returns
    print("Calculating portfolio returns...")
    results = calculate_portfolio_returns(data)
    
    # Get positions for analysis
    positions = data[data['signal'] != 0].copy()
    
    # Remove last month (no forward return)
    results = results.iloc[:-1]
    
    return results, positions


def performance_summary(results: pd.DataFrame):
    """Print performance statistics."""
    
    returns = results['portfolio_return'].dropna()
    
    print(f"\n{'='*80}")
    print(f"PERFORMANCE SUMMARY")
    print(f"{'='*80}")
    
    # Basic stats
    print(f"\nReturns Statistics:")
    print(f"  Total months:        {len(returns)}")
    print(f"  Mean return:         {returns.mean():>8.2%}")
    print(f"  Median return:       {returns.median():>8.2%}")
    print(f"  Std deviation:       {returns.std():>8.2%}")
    print(f"  Min return:          {returns.min():>8.2%}")
    print(f"  Max return:          {returns.max():>8.2%}")
    
    # Risk-adjusted metrics
    print(f"\nRisk-Adjusted Metrics:")
    sharpe = returns.mean() / returns.std() * np.sqrt(12)
    print(f"  Sharpe ratio (ann):  {sharpe:>8.2f}")
    
    # Win rate
    win_rate = (returns > 0).sum() / len(returns)
    print(f"  Win rate:            {win_rate:>8.2%}")
    
    # Cumulative return
    cumulative = (1 + returns).prod() - 1
    print(f"  Cumulative return:   {cumulative:>8.2%}")
    
    # Annualized return
    n_years = len(returns) / 12
    annual_return = (1 + cumulative) ** (1 / n_years) - 1
    print(f"  Annualized return:   {annual_return:>8.2%}")
    
    # Position counts
    print(f"\nPosition Statistics:")
    print(f"  Avg positions/month: {results['n_total'].mean():>8.1f}")
    print(f"  Avg longs/month:     {results['n_long'].mean():>8.1f}")
    print(f"  Avg shorts/month:    {results['n_short'].mean():>8.1f}")
    
    print(f"\n{'='*80}\n")


def main():
    # Setup paths
    src_dir = Path(__file__).parent
    project_dir = src_dir.parent
    data_dir = project_dir / "data"
    
    # Load clustered data (default: K-Means)
    method = 'kmeans'  # Change to 'gmm' or 'pca_kmeans' to test other methods
    
    file_path = data_dir / f"clustered_{method}.csv"
    
    if not file_path.exists():
        print(f"Error: {file_path} not found!")
        print("Please run generate_clusters.py first")
        return
    
    print(f"Loading {file_path.name}...")
    data = pd.read_csv(file_path, parse_dates=['Date'])
    data = data.set_index(['Date', 'Ticker']).sort_index()
    
    print(f"✓ Loaded {len(data):,} observations")
    print(f"  Date range: {data.index.get_level_values('Date').min()} to {data.index.get_level_values('Date').max()}")
    
    # Run backtest
    results, positions = backtest_strategy(
        data=data,
        momentum_feature='return_3m',  # Use 3-month momentum
        top_n=3,                        # Long top 3 per cluster
        bottom_n=3,                     # Short bottom 3 per cluster
    )
    
    # Show performance
    performance_summary(results)
    
    # Save results
    output_dir = data_dir
    results.to_csv(output_dir / f"strategy_results_{method}.csv")
    positions.to_csv(output_dir / f"strategy_positions_{method}.csv")
    
    print(f"✓ Results saved to:")
    print(f"  - strategy_results_{method}.csv")
    print(f"  - strategy_positions_{method}.csv")


if __name__ == "__main__":
    main()
