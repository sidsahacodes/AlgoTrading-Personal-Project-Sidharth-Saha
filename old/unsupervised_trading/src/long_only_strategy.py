"""
Long-Only Momentum Within Clusters Strategy

Strategy Logic:
- Each month, within each cluster, rank stocks by momentum signal
- Long top N stocks per cluster
- Equal-weight positions
- Monthly rebalancing
"""

from pathlib import Path
import pandas as pd
import numpy as np


def calculate_forward_returns(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate forward 1-month returns for each stock."""
    data = data.copy()
    data = data.sort_index()
    
    data['forward_return_1m'] = (
        data.groupby('Ticker')['return_1m']
        .shift(-1)
    )
    
    return data


def generate_long_only_signals(
    data: pd.DataFrame,
    momentum_feature: str = 'return_3m',
    top_n: int = 5,
) -> pd.DataFrame:
    """
    Generate long-only signals based on momentum within clusters.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Clustered data with Date and Ticker index
    momentum_feature : str
        Feature to rank on (e.g., 'return_3m', 'return_6m')
    top_n : int
        Number of top stocks to long per cluster
    
    Returns:
    --------
    pd.DataFrame with 'signal' column: 1 (long), 0 (no position)
    """
    
    data = data.copy()
    data['signal'] = 0
    
    for date in data.index.get_level_values('Date').unique():
        month_data = data.xs(date, level='Date')
        
        for cluster_id in month_data['cluster'].unique():
            cluster_mask = (
                (data.index.get_level_values('Date') == date) &
                (data['cluster'] == cluster_id)
            )
            
            cluster_data = month_data[month_data['cluster'] == cluster_id]
            
            if len(cluster_data) < top_n:
                continue
            
            # Rank by momentum (higher is better)
            cluster_data = cluster_data.sort_values(momentum_feature, ascending=False)
            
            # Get top stocks
            top_stocks = cluster_data.head(top_n).index
            
            # Assign signals
            for ticker in top_stocks:
                data.loc[(date, ticker), 'signal'] = 1  # Long
    
    return data


def calculate_portfolio_returns(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate portfolio returns from signals and forward returns.
    """
    
    positions = data[data['signal'] == 1].copy()
    
    positions['position_return'] = positions['forward_return_1m']
    
    portfolio_returns = (
        positions.groupby('Date')
        .agg({
            'position_return': 'mean',
            'signal': 'sum',  # Count positions
        })
        .rename(columns={'signal': 'n_positions', 'position_return': 'portfolio_return'})
    )
    
    return portfolio_returns


def backtest_long_only(
    data: pd.DataFrame,
    momentum_feature: str = 'return_3m',
    top_n: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run full backtest of long-only momentum strategy.
    """
    
    print(f"\n{'='*80}")
    print(f"BACKTESTING LONG-ONLY MOMENTUM WITHIN CLUSTERS")
    print(f"{'='*80}")
    print(f"Momentum signal: {momentum_feature}")
    print(f"Long top {top_n} stocks per cluster")
    print(f"{'='*80}\n")
    
    print("Calculating forward returns...")
    data = calculate_forward_returns(data)
    
    print("Generating trading signals...")
    data = generate_long_only_signals(data, momentum_feature, top_n)
    
    print("Calculating portfolio returns...")
    results = calculate_portfolio_returns(data)
    
    positions = data[data['signal'] == 1].copy()
    
    # Remove last month (no forward return)
    results = results.iloc[:-1]
    
    return results, positions


def performance_summary(results: pd.DataFrame):
    """Print performance statistics."""
    
    returns = results['portfolio_return'].dropna()
    
    print(f"\n{'='*80}")
    print(f"PERFORMANCE SUMMARY")
    print(f"{'='*80}")
    
    print(f"\nReturns Statistics:")
    print(f"  Total months:        {len(returns)}")
    print(f"  Mean return:         {returns.mean():>8.2%}")
    print(f"  Median return:       {returns.median():>8.2%}")
    print(f"  Std deviation:       {returns.std():>8.2%}")
    print(f"  Min return:          {returns.min():>8.2%}")
    print(f"  Max return:          {returns.max():>8.2%}")
    
    print(f"\nRisk-Adjusted Metrics:")
    sharpe = returns.mean() / returns.std() * np.sqrt(12)
    print(f"  Sharpe ratio (ann):  {sharpe:>8.2f}")
    
    win_rate = (returns > 0).sum() / len(returns)
    print(f"  Win rate:            {win_rate:>8.2%}")
    
    cumulative = (1 + returns).prod() - 1
    print(f"  Cumulative return:   {cumulative:>8.2%}")
    
    n_years = len(returns) / 12
    annual_return = (1 + cumulative) ** (1 / n_years) - 1
    print(f"  Annualized return:   {annual_return:>8.2%}")
    
    # Drawdown
    cumulative_wealth = (1 + returns).cumprod()
    running_max = cumulative_wealth.expanding().max()
    drawdown = (cumulative_wealth - running_max) / running_max
    print(f"  Max drawdown:        {drawdown.min():>8.2%}")
    
    print(f"\nPosition Statistics:")
    print(f"  Avg positions/month: {results['n_positions'].mean():>8.1f}")
    
    print(f"\n{'='*80}\n")


def main():
    src_dir = Path(__file__).parent
    project_dir = src_dir.parent
    data_dir = project_dir / "data"
    
    method = 'kmeans'
    
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
    results, positions = backtest_long_only(
        data=data,
        momentum_feature='return_3m',
        top_n=5,  # Top 5 per cluster
    )
    
    # Show performance
    performance_summary(results)
    
    # Save results
    output_dir = data_dir
    results.to_csv(output_dir / f"long_only_results_{method}.csv")
    positions.to_csv(output_dir / f"long_only_positions_{method}.csv")
    
    print(f"✓ Results saved to:")
    print(f"  - long_only_results_{method}.csv")
    print(f"  - long_only_positions_{method}.csv")


if __name__ == "__main__":
    main()
