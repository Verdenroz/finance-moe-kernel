import os
import warnings
from datetime import datetime, timedelta

import torch
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

from model import FinanceMoEModel
plt.ion()

class RealDataFinanceDemo:
    """
    Demo class that tests the Finance MoE Router with real market data from Yahoo Finance.
    
    This demo fetches live market data, processes it into model inputs, and creates
    beautiful visualizations showing how the router intelligently selects experts
    based on different market conditions and asset classes.
    """
    
    def __init__(self):
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🚀 Demo starting on: {self.device}")
        
        # Real ETFs and stocks representing different asset classes
        self.asset_categories = {
            'Equities': ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA'],
            'Fixed Income': ['TLT', 'IEF', 'SHY', 'HYG', 'LQD'],
            'Commodities': ['GLD', 'SLV', 'USO', 'DBA', 'PDBC'],
            'FX': ['UUP', 'FXE', 'FXY', 'EWJ', 'EWZ'],
            'Derivatives': ['VXX', 'UVXY', 'SQQQ', 'TQQQ', 'SPXL'],
            'Credit': ['HYG', 'JNK', 'EMB', 'LQD', 'VCIT']
        }
        
    def setup_model(self):
        """Load and prepare the MoE model"""
        print("🔧 Setting up Finance MoE Model...")
        self.model = FinanceMoEModel().to(self.device)
        print("✅ Model ready!")
        
    def fetch_real_market_data(self, lookback_days=60):
        """
        Download real market data from Yahoo Finance for all asset categories.
        
        Args:
            lookback_days: How many days of historical data to fetch
            
        Returns:
            all_data: Raw market data organized by asset category
            market_scenarios: Market condition analysis for each category
        """
        print(f"📡 Fetching {lookback_days} days of real market data...")
        
        # Get dates - we fetch extra days to account for weekends/holidays
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days + 30)
        
        all_data = {}
        market_scenarios = {}
        
        for category, tickers in self.asset_categories.items():
            print(f"   📊 Getting {category} data...")
            category_data = []
            
            for ticker in tickers:
                try:
                    # Download price data
                    stock = yf.Ticker(ticker)
                    hist = stock.history(start=start_date, end=end_date)
                    
                    if len(hist) > lookback_days:
                        # Keep only the most recent data
                        hist = hist.tail(lookback_days)
                        
                        # Calculate basic technical indicators
                        hist['Returns'] = hist['Close'].pct_change()
                        hist['Volatility'] = hist['Returns'].rolling(window=5).std()
                        hist['RSI'] = self.calculate_rsi(hist['Close'])
                        hist['SMA_10'] = hist['Close'].rolling(window=10).mean()
                        hist['SMA_20'] = hist['Close'].rolling(window=20).mean()
                        
                        # Create normalized features for the model
                        features = pd.DataFrame({
                            'price_norm': (hist['Close'] - hist['Close'].mean()) / hist['Close'].std(),
                            'volume_norm': (hist['Volume'] - hist['Volume'].mean()) / hist['Volume'].std(),
                            'volatility': hist['Volatility'].fillna(0),
                            'rsi': hist['RSI'].fillna(50) / 100,
                            'sma_ratio': (hist['Close'] / hist['SMA_10']).fillna(1),
                            'momentum': hist['Returns'].fillna(0)
                        })
                        
                        category_data.append({
                            'ticker': ticker,
                            'features': features.fillna(0),
                            'volatility': hist['Volatility'].fillna(0).values,
                            'returns': hist['Returns'].fillna(0).values,
                            'volume': hist['Volume'].values
                        })
                        
                except Exception as e:
                    print(f"   ⚠️  Couldn't get {ticker}: {e}")
                    continue
            
            if category_data:
                all_data[category] = category_data
                
                # Figure out what kind of market we're in
                recent_volatility = np.mean([
                    np.mean(data['volatility'][-10:]) 
                    for data in category_data if len(data['volatility']) >= 10
                ])
                
                recent_returns = np.mean([
                    np.mean(data['returns'][-10:]) 
                    for data in category_data if len(data['returns']) >= 10
                ])
                
                # Simple market condition classifier
                if recent_volatility > 0.03 and recent_returns < -0.01:
                    condition = 'crisis'
                elif recent_volatility > 0.02:
                    condition = 'high_volatility'
                elif recent_returns > 0.01:
                    condition = 'bull_market'
                elif recent_returns < -0.005:
                    condition = 'bear_market'
                else:
                    condition = 'sideways'
                    
                market_scenarios[category] = {
                    'condition': condition,
                    'volatility': recent_volatility,
                    'returns': recent_returns,
                    'description': f"{category} - {condition.replace('_', ' ').title()}"
                }
        
        return all_data, market_scenarios
    
    def calculate_rsi(self, prices, window=14):
        """Calculate RSI indicator (0-100 scale)"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def prepare_model_inputs(self, market_data, target_length=50):
        """
        Convert raw market data into the format our MoE model expects.
        
        Args:
            market_data: Raw market data by category
            target_length: How many time steps to use
            
        Returns:
            Dictionary of processed tensors ready for the model
        """
        print("🔄 Converting real data to model inputs...")
        
        processed_data = {}
        
        for category, category_data in market_data.items():
            if not category_data:
                continue
                
            # Collect features from all tickers in this category
            all_features = []
            all_volatility = []
            all_risk_factors = []
            
            for ticker_data in category_data:
                features = ticker_data['features'].values
                if len(features) >= target_length:
                    # Use the most recent data
                    features = features[-target_length:]
                    volatility = ticker_data['volatility'][-target_length:]
                    returns = ticker_data['returns'][-target_length:]
                    
                    # Our model expects exactly 16 features
                    if features.shape[1] < 16:
                        # Add zeros if we don't have enough
                        padding = np.zeros((features.shape[0], 16 - features.shape[1]))
                        features = np.hstack([features, padding])
                    elif features.shape[1] > 16:
                        # Use PCA to reduce if we have too many
                        pca = PCA(n_components=16)
                        features = pca.fit_transform(features)
                    
                    all_features.append(features)
                    all_volatility.append(volatility)
                    
                    # Risk is just absolute returns for now
                    risk = np.abs(returns).reshape(-1, 1)
                    all_risk_factors.append(risk)
            
            if all_features:
                # Average everything across tickers in the same category
                avg_features = np.mean(all_features, axis=0)
                avg_volatility = np.mean(all_volatility, axis=0)
                avg_risk = np.mean(all_risk_factors, axis=0)
                
                # Convert to PyTorch tensors
                processed_data[category] = {
                    'embeddings': torch.tensor(
                        avg_features.reshape(1, -1, 16), 
                        dtype=torch.float32, 
                        device=self.device
                    ),
                    'volatility': torch.tensor(
                        avg_volatility.reshape(1, -1), 
                        dtype=torch.float32, 
                        device=self.device
                    ),
                    'risk_factors': torch.tensor(
                        avg_risk.reshape(1, -1, 1), 
                        dtype=torch.float32, 
                        device=self.device
                    )
                }
        
        return processed_data
    
    def run_real_data_analysis(self, processed_data, market_scenarios):
        """
        Run the MoE model on real market data and see which experts it picks.
        
        Args:
            processed_data: Tensors ready for the model
            market_scenarios: Market condition info for context
            
        Returns:
            Dictionary with predictions and routing results for each asset category
        """
        print("🎯 Running real data through the MoE router...")
        
        results = {}
        
        with torch.no_grad():
            for category, data in processed_data.items():
                try:
                    predictions, routing_info = self.model(
                        data['embeddings'],
                        data['volatility'], 
                        data['risk_factors']
                    )
                    
                    results[category] = {
                        'predictions': predictions.cpu().numpy(),
                        'domain_assignments': routing_info['domain_assignments'].cpu().numpy(),
                        'routing_probs': routing_info['routing_probs'].cpu().numpy(),
                        'domain_names': routing_info['domain_names'],
                        'market_scenario': market_scenarios.get(category, {}),
                        'volatility': data['volatility'].cpu().numpy(),
                        'tickers': list(self.asset_categories[category])
                    }
                    
                except Exception as e:
                    print(f"   ⚠️  Couldn't process {category}: {e}")
                    continue
                    
        return results
        
    def create_matplotlib_visualizations(self, results):
        """Create comprehensive and beautiful matplotlib visualizations in separate figures"""
        print("Creating comprehensive visualizations...")
        
        # Set global matplotlib style for better aesthetics
        plt.style.use('default')
        plt.rcParams.update({
            'font.size': 12,
            'font.family': 'sans-serif',
            'axes.labelsize': 13,
            'axes.titlesize': 16,
            'xtick.labelsize': 11,
            'ytick.labelsize': 11,
            'legend.fontsize': 11,
            'figure.titlesize': 20,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.grid': True,
            'grid.alpha': 0.3
        })

        figures = []
        
        # ===== FIGURE 1: Expert Routing Patterns =====
        print("   Generating expert routing analysis...")
        fig1 = plt.figure(figsize=(16, 8), facecolor='white')
        fig1.suptitle('Expert Routing Analysis - Finance MoE Router', 
                     fontsize=20, fontweight='bold', y=0.95, color='#2E3440')
        
        gs1 = fig1.add_gridspec(1, 2, hspace=0.3, wspace=0.4, 
                               left=0.08, right=0.92, top=0.85, bottom=0.15)
        
        # Left plot: Which expert does each asset category prefer?
        ax1a = fig1.add_subplot(gs1[0, 0])
        categories = list(results.keys())
        expert_counts = []
        
        for category, data in results.items():
            assignments = data['domain_assignments'].flatten()
            domain_counts = np.bincount(assignments, minlength=6)
            expert_counts.append(domain_counts)
        
        if expert_counts:
            expert_matrix = np.array(expert_counts)
            im = ax1a.imshow(expert_matrix, cmap='Blues', aspect='auto', 
                           interpolation='nearest', alpha=0.9)
            ax1a.set_title('Expert Selection by Asset Class', 
                          fontsize=16, fontweight='bold', pad=20, color='#2E3440')
            ax1a.set_xlabel('Expert Domain', fontsize=13, color='#2E3440')
            ax1a.set_ylabel('Asset Category', fontsize=13, color='#2E3440')
            ax1a.set_yticks(range(len(categories)))
            ax1a.set_yticklabels(categories, fontsize=11, color='#2E3440')
            ax1a.set_xticks(range(6))
            
            domain_names = list(results.values())[0]['domain_names']
            ax1a.set_xticklabels(domain_names, rotation=45, ha='right', 
                               fontsize=11, color='#2E3440')
            
            cbar = plt.colorbar(im, ax=ax1a, shrink=0.8, pad=0.02)
            cbar.set_label('Times Selected', fontsize=12, color='#2E3440')
            cbar.ax.tick_params(labelsize=10, colors='#2E3440')
            
            # Add numbers on the heatmap
            for i in range(len(categories)):
                for j in range(6):
                    value = expert_matrix[i, j]
                    color = 'white' if value > expert_matrix.max() * 0.6 else '#2E3440'
                    ax1a.text(j, i, f'{value}', ha="center", va="center", 
                            color=color, fontweight='bold', fontsize=12)
        
        # Right plot: Overall popularity of each expert
        ax1b = fig1.add_subplot(gs1[0, 1])
        all_assignments = []
        for data in results.values():
            all_assignments.extend(data['domain_assignments'].flatten())
        
        if all_assignments:
            domain_counts = np.bincount(all_assignments, minlength=6)
            domain_names = list(results.values())[0]['domain_names']
            
            colors_bar = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3']
            bars = ax1b.bar(range(len(domain_names)), domain_counts, 
                           color=colors_bar[:len(domain_names)], alpha=0.8, 
                           edgecolor='white', linewidth=2)
            
            ax1b.set_title('Overall Expert Popularity', 
                          fontsize=16, fontweight='bold', pad=20, color='#2E3440')
            ax1b.set_xlabel('Expert Domain', fontsize=13, color='#2E3440')
            ax1b.set_ylabel('Total Times Selected', fontsize=13, color='#2E3440')
            ax1b.set_xticks(range(len(domain_names)))
            ax1b.set_xticklabels(domain_names, rotation=45, ha='right', 
                               fontsize=11, color='#2E3440')
            ax1b.grid(True, axis='y', alpha=0.4, linestyle='--')
            ax1b.set_facecolor('#FAFAFA')
            
            # Add count labels on bars
            for bar, count in zip(bars, domain_counts):
                height = bar.get_height()
                ax1b.text(bar.get_x() + bar.get_width()/2., height + max(domain_counts)*0.02,
                         f'{count}', ha='center', va='bottom', fontweight='bold', 
                         fontsize=12, color='#2E3440',
                         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        filename1 = 'finance_moe_expert_routing.png'
        fig1.savefig(filename1, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.3)
        print(f"   Saved: {filename1}")
        figures.append(fig1)
        
        # ===== FIGURE 2: Current Market Conditions =====
        print("   Generating market conditions analysis...")
        fig2 = plt.figure(figsize=(16, 8), facecolor='white')
        fig2.suptitle('Market Conditions Analysis - Finance MoE Router', 
                     fontsize=20, fontweight='bold', y=0.95, color='#2E3440')
        
        gs2 = fig2.add_gridspec(1, 2, hspace=0.3, wspace=0.4, 
                               left=0.08, right=0.92, top=0.85, bottom=0.15)
        
        # Left: Scatter plot showing market conditions
        ax2a = fig2.add_subplot(gs2[0, 0])
        conditions = []
        volatilities = []
        returns = []
        colors_list = []
        
        # Color code by market condition
        color_map = {
            'crisis': '#D32F2F', 'high_volatility': '#FF9800', 'bull_market': '#4CAF50',
            'bear_market': '#9C27B0', 'sideways': '#2196F3', 'stable': '#2196F3', 'unknown': '#607D8B'
        }
        
        for category, data in results.items():
            scenario = data['market_scenario']
            conditions.append(category)
            volatilities.append(scenario.get('volatility', 0))
            returns.append(scenario.get('returns', 0) * 100)
            condition = scenario.get('condition', 'unknown')
            colors_list.append(color_map.get(condition, color_map['unknown']))
        
        ax2a.scatter(returns, volatilities, c=colors_list, s=200, alpha=0.8, 
                    edgecolors='white', linewidth=3, zorder=5)
        ax2a.set_title('Returns vs Volatility by Asset Class', 
                      fontsize=16, fontweight='bold', pad=20, color='#2E3440')
        ax2a.set_xlabel('Recent Returns (%)', fontsize=13, color='#2E3440')
        ax2a.set_ylabel('Volatility', fontsize=13, color='#2E3440')
        ax2a.grid(True, alpha=0.4, linestyle='--')
        ax2a.set_facecolor('#FAFAFA')
        
        # Add labels for each point
        for i, category in enumerate(conditions):
            ax2a.annotate(category, (returns[i], volatilities[i]), 
                         xytext=(10, 10), textcoords='offset points', fontsize=11,
                         bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                                 alpha=0.9, edgecolor='#E0E0E0'),
                         color='#2E3440', fontweight='medium')
        
        # Right: Bar chart comparing volatility and returns
        ax2b = fig2.add_subplot(gs2[0, 1])
        categories = list(results.keys())
        avg_volatility = [results[cat]['market_scenario'].get('volatility', 0) for cat in categories]
        avg_returns = [results[cat]['market_scenario'].get('returns', 0) * 100 for cat in categories]
        
        x_pos = np.arange(len(categories))
        width = 0.35
        
        ax2b.bar(x_pos - width/2, avg_volatility, width, label='Volatility', 
                color='#FF6B6B', alpha=0.8, edgecolor='white', linewidth=2)
        ax2b.bar(x_pos + width/2, avg_returns, width, label='Returns (%)', 
                color='#4ECDC4', alpha=0.8, edgecolor='white', linewidth=2)
        
        ax2b.set_title('Market Stats by Category', 
                      fontsize=16, fontweight='bold', pad=20, color='#2E3440')
        ax2b.set_xlabel('Asset Categories', fontsize=13, color='#2E3440')
        ax2b.set_ylabel('Values', fontsize=13, color='#2E3440')
        ax2b.set_xticks(x_pos)
        ax2b.set_xticklabels([cat[:10] + '...' if len(cat) > 10 else cat 
                            for cat in categories], rotation=45, ha='right', 
                           fontsize=11, color='#2E3440')
        ax2b.legend(fontsize=12, frameon=True, fancybox=True, shadow=True)
        ax2b.grid(True, alpha=0.4, linestyle='--')
        ax2b.set_facecolor('#FAFAFA')
        
        plt.tight_layout()
        filename2 = 'finance_moe_market_conditions.png'
        fig2.savefig(filename2, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.3)
        print(f"   Saved: {filename2}")
        figures.append(fig2)
        
        # ===== FIGURE 3: Time Series Analysis =====
        print("   Generating time series analysis...")
        fig3 = plt.figure(figsize=(18, 10), facecolor='white')
        fig3.suptitle('Time Series Analysis - Model Confidence & Market Volatility', 
                     fontsize=20, fontweight='bold', y=0.95, color='#2E3440')
        
        gs3 = fig3.add_gridspec(2, 1, hspace=0.4, 
                               left=0.08, right=0.85, top=0.85, bottom=0.10)
        
        # Top: How confident is the model in its routing decisions?
        ax3a = fig3.add_subplot(gs3[0, 0])
        colors = ['#E91E63', '#9C27B0', '#673AB7', '#3F51B5', '#2196F3', '#00BCD4']
        
        for i, (category, data) in enumerate(results.items()):
            probs = data['routing_probs'][0]
            confidence = np.max(probs, axis=1)  # Highest probability = confidence
            color = colors[i % len(colors)]
            
            ax3a.plot(confidence, label=f'{category}', 
                     color=color, linewidth=3, alpha=0.9, marker='o', 
                     markersize=5, markevery=max(1, len(confidence)//15))
        
        ax3a.set_title('How Confident is the Model?', 
                      fontsize=16, fontweight='bold', pad=20, color='#2E3440')
        ax3a.set_xlabel('Time Steps (Trading Days)', fontsize=13, color='#2E3440')
        ax3a.set_ylabel('Routing Confidence', fontsize=13, color='#2E3440')
        ax3a.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=11)
        ax3a.grid(True, alpha=0.4, linestyle='-', linewidth=0.5)
        ax3a.set_facecolor('#FAFAFA')
        ax3a.set_ylim(0, 1.05)
        
        # Add background zones to show confidence levels
        ax3a.axhspan(0.8, 1.0, alpha=0.1, color='green', label='High Confidence')
        ax3a.axhspan(0.6, 0.8, alpha=0.1, color='yellow', label='Medium Confidence')
        ax3a.axhspan(0.0, 0.6, alpha=0.1, color='red', label='Low Confidence')
        
        # Bottom: How volatile are the markets?
        ax3b = fig3.add_subplot(gs3[1, 0])
        
        for i, (category, data) in enumerate(results.items()):
            volatility = data['volatility'][0]
            color = colors[i % len(colors)]
            
            ax3b.plot(volatility, label=f'{category}', 
                     color=color, linewidth=2.5, alpha=0.8, linestyle='--')
        
        ax3b.set_title('Market Volatility Over Time', 
                      fontsize=16, fontweight='bold', pad=20, color='#2E3440')
        ax3b.set_xlabel('Time Steps (Trading Days)', fontsize=13, color='#2E3440')
        ax3b.set_ylabel('Volatility', fontsize=13, color='#2E3440')
        ax3b.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=11)
        ax3b.grid(True, alpha=0.4, linestyle='-', linewidth=0.5)
        ax3b.set_facecolor('#FAFAFA')
        
        plt.tight_layout()
        filename3 = 'finance_moe_time_series.png'
        fig3.savefig(filename3, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.3)
        print(f"   Saved: {filename3}")
        figures.append(fig3)
        
        # ===== FIGURE 4: Performance Analysis =====
        print("   Generating performance analysis...")
        fig4 = plt.figure(figsize=(12, 8), facecolor='white')
        fig4.suptitle('Performance Analysis - Asset Returns vs Expert Preferences', 
                     fontsize=18, fontweight='bold', y=0.95, color='#2E3440')
        
        ax4 = fig4.add_subplot(111)
        performance_data = []
        expert_preference = []
        category_names = []
        
        # For each asset category, see which expert it likes most
        for category, data in results.items():
            recent_return = data['market_scenario'].get('returns', 0) * 100
            performance_data.append(recent_return)
            
            # Find the most popular expert for this category
            assignments = data['domain_assignments'].flatten()
            domain_counts = np.bincount(assignments, minlength=6)
            most_used_expert = np.argmax(domain_counts)
            expert_preference.append(most_used_expert)
            category_names.append(category)
        
        colors_scatter = ['#E91E63', '#9C27B0', '#673AB7', '#3F51B5', '#2196F3', '#00BCD4']
        ax4.scatter(performance_data, expert_preference, 
                   c=[colors_scatter[i % len(colors_scatter)] for i in expert_preference], 
                   s=250, alpha=0.8, edgecolors='white', linewidth=3, zorder=5)
        
        ax4.set_title('Do Better Assets Prefer Certain Experts?', 
                     fontsize=16, fontweight='bold', pad=20, color='#2E3440')
        ax4.set_xlabel('Recent Returns (%)', fontsize=13, color='#2E3440')
        ax4.set_ylabel('Most Preferred Expert', fontsize=13, color='#2E3440')
        ax4.grid(True, alpha=0.4, linestyle='--')
        ax4.set_facecolor('#FAFAFA')
        ax4.set_yticks(range(6))
        
        domain_names = list(results.values())[0]['domain_names']
        ax4.set_yticklabels(domain_names, fontsize=11, color='#2E3440')
        
        # Label each point with the asset category
        for i, category in enumerate(category_names):
            ax4.annotate(category, (performance_data[i], expert_preference[i]), 
                        xytext=(15, 15), textcoords='offset points', fontsize=12,
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                                alpha=0.9, edgecolor='#E0E0E0'),
                        color='#2E3440', fontweight='medium')
        
        plt.tight_layout()
        filename4 = 'finance_moe_performance.png'
        fig4.savefig(filename4, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.3)
        print(f"   Saved: {filename4}")
        figures.append(fig4)
        
        # Display all figures and keep them open
        plt.show(block=True)
        
        print(f"\nCreated {len(figures)} visualization files:")
        for i, filename in enumerate([filename1, filename2, filename3, filename4], 1):
            print(f"   {i}. {filename}")
        
        return figures
    
    def print_real_data_insights(self, results):
        """Print a human-readable summary of what we learned"""
        print("\n" + "="*70)
        print("🧠 WHAT WE LEARNED FROM REAL MARKET DATA")
        print("="*70)
        
        for category, data in results.items():
            scenario = data['market_scenario']
            print(f"\n📊 {category.upper()}:")
            print(f"   Market condition: {scenario.get('condition', 'unknown').replace('_', ' ').title()}")
            print(f"   Volatility: {scenario.get('volatility', 0):.4f}")
            print(f"   Recent returns: {scenario.get('returns', 0)*100:.2f}%")
            print(f"   Example stocks: {', '.join(data['tickers'][:3])}...")
            
            # Which expert did this category prefer?
            assignments = data['domain_assignments'].flatten()
            domain_counts = np.bincount(assignments, minlength=6)
            
            most_used = np.argmax(domain_counts)
            least_used = np.argmin(domain_counts)
            
            print(f"   🎯 Favorite expert: {data['domain_names'][most_used]} ({domain_counts[most_used]} times)")
            print(f"   🔽 Least favorite expert: {data['domain_names'][least_used]} ({domain_counts[least_used]} times)")
            
            # How confident was the model?
            probs = data['routing_probs'][0]
            avg_confidence = np.mean(np.max(probs, axis=1))
            print(f"   🎪 Average confidence: {avg_confidence:.3f}")


def main():
    """
    Main demo function - downloads real market data and demonstrates MoE router performance.
    """
    print("FINANCE MOE ROUTER - REAL DATA ANALYSIS")
    print("=" * 50)
    print("Analyzing financial market data with MoE routing")
    print("Data source: Yahoo Finance API")
    print()
    
    # Initialize demo
    demo = RealDataFinanceDemo()
    demo.setup_model()
    
    # Fetch market data
    market_data, market_scenarios = demo.fetch_real_market_data(lookback_days=50)
    
    if not market_data:
        print("Error: Unable to fetch market data. Please check internet connection.")
        return
    
    # Process data for model input
    processed_data = demo.prepare_model_inputs(market_data)
    
    if not processed_data:
        print("Error: Data processing failed.")
        return
    
    # Run analysis
    results = demo.run_real_data_analysis(processed_data, market_scenarios)
    
    if not results:
        print("Error: Analysis execution failed.")
        return
    
    # Generate visualizations
    demo.create_matplotlib_visualizations(results)
    
    # Display insights
    demo.print_real_data_insights(results)
    
if __name__ == "__main__":
    main()
