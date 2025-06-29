import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yfinance as yf
from sklearn.decomposition import PCA

from model import FinanceMoEModel

warnings.filterwarnings('ignore')
plt.ion()


class RealDataFinanceDemo:
    """Tests the Finance MoE Router with real-world market data.

    This demo fetches live market data from Yahoo Finance, processes it into
    model-compatible inputs, and generates visualizations to show how the
    router intelligently selects experts based on different market conditions
    and asset classes.

    Attributes:
        model: The initialized FinanceMoEModel.
        device: The torch device ('cuda' or 'cpu').
        asset_categories: A dictionary mapping asset classes to representative
            tickers.
    """

    def __init__(self, lookback_days: int = 50, hidden_size: int = 16):
        """Initializes the RealDataFinanceDemo.

        Args:
            lookback_days (int): The number of historical days to fetch data for.
            hidden_size (int): The feature dimension expected by the model.
        """
        self.model: Optional[FinanceMoEModel] = None
        self.device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._lookback_days = lookback_days
        self._hidden_size = hidden_size
        print(f"🚀 Demo starting on: {self.device}")

        # Real ETFs and stocks representing different asset classes
        self.asset_categories: Dict[str, List[str]] = {
            # Large-cap growth, value, and sector leaders across U.S. and global markets
            'Equities': [
                'AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA',
                'AMZN', 'META', 'LLY', 'JNJ', 'UNH',
                'XOM', 'PG', 'HD', 'V', 'MA',
                'BABA', 'NVO', 'ADBE', 'ASML', 'NFLX'
            ],

            # Blend of Treasuries, TIPS, aggregate, and short-duration cash proxies
            'Fixed Income': [
                'TLT', 'IEF', 'SHY', 'HYG', 'LQD',
                'BND', 'AGG', 'TIP', 'VGIT', 'MUB',
                'BIL', 'SHV', 'VGLT', 'ZROZ', 'IBND'
            ],

            # Broad commodity basket plus targeted metal, energy, and ag exposure
            'Commodities': [
                'GLD', 'SLV', 'USO', 'DBA', 'PDBC',
                'DBC', 'COPX', 'URA', 'PALL', 'CANE',
                'CORN', 'WEAT', 'UGA', 'KOLD', 'UNG'
            ],

            # Major currency ETFs and a few country-currency proxies
            'FX': [
                'UUP', 'FXE', 'FXY', 'EWJ', 'EWZ',
                'FXB', 'FXF', 'FXA', 'CYB', 'CEW',
                'EUO', 'YCS', 'UDN', 'USDU', 'FXC'
            ],

            # Leveraged & inverse equity / volatility products for stress-testing the router
            'Derivatives': [
                'VXX', 'UVXY', 'SQQQ', 'TQQQ', 'SPXL',
                'SPXS', 'SOXL', 'SOXS', 'UPRO', 'TMF',
                'TZA', 'URTY', 'LABU', 'LABD', 'DRV'
            ],

            # Corporate, high-yield, emerging-market, and bank-loan credit risk
            'Credit': [
                'HYG', 'JNK', 'EMB', 'LQD', 'VCIT',
                'VCSH', 'SJNK', 'BKLN', 'ANGL', 'HYLB',
                'IGIB', 'IGSB', 'SHYG', 'HYGH', 'LQDH'
            ]
        }

    def setup_model(self) -> None:
        """Loads and prepares the FinanceMoEModel."""
        print("🔧 Setting up Finance MoE Model...")
        if self.device == 'cpu':
            raise RuntimeError("CUDA GPU is required to run the model.")
        self.model = FinanceMoEModel(hidden_size=self._hidden_size).to(self.device)
        print("✅ Model ready!")

    def _fetch_ticker_data(self, ticker: str, start_date: datetime,
                           end_date: datetime) -> Optional[Dict[str, Any]]:
        """Fetches and processes data for a single ticker."""
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(start=start_date, end=end_date)

            if len(hist) < self._lookback_days:
                return None

            hist = hist.tail(self._lookback_days)
            hist['Returns'] = hist['Close'].pct_change()
            hist['Volatility'] = hist['Returns'].rolling(window=5).std()
            hist['RSI'] = self._calculate_rsi(hist['Close'])

            features = pd.DataFrame({
                'price_norm': (hist['Close'] - hist['Close'].mean()) / hist['Close'].std(),
                'volume_norm': (hist['Volume'] - hist['Volume'].mean()) / hist['Volume'].std(),
                'volatility': hist['Volatility'],
                'rsi': hist['RSI'] / 100.0,
                'momentum': hist['Returns']
            }).fillna(0)

            return {
                'ticker': ticker,
                'features': features,
                'volatility': hist['Volatility'].fillna(0).values,
                'returns': hist['Returns'].fillna(0).values,
            }
        except Exception as e:
            print(f"   ⚠️  Couldn't get {ticker}: {e}")
            return None

    def _classify_market_condition(self, category_data: List[Dict]) -> Dict[str, Any]:
        """Classifies the market condition for an asset category."""
        recent_vol = np.mean([
            np.mean(data['volatility'][-10:]) for data in category_data
            if len(data['volatility']) >= 10
        ])
        recent_ret = np.mean([
            np.mean(data['returns'][-10:]) for data in category_data
            if len(data['returns']) >= 10
        ])

        if recent_vol > 0.03 and recent_ret < -0.01:
            condition = 'crisis'
        elif recent_vol > 0.02:
            condition = 'high_volatility'
        elif recent_ret > 0.01:
            condition = 'bull_market'
        elif recent_ret < -0.005:
            condition = 'bear_market'
        else:
            condition = 'sideways'

        return {
            'condition': condition,
            'volatility': recent_vol,
            'returns': recent_ret,
            'description': f"{condition.replace('_', ' ').title()}"
        }

    def fetch_real_market_data(self) -> Tuple[Dict, Dict]:
        """Downloads and processes real market data from Yahoo Finance.

        Returns:
            A tuple containing:
                - Raw market data organized by asset category.
                - Market condition analysis for each category.
        """
        print(f"📡 Fetching {self._lookback_days} days of real market data...")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self._lookback_days + 30)

        all_data = {}
        market_scenarios = {}

        for category, tickers in self.asset_categories.items():
            print(f"   📊 Getting {category} data...")
            category_data = [
                data for ticker in tickers
                if (data := self._fetch_ticker_data(ticker, start_date, end_date))
            ]

            if category_data:
                all_data[category] = category_data
                market_scenarios[category] = self._classify_market_condition(category_data)
                market_scenarios[category]['description'] = (
                    f"{category} - {market_scenarios[category]['description']}"
                )

        return all_data, market_scenarios

    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculates the Relative Strength Index (RSI).

        Args:
            prices (pd.Series): A series of prices.
            window (int): The lookback window for RSI calculation.

        Returns:
            pd.Series: The calculated RSI values.
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def prepare_model_inputs(self, market_data: Dict) -> Dict[str, Dict[str, torch.Tensor]]:
        """Converts raw market data into model-ready tensors.

        Args:
            market_data: A dictionary of raw market data by category.

        Returns:
            A dictionary of processed tensors ready for the model.
        """
        print("🔄 Converting real data to model inputs...")
        processed_data = {}

        for category, category_data in market_data.items():
            if not category_data:
                continue

            all_features, all_volatility, all_risk_factors = [], [], []

            for ticker_data in category_data:
                features = ticker_data['features'].values
                if features.shape[1] < self._hidden_size:
                    pad = np.zeros((features.shape[0], self._hidden_size - features.shape[1]))
                    features = np.hstack([features, pad])
                elif features.shape[1] > self._hidden_size:
                    pca = PCA(n_components=self._hidden_size)
                    features = pca.fit_transform(features)

                all_features.append(features)
                all_volatility.append(ticker_data['volatility'])
                all_risk_factors.append(np.abs(ticker_data['returns']).reshape(-1, 1))

            if all_features:
                avg_features = np.mean(all_features, axis=0)
                avg_volatility = np.mean(all_volatility, axis=0)
                avg_risk = np.mean(all_risk_factors, axis=0)

                processed_data[category] = {
                    'embeddings': torch.tensor(
                        avg_features.reshape(1, -1, self._hidden_size),
                        dtype=torch.float32, device=self.device
                    ),
                    'volatility': torch.tensor(
                        avg_volatility.reshape(1, -1, 1),
                        dtype=torch.float32, device=self.device
                    ),
                    'risk_factors': torch.tensor(
                        avg_risk.reshape(1, -1, 1),
                        dtype=torch.float32, device=self.device
                    )
                }
        return processed_data

    def run_real_data_analysis(self, processed_data: Dict, market_scenarios: Dict) -> Dict:
        """Runs the MoE model on real market data.

        Args:
            processed_data: A dictionary of tensors ready for the model.
            market_scenarios: A dictionary of market condition info for context.

        Returns:
            A dictionary with predictions and routing results for each category.
        """
        print("🎯 Running real data through the MoE router...")
        results = {}
        if not self.model:
            raise RuntimeError("Model is not set up. Call setup_model() first.")

        with torch.no_grad():
            for category, data in processed_data.items():
                try:
                    predictions, routing_info = self.model(
                        data['embeddings'], data['volatility'], data['risk_factors']
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
        return results

    def create_matplotlib_visualizations(self, results: Dict) -> List[plt.Figure]:
        """Creates comprehensive and beautiful matplotlib visualizations in separate figures.

        Args:
            results: A dictionary of analysis results from the model.

        Returns:
            A list of matplotlib Figure objects.
        """
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
        print("   - Generating expert routing analysis...")
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
                ax1b.text(bar.get_x() + bar.get_width() / 2., height + max(domain_counts) * 0.02,
                          f'{count}', ha='center', va='bottom', fontweight='bold',
                          fontsize=12, color='#2E3440',
                          bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        plt.tight_layout()
        filename1 = 'finance_moe_expert_routing.png'
        fig1.savefig(filename1, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.3)
        print(f"   ✅ Saved: {filename1}")
        figures.append(fig1)

        # ===== FIGURE 2: Current Market Conditions =====
        print("   - Generating market conditions analysis...")
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

        ax2b.bar(x_pos - width / 2, avg_volatility, width, label='Volatility',
                 color='#FF6B6B', alpha=0.8, edgecolor='white', linewidth=2)
        ax2b.bar(x_pos + width / 2, avg_returns, width, label='Returns (%)',
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
        print(f"   ✅ Saved: {filename2}")
        figures.append(fig2)

        # ===== FIGURE 3: Market Volatility Analysis =====
        print("   - Generating market volatility analysis...")
        fig3 = plt.figure(figsize=(16, 8), facecolor='white')
        fig3.suptitle('Market Volatility Analysis - Finance MoE Router',
                      fontsize=20, fontweight='bold', y=0.95, color='#2E3440')

        ax3 = fig3.add_subplot(111)
        colors = ['#E91E63', '#9C27B0', '#673AB7', '#3F51B5', '#2196F3', '#00BCD4']

        for i, (category, data) in enumerate(results.items()):
            volatility = data['volatility'][0].flatten()
            color = colors[i % len(colors)]

            ax3.plot(volatility, label=f'{category}',
                     color=color, linewidth=3, alpha=0.9, marker='o',
                     markersize=4, markevery=max(1, len(volatility) // 20))

        ax3.set_title('Market Volatility Over Time',
                      fontsize=16, fontweight='bold', pad=20, color='#2E3440')
        ax3.set_xlabel('Time Steps (Trading Days)', fontsize=13, color='#2E3440')
        ax3.set_ylabel('Volatility', fontsize=13, color='#2E3440')
        ax3.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=11)
        ax3.grid(True, alpha=0.4, linestyle='-', linewidth=0.5)
        ax3.set_facecolor('#FAFAFA')

        # Add volatility zones
        max_vol = max([data['volatility'][0].flatten().max() for data in results.values()])
        ax3.axhspan(0, max_vol * 0.3, alpha=0.1, color='green', label='Low Volatility')
        ax3.axhspan(max_vol * 0.3, max_vol * 0.7, alpha=0.1, color='yellow', label='Medium Volatility')
        ax3.axhspan(max_vol * 0.7, max_vol, alpha=0.1, color='red', label='High Volatility')

        plt.tight_layout()
        filename3 = 'finance_moe_volatility.png'
        fig3.savefig(filename3, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.3)
        print(f"   ✅ Saved: {filename3}")
        figures.append(fig3)

        # ===== FIGURE 4: Performance Analysis =====
        print("   - Generating performance analysis...")
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
        print(f"   ✅ Saved: {filename4}")
        figures.append(fig4)

        # Display all figures and keep them open
        plt.show(block=True)

        print(f"\n🎨 Created {len(figures)} visualization files:")
        for i, filename in enumerate([filename1, filename2, filename3, filename4], 1):
            print(f"   {i}. {filename}")

        return figures

    def print_real_data_insights(self, results: Dict) -> None:
        """Prints a human-readable summary of the analysis.

        Args:
            results: A dictionary of analysis results from the model.
        """
        print("\n" + "=" * 70)
        print("🧠 REAL-WORLD DATA INSIGHTS")
        print("=" * 70)

        for category, data in results.items():
            scenario = data['market_scenario']
            assignments = data['domain_assignments'].flatten()
            domain_counts = np.bincount(assignments, minlength=6)
            most_used_idx = np.argmax(domain_counts)
            avg_confidence = np.mean(np.max(data['routing_probs'][0], axis=1))

            print(f"\n📊 {category.upper()}:")
            print(f"   - Market: {scenario.get('description', 'N/A')}")
            print(f"   - Tickers: {', '.join(data['tickers'][:3])}...")
            print(f"   - Favorite Expert: {data['domain_names'][most_used_idx]} ({domain_counts[most_used_idx]} times)")
            print(f"   - Avg. Confidence: {avg_confidence:.2%}")


def main():
    """Main function to run the real-data financial MoE demo."""
    print("=" * 50)
    print("FINANCE MOE ROUTER - REAL DATA ANALYSIS")
    print("=" * 50)

    try:
        demo = RealDataFinanceDemo(lookback_days=50)
        demo.setup_model()

        market_data, market_scenarios = demo.fetch_real_market_data()
        if not market_data:
            print("❌ Error: Unable to fetch market data. Check connection or tickers.")
            return

        processed_data = demo.prepare_model_inputs(market_data)
        if not processed_data:
            print("❌ Error: Data processing failed.")
            return

        results = demo.run_real_data_analysis(processed_data, market_scenarios)
        if not results:
            print("❌ Error: Model analysis failed.")
            return

        demo.create_matplotlib_visualizations(results)
        demo.print_real_data_insights(results)

    except Exception as e:
        print(f"\n🚨 An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
