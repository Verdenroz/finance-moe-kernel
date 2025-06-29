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


class RealDataFinanceDemo:
    """Tests the Finance MoE Router with real-world market data.

    This demo fetches live market data from Yahoo Finance, processes it into
    model-compatible inputs, and generates visualizations to show how the
    router intelligently selects experts based on different market conditions
    and asset classes.

    Attributes:
        model: The initialized and trained FinanceMoEModel.
        device: The torch device ('cuda' or 'cpu').
        asset_categories: A dictionary mapping asset classes to representative
            tickers.
    """

    def __init__(self, lookback_days: int = 50, hidden_size: int = 32):
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

            # Blend of Treasuries, TIPS, aggregate, short-duration, credit, and high-yield
            'Fixed Income': [
                'TLT', 'IEF', 'SHY', 'HYG', 'LQD',
                'BND', 'AGG', 'TIP', 'VGIT', 'MUB',
                'BIL', 'SHV', 'VGLT', 'ZROZ', 'IBND',
                'JNK', 'EMB', 'VCIT', 'VCSH', 'SJNK',
                'BKLN', 'ANGL', 'HYLB', 'IGIB', 'IGSB'
            ],

            # Broad commodity basket plus targeted metal, energy, and ag exposure
            'Commodities': [
                'GLD', 'SLV', 'USO', 'DBA', 'PDBC',
                'DBC', 'COPX', 'URA', 'PALL', 'CANE',
                'CORN', 'WEAT', 'UGA', 'KOLD', 'UNG'
            ],

            # Leveraged & inverse products, volatility, FX, and complex derivatives
            'Derivatives': [
                'VXX', 'UVXY', 'SQQQ', 'TQQQ', 'SPXL',
                'SPXS', 'SOXL', 'SOXS', 'UPRO', 'TMF',
                'TZA', 'URTY', 'LABU', 'LABD', 'DRV',
                'UUP', 'FXE', 'FXY', 'EWJ', 'EWZ',
                'FXB', 'FXF', 'FXA', 'CYB', 'CEW'
            ]
        }

    def setup_and_train_model(self, real_market_data: Optional[Dict] = None) -> None:
        """Initializes, trains, and optionally calibrates the FinanceMoEModel.

        This method performs the following steps:
        1. Initializes the model on the specified device.
        2. Trains the model's router using synthetically generated data.
        3. If real market data is provided, it calibrates the router on this
           data to fine-tune its performance on real-world distributions.

        Args:
            real_market_data: A dictionary of raw market data, used for
                optional calibration.
        """
        print("🔧 Setting up Finance MoE Model...")
        if self.device == 'cpu':
            raise RuntimeError("CUDA GPU is required to run the model.")

        # Initialize and train model
        self.model = FinanceMoEModel(hidden_size=self._hidden_size).to(self.device)
        print("✅ Model initialized!")

        print("🎯 Training the MoE router on synthetic data...")
        self.model.train_router(num_epochs=120, batch_size=32, seq_len=60)
        self.model.eval()

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

            # Feature engineering: Create a feature set for the model
            features = pd.DataFrame({
                'price_norm': (hist['Close'] - hist['Close'].mean()) / hist['Close'].std(),
                'volume_norm': (hist['Volume'] - hist['Volume'].mean()) / hist['Volume'].std(),
                'volatility': hist['Volatility'],
                'rsi': hist['RSI'] / 100.0,  # Normalize RSI to be between 0 and 1
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
        """Classifies the market condition for an asset category based on heuristics.

        Args:
            category_data: A list of processed data dictionaries for tickers
                in a single category.

        Returns:
            A dictionary describing the classified market condition.
        """
        # Aggregate recent volatility and returns across all tickers in the category
        recent_vol = np.mean([
            np.mean(data['volatility'][-10:]) for data in category_data
            if len(data['volatility']) >= 10
        ])
        recent_ret = np.mean([
            np.mean(data['returns'][-10:]) for data in category_data
            if len(data['returns']) >= 10
        ])

        # Heuristic rules for market classification. These thresholds could be
        # made configurable for more advanced analysis.
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
                data for ticker in tickers[:10]  # Limit to first 10 tickers for faster demo
                if (data := self._fetch_ticker_data(ticker, start_date, end_date))
            ]

            if category_data:
                all_data[category] = category_data
                # Classify and store the market condition for the category
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

                # Ensure feature dimensions match the model's expected hidden_size
                if features.shape[1] < self._hidden_size:
                    # Pad with zeros if there are fewer features than hidden_size
                    pad = np.zeros((features.shape[0], self._hidden_size - features.shape[1]))
                    features = np.hstack([features, pad])
                elif features.shape[1] > self._hidden_size:
                    # Use PCA to reduce dimensions if there are more features
                    pca = PCA(n_components=self._hidden_size)
                    features = pca.fit_transform(features)

                all_features.append(features)
                all_volatility.append(ticker_data['volatility'])
                # Use absolute returns as a simple proxy for risk
                all_risk_factors.append(np.abs(ticker_data['returns']).reshape(-1, 1))

            if all_features:
                # Aggregate data by averaging across all tickers in the category
                # to create a single representative time series for the asset class.
                avg_features = np.mean(all_features, axis=0)
                avg_volatility = np.mean(all_volatility, axis=0)
                avg_risk = np.mean(all_risk_factors, axis=0)

                # Ensure we have a consistent sequence length for the model
                min_seq_len = min(self._lookback_days, avg_features.shape[0])

                processed_data[category] = {
                    'embeddings': torch.tensor(
                        avg_features[-min_seq_len:].reshape(1, min_seq_len, self._hidden_size),
                        dtype=torch.float32, device=self.device
                    ),
                    'volatility': torch.tensor(
                        avg_volatility[-min_seq_len:].reshape(1, min_seq_len, 1),
                        dtype=torch.float32, device=self.device
                    ),
                    'risk_factors': torch.tensor(
                        avg_risk[-min_seq_len:].reshape(1, min_seq_len, 1),
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
        print("🎯 Running real data through the trained MoE router...")
        results = {}
        if not self.model:
            raise RuntimeError("Model is not set up. Call setup_and_train_model() first.")

        with torch.no_grad():
            for category, data in processed_data.items():
                try:
                    # Run the trained model forward pass
                    predictions, routing_info = self.model(
                        data['embeddings'], data['volatility'], data['risk_factors']
                    )

                    # Store results with context for visualization and analysis
                    results[category] = {
                        'predictions': predictions.cpu().numpy(),
                        'domain_assignments': routing_info['domain_assignments'].cpu().numpy(),
                        'routing_probs': routing_info['routing_probs'].cpu().numpy(),
                        'domain_names': routing_info['domain_names'],
                        'market_scenario': market_scenarios.get(category, {}),
                        'volatility': data['volatility'].cpu().numpy(),
                        'tickers': list(self.asset_categories[category][:10])
                    }

                    # Print a quick summary of the routing decision for this category
                    assignments = routing_info['domain_assignments'].cpu().numpy().flatten()
                    most_common_domain = np.bincount(assignments).argmax()
                    confidence = np.mean(np.max(routing_info['routing_probs'].cpu().numpy()[0], axis=1))

                    print(f"   ✅ {category}: Routed to '{routing_info['domain_names'][most_common_domain]}' "
                          f"expert (confidence: {confidence:.2%})")

                except Exception as e:
                    print(f"   ⚠️  Couldn't process {category}: {e}")

        return results

    def create_matplotlib_visualizations(self, results: Dict) -> List[plt.Figure]:
        """Creates comprehensive matplotlib visualizations in separate figures.

        Each figure is designed to analyze a different aspect of the model's
        performance on real-world data, from expert selection patterns to
        market condition analysis.

        Args:
            results: A dictionary of analysis results from the model.

        Returns:
            A list of matplotlib Figure objects, which are also saved to disk.
        """
        # Set global matplotlib style for professional-looking plots
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
        # This figure shows which experts are chosen for different asset classes.
        print("   - Generating expert routing analysis...")
        fig1 = plt.figure(figsize=(16, 8), facecolor='white')
        fig1.suptitle('Expert Routing Analysis - Trained Finance MoE Router',
                      fontsize=20, fontweight='bold', y=0.95, color='#2E3440')

        # Use GridSpec for a more controlled layout
        gs1 = fig1.add_gridspec(1, 2, hspace=0.3, wspace=0.4,
                                left=0.08, right=0.92, top=0.85, bottom=0.15)

        # Left plot: Heatmap of expert selection by asset category
        ax1a = fig1.add_subplot(gs1[0, 0])
        categories = list(results.keys())
        expert_counts = []

        for category, data in results.items():
            assignments = data['domain_assignments'].flatten()
            domain_counts = np.bincount(assignments, minlength=4)
            expert_counts.append(domain_counts)

        if expert_counts:
            expert_matrix = np.array(expert_counts)
            im = ax1a.imshow(expert_matrix, cmap='Blues', aspect='auto',
                             interpolation='nearest', alpha=0.9)
            ax1a.set_title('Expert Selection by Real Asset Class',
                           fontsize=16, fontweight='bold', pad=20, color='#2E3440')
            ax1a.set_xlabel('Expert Domain', fontsize=13, color='#2E3440')
            ax1a.set_ylabel('Asset Category', fontsize=13, color='#2E3440')
            ax1a.set_yticks(range(len(categories)))
            ax1a.set_yticklabels(categories, fontsize=11, color='#2E3440')
            ax1a.set_xticks(range(4))

            domain_names = list(results.values())[0]['domain_names']
            ax1a.set_xticklabels(domain_names, rotation=45, ha='right',
                                 fontsize=11, color='#2E3440')

            cbar = plt.colorbar(im, ax=ax1a, shrink=0.8, pad=0.02)
            cbar.set_label('Times Selected', fontsize=12, color='#2E3440')
            cbar.ax.tick_params(labelsize=10, colors='#2E3440')

            # Add numbers on the heatmap
            for i in range(len(categories)):
                for j in range(4):
                    value = expert_matrix[i, j]
                    color = 'white' if value > expert_matrix.max() * 0.6 else '#2E3440'
                    ax1a.text(j, i, f'{value}', ha="center", va="center",
                              color=color, fontweight='bold', fontsize=12)

        # Right plot: Bar chart of overall expert popularity
        ax1b = fig1.add_subplot(gs1[0, 1])
        all_assignments = []
        for data in results.values():
            all_assignments.extend(data['domain_assignments'].flatten())

        if all_assignments:
            domain_counts = np.bincount(all_assignments, minlength=4)
            domain_names = list(results.values())[0]['domain_names']

            colors_bar = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
            bars = ax1b.bar(range(len(domain_names)), domain_counts,
                            color=colors_bar[:len(domain_names)], alpha=0.8,
                            edgecolor='white', linewidth=2)

            ax1b.set_title('Overall Expert Popularity (Real Data)',
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
        filename1 = 'finance_moe_expert_routing_real_data.png'
        fig1.savefig(filename1, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.3)
        print(f"   ✅ Saved: {filename1}")
        figures.append(fig1)

        # ===== FIGURE 2: Current Market Conditions =====
        # This figure visualizes the market conditions (volatility vs. returns)
        # for each asset class at the time of the analysis.
        print("   - Generating market conditions analysis...")
        fig2 = plt.figure(figsize=(16, 8), facecolor='white')
        fig2.suptitle('Real Market Conditions Analysis - Trained MoE Router',
                      fontsize=20, fontweight='bold', y=0.95, color='#2E3440')

        gs2 = fig2.add_gridspec(1, 2, hspace=0.3, wspace=0.4,
                                left=0.08, right=0.92, top=0.85, bottom=0.15)

        # Left: Scatter plot of returns vs. volatility for each asset class
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
        ax2a.set_title('Returns vs Volatility by Asset Class (Real Data)',
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

        # Right: Bar chart comparing volatility and returns side-by-side
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

        ax2b.set_title('Market Stats by Category (Real Data)',
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
        filename2 = 'finance_moe_market_conditions_real_data.png'
        fig2.savefig(filename2, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.3)
        print(f"   ✅ Saved: {filename2}")
        figures.append(fig2)

        # ===== FIGURE 3: Market Volatility Analysis =====
        # This figure plots the volatility time series for each asset class.
        print("   - Generating market volatility analysis...")
        fig3 = plt.figure(figsize=(16, 8), facecolor='white')
        fig3.suptitle('Real Market Volatility Analysis - Trained MoE Router',
                      fontsize=20, fontweight='bold', y=0.95, color='#2E3440')

        ax3 = fig3.add_subplot(111)
        colors = ['#E91E63', '#9C27B0', '#673AB7', '#3F51B5', '#2196F3', '#00BCD4']

        for i, (category, data) in enumerate(results.items()):
            volatility = data['volatility'][0].flatten()
            color = colors[i % len(colors)]

            ax3.plot(volatility, label=f'{category}',
                     color=color, linewidth=3, alpha=0.9, marker='o',
                     markersize=4, markevery=max(1, len(volatility) // 20))

        ax3.set_title('Market Volatility Over Time (Real Data)',
                      fontsize=16, fontweight='bold', pad=20, color='#2E3440')
        ax3.set_xlabel('Time Steps (Trading Days)', fontsize=13, color='#2E3440')
        ax3.set_ylabel('Volatility', fontsize=13, color='#2E3440')
        ax3.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=11)
        ax3.grid(True, alpha=0.4, linestyle='-', linewidth=0.5)
        ax3.set_facecolor('#FAFAFA')

        # Add horizontal bands to indicate volatility levels
        max_vol = max([data['volatility'][0].flatten().max() for data in results.values()]) if results else 0.1
        ax3.axhspan(0, max_vol * 0.3, alpha=0.1, color='green', label='Low Volatility')
        ax3.axhspan(max_vol * 0.3, max_vol * 0.7, alpha=0.1, color='yellow', label='Medium Volatility')
        ax3.axhspan(max_vol * 0.7, max_vol, alpha=0.1, color='red', label='High Volatility')

        plt.tight_layout()
        filename3 = 'finance_moe_volatility_real_data.png'
        fig3.savefig(filename3, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.3)
        print(f"   ✅ Saved: {filename3}")
        figures.append(fig3)

        # ===== FIGURE 4: Performance Analysis =====
        # This figure explores the relationship between asset performance (returns)
        # and the expert chosen by the router.
        print("   - Generating performance analysis...")
        fig4 = plt.figure(figsize=(12, 8), facecolor='white')
        fig4.suptitle('Performance Analysis - Real Asset Returns vs Expert Preferences',
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
            domain_counts = np.bincount(assignments, minlength=4)
            most_used_expert = np.argmax(domain_counts)
            expert_preference.append(most_used_expert)
            category_names.append(category)

        colors_scatter = ['#E91E63', '#9C27B0', '#673AB7', '#3F51B5']
        ax4.scatter(performance_data, expert_preference,
                    c=[colors_scatter[i % len(colors_scatter)] for i in expert_preference],
                    s=250, alpha=0.8, edgecolors='white', linewidth=3, zorder=5)

        ax4.set_title('Do Better Assets Prefer Certain Experts? (Real Data)',
                      fontsize=16, fontweight='bold', pad=20, color='#2E3440')
        ax4.set_xlabel('Recent Returns (%)', fontsize=13, color='#2E3440')
        ax4.set_ylabel('Most Preferred Expert', fontsize=13, color='#2E3440')
        ax4.grid(True, alpha=0.4, linestyle='--')
        ax4.set_facecolor('#FAFAFA')
        ax4.set_yticks(range(4))

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
        filename4 = 'finance_moe_performance_real_data.png'
        fig4.savefig(filename4, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.3)
        print(f"   ✅ Saved: {filename4}")
        figures.append(fig4)

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
        print("🧠 REAL-WORLD DATA INSIGHTS FROM TRAINED MODEL")
        print("=" * 70)

        # Overall model performance summary
        total_predictions = sum(len(data['domain_assignments'].flatten()) for data in results.values())
        print(f"\n📈 ANALYSIS SUMMARY:")
        print(f"   - Total predictions made: {total_predictions}")
        print(f"   - Asset categories analyzed: {len(results)}")

        # Detailed analysis for each category
        for category, data in results.items():
            scenario = data['market_scenario']
            assignments = data['domain_assignments'].flatten()
            domain_counts = np.bincount(assignments, minlength=4)
            most_used_idx = np.argmax(domain_counts)
            usage_percentage = (domain_counts[most_used_idx] / len(assignments)) * 100
            avg_confidence = np.mean(np.max(data['routing_probs'][0], axis=1))

            print(f"\n📊 {category.upper()}:")
            print(f"   - Market Condition: {scenario.get('description', 'N/A')}")
            print(f"   - Recent Volatility: {scenario.get('volatility', 0):.4f}")
            print(f"   - Recent Returns: {scenario.get('returns', 0) * 100:.2f}%")
            print(f"   - Sample Tickers: {', '.join(data['tickers'][:5])}...")
            print(f"   - Primary Expert: {data['domain_names'][most_used_idx]} ({usage_percentage:.1f}% of time)")
            print(f"   - Average Confidence: {avg_confidence:.2%}")

            # Show expert distribution
            expert_distribution = []
            for i, count in enumerate(domain_counts):
                if count > 0:
                    pct = (count / len(assignments)) * 100
                    expert_distribution.append(f"{data['domain_names'][i]} ({pct:.1f}%)")
            print(f"   - Expert Distribution: {', '.join(expert_distribution)}")

        # Cross-category insights summarizing overall trends
        print(f"\n🔍 CROSS-CATEGORY INSIGHTS:")

        # Find which expert is most popular overall
        all_assignments = []
        for data in results.values():
            all_assignments.extend(data['domain_assignments'].flatten())

        if all_assignments:
            overall_counts = np.bincount(all_assignments, minlength=4)
            most_popular_expert_idx = np.argmax(overall_counts)
            domain_names = list(results.values())[0]['domain_names']

            print(f"   - Most utilized expert: {domain_names[most_popular_expert_idx]} "
                  f"({(overall_counts[most_popular_expert_idx] / len(all_assignments) * 100):.1f}% overall)")

            # Show expert usage ranking
            expert_ranking = sorted(enumerate(overall_counts), key=lambda x: x[1], reverse=True)
            print(f"   - Expert popularity ranking:")
            for rank, (expert_idx, count) in enumerate(expert_ranking, 1):
                if count > 0 and expert_idx < len(domain_names):
                    pct = (count / len(all_assignments)) * 100
                    print(f"      {rank}. {domain_names[expert_idx]}: {pct:.1f}%")

        # Market condition insights
        high_vol_categories = [cat for cat, data in results.items()
                               if data['market_scenario'].get('volatility', 0) > 0.02]
        low_vol_categories = [cat for cat, data in results.items()
                              if data['market_scenario'].get('volatility', 0) < 0.01]

        if high_vol_categories:
            print(f"   - High volatility categories: {', '.join(high_vol_categories)}")
        if low_vol_categories:
            print(f"   - Low volatility categories: {', '.join(low_vol_categories)}")

        # Model confidence insights
        confidence_by_category = {cat: np.mean(np.max(data['routing_probs'][0], axis=1))
                                  for cat, data in results.items()}
        most_confident_cat = max(confidence_by_category, key=confidence_by_category.get)
        least_confident_cat = min(confidence_by_category, key=confidence_by_category.get)

        print(f"   - Highest confidence routing: {most_confident_cat} "
              f"({confidence_by_category[most_confident_cat]:.2%})")
        print(f"   - Lowest confidence routing: {least_confident_cat} "
              f"({confidence_by_category[least_confident_cat]:.2%})")

        print("\n" + "=" * 70)


def main() -> None:
    """Main function to run the real-data financial MoE demo with trained model."""
    print("=" * 60)
    print("FINANCE MOE ROUTER - REAL DATA ANALYSIS WITH TRAINED MODEL")
    print("=" * 60)

    try:
        # 1. Initialize the demo environment
        demo = RealDataFinanceDemo(lookback_days=40, hidden_size=32)

        # 2. Fetch real-time market data from Yahoo Finance
        market_data, market_scenarios = demo.fetch_real_market_data()
        if not market_data:
            print("❌ Error: Unable to fetch market data. Aborting.")
            return

        # 3. Setup the model and train it on synthetic data.
        #    Optionally, calibrate it with the fetched real data.
        demo.setup_and_train_model(real_market_data=market_data)

        # 4. Prepare the fetched data into a model-compatible format
        processed_data = demo.prepare_model_inputs(market_data)

        # 5. Run the trained model on the processed real-world data
        results = demo.run_real_data_analysis(processed_data, market_scenarios)
        if not results:
            print("❌ Error: Model analysis failed. Aborting.")
            return

        # 6. Generate and save detailed visualizations
        demo.create_matplotlib_visualizations(results)

        # 7. Print a text-based summary of insights to the console
        demo.print_real_data_insights(results)

        print("\n🎉 Demo completed successfully!")
        print("📊 Check the generated PNG files for detailed visualizations.")
        print("🔧 The model was trained on synthetic data and tested on real market data.")
        print("📈 Routing decisions show how different asset classes are classified by the MoE system.")

    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted by user.")
    except Exception as e:
        print(f"\n🚨 An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
        print("\n💡 Troubleshooting tips:")
        print("   - Ensure you have a CUDA-capable GPU")
        print("   - Check internet connection for market data")
        print("   - Verify all required Python packages are installed")


if __name__ == "__main__":
    main()
