'''
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime
import os
from django.conf import settings

class FraudVisualizer:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
    def generate_all_visualizations(self, df):
        """Generate all visualization files"""
        visualizations = {}
        
        # Risk distribution pie chart
        viz_path = self.create_risk_distribution_chart(df)
        if viz_path:
            visualizations['risk_distribution'] = viz_path
        
        # Days to report histogram
        viz_path = self.create_days_to_report_histogram(df)
        if viz_path:
            visualizations['days_to_report'] = viz_path
        
        # Fraud score distribution
        viz_path = self.create_fraud_score_distribution(df)
        if viz_path:
            visualizations['fraud_score_dist'] = viz_path
        
        # Red flags frequency
        viz_path = self.create_red_flags_chart(df)
        if viz_path:
            visualizations['red_flags'] = viz_path
        
        # Monthly trend
        viz_path = self.create_monthly_trend(df)
        if viz_path:
            visualizations['monthly_trend'] = viz_path
        
        # Correlation heatmap
        viz_path = self.create_correlation_heatmap(df)
        if viz_path:
            visualizations['correlation'] = viz_path
        
        return visualizations
    
    def create_risk_distribution_chart(self, df):
        """Create risk level distribution pie chart"""
        try:
            plt.figure(figsize=(10, 8))
            
            risk_counts = df['risk_level'].value_counts()
            colors = {'Low': '#2ecc71', 'Medium': '#f39c12', 'High': '#e74c3c', 'Critical': '#c0392b'}
            
            plt.pie(risk_counts.values, labels=risk_counts.index, autopct='%1.1f%%',
                   colors=[colors.get(level, '#95a5a6') for level in risk_counts.index],
                   startangle=90)
            
            plt.title('Risk Level Distribution', fontsize=16, fontweight='bold')
            
            filename = 'risk_distribution.png'
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            return filename
        except Exception as e:
            print(f"Error creating risk distribution chart: {e}")
            return None
    
    def create_days_to_report_histogram(self, df):
        """Create histogram of days to report"""
        try:
            if 'days_to_report' not in df.columns:
                return None
            
            plt.figure(figsize=(12, 8))
            
            # Filter out extreme values for better visualization
            days_data = df['days_to_report'].dropna()
            days_data = days_data[days_data <= 365]  # Cap at 1 year
            
            plt.hist(days_data, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
            plt.axvline(x=30, color='red', linestyle='--', linewidth=2, label='30-day threshold')
            
            plt.xlabel('Days to Report', fontsize=12)
            plt.ylabel('Number of Claims', fontsize=12)
            plt.title('Distribution of Days to Report Claim', fontsize=16, fontweight='bold')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            filename = 'days_to_report.png'
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            return filename
        except Exception as e:
            print(f"Error creating days to report histogram: {e}")
            return None
    
    def create_fraud_score_distribution(self, df):
        """Create fraud score distribution plot"""
        try:
            plt.figure(figsize=(12, 8))
            
            # Create histogram with KDE
            plt.hist(df['fraud_score'], bins=30, density=True, alpha=0.7, color='coral', edgecolor='black')
            
            # Add KDE line
            from scipy import stats
            kde = stats.gaussian_kde(df['fraud_score'])
            x_range = np.linspace(0, 100, 200)
            plt.plot(x_range, kde(x_range), 'b-', linewidth=2, label='Density')
            
            # Add risk level thresholds
            plt.axvline(x=30, color='green', linestyle='--', linewidth=2, label='Low/Medium')
            plt.axvline(x=50, color='orange', linestyle='--', linewidth=2, label='Medium/High')
            plt.axvline(x=70, color='red', linestyle='--', linewidth=2, label='High/Critical')
            
            plt.xlabel('Fraud Score', fontsize=12)
            plt.ylabel('Density', fontsize=12)
            plt.title('Fraud Score Distribution', fontsize=16, fontweight='bold')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            filename = 'fraud_score_distribution.png'
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            return filename
        except Exception as e:
            print(f"Error creating fraud score distribution: {e}")
            return None
    
    def create_red_flags_chart(self, df):
        """Create bar chart of most common red flags"""
        try:
            # Count all red flags
            all_flags = []
            for flags in df['red_flags']:
                all_flags.extend(flags)
            
            from collections import Counter
            flag_counts = Counter(all_flags)
            top_flags = dict(flag_counts.most_common(10))
            
            if not top_flags:
                return None
            
            plt.figure(figsize=(12, 8))
            
            flags = list(top_flags.keys())
            counts = list(top_flags.values())
            
            bars = plt.barh(flags, counts, color='steelblue')
            
            # Add value labels
            for bar in bars:
                width = bar.get_width()
                plt.text(width, bar.get_y() + bar.get_height()/2, 
                        f'{int(width)}', ha='left', va='center', fontweight='bold')
            
            plt.xlabel('Number of Claims', fontsize=12)
            plt.title('Top 10 Most Common Red Flags', fontsize=16, fontweight='bold')
            plt.grid(True, alpha=0.3, axis='x')
            
            filename = 'red_flags_frequency.png'
            filepath = os.path.join(self.output_dir, filename)
            plt.tight_layout()
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            return filename
        except Exception as e:
            print(f"Error creating red flags chart: {e}")
            return None
    
    def create_monthly_trend(self, df):
        """Create monthly trend chart"""
        try:
            if 'Date of Loss' not in df.columns:
                return None
            
            # Create monthly aggregation
            df['month'] = pd.to_datetime(df['Date of Loss']).dt.to_period('M')
            monthly_stats = df.groupby('month').agg({
                'fraud_score': 'mean',
                'risk_level': lambda x: (x.isin(['High', 'Critical']).sum() / len(x) * 100) if len(x) > 0 else 0
            }).reset_index()
            
            if len(monthly_stats) < 2:
                return None
            
            fig, ax1 = plt.subplots(figsize=(14, 8))
            
            # Plot average fraud score
            color = 'tab:blue'
            ax1.set_xlabel('Month', fontsize=12)
            ax1.set_ylabel('Average Fraud Score', color=color, fontsize=12)
            ax1.plot(monthly_stats.index, monthly_stats['fraud_score'], 
                    color=color, marker='o', linewidth=2, markersize=8)
            ax1.tick_params(axis='y', labelcolor=color)
            ax1.grid(True, alpha=0.3)
            
            # Create second y-axis for high-risk percentage
            ax2 = ax1.twinx()
            color = 'tab:red'
            ax2.set_ylabel('High Risk Claims (%)', color=color, fontsize=12)
            ax2.plot(monthly_stats.index, monthly_stats['risk_level'], 
                    color=color, marker='s', linewidth=2, markersize=8)
            ax2.tick_params(axis='y', labelcolor=color)
            
            # Format x-axis
            ax1.set_xticks(monthly_stats.index)
            ax1.set_xticklabels([str(m) for m in monthly_stats['month']], rotation=45)
            
            plt.title('Monthly Fraud Trends', fontsize=16, fontweight='bold', pad=20)
            fig.tight_layout()
            
            filename = 'monthly_trend.png'
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            return filename
        except Exception as e:
            print(f"Error creating monthly trend: {e}")
            return None
    

    def create_correlation_heatmap(self, df):
        """Create correlation heatmap of fraud indicators"""
        try:
            # Select numeric fraud indicator columns
            indicator_cols = [
                'delayed_reporting', 'near_birthday', 'new_employee_30d', 
                'new_employee_90d', 'multiple_claims', 'soft_tissue_injury',
                'no_witness', 'suspicious_body_part', 'weekend_injury',
                'summer_claim', 'fraud_score'
            ]
            
            available_cols = [col for col in indicator_cols if col in df.columns]
            
            if len(available_cols) < 3:
                return None
            
            # Convert boolean to int for correlation
            corr_data = df[available_cols].astype(float)
            correlation_matrix = corr_data.corr()
            
            plt.figure(figsize=(12, 10))
            
            # Create heatmap
            mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
            sns.heatmap(correlation_matrix, mask=mask, annot=True, fmt='.2f', 
                       cmap='coolwarm', center=0, square=True, linewidths=1,
                       cbar_kws={"shrink": .8})
            
            plt.title('Fraud Indicator Correlation Matrix', fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            filename = 'correlation_heatmap.png'
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            return filename
        except Exception as e:
            print(f"Error creating correlation heatmap: {e}")
            return None
'''

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime
import os
from django.contrib.humanize.templatetags.humanize import intcomma
from django.conf import settings

class FraudVisualizer:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Modern color palette
        self.colors = {
            'primary': '#667eea',
            'secondary': '#764ba2',
            'success': '#10b981',
            'warning': '#f59e0b',
            'danger': '#ef4444',
            'info': '#3b82f6',
            'gray': '#6b7280',
            'light': '#f3f4f6',
            'dark': '#1f2937'
        }
        
        # Risk level colors
        self.risk_colors = {
            'Low': '#10b981',
            'Medium': '#f59e0b',
            'High': '#ef4444',
            'Critical': '#dc2626'
        }
        
        # Set modern style
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # Configure matplotlib for consistent appearance
        plt.rcParams.update({
            'figure.facecolor': 'white',
            'axes.facecolor': 'white',
            'axes.edgecolor': '#e5e7eb',
            'axes.linewidth': 1,
            'grid.color': '#f3f4f6',
            'grid.linewidth': 1,
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial', 'DejaVu Sans'],
            'font.size': 11,
            'axes.titlesize': 14,
            'axes.titleweight': 'bold',
            'axes.labelsize': 12,
            'axes.labelweight': 'medium',
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'legend.frameon': True,
            'legend.fancybox': True,
            'legend.shadow': False,
            'legend.edgecolor': '#e5e7eb',
            'figure.autolayout': True
        })
        
    def generate_all_visualizations(self, df):
        """Generate all visualization files with consistent styling"""
        visualizations = {}
        
        # Risk distribution pie chart
        viz_path = self.create_risk_distribution_chart(df)
        if viz_path:
            visualizations['risk_distribution'] = viz_path
        
        # Days to report histogram
        viz_path = self.create_days_to_report_histogram(df)
        if viz_path:
            visualizations['days_to_report'] = viz_path
        
        # Fraud score distribution
        viz_path = self.create_fraud_score_distribution(df)
        if viz_path:
            visualizations['fraud_score_dist'] = viz_path
        
        # Red flags frequency
        viz_path = self.create_red_flags_chart(df)
        if viz_path:
            visualizations['red_flags'] = viz_path
        
        # Monthly trend
        viz_path = self.create_monthly_trend(df)
        if viz_path:
            visualizations['monthly_trend'] = viz_path
        
        # Correlation heatmap
        viz_path = self.create_correlation_heatmap(df)
        if viz_path:
            visualizations['correlation'] = viz_path
        
        return visualizations
    
    def create_risk_distribution_chart(self, df):
        """Create modern risk level distribution pie chart"""
        try:
            fig, ax = plt.subplots(figsize=(10, 8))
            fig.patch.set_facecolor('white')
            
            risk_counts = df['risk_level'].value_counts()
            colors = [self.risk_colors.get(level, self.colors['gray']) for level in risk_counts.index]
            
            # Create pie chart with modern styling
            wedges, texts, autotexts = ax.pie(
                risk_counts.values, 
                labels=risk_counts.index, 
                autopct='%1.1f%%',
                colors=colors,
                startangle=90,
                explode=[0.05 if level in ['High', 'Critical'] else 0 for level in risk_counts.index],
                shadow=False,
                textprops={'fontsize': 12, 'weight': 'medium'}
            )
            
            # Enhance text appearance
            for text in texts:
                text.set_fontsize(13)
                text.set_weight('bold')
            
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontsize(12)
                autotext.set_weight('bold')
            
            ax.set_title('Risk Level Distribution', fontsize=16, weight='bold', pad=20, color=self.colors['dark'])
            
            # Add total count in center
            total = risk_counts.sum()
            ax.text(0, 0, f'{total:,}\nTotal Claims', ha='center', va='center', 
                   fontsize=14, weight='bold', color=self.colors['dark'])
            
            plt.tight_layout()
            filename = 'risk_distribution.png'
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=150, facecolor='white', edgecolor='none')
            plt.close()
            
            return filename
        except Exception as e:
            print(f"Error creating risk distribution chart: {e}")
            plt.close()
            return None
    
    def create_days_to_report_histogram(self, df):
        """Create modern histogram of days to report"""
        try:
            if 'days_to_report' not in df.columns:
                return None
            
            fig, ax = plt.subplots(figsize=(12, 8))
            fig.patch.set_facecolor('white')
            
            # Filter data
            days_data = df['days_to_report'].dropna()
            days_data = days_data[days_data <= 365]
            
            if len(days_data) == 0:
                plt.close()
                return None
            
            # Create histogram with gradient effect
            n, bins, patches = ax.hist(
                days_data, 
                bins=30, 
                color=self.colors['primary'],
                alpha=0.8,
                edgecolor='white',
                linewidth=1.5
            )
            
            # Add gradient effect to bars
            for i, patch in enumerate(patches):
                if bins[i] > 30:
                    patch.set_facecolor(self.colors['warning'])
                    patch.set_alpha(0.9)
            
            # Add threshold line
            ax.axvline(x=30, color=self.colors['danger'], linestyle='--', linewidth=2.5, 
                      label='30-day threshold', alpha=0.8)
            
            # Styling
            ax.set_xlabel('Days to Report', fontsize=13, weight='medium', color=self.colors['dark'])
            ax.set_ylabel('Number of Claims', fontsize=13, weight='medium', color=self.colors['dark'])
            ax.set_title('Distribution of Days to Report Claim', fontsize=16, weight='bold', 
                        pad=20, color=self.colors['dark'])
            
            # Add statistics box
            mean_days = days_data.mean()
            median_days = days_data.median()
            textstr = f'Mean: {mean_days:.1f} days\nMedian: {median_days:.1f} days'
            props = dict(boxstyle='round,pad=0.5', facecolor=self.colors['light'], 
                        edgecolor=self.colors['gray'], alpha=0.8)
            ax.text(0.70, 0.95, textstr, transform=ax.transAxes, fontsize=11,
                   verticalalignment='top', bbox=props)
            
            ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=False)
            ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
            ax.set_axisbelow(True)
            
            plt.tight_layout()
            filename = 'days_to_report.png'
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=150, facecolor='white', edgecolor='none')
            plt.close()
            
            return filename
        except Exception as e:
            print(f"Error creating days to report histogram: {e}")
            plt.close()
            return None
    
    def create_fraud_score_distribution(self, df):
        """Create modern fraud score distribution plot"""
        try:
            fig, ax = plt.subplots(figsize=(12, 8))
            fig.patch.set_facecolor('white')
            
            # Create histogram
            n, bins, patches = ax.hist(
                df['fraud_score'], 
                bins=25, 
                density=True, 
                alpha=0.7,
                edgecolor='white',
                linewidth=1.5
            )
            
            # Color bars based on risk levels
            for i, patch in enumerate(patches):
                if bins[i] >= 70:
                    patch.set_facecolor(self.colors['danger'])
                elif bins[i] >= 50:
                    patch.set_facecolor(self.colors['warning'])
                elif bins[i] >= 30:
                    patch.set_facecolor(self.colors['info'])
                else:
                    patch.set_facecolor(self.colors['success'])
            
            # Add KDE line
            from scipy import stats
            kde = stats.gaussian_kde(df['fraud_score'])
            x_range = np.linspace(0, 100, 200)
            ax.plot(x_range, kde(x_range), color=self.colors['secondary'], 
                   linewidth=3, label='Density curve', alpha=0.8)
            
            # Add risk level thresholds
            threshold_lines = [
                (30, 'Low/Medium', self.colors['success']),
                (50, 'Medium/High', self.colors['warning']),
                (70, 'High/Critical', self.colors['danger'])
            ]
            
            for threshold, label, color in threshold_lines:
                ax.axvline(x=threshold, color=color, linestyle='--', 
                          linewidth=2, label=label, alpha=0.7)
            
            # Styling
            ax.set_xlabel('Fraud Score', fontsize=13, weight='medium', color=self.colors['dark'])
            ax.set_ylabel('Density', fontsize=13, weight='medium', color=self.colors['dark'])
            ax.set_title('Fraud Score Distribution', fontsize=16, weight='bold', 
                        pad=20, color=self.colors['dark'])
            
            ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=False)
            ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
            ax.set_axisbelow(True)
            
            plt.tight_layout()
            filename = 'fraud_score_distribution.png'
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=150, facecolor='white', edgecolor='none')
            plt.close()
            
            return filename
        except Exception as e:
            print(f"Error creating fraud score distribution: {e}")
            plt.close()
            return None
    
    def create_red_flags_chart(self, df):
        """Create modern bar chart of most common red flags"""
        try:
            # Count red flags
            all_flags = []
            for flags in df['red_flags']:
                if isinstance(flags, list):
                    all_flags.extend(flags)
            
            if not all_flags:
                return None
            
            from collections import Counter
            flag_counts = Counter(all_flags)
            top_flags = dict(flag_counts.most_common(10))
            
            if not top_flags:
                return None
            
            fig, ax = plt.subplots(figsize=(12, 8))
            fig.patch.set_facecolor('white')
            
            flags = list(top_flags.keys())
            counts = list(top_flags.values())
            
            # Create horizontal bar chart with gradient
            y_pos = np.arange(len(flags))
            bars = ax.barh(y_pos, counts)
            
            # Apply gradient colors
            max_count = max(counts)
            for i, (bar, count) in enumerate(zip(bars, counts)):
                intensity = count / max_count
                if intensity > 0.7:
                    bar.set_facecolor(self.colors['danger'])
                elif intensity > 0.4:
                    bar.set_facecolor(self.colors['warning'])
                else:
                    bar.set_facecolor(self.colors['info'])
                bar.set_alpha(0.8)
                bar.set_edgecolor('white')
                bar.set_linewidth(1.5)
                
                # Add value labels
                ax.text(count + max_count * 0.01, bar.get_y() + bar.get_height()/2, 
                       f'{count:,}', va='center', fontsize=11, weight='bold', 
                       color=self.colors['dark'])
            
            # Styling
            ax.set_yticks(y_pos)
            ax.set_yticklabels(flags, fontsize=11)
            ax.set_xlabel('Number of Claims', fontsize=13, weight='medium', color=self.colors['dark'])
            ax.set_title('Top 10 Most Common Red Flags', fontsize=16, weight='bold', 
                        pad=20, color=self.colors['dark'])
            
            ax.grid(True, alpha=0.3, axis='x', linestyle='-', linewidth=0.5)
            ax.set_axisbelow(True)
            
            # Remove top and right spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            plt.tight_layout()
            filename = 'red_flags_frequency.png'
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=150, facecolor='white', edgecolor='none')
            plt.close()
            
            return filename
        except Exception as e:
            print(f"Error creating red flags chart: {e}")
            plt.close()
            return None
    
    def create_monthly_trend(self, df):
        """Create modern monthly trend chart"""
        try:
            if 'Date of Loss' not in df.columns:
                return None
            
            # Process data
            df['Date of Loss'] = pd.to_datetime(df['Date of Loss'], errors='coerce')
            valid_dates = df['Date of Loss'].notna()
            
            if valid_dates.sum() == 0:
                return None
            
            df_valid = df[valid_dates].copy()
            df_valid['month'] = df_valid['Date of Loss'].dt.to_period('M')
            
            monthly_stats = df_valid.groupby('month').agg({
                'fraud_score': 'mean',
                'risk_level': lambda x: (x.isin(['High', 'Critical']).sum() / len(x) * 100) if len(x) > 0 else 0
            }).reset_index()
            
            if len(monthly_stats) < 2:
                return None
            
            fig, ax1 = plt.subplots(figsize=(14, 8))
            fig.patch.set_facecolor('white')
            
            # Plot average fraud score
            line1 = ax1.plot(monthly_stats.index, monthly_stats['fraud_score'], 
                            color=self.colors['primary'], marker='o', linewidth=3, 
                            markersize=10, label='Average Fraud Score', alpha=0.8)
            
            # Fill area under the line
            ax1.fill_between(monthly_stats.index, monthly_stats['fraud_score'], 
                           alpha=0.2, color=self.colors['primary'])
            
            ax1.set_xlabel('Month', fontsize=13, weight='medium', color=self.colors['dark'])
            ax1.set_ylabel('Average Fraud Score', fontsize=13, weight='medium', 
                          color=self.colors['primary'])
            ax1.tick_params(axis='y', labelcolor=self.colors['primary'])
            ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
            
            # Create second y-axis
            ax2 = ax1.twinx()
            line2 = ax2.plot(monthly_stats.index, monthly_stats['risk_level'], 
                            color=self.colors['danger'], marker='s', linewidth=3, 
                            markersize=10, label='High Risk Claims (%)', alpha=0.8)
            
            ax2.set_ylabel('High Risk Claims (%)', fontsize=13, weight='medium', 
                          color=self.colors['danger'])
            ax2.tick_params(axis='y', labelcolor=self.colors['danger'])
            
            # Format x-axis
            ax1.set_xticks(monthly_stats.index)
            ax1.set_xticklabels([str(m) for m in monthly_stats['month']], rotation=45, ha='right')
            
            # Combined legend
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax1.legend(lines, labels, loc='upper left', frameon=True, fancybox=True, shadow=False)
            
            plt.title('Monthly Fraud Trends', fontsize=16, weight='bold', 
                     pad=20, color=self.colors['dark'])
            
            # Remove top spine
            ax1.spines['top'].set_visible(False)
            ax2.spines['top'].set_visible(False)
            
            plt.tight_layout()
            filename = 'monthly_trend.png'
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=150, facecolor='white', edgecolor='none')
            plt.close()
            
            return filename
        except Exception as e:
            print(f"Error creating monthly trend: {e}")
            plt.close()
            return None
    
    def create_correlation_heatmap(self, df):
        """Create modern correlation heatmap"""
        try:
            # Select indicator columns
            indicator_cols = [
                'delayed_reporting', 'near_birthday', 'new_employee_30d', 
                'new_employee_90d', 'multiple_claims', 'soft_tissue_injury',
                'no_witness', 'suspicious_body_part', 'weekend_injury',
                'summer_claim', 'fraud_score'
            ]
            
            available_cols = [col for col in indicator_cols if col in df.columns]
            
            if len(available_cols) < 3:
                return None
            
            # Prepare data
            corr_data = df[available_cols].copy()
            for col in corr_data.columns:
                if corr_data[col].dtype == 'bool':
                    corr_data[col] = corr_data[col].astype(float)
            
            correlation_matrix = corr_data.corr()
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 10))
            fig.patch.set_facecolor('white')
            
            # Create mask for upper triangle
            mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
            
            # Create custom colormap
            cmap = sns.diverging_palette(250, 10, as_cmap=True)
            
            # Create heatmap
            sns.heatmap(correlation_matrix, mask=mask, annot=True, fmt='.2f', 
                       cmap=cmap, center=0, square=True, linewidths=2,
                       cbar_kws={"shrink": .8, "label": "Correlation"},
                       vmin=-1, vmax=1, ax=ax,
                       annot_kws={"fontsize": 10, "weight": "medium"})
            
            # Styling
            ax.set_title('Fraud Indicator Correlation Matrix', fontsize=16, weight='bold', 
                        pad=20, color=self.colors['dark'])
            
            # Rotate labels
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=11)
            plt.setp(ax.get_yticklabels(), rotation=0, fontsize=11)
            
            # Add border
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor(self.colors['gray'])
                spine.set_linewidth(1)
            
            plt.tight_layout()
            filename = 'correlation_heatmap.png'
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=150, facecolor='white', edgecolor='none', bbox_inches='tight')
            plt.close()
            
            return filename
        except Exception as e:
            print(f"Error creating correlation heatmap: {e}")
            plt.close()
            return None