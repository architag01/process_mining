import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (18, 12)
plt.rcParams['font.size'] = 9

# Load data
df = pd.read_csv('purchase_order_events.csv')
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# Create figure with subplots
fig = plt.figure(figsize=(18, 12))

# ============================================
# 1. ACTIVITY FREQUENCY BAR CHART
# ============================================
ax1 = plt.subplot(3, 3, 1)
activity_counts = df['Activity'].value_counts()
colors = sns.color_palette("husl", len(activity_counts))
activity_counts.plot(kind='barh', color=colors, ax=ax1)
ax1.set_title('Activity Frequency', fontsize=14, fontweight='bold')
ax1.set_xlabel('Count')
ax1.set_ylabel('Activity')

# ============================================
# 2. CYCLE TIME DISTRIBUTION
# ============================================
ax2 = plt.subplot(3, 3, 2)
cycle_times = df.groupby('Case_ID').agg({
    'Timestamp': ['min', 'max'],
    'Amount': 'first',
    'Department': 'first'
})
cycle_times.columns = ['Start', 'End', 'Amount', 'Department']
cycle_times['Duration_Days'] = (cycle_times['End'] - cycle_times['Start']).dt.total_seconds() / (3600 * 24)

ax2.hist(cycle_times['Duration_Days'], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
ax2.axvline(cycle_times['Duration_Days'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {cycle_times["Duration_Days"].mean():.1f} days')
ax2.axvline(cycle_times['Duration_Days'].median(), color='green', linestyle='--', linewidth=2, label=f'Median: {cycle_times["Duration_Days"].median():.1f} days')
ax2.set_title('Cycle Time Distribution', fontsize=14, fontweight='bold')
ax2.set_xlabel('Duration (Days)')
ax2.set_ylabel('Number of Cases')
ax2.legend()

# ============================================
# 3. CYCLE TIME BY AMOUNT BRACKET
# ============================================
ax3 = plt.subplot(3, 3, 3)
cycle_times['Amount_Bracket'] = pd.cut(cycle_times['Amount'], 
                                        bins=[0, 5000, 10000, 100000],
                                        labels=['<$5K', '$5K-$10K', '>$10K'])
cycle_by_amount = cycle_times.groupby('Amount_Bracket')['Duration_Days'].mean()
cycle_by_amount.plot(kind='bar', color=['#2ecc71', '#f39c12', '#e74c3c'], ax=ax3)
ax3.set_title('Avg Cycle Time by Order Amount', fontsize=14, fontweight='bold')
ax3.set_xlabel('Amount Bracket')
ax3.set_ylabel('Avg Duration (Days)')
ax3.set_xticklabels(ax3.get_xticklabels(), rotation=0)

# ============================================
# 4. DEPARTMENT PERFORMANCE
# ============================================
ax4 = plt.subplot(3, 3, 4)
dept_cycle = cycle_times.groupby('Department')['Duration_Days'].mean().sort_values()
dept_cycle.plot(kind='barh', color=sns.color_palette("viridis", len(dept_cycle)), ax=ax4)
ax4.set_title('Avg Cycle Time by Department', fontsize=14, fontweight='bold')
ax4.set_xlabel('Avg Duration (Days)')
ax4.set_ylabel('Department')

# ============================================
# 5. PROCESS FLOW (Process Variants)
# ============================================
ax5 = plt.subplot(3, 3, 5)
variants = df.groupby('Case_ID')['Activity'].apply(lambda x: ' → '.join(x)).value_counts().head(10)
colors_variants = sns.color_palette("Set3", len(variants))

# Shorten variant labels for better readability
variant_labels = []
for v in variants.index:
    activities = v.split(' → ')
    if len(activities) > 4:
        short_label = ' → '.join(activities[:2]) + '...' + activities[-1]
    else:
        short_label = v
    variant_labels.append(short_label)

y_pos = range(len(variants))
ax5.barh(y_pos, variants.values, color=colors_variants)
ax5.set_yticks(y_pos)
ax5.set_yticklabels(variant_labels, fontsize=8)
ax5.set_title('Top 10 Process Variants', fontsize=14, fontweight='bold')
ax5.set_xlabel('Number of Cases')
ax5.set_ylabel('')

# ============================================
# 6. REWORK & REJECTION RATES
# ============================================
ax6 = plt.subplot(3, 3, 6)
total_cases = df['Case_ID'].nunique()
rework_cases = len(df[df['Activity'] == 'Rework Request']['Case_ID'].unique())
rejected_cases = len(df[df['Activity'] == 'Reject']['Case_ID'].unique())
completed_cases = total_cases - rejected_cases

categories = ['Completed', 'With Rework', 'Rejected']
values = [completed_cases, rework_cases, rejected_cases]
colors_pie = ['#2ecc71', '#f39c12', '#e74c3c']

wedges, texts, autotexts = ax6.pie(values, labels=categories, autopct='%1.1f%%', 
                                     colors=colors_pie, startangle=90,
                                     textprops={'fontsize': 10, 'fontweight': 'bold'})
ax6.set_title('Case Outcomes', fontsize=14, fontweight='bold')

# ============================================
# 7. RESOURCE WORKLOAD
# ============================================
ax7 = plt.subplot(3, 3, 7)
resource_counts = df['Resource'].value_counts().head(10)
resource_counts.plot(kind='bar', color='coral', ax=ax7)
ax7.set_title('Top 10 Resource Workload', fontsize=14, fontweight='bold')
ax7.set_xlabel('Resource')
ax7.set_ylabel('Number of Activities')
ax7.set_xticklabels(ax7.get_xticklabels(), rotation=45, ha='right', fontsize=9)
ax7.tick_params(axis='x', labelsize=9)

# ============================================
# 8. BOTTLENECK ANALYSIS
# ============================================
ax8 = plt.subplot(3, 3, 8)
df_sorted = df.sort_values(['Case_ID', 'Timestamp'])
df_sorted['Next_Timestamp'] = df_sorted.groupby('Case_ID')['Timestamp'].shift(-1)
df_sorted['Next_Activity'] = df_sorted.groupby('Case_ID')['Activity'].shift(-1)
df_sorted['Waiting_Time_Hours'] = (df_sorted['Next_Timestamp'] - df_sorted['Timestamp']).dt.total_seconds() / 3600
df_sorted = df_sorted[df_sorted['Next_Activity'].notna()]

bottlenecks = df_sorted.groupby('Activity')['Waiting_Time_Hours'].mean().sort_values(ascending=False).head(8)

# Create horizontal bar chart with better spacing
y_pos = range(len(bottlenecks))
ax8.barh(y_pos, bottlenecks.values, color='crimson')
ax8.set_yticks(y_pos)
ax8.set_yticklabels(bottlenecks.index, fontsize=9)
ax8.set_title('Avg Waiting Time After Activity (Top 8)', fontsize=14, fontweight='bold')
ax8.set_xlabel('Waiting Time (Hours)')
ax8.set_ylabel('')
ax8.invert_yaxis()

# ============================================
# 9. TIMELINE - Orders Over Time
# ============================================
ax9 = plt.subplot(3, 3, 9)
df['Date'] = df['Timestamp'].dt.date
daily_orders = df[df['Activity'] == 'Submit Purchase Request'].groupby('Date').size()
ax9.plot(daily_orders.index, daily_orders.values, color='steelblue', linewidth=2, marker='o', markersize=3)
ax9.set_title('Purchase Orders Over Time', fontsize=14, fontweight='bold')
ax9.set_xlabel('Date')
ax9.set_ylabel('Number of Orders')
# Improve x-axis label spacing
ax9.tick_params(axis='x', rotation=45, labelsize=8)
# Show every 30th label to avoid overlap
n = len(daily_orders) // 6  # Show ~6 labels
ax9.set_xticks(ax9.get_xticks()[::max(1, n)])
ax9.grid(True, alpha=0.3)

plt.tight_layout(pad=2.0)
plt.savefig('process_mining_dashboard.png', dpi=300, bbox_inches='tight')
print("✅ Dashboard saved as 'process_mining_dashboard.png'")
plt.show()

# ============================================
# EXPORT SUMMARY METRICS
# ============================================
summary_metrics = {
    'Total_Cases': df['Case_ID'].nunique(),
    'Total_Events': len(df),
    'Avg_Cycle_Time_Days': cycle_times['Duration_Days'].mean(),
    'Median_Cycle_Time_Days': cycle_times['Duration_Days'].median(),
    'Rework_Rate_%': (rework_cases / total_cases * 100),
    'Rejection_Rate_%': (rejected_cases / total_cases * 100),
    'Avg_Activities_Per_Case': len(df) / df['Case_ID'].nunique(),
    'Top_Bottleneck': bottlenecks.index[0],
    'Top_Bottleneck_Wait_Hours': bottlenecks.values[0]
}

summary_df = pd.DataFrame([summary_metrics])
summary_df.to_csv('process_summary_metrics.csv', index=False)
print("\n✅ Summary metrics saved as 'process_summary_metrics.csv'")
print("\n📊 Key Metrics:")
for key, value in summary_metrics.items():
    if isinstance(value, float):
        print(f"   {key}: {value:.2f}")
    else:
        print(f"   {key}: {value}")