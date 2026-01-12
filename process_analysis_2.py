import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('purchase_order_events.csv')
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

print("="*60)
print("PURCHASE ORDER PROCESS ANALYSIS")
print("="*60)

# ============================================
# 1. BASIC STATISTICS
# ============================================
print("\n📊 BASIC STATISTICS")
print(f"Total Events: {len(df)}")
print(f"Total Cases: {df['Case_ID'].nunique()}")
print(f"Date Range: {df['Timestamp'].min()} to {df['Timestamp'].max()}")
print(f"Departments: {df['Department'].unique()}")

# ============================================
# 2. ACTIVITY FREQUENCY
# ============================================
print("\n📈 ACTIVITY FREQUENCY")
activity_counts = df['Activity'].value_counts()
print(activity_counts)

# ============================================
# 3. PROCESS VARIANTS (Most Common Paths)
# ============================================
print("\n🔄 PROCESS VARIANTS")
variants = df.groupby('Case_ID')['Activity'].apply(lambda x: ' → '.join(x)).value_counts()
print(f"\nTotal Unique Variants: {len(variants)}")
print("\nTop 5 Most Common Process Paths:")
for i, (path, count) in enumerate(variants.head(5).items(), 1):
    print(f"\n{i}. Count: {count} cases")
    print(f"   Path: {path}")

# ============================================
# 4. CYCLE TIME ANALYSIS
# ============================================
print("\n⏱️  CYCLE TIME ANALYSIS")
cycle_times = df.groupby('Case_ID').agg({
    'Timestamp': ['min', 'max'],
    'Amount': 'first',
    'Department': 'first'
})
cycle_times.columns = ['Start', 'End', 'Amount', 'Department']
cycle_times['Duration_Hours'] = (cycle_times['End'] - cycle_times['Start']).dt.total_seconds() / 3600
cycle_times['Duration_Days'] = cycle_times['Duration_Hours'] / 24

print(f"\nAverage Cycle Time: {cycle_times['Duration_Days'].mean():.2f} days")
print(f"Median Cycle Time: {cycle_times['Duration_Days'].median():.2f} days")
print(f"Min Cycle Time: {cycle_times['Duration_Days'].min():.2f} days")
print(f"Max Cycle Time: {cycle_times['Duration_Days'].max():.2f} days")

# Cycle time by amount bracket
cycle_times['Amount_Bracket'] = pd.cut(cycle_times['Amount'], 
                                        bins=[0, 5000, 10000, 100000],
                                        labels=['<$5K', '$5K-$10K', '>$10K'])
print("\n📊 Average Cycle Time by Amount:")
print(cycle_times.groupby('Amount_Bracket')['Duration_Days'].mean().round(2))

# ============================================
# 5. IDENTIFY REWORK CASES
# ============================================
print("\n🔁 REWORK ANALYSIS")
rework_cases = df[df['Activity'] == 'Rework Request']['Case_ID'].unique()
print(f"Cases with Rework: {len(rework_cases)} ({len(rework_cases)/df['Case_ID'].nunique()*100:.1f}%)")

# ============================================
# 6. REJECTION ANALYSIS
# ============================================
print("\n❌ REJECTION ANALYSIS")
rejected_cases = df[df['Activity'] == 'Reject']['Case_ID'].unique()
print(f"Rejected Cases: {len(rejected_cases)} ({len(rejected_cases)/df['Case_ID'].nunique()*100:.1f}%)")

rejected_data = cycle_times[cycle_times.index.isin(rejected_cases)]
print("\nRejections by Department:")
print(rejected_data['Department'].value_counts())

# ============================================
# 7. BOTTLENECK DETECTION
# ============================================
print("\n🚧 BOTTLENECK ANALYSIS")
# Calculate time between consecutive activities
df_sorted = df.sort_values(['Case_ID', 'Timestamp'])
df_sorted['Next_Timestamp'] = df_sorted.groupby('Case_ID')['Timestamp'].shift(-1)
df_sorted['Next_Activity'] = df_sorted.groupby('Case_ID')['Activity'].shift(-1)
df_sorted['Waiting_Time_Hours'] = (df_sorted['Next_Timestamp'] - df_sorted['Timestamp']).dt.total_seconds() / 3600

# Remove last activity of each case
df_sorted = df_sorted[df_sorted['Next_Activity'].notna()]

# Average waiting time after each activity
bottlenecks = df_sorted.groupby('Activity')['Waiting_Time_Hours'].agg(['mean', 'median', 'count'])
bottlenecks = bottlenecks.sort_values('mean', ascending=False)
print("\nAverage Waiting Time After Each Activity:")
print(bottlenecks.round(2))

# ============================================
# 8. RESOURCE UTILIZATION
# ============================================
print("\n👥 RESOURCE UTILIZATION")
resource_counts = df['Resource'].value_counts()
print("\nTop 10 Most Active Resources:")
print(resource_counts.head(10))

# ============================================
# 9. KEY INSIGHTS SUMMARY
# ============================================
print("\n" + "="*60)
print("🎯 KEY INSIGHTS & RECOMMENDATIONS")
print("="*60)

# Finding 1: Longest bottleneck
longest_bottleneck = bottlenecks.index[0]
longest_wait = bottlenecks.iloc[0]['mean']
print(f"\n1. MAIN BOTTLENECK: '{longest_bottleneck}'")
print(f"   Average waiting time: {longest_wait:.1f} hours")
print(f"   💡 Recommendation: Add automation or resources to this step")

# Finding 2: Rework impact
avg_cycle_with_rework = cycle_times[cycle_times.index.isin(rework_cases)]['Duration_Days'].mean()
avg_cycle_without_rework = cycle_times[~cycle_times.index.isin(rework_cases)]['Duration_Days'].mean()
print(f"\n2. REWORK IMPACT:")
print(f"   Cases with rework: {avg_cycle_with_rework:.2f} days avg")
print(f"   Cases without rework: {avg_cycle_without_rework:.2f} days avg")
print(f"   💡 Recommendation: Improve initial request quality and validation")

# Finding 3: High-value processing
high_value_cycle = cycle_times[cycle_times['Amount'] > 10000]['Duration_Days'].mean()
print(f"\n3. HIGH-VALUE ORDERS (>$10K):")
print(f"   Average cycle time: {high_value_cycle:.2f} days")
print(f"   💡 Recommendation: Streamline approval workflow for urgent high-value orders")

# Finding 4: Department performance
dept_performance = cycle_times.groupby('Department')['Duration_Days'].mean().sort_values(ascending=False)
print(f"\n4. DEPARTMENT PERFORMANCE:")
for dept, days in dept_performance.items():
    print(f"   {dept}: {days:.2f} days avg")
slowest_dept = dept_performance.index[0]
print(f"   💡 Recommendation: Investigate delays in {slowest_dept} department")

print("\n" + "="*60)
print("✅ Analysis Complete!")
print("="*60)