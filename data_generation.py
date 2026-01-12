import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

# Define process activities
activities = [
    'Submit Purchase Request',
    'Manager Review',
    'Finance Check',
    'Approve',
    'Reject',
    'Rework Request',
    'Final Approval',
    'Send to Vendor'
]

# Generate 500 purchase orders
num_cases = 500
data = []

for case_id in range(1, num_cases + 1):
    case = f"PO_{case_id:04d}"
    
    # Random characteristics
    amount = random.choice([500, 1500, 3000, 5500, 8000, 15000, 25000, 50000])
    department = random.choice(['IT', 'Marketing', 'Operations', 'HR', 'Finance'])
    
    # Start time
    start_date = datetime(2024, 1, 1) + timedelta(days=random.randint(0, 180))
    current_time = start_date
    
    # Determine process path based on amount and random factors
    needs_rework = random.random() < 0.15  # 15% need rework
    will_reject = random.random() < 0.08   # 8% get rejected
    
    # Activity 1: Submit Purchase Request
    data.append({
        'Case_ID': case,
        'Activity': 'Submit Purchase Request',
        'Timestamp': current_time,
        'Resource': random.choice(['Employee_A', 'Employee_B', 'Employee_C', 'Employee_D']),
        'Amount': amount,
        'Department': department
    })
    current_time += timedelta(hours=random.randint(1, 4))
    
    # Activity 2: Manager Review
    data.append({
        'Case_ID': case,
        'Activity': 'Manager Review',
        'Timestamp': current_time,
        'Resource': random.choice(['Manager_1', 'Manager_2', 'Manager_3']),
        'Amount': amount,
        'Department': department
    })
    current_time += timedelta(hours=random.randint(2, 24))
    
    # Rework loop (if needed)
    if needs_rework:
        data.append({
            'Case_ID': case,
            'Activity': 'Rework Request',
            'Timestamp': current_time,
            'Resource': random.choice(['Employee_A', 'Employee_B', 'Employee_C', 'Employee_D']),
            'Amount': amount,
            'Department': department
        })
        current_time += timedelta(hours=random.randint(4, 48))
        
        # Re-submit after rework
        data.append({
            'Case_ID': case,
            'Activity': 'Manager Review',
            'Timestamp': current_time,
            'Resource': random.choice(['Manager_1', 'Manager_2', 'Manager_3']),
            'Amount': amount,
            'Department': department
        })
        current_time += timedelta(hours=random.randint(2, 12))
    
    # Activity 3: Finance Check (for amounts > 5000)
    if amount > 5000:
        data.append({
            'Case_ID': case,
            'Activity': 'Finance Check',
            'Timestamp': current_time,
            'Resource': random.choice(['Finance_X', 'Finance_Y', 'Finance_Z']),
            'Amount': amount,
            'Department': department
        })
        current_time += timedelta(hours=random.randint(4, 72))
    
    # Decision: Approve or Reject
    if will_reject:
        data.append({
            'Case_ID': case,
            'Activity': 'Reject',
            'Timestamp': current_time,
            'Resource': random.choice(['Manager_1', 'Manager_2', 'Manager_3']),
            'Amount': amount,
            'Department': department
        })
    else:
        data.append({
            'Case_ID': case,
            'Activity': 'Approve',
            'Timestamp': current_time,
            'Resource': random.choice(['Manager_1', 'Manager_2', 'Manager_3']),
            'Amount': amount,
            'Department': department
        })
        current_time += timedelta(hours=random.randint(1, 8))
        
        # Final Approval (for high amounts)
        if amount > 10000:
            data.append({
                'Case_ID': case,
                'Activity': 'Final Approval',
                'Timestamp': current_time,
                'Resource': random.choice(['Director_Alpha', 'Director_Beta']),
                'Amount': amount,
                'Department': department
            })
            current_time += timedelta(hours=random.randint(2, 48))
        
        # Send to Vendor
        data.append({
            'Case_ID': case,
            'Activity': 'Send to Vendor',
            'Timestamp': current_time,
            'Resource': 'System',
            'Amount': amount,
            'Department': department
        })

# Create DataFrame
df = pd.DataFrame(data)

# Sort by case and timestamp
df = df.sort_values(['Case_ID', 'Timestamp']).reset_index(drop=True)

# Save to CSV
df.to_csv('purchase_order_events.csv', index=False)

print(f"✅ Generated {len(df)} events for {num_cases} purchase orders")
print(f"\nDataset shape: {df.shape}")
print(f"\nFirst few rows:")
print(df.head(10))
print(f"\nActivity distribution:")
print(df['Activity'].value_counts())