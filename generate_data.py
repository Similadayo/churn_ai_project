import pandas as pd
import numpy as np
import os

# Set random seed for reproducibility
np.random.seed(42)

# Number of samples
num_samples = 1000

# Generate customer_id as CUST0001, CUST0002, ...
customer_id = [f"CUST{i:04d}" for i in range(1, num_samples + 1)]

# Membership level - assign with weights for more Bas/Sil vs Gold
membership_levels = np.random.choice(['Basic', 'Silver', 'Gold'], size=num_samples, p=[0.5, 0.35, 0.15])

# Region selection
regions = np.random.choice(['North', 'South', 'East', 'West'], size=num_samples)

# Days since signup (1 to 1825 days ~5 years)
days_since_signup = np.random.randint(1, 1826, size=num_samples)

# Number of visits depends slightly on membership
num_visits = np.random.poisson(lam=9, size=num_samples) + (membership_levels == 'Gold') * 3 + (membership_levels == 'Silver') * 1
num_visits = np.clip(num_visits, 1, 50)  # 1 to 50 visits

# Average order value ($10-$500, with Gold tending higher)
base_aov = np.random.normal(80, 25, num_samples) + (membership_levels == 'Gold') * 60 + (membership_levels == 'Silver') * 20
avg_order_value = np.clip(base_aov, 10, 500)

# Total spent correlates with num_visits and AOV
# Some random noise is added
total_spent = avg_order_value * num_visits + np.random.normal(0, 80, num_samples)
total_spent = np.clip(total_spent, 20, None)

# Recency in days (days since last visit). High = less recent / more likely churn
recency_days = np.random.randint(1, 370, size=num_samples) + (membership_levels == 'Basic') * 20
recency_days = np.clip(recency_days, 1, 730)  # up to 2 years

# Number of support tickets (most are 0-2)
support_tickets = np.random.poisson(lam=0.8, size=num_samples)
support_tickets = np.clip(support_tickets, 0, 8)

# Complaint flag (10% chance overall, reduced for Gold)
complaint_prob = 0.1 - (membership_levels == 'Gold') * 0.06
complaint_flag = np.random.binomial(1, complaint_prob)

# Churn calculation: base risk, up, or down for features
churn_risk = (
    0.18 +
    (total_spent < 200) * 0.20 +
    (num_visits < 6) * 0.20 +
    (recency_days > 90) * 0.20 +
    (complaint_flag > 0) * 0.15 +
    (membership_levels == 'Gold') * -0.18 +
    (membership_levels == 'Silver') * -0.08 +
    (support_tickets > 2) * 0.08
)
churn_risk = np.clip(churn_risk, 0.02, 0.85)

# Bernoulli draw for churned (1 = churn)
churned = np.random.binomial(1, churn_risk)

# Assemble data frame
df = pd.DataFrame({
    'customer_id': customer_id,
    'total_spent': np.round(total_spent, 2),
    'avg_order_value': np.round(avg_order_value, 2),
    'num_visits': num_visits,
    'recency_days': recency_days,
    'complaint_flag': complaint_flag,
    'membership_level': membership_levels,
    'region': regions,
    'support_tickets': support_tickets,
    'days_since_signup': days_since_signup,
    'churned': churned
})

# Ensure output directory exists
os.makedirs('data', exist_ok=True)

# Save to CSV
output_path = os.path.join('data', 'customers.csv')
df.to_csv(output_path, index=False)

print(f"Synthetic customer data saved to {output_path}")
