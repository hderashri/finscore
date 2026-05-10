import numpy as np
import pandas as pd
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, classification_report
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic credit data
def generate_synthetic_data(n_samples=15000):
    """
    Generate synthetic credit data with realistic patterns
    """
    print(f"Generating {n_samples} synthetic credit profiles...")
    
    data = []
    
    for i in range(n_samples):
        # Demographics
        age = np.random.randint(21, 65)
        employment_type = np.random.choice(['Salaried', 'Self Employed'], p=[0.7, 0.3])
        city_tier = np.random.choice(['Tier 1', 'Tier 2', 'Tier 3'], p=[0.4, 0.35, 0.25])
        
        # Income based on age and employment
        base_income = np.random.normal(50000, 20000)
        if employment_type == 'Self Employed':
            base_income *= 1.2
        if age > 40:
            base_income *= 1.3
        monthly_income = max(10000, base_income + np.random.normal(0, 10000))
        
        # Credit cards
        credit_cards = np.random.randint(1, 6)
        credit_limit_total = credit_cards * np.random.uniform(50000, 200000)
        
        # Spending patterns
        total_spend = monthly_income * np.random.uniform(0.5, 1.5)
        avg_spend = total_spend / np.random.randint(10, 30)
        spend_volatility = avg_spend * np.random.uniform(0.3, 1.5)
        max_spend = max(avg_spend * 1.5, total_spend * np.random.uniform(0.2, 0.5))
        
        # BNPL usage
        bnpl_user = np.random.choice([0, 1], p=[0.6, 0.4])
        if bnpl_user:
            bnpl_total = total_spend * np.random.uniform(0.1, 0.6)
            bnpl_transactions = np.random.randint(2, 15)
        else:
            bnpl_total = 0
            bnpl_transactions = 0
        
        # Cash advances
        cash_advance_user = np.random.choice([0, 1], p=[0.8, 0.2])
        if cash_advance_user:
            credit_cash_total = monthly_income * np.random.uniform(0.1, 0.5)
            credit_cash_transactions = np.random.randint(1, 10)
        else:
            credit_cash_total = 0
            credit_cash_transactions = 0
        
        # External support
        external_support_user = np.random.choice([0, 1], p=[0.85, 0.15])
        if external_support_user:
            external_support_total = monthly_income * np.random.uniform(0.2, 1.0)
            external_support_transactions = np.random.randint(1, 8)
        else:
            external_support_total = 0
            external_support_transactions = 0
        
        # Account balance
        min_balance = np.random.normal(monthly_income * 0.2, monthly_income * 0.3)
        if cash_advance_user or external_support_user:
            min_balance -= np.random.uniform(0, monthly_income * 0.5)
        
        # Current loan
        loan_user = np.random.choice([0, 1], p=[0.7, 0.3])
        if loan_user:
            current_active_loan = monthly_income * np.random.uniform(0.5, 5.0)
        else:
            current_active_loan = 0
        
        # Feature engineering
        bnpl_usage_ratio = bnpl_total / total_spend if total_spend > 0 else 0
        cash_advance_ratio = credit_cash_total / total_spend if total_spend > 0 else 0
        external_support_ratio = external_support_total / monthly_income if monthly_income > 0 else 0
        loan_income_ratio = current_active_loan / monthly_income if monthly_income > 0 else 0
        bnpl_income_ratio = bnpl_total / monthly_income if monthly_income > 0 else 0
        credit_utilisation = max_spend / credit_limit_total if credit_limit_total > 0 else 0
        
        spend_income_ratio = total_spend / monthly_income if monthly_income > 0 else 0
        overspend_ratio = max(0, (total_spend - monthly_income) / monthly_income) if monthly_income > 0 else 0
        debt_to_income = current_active_loan / monthly_income if monthly_income > 0 else 0
        buffer_ratio = min_balance / monthly_income if monthly_income > 0 else 0
        revolving_utilisation = total_spend / credit_limit_total if credit_limit_total > 0 else 0
        payment_ratio = max(0, 1 - loan_income_ratio) if monthly_income > 0 else 0
        payment_delay = int(min_balance < 0)
        bnpl_installment_ratio = bnpl_total / monthly_income if monthly_income > 0 else 0
        credit_history_years = (age - 21) / 10
        hard_inquiry_count = int(credit_cards > 3)
        
        # Behavioral segments
        segment_cash_dependent = int(cash_advance_ratio > 0.25)
        segment_external_support = int(external_support_ratio > 1)
        
        # Categorical encoding
        employment_type_self_employed = 1 if employment_type == "Self Employed" else 0
        city_tier_tier2 = 1 if city_tier == "Tier 2" else 0
        city_tier_tier3 = 1 if city_tier == "Tier 3" else 0
        
        # Calculate default probability based on risk factors with realistic weights
        # Based on behavioral finance and credit scoring best practices
        risk_score = 0
        
        # BNPL dependency - high weight as it's a new behavioral risk factor
        risk_score += bnpl_income_ratio * 0.35  # BNPL spending relative to income
        risk_score += bnpl_usage_ratio * 0.25   # BNPL as % of total spend
        risk_score += bnpl_installment_ratio * 0.30  # BNPL installment burden
        
        # Credit utilization - traditional major risk factor
        risk_score += credit_utilisation * 0.20
        risk_score += revolving_utilisation * 0.25
        
        # Payment behavior - critical for credit scoring
        risk_score += payment_delay * 0.40      # Binary: negative balance
        risk_score += payment_ratio * 0.15       # Payment capacity
        
        # Debt burden
        risk_score += loan_income_ratio * 0.25   # Current loan burden
        risk_score += debt_to_income * 0.30      # Total debt including BNPL
        
        # Spending patterns
        risk_score += overspend_ratio * 0.35     # Spending > income
        risk_score += spend_income_ratio * 0.15   # Overall spending relative to income
        
        # Cash advances - indicates liquidity stress
        risk_score += cash_advance_ratio * 0.30
        risk_score += segment_cash_dependent * 0.25
        
        # External borrowing - financial dependence
        risk_score += external_support_ratio * 0.20
        risk_score += segment_external_support * 0.25
        
        # Liquidity buffer
        risk_score -= buffer_ratio * 0.15        # Negative if low buffer
        
        # Credit behavior
        risk_score += hard_inquiry_count * 0.10  # Multiple credit applications
        
        # Age factor - younger profiles may have less history
        if age < 30:
            risk_score += 0.10
        elif age > 50:
            risk_score -= 0.05
        
        # Convert to probability with some randomness
        prob_default = 1 / (1 + np.exp(-(risk_score - 0.5) * 3))
        prob_default = np.clip(prob_default + np.random.normal(0, 0.1), 0, 1)
        
        # Determine default label
        default_label = 1 if np.random.random() < prob_default else 0
        
        row = {
            'age': age,
            'monthly_income': monthly_income,
            'credit_cards': credit_cards,
            'credit_limit_total': credit_limit_total,
            'total_spend': total_spend,
            'avg_spend': avg_spend,
            'spend_volatility': spend_volatility,
            'max_spend': max_spend,
            'bnpl_total': bnpl_total,
            'bnpl_transactions': bnpl_transactions,
            'min_balance': min_balance,
            'bnpl_usage_ratio': bnpl_usage_ratio,
            'credit_cash_total': credit_cash_total,
            'credit_cash_transactions': credit_cash_transactions,
            'external_support_total': external_support_total,
            'external_support_transactions': external_support_transactions,
            'cash_advance_ratio': cash_advance_ratio,
            'external_support_ratio': external_support_ratio,
            'employment_type_self_employed': employment_type_self_employed,
            'city_tier_tier2': city_tier_tier2,
            'city_tier_tier3': city_tier_tier3,
            'segment_cash_dependent': segment_cash_dependent,
            'segment_external_support': segment_external_support,
            'current_active_loan': current_active_loan,
            'loan_income_ratio': loan_income_ratio,
            'spend_income_ratio': spend_income_ratio,
            'overspend_ratio': overspend_ratio,
            'payment_ratio': payment_ratio,
            'payment_delay': payment_delay,
            'bnpl_income_ratio': bnpl_income_ratio,
            'credit_utilisation': credit_utilisation,
            'debt_to_income': debt_to_income,
            'buffer_ratio': buffer_ratio,
            'revolving_utilisation': revolving_utilisation,
            'credit_history_years': credit_history_years,
            'hard_inquiry_count': hard_inquiry_count,
            'bnpl_installment_ratio': bnpl_installment_ratio,
            'default': default_label
        }
        
        data.append(row)
    
    df = pd.DataFrame(data)
    print(f"Generated {len(df)} samples")
    print(f"Default rate: {df['default'].mean():.2%}")
    return df

# Train LightGBM model
def train_model(df):
    """
    Train LightGBM model with calibrated probabilities
    """
    print("\nTraining LightGBM model...")
    
    # Features to use
    feature_cols = [
        'age', 'monthly_income', 'credit_cards', 'credit_limit_total',
        'total_spend', 'avg_spend', 'spend_volatility', 'max_spend',
        'bnpl_total', 'bnpl_transactions', 'min_balance', 'bnpl_usage_ratio',
        'credit_cash_total', 'credit_cash_transactions',
        'external_support_total', 'external_support_transactions',
        'cash_advance_ratio', 'external_support_ratio',
        'employment_type_self_employed', 'city_tier_tier2', 'city_tier_tier3',
        'segment_cash_dependent', 'current_active_loan', 'loan_income_ratio',
        'spend_income_ratio', 'overspend_ratio', 'payment_ratio', 'payment_delay',
        'bnpl_income_ratio', 'credit_utilisation', 'debt_to_income', 'buffer_ratio',
        'revolving_utilisation', 'credit_history_years', 'hard_inquiry_count',
        'bnpl_installment_ratio'
    ]
    
    X = df[feature_cols]
    y = df['default']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Train LightGBM using sklearn API
    model = LGBMClassifier(
        objective='binary',
        metric='auc',
        num_leaves=31,
        learning_rate=0.05,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5,
        verbose=-1,
        random_state=42,
        n_estimators=500
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        eval_names=['train', 'test'],
        callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(100)]
    )
    
    # Predict on test set
    y_pred = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred)
    print(f"\nTest AUC: {auc:.4f}")
    
    # Calibrate the model
    print("\nCalibrating model...")
    calibrated_model = CalibratedClassifierCV(model, method='sigmoid', cv='prefit')
    calibrated_model.fit(X_train, y_train)
    
    # Evaluate calibrated model
    y_pred_cal = calibrated_model.predict_proba(X_test)[:, 1]
    auc_cal = roc_auc_score(y_test, y_pred_cal)
    print(f"Calibrated Test AUC: {auc_cal:.4f}")
    
    return calibrated_model, feature_cols

# Main execution
if __name__ == "__main__":
    # Generate data
    df = generate_synthetic_data(n_samples=15000)
    
    # Save raw data for reference
    df.to_csv('synthetic_credit_data.csv', index=False)
    print("\nSaved synthetic data to synthetic_credit_data.csv")
    
    # Train model
    model, feature_cols = train_model(df)
    
    # Save model
    joblib.dump(model, 'models/model_lgb.pkl')
    print("\nSaved model to models/model_lgb.pkl")
    
    # Save feature names
    with open('models/feature_names.txt', 'w') as f:
        f.write('\n'.join(feature_cols))
    print("Saved feature names to models/feature_names.txt")
    
    print("\nModel training complete!")
