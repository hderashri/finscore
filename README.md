# FINSCORE: BNPL-Aware Credit Scoring System

A machine learning-powered credit scoring application that evaluates consumer creditworthiness with special focus on Buy Now Pay Later (BNPL) usage patterns. Built with Streamlit for easy deployment and LightGBM for high-performance prediction.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/your-username/finscore/main/app.py)

## 🚀 Features

- **Behavioral Credit Scoring**: Analyzes spending patterns, BNPL usage, credit utilization, and liquidity indicators
- **Interactive Web App**: User-friendly Streamlit interface for real-time credit score calculation
- **Comprehensive Risk Assessment**: Evaluates multiple risk factors including BNPL dependency, cash advance behavior, and external borrowing
- **Synthetic Dataset**: Includes 15,000 synthetic customer profiles with 3 years of transaction history
- **Model Interpretability**: Provides clear explanations of risk factors and behavioral insights
- **FINSCORE Calculation**: Ranges from 300-900, similar to traditional credit scores

## 🛠 Technology Stack

- **Frontend**: Streamlit
- **ML Framework**: LightGBM with scikit-learn
- **Data Processing**: Pandas, NumPy
- **Visualization**: Streamlit components
- **Deployment**: Ready for Streamlit Cloud, Heroku, or local deployment

## 📊 Data Source

The project uses **synthetic financial data** generated to simulate real-world credit behaviors. This synthetic dataset includes:

- 5,000 customers with various financial profiles (stable, BNPL-heavy, credit card revolvers, etc.)
- 3 years of transaction history including income, spending, BNPL usage, credit card payments, cash advances, and external support
- Behavioral features like BNPL ratios, credit utilization, spending volatility, and liquidity stress indicators

The synthetic data is inspired by the **UCI Credit Card Default Dataset**, which contains real Taiwanese credit card client data from 2005. The UCI dataset was used as a reference for feature engineering and model validation, but the main training data is synthetic to avoid privacy issues and allow for controlled experimentation with BNPL behaviors.

## 🧠 Model Details

### Features Engineered (25+ features)
- **Demographic**: Age, monthly income, employment type, city tier
- **Credit Profile**: Number of credit cards, total credit limit, utilization ratio
- **Spending Behavior**: Total spend, average spend, spend volatility, max spend
- **BNPL Usage**: BNPL total amount, transaction count, usage ratio, income ratio
- **Cash Advances**: Total cash withdrawal, transaction count, advance ratio
- **External Support**: Money borrowed from family/friends, transaction count, support ratio
- **Liquidity**: Minimum account balance
- **Behavioral Segments**: Cash dependent, external support dependent

### Model Performance
- **Algorithm**: LightGBM (Gradient Boosting)
- **Default Detection Rate**: 94.3%
- **AUC-ROC**: High accuracy on synthetic validation sets
- **Calibration**: Platt scaling for probability calibration

### Risk Categories
- **800-900**: Excellent credit health
- **740-799**: Very good
- **670-739**: Good
- **580-669**: Risky
- **Below 580**: High default risk

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/hderashri/finscore.git
   cd finscore
   ```

2. **Create virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

   **Note**: If `streamlit` command is not found even after activating the venv, use:
   ```bash
   python -m streamlit run app.py
   ```

## 📖 Usage

1. **Enter Financial Details**: Fill in your monthly income, spending patterns, credit card usage, and BNPL activity
2. **Add Current Loan**: Include your current active loan balance for comprehensive assessment
3. **Calculate FINSCORE**: Click the button to get your credit score and risk analysis
4. **Review Results**: See your FINSCORE, risk category, default probability, and behavioral insights

### Input Parameters
- **Demographics**: Age, monthly income
- **Spending**: Total monthly spending, BNPL usage
- **Credit**: Number of cards, credit limits, cash advances
- **Support**: External borrowing from family/friends
- **Liquidity**: Current account balance
- **Loans**: Current active loan balance

## 📈 Sample Scenarios

The app includes test scenarios demonstrating different risk profiles:

1. **Low Risk**: Stable income, moderate spending, no BNPL dependency
2. **Medium Risk**: High credit utilization, occasional cash advances
3. **High Risk**: BNPL dependent, negative balance, frequent external borrowing

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- UCI Machine Learning Repository for the Credit Card Default Dataset
- Streamlit for the amazing web app framework
- LightGBM for efficient gradient boosting implementation

## 📞 Contact

Harshit Derashree - [GitHub](https://github.com/hderashri)

Project Link: [https://github.com/hderashri/finscore](https://github.com/hderashri/finscore)

---

**Note**: This is an educational project demonstrating credit scoring concepts. Not intended for actual financial decision-making.

## 📈 Final Submission Updates

### Changes Since Mid-Sem Submission
- **Fixed Streamlit Execution**: Resolved command not found error by using `python -m streamlit run app.py` as an alternative to direct `streamlit run app.py` command.
- **Verified Model Loading**: Confirmed the LightGBM model (model_lgb.pkl) loads correctly with calibrated classifiers for accurate probability predictions.
- **App Functionality**: The web app runs successfully on `http://localhost:8501`, with all input forms, feature engineering, predictions, and explanations working as intended.
- **Dependencies**: All packages in requirements.txt are installed and functional; Watchdog module is recommended for improved performance during development.
- **Data Handling**: Synthetic data is generated on-the-fly in the app; no external data files are required for deployment.
- **Deployment Readiness**: App is ready for Streamlit Cloud or Heroku deployment; consider adding environment variables for production (e.g., model path).

### Known Limitations and Future Improvements
- Model interpretability could be enhanced with SHAP/LIME explanations for feature importance.
- Add input validation to prevent unrealistic financial inputs.
- Implement user data persistence or export functionality for results.
- Deploy to Streamlit Cloud for public access (update badge URL accordingly).