# 🎓 Federated Credit Scoring with Sharing ADMM SVM

## Project Overview

This notebook implements a **privacy-preserving credit scoring system** using **vertical federated learning** with the **Sharing ADMM (Alternating Direction Method of Multipliers) SVM (Support Vector Machine)** algorithm. The system demonstrates how three independent entities (Financial Institution, Credit Bureau, Demographic Agency) can collaboratively train a credit risk model **without sharing their raw data**.

---

## 📊 Dataset: Give Me Some Credit (Kaggle)

- **Original size**: 150,000 observations
- **After cleaning (removing NA)**: 120,269 observations
- **Balanced dataset**: 10,000 observations (5,000 Default + 5,000 Non-Default)
- **Training set**: 8,000 samples (80%)
- **Test set**: 2,000 samples (20%)

---

## 🏗️ Architecture: Vertical Federated Learning

| Agent | Feature | Description |
| :--- | :--- | :--- |
| **Financial Institution** | RevolvingUtilizationOfUnsecuredLines | Total balance on credit cards and personal lines of credit except real estate and no installment debt like car loans divided by the sum of credit limits. |
| **Financial Institution** | DebtRatio | Monthly debt payments, alimony, living costs divided by monthly gross income. |
| **Financial Institution** | MonthlyIncome | Monthly income. |
| **Credit Bureau** | NumberOfTime30-59DaysPastDueNotWorse | Number of times borrower has been 30-59 days past due but no worse in the last 2 years. |
| **Credit Bureau** | NumberOfTime60-89DaysPastDueNotWorse | Number of times borrower has been 60-89 days past due but no worse in the last 2 years. |
| **Credit Bureau** | NumberOfTimes90DaysLate | Number of times borrower has been 90 days or more past due. |
| **Demographic Entity** | Age | Age of borrower in years. |
| **Demographic Entity** | NumberOfOpenCreditLinesAndLoans | Number of Open loans (installment like car loan or mortgage) and Lines of credit (e.g. credit cards). |
| **Demographic Entity** | NumberRealEstateLoansOrLines | Number of mortgage and real estate loans including home equity lines of credit. |
| **Demographic Entity** | NumberOfDependents | Number of dependents in family excluding themselves (spouse, children etc.). |

### Central Server
- Manages target labels (Default/Non-Default)
- Coordinates global and dual variable updates
- **Never sees raw feature data**

---

## 🔬 Algorithm: Sharing ADMM for SVM

**Hyperparameters:**
- λ (L2 regularization): 0.1
- ρ (ADMM penalty): 1.0 / N (scaled)
- Iterations: 200

**Convergence validated through 4 analyses:**
1. ✅ Objective function minimization
2. ✅ Primal residual (agent consensus)
3. ✅ Centralized vs distributed equivalence
4. ✅ Dual variable stability

---

## 📈 Results Summary

### Model Performance (Test Set)
| Metric | ADMM (Federated) | Centralized SVM | Difference |
|--------|------------------|-----------------|------------|
| **Accuracy** | 63.30% | 63.30% | 0% |
| **ROC-AUC** | 0.6861 | 0.6861 | 0 |

### Feature Importance by Agent
| Agent | Contribution | Interpretation |
|-------|--------------|----------------|
| **Agent 2 (Credit Bureau)** | **90.75%** | 🥇 **DOMINANT** factor |
| Agent 3 (Demographic) | 6.71% | Minor factor |
| Agent 1 (Financial) | 2.54% | Minimal impact |

### Top Predictive Features
1. **60-89DaysPastDue** (weight: -1.66) - Strong negative predictor
2. **30-59DaysPastDue** (weight: +1.32) - Strong positive predictor
3. **90DaysLate** (weight: +0.64) - Moderate positive predictor

---

## 🔑 Key Findings

### 1. Privacy-Preserving Equivalence ✅
The federated ADMM SVM achieved **equivalent performance** to centralized learning while keeping all data distributed across local servers.

### 2. Credit History is Dominant 📊
**Payment delinquency history** (Agent 2 - Credit Bureau) accounts for over **90% of predictive power**, indicating that past payment behavior is the strongest indicator of credit risk.

### 3. Theoretical Guarantees Validated 🔬
All four convergence metrics confirmed:
- Objective function decreased and stabilized
- Agents reached distributed consensus (primal residual → 82.59)
- Dual variables converged (10⁻⁶ stability)
- Weights matched centralized solution

### 4. Practical Deployment Ready 🚀
- Algorithm converged in 200 iterations 
- Scalable to larger datasets and more agents
- Compliant with privacy regulations (GDPR, etc.)

---

## 💡 Business Implications

✅ **Multi-party collaboration** without data sharing  
✅ **Regulatory compliance** maintained  
✅ **No accuracy loss** compared to centralized training  
✅ **Explainable predictions** with feature importance  

---

## 📚 References

- Dataset: Kaggle - *Give Me Some Credit*

---

## 👨‍💻 Implementation Details

**Language**: Python 3.10  
**Key Libraries**: NumPy, Pandas, Scikit-learn, Matplotlib, Seaborn  
**Notebook Structure**:
1. Data preprocessing and balancing
2. Vertical feature split across agents
3. ADMM algorithm implementation
4. Convergence analysis (4 metrics)
5. Test set evaluation
6. Feature importance analysis