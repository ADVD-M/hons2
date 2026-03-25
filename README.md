# Robustness Analysis of Honest Random Forests
### Evaluating Stability Under Covariate Shift in Tabular Data

## Project Overview

This research-focused platform provides an interactive environment to evaluate the performance of Honest Random Forests against standard ensemble methods. The core focus of this study is the "Honesty" property — an architectural constraint where the data used to determine tree structure is disjoint from the data used to estimate leaf-node probabilities.

While standard Random Forests are high-performing on static data, they often rely on spurious correlations that dissolve when the data distribution evolves. This project demonstrates that the Honest Random Forest implementation offers superior Accuracy Retention and structural stability when subjected to extreme Covariate Shift.

## Interactive Robustness Lab

You can test the models in real-time using the interactive dashboard. Upload any classification dataset, select a target feature, and observe how the models degrade as the distribution is artificially shifted. (Graphs and results may vary largely based on dataset cleanliness and features.)

🔗 **[Try the app here](https://honestrandomforests.streamlit.app/)**

## Technical Stack


Python | Machine Learning | Scikit-learn, NumPy | Pandas | Streamlit | Plotly |

## Methodology

The implementation utilizes a split-sample regime to eliminate in-sample prediction bias:

1. **Structure Phase** — The first half of the training data determines the optimal splits and tree architecture.
2. **Estimation Phase** — The second half of the data is passed through the existing structure to provide unbiased class probability estimates for each leaf.
3. **Stress Testing** — The system automatically identifies high-importance features and applies a mean shift (measured in standard deviations) to evaluate which model better maintains its predictive integrity.

## Evaluated Benchmarks

The framework has been validated across several high-accuracy classification domains to prove that "Honesty" acts as a universal regularizer:

- **Astrophysics:** Star Classification
- **Cybersecurity:** Phishing Website Detection

## Key Findings

Experimental results consistently show a "Crossover Point" where the Standard Random Forest's accuracy collapses due to its reliance on training-specific noise. In contrast, the Honest Random Forest maintains a significantly higher percentage of its baseline performance, proving its reliability for deployment in non-stationary environments.
