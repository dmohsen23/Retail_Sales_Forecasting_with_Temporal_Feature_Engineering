Overview
This project tackles next-day sales forecasting for dairy products across 54 Corporación Favorita supermarket stores in Ecuador. Rather than relying on raw date features, the pipeline emphasises causal, domain-informed feature engineering — encoding real-world drivers of consumer behaviour such as proximity to public-sector paydays and cross-product family correlations.
Three regression models are systematically benchmarked via Grid Search cross-validation: SVR, MLP Regressor, and Gradient Boosting Regressor. The best model achieves R² = 0.948 on the held-out test set.
