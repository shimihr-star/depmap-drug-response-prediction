import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn. preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.decomposition import PCA

# ============================================================
# 1. LOAD CLEAN MODELING DATA
# ============================================================
# Load the preprocessed AZD2014 modeling dataset.
# Each row represents one cell line and contains:
#   - ModelID: DepMap cell-line identifier
#   - 19,206 baseline gene-expression features
#   - auc: AZD2014 drug-response target
modeling_data = pd.read_csv(r"data\azd2014_modeling_data.csv")
print("Dataset shape:", modeling_data.shape)


# ============================================================
# 2. DEFINE FEATURES (X) AND TARGET (y)
# ============================================================

# Use ModelID as the row index rather than as a predictive feature.
modeling_data = modeling_data.set_index("ModelID")

# X contains gene-expression features only.
# y contains the continuous AZD2014 AUC response.
X = modeling_data.drop(columns="auc")
y = modeling_data["auc"]

print("X shape:", X.shape)
print("y shape:", y.shape)

# Data split
# holding a hold out set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=101)


# ============================================================
# 3. Dimension reduction - PCA
# ============================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
pca = PCA()
X_train_pca = pca.fit_transform(X_train_scaled)
pca.explained_variance_ratio_
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

plt.plot(range(1, len(cumulative_variance) + 1),
         cumulative_variance)

plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.show()

plt.scatter(X_train_pca[:, 0], y_train)
plt.xlabel("PC1")
plt.ylabel("AZD2014 AUC")
plt.show()

# make 100 scatterplots. Calculate the correlation of each PC with AUC:
correlations = [np.corrcoef(X_train_pca[:, i], y_train)[0, 1]for i in range(X_train_pca.shape[1])]

plt.plot(range(1, len(correlations) + 1), correlations)
plt.xlabel("Principal Component")
plt.ylabel("Correlation with AZD2014 AUC")
plt.show()

#  The main result is that none of the PCs appears strongly correlated with AZD2014 AUC.

# ============================================================
# 4. MODELING - - Linear Regression with elastic net
# ============================================================
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('elastic_net', ElasticNet(max_iter=10000)),
])

param_grid = {
    'elastic_net__alpha': [0.1, 1, 10, 50, 100],
    'elastic_net__l1_ratio': [0.1, 0.5, 0.7, 1]
}
grid_model = GridSearchCV(estimator=pipe, param_grid=param_grid)
grid_model.fit(X_train, y_train)
grid_model.best_params_  # {'elastic_net__alpha': 0.1, 'elastic_net__l1_ratio': 0.1}

# alpha = 0 → essentially ordinary linear regression, no regularization
# small alpha → weak regularization
# large alpha → strong regularization → coefficients pushed more strongly toward zero

# l1_ratio = 0     → pure Ridge (L2)
# l1_ratio = 0.1   → 10% Lasso + 90% Ridge
# l1_ratio = 0.5   → 50% Lasso + 50% Ridge
# l1_ratio = 1     → pure Lasso (L1)



# Model Evaluation
pred = grid_model.predict(X_test)
pred.mean()
MAE = mean_absolute_error(y_test, pred)
RMSE = np.sqrt(mean_squared_error(y_test, pred))
residuals = y_test - pred

plt.scatter(pred, residuals)
plt.axhline(y=0, linestyle='--')
plt.xlabel('Fitted values')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted Values')
plt.show()