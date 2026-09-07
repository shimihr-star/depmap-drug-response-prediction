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
modeling_data = pd.read_csv(r"data/azd2014_modeling_data.csv")
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

# The data distribution
modeling_data["auc"].describe()
sns.displot(data=modeling_data, x='auc')

# ============================================================
# 4.1. MODELING - - Linear Regression with elastic net
# ============================================================
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('elastic_net', ElasticNet(max_iter=20000)),
])

param_grid = {
    'elastic_net__alpha': [0.001, 0.01, 0.1, 1, 10],
    'elastic_net__l1_ratio': [0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 1]
}
grid_model = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    cv=5,
    scoring='neg_root_mean_squared_error',
    n_jobs=-1
)
grid_model.fit(X_train, y_train)
grid_model.best_params_  # {'elastic_net__alpha': 0.1, 'elastic_net__l1_ratio': 0.1}

# alpha = 0 → essentially ordinary linear regression, no regularization
# small alpha → weak regularization
# large alpha → strong regularization → coefficients pushed more strongly toward zero

# l1_ratio = 0     → pure Ridge (L2)
# l1_ratio = 0.1   → 10% Lasso + 90% Ridge
# l1_ratio = 0.5   → 50% Lasso + 50% Ridge
# l1_ratio = 1     → pure Lasso (L1


# Model Evaluation
pred = grid_model.predict(X_test)
pred.mean()
MAE = mean_absolute_error(y_test, pred)
RMSE = np.sqrt(mean_squared_error(y_test, pred))
print(f'MAE:',MAE)
print(f'RMSE:', RMSE)

r2 = r2_score(y_test, pred)
print("R2:", r2)

residuals = y_test - pred
plt.scatter(pred, residuals)
plt.axhline(y=0, linestyle='--')
plt.xlabel('Fitted values')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted Values')
plt.show()


best_enet = grid_model.best_estimator_.named_steps['elastic_net']

print("Iterations:", best_enet.n_iter_)
print("Non-zero coefficients:", np.sum(best_enet.coef_ != 0))
print("Total coefficients:", len(best_enet.coef_))
# Iterations: 18
# Non-zero coefficients: 38
# Total coefficients: 19206

# The non zero coefficients:
best_model = grid_model.best_estimator_
enet = best_model.named_steps['elastic_net']
coef_df = pd.DataFrame({'gene': X_train.columns,'coefficient': enet.coef_})
selected_genes = (coef_df[coef_df['coefficient'] != 0].sort_values('coefficient', key=abs, ascending=False))
print(selected_genes)


top_genes = selected_genes.head(20)
plt.figure(figsize=(8, 7))
plt.barh(top_genes['gene'], top_genes['coefficient'])
plt.axvline(0, linestyle='--')
plt.xlabel('Elastic Net coefficient')
plt.ylabel('Gene')
plt.title('Top Elastic Net Selected Genes')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# CONCLUSION: The model is not good
# Elastic Net provided a sparse but weak predictive baseline. Cross-validation selected \(\alpha=0.1\) and an L1 ratio
# of 0.1, retaining 38 of 19,206 gene-expression features. However, test performance was poor (\(R^2=0.029\),
# RMSE = 0.093), with RMSE approximately equal to the standard deviation of the response variable (0.092).
# Thus, the linear model captured little of the variation in drug response.

# ============================================================
# 4.2. Modeling - - SVR
# ============================================================
pipe_svr = Pipeline([
    ('scaler', StandardScaler()),
    ('svr', SVR(kernel='rbf'))
])

param_grid_svr = {
    'svr__C': [0.1, 1, 10, 100],
    'svr__gamma': ['scale', 0.001, 0.01, 0.1],
    'svr__epsilon': [0.01, 0.05, 0.1]
}

grid_svr = GridSearchCV(
    pipe_svr,
    param_grid_svr,
    cv=5,
    scoring='neg_root_mean_squared_error',
    n_jobs=-1)

grid_svr.fit(X_train, y_train)

# Model Evaluation
pred = grid_svr.predict(X_test)
pred.mean()
MAE = mean_absolute_error(y_test, pred)
RMSE = np.sqrt(mean_squared_error(y_test, pred))
print(f'MAE:',MAE)
print(f'RMSE:', RMSE)

r2_svr = r2_score(y_test, pred)
print("R2:", r2_svr)

# SVR Summary:
# RBF-SVR was used to test whether a nonlinear relationship could improve
# drug-response prediction compared with Elastic Net.
#
# Test performance:
# MAE  = 0.07567
# RMSE = 0.09390
# R2   = 0.0169
#
# Conclusion:
# SVR showed weak predictive performance and did not improve over Elastic Net.
# Only ~1.7% of the variance in drug-response AUC was explained.

# ============================================================
# 4.3. Modeling - - Random Forest
# ============================================================
rf = RandomForestRegressor(random_state=42,n_jobs=-1)

param_grid_rf = {
    'n_estimators': [300, 500],
    'max_features': ['sqrt', 0.1, 0.3],
    'max_depth': [None, 10, 20],
    'min_samples_leaf': [1, 3, 5]
}

grid_rf = GridSearchCV(
    estimator=rf,
    param_grid=param_grid_rf,
    cv=5,
    scoring='neg_root_mean_squared_error',
    n_jobs=-1
)

grid_rf.fit(X_train, y_train)
grid_rf.best_params_
# {'max_depth': None, 'max_features': 0.3, 'min_samples_leaf': 1, 'n_estimators': 300}

# Model evaluation
pred_rf = grid_rf.predict(X_test)
MAE_rf = mean_absolute_error(y_test, pred_rf)
RMSE_rf = np.sqrt(mean_squared_error(y_test, pred_rf))
R2_rf = r2_score(y_test, pred_rf)

print("MAE:", MAE_rf)
print("RMSE:", RMSE_rf)
print("R2:", R2_rf)
# MAE: 0.07311581417568662
# RMSE: 0.09109493313860467
# R2: 0.07472254283550073

pred_train_rf = grid_rf.predict(X_train)
print("Train R2:", r2_score(y_train, pred_train_rf))
print("Test R2:", r2_score(y_test, pred_rf))
# Train R2: 0.8714003508206001
# Test R2: 0.07472254283550073

# Random Forest Summary:
# Random Forest was used to test whether a nonlinear tree-based model could
# improve drug-response prediction.
#
# Test performance:
# MAE  = 0.07312
# RMSE = 0.09109
# R2   = 0.0747
#
# Train R2 = 0.8714
#
# Conclusion:
# Random Forest showed weak predictive performance, explaining only ~7.5%
# of the variance in drug-response AUC on the test set.
# The large gap between train R2 (~87%) and test R2 (~7.5%) indicates
# substantial overfitting and poor generalization to unseen cell lines.