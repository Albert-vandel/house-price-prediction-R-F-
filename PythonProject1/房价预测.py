# 1. 环境准备
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

# 设置图片清晰度
plt.rcParams['figure.dpi'] = 300

# 设置 matplotlib 的字体
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']

# 2. 数据加载与初步分析
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

# 数据可视化：房价分布（直方图、箱线图）
fig, axes = plt.subplots(1, 2, figsize=(9, 4))

# 绘制直方图
sns.histplot(train_data['SalePrice'], kde=True, color='skyblue', ax=axes[0])
axes[0].set_title('Histogram of House Sale Prices', fontsize=10, fontweight='bold')
axes[0].set_xlabel('Sale Price', fontsize=8)
axes[0].set_ylabel('Frequency', fontsize=8)
axes[0].tick_params(axis='both', which='major', labelsize=6)

# 绘制箱线图
sns.boxplot(y=train_data['SalePrice'], color='lightgreen', ax=axes[1])
axes[1].set_title('Boxplot of House Sale Prices', fontsize=10, fontweight='bold')
axes[1].set_ylabel('Sale Price', fontsize=8)
axes[1].tick_params(axis='y', which='major', labelsize=6)

plt.tight_layout()
plt.show()

# 数据可视化：特征与房价的相关性（热力图）
# 先筛选出数值型特征
numeric_features = train_data.select_dtypes(include=['int64', 'float64'])
correlation_matrix = numeric_features.corr()

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=False, cmap='viridis', linewidths=0.5)
plt.title('Correlation Heatmap of Numeric Features with Sale Price', fontsize=10, fontweight='bold')
plt.xticks(fontsize=6, rotation=45)
plt.yticks(fontsize=6)
plt.show()

# 3. 数据预处理
# 3.1 处理缺失值
def handle_missing_values(df):
    # 数值型特征用中位数填充
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    # 类别型特征用众数填充
    cat_cols = df.select_dtypes(include=["object"]).columns
    df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])
    return df

train_data = handle_missing_values(train_data)
test_data = handle_missing_values(test_data)

# 3.2 特征工程
# 创建新特征：房屋总面积
train_data["TotalSF"] = train_data["1stFlrSF"] + train_data["2ndFlrSF"] + train_data["TotalBsmtSF"]
test_data["TotalSF"] = test_data["1stFlrSF"] + test_data["2ndFlrSF"] + test_data["TotalBsmtSF"]

# 类别型特征使用标签编码（替代独热编码）
cat_cols = train_data.select_dtypes(include=["object"]).columns
for col in cat_cols:
    le = LabelEncoder()
    le.fit(train_data[col].astype(str))  # 防止测试集出现新类别
    train_data[col] = le.transform(train_data[col].astype(str))
    test_data[col] = le.transform(test_data[col].astype(str))

# 3.3 数据划分
X = train_data.drop("SalePrice", axis=1)
y = train_data["SalePrice"]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. 随机森林模型训练与调优
# 4.1 基线模型
rf_baseline = RandomForestRegressor(n_estimators=100, random_state=42)
rf_baseline.fit(X_train, y_train)
y_pred = rf_baseline.predict(X_val)
# 手动计算均方根误差
mse_baseline = mean_squared_error(y_val, y_pred)
rmse_baseline = np.sqrt(mse_baseline)
print(f"Baseline RMSE: {rmse_baseline}")

# 4.2 交叉验证评估
cv_scores = cross_val_score(rf_baseline, X_train, y_train,
                            scoring="neg_root_mean_squared_error", cv=5)
print(f"Cross-Validation RMSE: {-cv_scores.mean():.2f} ± {cv_scores.std():.2f}")

# 4.3 超参数调优（网格搜索）
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2]
}

grid_search = GridSearchCV(
    estimator=RandomForestRegressor(random_state=42),
    param_grid=param_grid,
    scoring="neg_root_mean_squared_error",
    cv=3,
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

# 输出最佳参数
best_params = grid_search.best_params_
print(f"Best Parameters: {best_params}")

# 4.4 使用最佳模型重新训练
best_rf = grid_search.best_estimator_
best_rf.fit(X_train, y_train)
y_pred_tuned = best_rf.predict(X_val)
# 手动计算均方根误差
mse_tuned = mean_squared_error(y_val, y_pred_tuned)
rmse_tuned = np.sqrt(mse_tuned)
print(f"Tuned RMSE: {rmse_tuned}")

# 模型结果可视化：预测值与真实值的对比图（散点图）
plt.figure(figsize=(5, 5))
plt.scatter(y_val, y_pred_tuned, color='orange', alpha=0.6)
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2)
plt.xlabel('True Sale Prices', fontsize=8)
plt.ylabel('Predicted Sale Prices', fontsize=8)
plt.title('True vs Predicted Sale Prices', fontsize=10, fontweight='bold')
plt.xticks(fontsize=6)
plt.yticks(fontsize=6)
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# 5. 特征重要性分析及可视化
feature_importance = best_rf.feature_importances_
sorted_idx = np.argsort(feature_importance)[-20:]  # 取前20个重要特征

plt.figure(figsize=(6, 5))
bars = plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align="center", color='purple')
plt.yticks(range(len(sorted_idx)), X.columns[sorted_idx], fontsize=6)
plt.xlabel("Feature Importance", fontsize=8)
plt.title("Top 20 Important Features (Random Forest)", fontsize=10, fontweight='bold')
plt.xticks(fontsize=6)
# 为每个条形图添加数值标签
for bar in bars:
    width = bar.get_width()
    plt.text(width, bar.get_y() + bar.get_height()/2, f'{width:.4f}', ha='left', va='center', fontsize=5)

plt.show()

# 6. 测试集预测与结果保存
test_predictions = best_rf.predict(test_data)
submission = pd.DataFrame({"Id": test_data["Id"], "SalePrice": test_predictions})
submission.to_csv("submission_rf_tuned.csv", index=False)