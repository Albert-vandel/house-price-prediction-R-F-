# 房价预测模型 - 随机森林实现

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.2.2-green)
![License](https://img.shields.io/badge/License-MIT-orange)

基于Kaggle房价预测竞赛的机器学习解决方案，使用随机森林算法实现高精度房价预测。

## 项目目录
* data/ # 原始数据集
* images/ # 可视化图表
* submission_rf_tuned.csv # 预测结果文件
* 房价预测.py # 完整代码

## 快速开始

### 环境要求
```bash
pandas>=1.5.3
numpy>=1.24.3
scikit-learn>=1.2.2
matplotlib>=3.7.1
seaborn>=0.12.2
```
### 运行步骤
* 下载Kaggle数据集到data/目录
* 执行Jupyter Notebook：
  ```bash
  jupyter notebook house_price_prediction.ipynb
  ```
## 技术实现
### 数据预处理
|  处理步骤   | 实现方法  |
|  ----  | ----  |
| 缺失值填充  | 数值特征中位数填充/类别特征众数填充 |
| 特征工程  | 创建TotalSF总面积特征 |
| 类别编码  | LabelEncoder标签编码 |
### 模型架构
```bash
RandomForestRegressor(
    n_estimators=200, 
    max_depth=20,
    min_samples_split=2,
    min_samples_leaf=1
)
 ```
### 性能表现
| 评估指标 | 基线模型 | 调优模型 |
| :-----:| :----: | :----: |
|验证集RMSE | 29176 | 28432 |
| 交叉验证RMSE | 29350±800 | 28520±750 |

## 关键可视化
* 房价分布：images/price_distribution.png
* 特征重要性：images/feature_importance.png



