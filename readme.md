# H&M 个性化推荐系统

基于 LightGBM LambdaRank 的商品排序推荐管线，完整覆盖数据处理、特征工程、模型训练到提交文件生成的全流程。

## 项目结构

```
F:\HM-recommendation/
├── config.py              # 全局配置 (路径/参数/随机种子/特征列定义)
├── utils.py               # 工具函数 (MAP@K 评估/计时器/随机种子)
├── requirements.txt       # Python 依赖清单
├── .gitignore
├── step1_load_data.py     # 数据加载与验证集划分
├── step2_features.py      # 特征工程 (5类 46维特征)
├── step3_baseline.py      # 基线模型评估 (流行度/Item-CF)
├── step4_candidates.py    # LTR 候选生成
├── step5_train.py         # LightGBM LambdaRank 排序模型训练
├── step6_infer.py         # 全量特征重建 + 分批推理 + 提交
└── output/                # 模型与提交文件 (自动生成)
```

## 运行流程

### 1. 安装依赖

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 2. 准备数据

从 [Kaggle H&M 竞赛](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations) 下载数据，放入 `F:/H&M_data/` 目录：

```
F:/H&M_data/
├── articles.csv
├── customers.csv
├── transactions_train.csv
└── sample_submission.csv
```

### 3. 按顺序运行

```bash
python step1_load_data.py    # 加载数据 → 按时间切分 train/val
python step2_features.py     # 特征工程 → 客户/商品/交互特征 + Item-CF + 用户历史
python step3_baseline.py     # 基线评估 → 流行度 + Item-CF MAP@12
python step4_candidates.py   # 候选生成 → 每用户候选商品列表
python step5_train.py        # LTR 训练 → Phase 1 早停 + Phase 2 全量训练
python step6_infer.py        # 全量推理 → 生成 submission.csv
```

## 管线详解

| Step | 功能 | 输入 | 输出 |
|------|------|------|------|
| 1 | 数据加载与切分 | CSV 原始数据 | train/val Parquet |
| 2 | 特征工程 | train_txn | 客户8维 + 商品33维 + 交互3维 + Item-CF + 用户历史 |
| 3 | 基线评估 | 特征 + val_txn | MAP@12 + baseline 提交文件 |
| 4 | 候选生成 | 特征 + val_txn | val_candidates.pkl |
| 5 | 排序训练 | 特征 + 候选 | model.txt + 评估报告 |
| 6 | 全量推理 | 全量数据 + model.txt | submission.csv |

### 特征维度 (46维)

- **客户特征 (8维)**: age, club_member_status_le, postal_le, R_days, F_count, M_spend, avg_price_user, n_unique_articles
- **商品特征 (33维)**: avg_price, sales_count, n_buyers, popularity_score, price_log, sales_log, 7个类别编码 + 20维文本嵌入
- **交互特征 (3维)**: buy_count, last_buy_days, first_buy_days
- **候选特征 (2维)**: cf_score, price_match

### 训练策略

- **Phase 1**: val 数据内再切分 (前5天训练 / 后2天 holdout 评估) → 早停 (patience=50) → 确定最佳迭代轮数
- **Phase 2**: 使用全部 val 数据重新训练固定轮数 → 保存最终模型

## 配置说明

主要参数在 `config.py` 中调整：

```python
SEED = 42                              # 全局随机种子
VAL_DAYS = 7                           # 验证集天数
LGB_PARAMS = {                         # LightGBM 参数
    'learning_rate': 0.05,
    'num_leaves': 127,
    'max_depth': 8,
    ...
}
INFER_BATCH_SIZE = 50000               # 推理批次大小
```

## 随机种子

所有 step 开头均调用 `set_seed(SEED)` 保证可复现。

## 环境要求

- Python >= 3.9
- 依赖见 requirements.txt
- GPU 可选 (LightGBM 自动检测 GPU/CPU)

## License

MIT
