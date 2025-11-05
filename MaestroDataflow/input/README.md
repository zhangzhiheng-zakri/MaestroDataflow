# 输入数据目录说明

将你的真实数据文件放在此目录中，示例脚本默认从 `input/` 读取。

支持的文件格式：
- CSV（推荐）
- XLSX（Excel）
- JSON / JSONL
- Parquet
- Pickle

快速开始：
- 将你的文件命名为 `my_dataset.csv`（或修改示例脚本中的路径为你的文件名）。
- 运行：`python -m examples.integrated_packaging_workflow` 或 `python -m examples.package_dataset_example`

注意：
- 此目录用于本地数据输入，不建议把大文件提交到版本库。
- 打包示例输出仅包含 `all_column_name.json`（列名意义JSON），不再生成或依赖 `column_template.json`。