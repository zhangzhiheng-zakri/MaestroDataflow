"""
HTML报告生成演示：仅输出HTML，不依赖WeasyPrint。

运行：
    python -m examples.html_report_demo
"""

import os
import pandas as pd
from maestro.utils.storage import FileStorage
from maestro.operators.report_ops import HTMLReportGeneratorOperator


def main():
    # 准备输入数据
    os.makedirs("output/reports", exist_ok=True)
    input_csv = "output/reports/demo_input.csv"
    df = pd.DataFrame({
        "年份": [2021, 2022, 2023, 2024],
        "GDP": [320000, 335000, 345000, 360000],
        "满意度": [4.5, 4.2, 4.7, 4.6],
    })
    df.to_csv(input_csv, index=False, encoding="utf-8")

    # 创建存储
    storage = FileStorage(
        input_file_path=input_csv,
        cache_path="./output/reports/cache",
        file_name_prefix="html_demo",
        cache_type="csv"
    )

    # 生成HTML报告
    output_html = "output/reports/demo_report.html"
    html_op = HTMLReportGeneratorOperator(
        report_title="HTML报告演示（无PDF依赖）",
        output_file=output_html,
        include_charts=False,
        include_data_table=True,
    )
    result = html_op.run(storage)

    print("✅ HTML 报告生成完成：")
    print(f"- {output_html}")


if __name__ == "__main__":
    main()