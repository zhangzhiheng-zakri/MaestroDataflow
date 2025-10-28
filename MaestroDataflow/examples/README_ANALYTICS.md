# MaestroDataflow 数据分析与可视化功能

## 概述

MaestroDataflow 现已支持强大的数据分析和可视化功能，可以帮助您快速生成专业的数据报告和图表。所有输出文件都会保存到 `../output/` 目录的相应子目录中。

## 输出目录结构

运行示例脚本后，输出文件将按以下结构组织：

```
output/
├── digital_economy_analysis/       # 数字经济分析示例输出
│   ├── charts/                     # 图表文件
│   ├── reports/                    # 报告文件
│   ├── data/                       # 数据文件
│   └── dashboard/                  # 仪表板文件
├── advanced_pipeline_example/      # 高级管道示例输出
│   ├── results/                    # 结果文件
│   ├── data/                       # 数据文件
│   └── cache/                      # 缓存文件
├── ai_operators_demo/              # AI操作符演示输出
│   └── cache/                      # 缓存文件
└── comprehensive_ai_workflow/      # 综合工作流输出
    ├── data/                       # 数据文件
    ├── cache/                      # 缓存文件
    ├── reports/                    # 报告文件
    └── results/                    # 结果文件
```

## 新增算子

### 1. 数据分析算子 (Analytics Operators)

#### DataAnalysisOperator
- **功能**: 执行全面的统计分析
- **支持**: 描述性统计、增长率计算、趋势分析
- **输出**: 统计摘要、增长率、趋势指标

```python
from maestro.operators.analytics_ops import DataAnalysisOperator

analysis_op = DataAnalysisOperator(
    analysis_columns=['销售额', '利润率'],
    time_column='日期',
    calculate_growth_rate=True
)
```

#### DataSummaryOperator
- **功能**: 生成数据摘要报告
- **支持**: 基础摘要、详细摘要、相关性分析
- **输出**: 数据概览、质量评估、相关性矩阵

```python
from maestro.operators.analytics_ops import DataSummaryOperator

summary_op = DataSummaryOperator(
    summary_type='comprehensive',
    include_correlations=True
)
```

### 2. 可视化算子 (Visualization Operators)

#### ChartGeneratorOperator
- **功能**: 生成各种类型的图表
- **支持图表类型**:
  - 线图 (line)
  - 柱状图 (bar)
  - 散点图 (scatter)
  - 饼图 (pie)
  - 热力图 (heatmap)
  - 箱线图 (box)

```python
from maestro.operators.visualization_ops import ChartGeneratorOperator

chart_op = ChartGeneratorOperator(
    chart_type='line',
    x_column='年份',
    y_columns=['数字经济规模'],
    title='数字经济发展趋势',
    output_file='trend_chart.png'
)
```

#### DashboardGeneratorOperator
- **功能**: 生成交互式仪表板
- **支持**: 多图表组合、响应式布局
- **输出**: HTML仪表板文件

```python
from maestro.operators.visualization_ops import DashboardGeneratorOperator

dashboard_op = DashboardGeneratorOperator(
    dashboard_title='销售数据仪表板',
    chart_configs=[
        {
            'type': 'line',
            'x_column': '月份',
            'y_columns': ['销售额'],
            'title': '月度销售趋势'
        }
    ]
)
```

### 3. 报告生成算子 (Report Operators)

#### HTMLReportGeneratorOperator
- **功能**: 生成专业的HTML报告
- **支持**: 现代化样式、图表嵌入、数据表格
- **模板**: modern, classic, minimal

```python
from maestro.operators.report_ops import HTMLReportGeneratorOperator

html_report = HTMLReportGeneratorOperator(
    report_title='数据分析报告',
    output_file='report.html',
    template_style='modern'
)
```

#### PDFReportGeneratorOperator
- **功能**: 生成PDF格式报告
- **支持**: A4/Letter页面、表格、图表
- **依赖**: reportlab库

```python
from maestro.operators.report_ops import PDFReportGeneratorOperator

pdf_report = PDFReportGeneratorOperator(
    report_title='数据分析报告',
    output_file='report.pdf'
)
```

#### ReportTemplateOperator
- **功能**: 使用预定义模板生成报告
- **模板类型**:
  - comprehensive: 综合报告
  - executive: 执行摘要
  - technical: 技术报告

```python
from maestro.operators.report_ops import ReportTemplateOperator

template_report = ReportTemplateOperator(
    template_name='executive',
    output_format='html'
)
```

## 使用示例

### 完整的数据分析流程

```python
from maestro.core import MaestroWorkflow
from maestro.operators.analytics_ops import DataAnalysisOperator
from maestro.operators.visualization_ops import ChartGeneratorOperator
from maestro.operators.report_ops import HTMLReportGeneratorOperator

# 创建工作流
workflow = MaestroWorkflow()

# 添加算子
workflow.add_operator("analysis", DataAnalysisOperator(
    analysis_columns=['销售额', '利润'],
    time_column='日期'
))

workflow.add_operator("chart", ChartGeneratorOperator(
    chart_type='line',
    x_column='日期',
    y_columns=['销售额']
), depends_on=["analysis"])

workflow.add_operator("report", HTMLReportGeneratorOperator(
    report_title='销售分析报告'
), depends_on=["chart"])

# 执行工作流
result = workflow.run(data)
```

### 数字经济数据分析示例

参考 `examples/digital_economy_analysis.py` 文件，该示例展示了：

1. **数据准备**: 创建数字经济发展数据
2. **统计分析**: 计算增长率、趋势分析
3. **可视化**: 生成多种图表类型
4. **仪表板**: 创建交互式仪表板
5. **报告生成**: 输出专业HTML报告

运行示例：
```bash
cd examples
python digital_economy_analysis.py
```

运行后，所有输出文件将保存到 `../output/digital_economy_analysis/` 目录：
- 图表文件保存到 `charts/` 子目录
- 报告文件保存到 `reports/` 子目录  
- 数据文件保存到 `data/` 子目录
- 仪表板文件保存到 `dashboard/` 子目录

### 其他示例

```bash
# 高级管道示例
python advanced_pipeline_example.py
# 输出到: ../output/advanced_pipeline_example/

# AI操作符演示
python ai_operators_demo.py  
# 输出到: ../output/ai_operators_demo/

# 综合AI工作流
python comprehensive_ai_workflow.py
# 输出到: ../output/comprehensive_ai_workflow/
```

## 输出文件

### 图表文件
- PNG格式的静态图表
- 支持高分辨率输出
- 自动优化图表样式

### 仪表板文件
- HTML格式的交互式仪表板
- 响应式设计，支持移动端
- 基于Plotly的交互功能

### 报告文件
- HTML报告：现代化样式，包含图表和数据
- PDF报告：适合打印和分享
- 支持自定义模板和样式

## 依赖库

### 必需依赖
```bash
pip install pandas matplotlib plotly
```

### 可选依赖
```bash
# PDF报告生成
pip install reportlab

# 高级PDF功能
pip install weasyprint
```

## 配置选项

### 图表配置
- 颜色主题自定义
- 图表尺寸设置
- 字体和样式配置

### 报告配置
- 模板样式选择
- 章节内容自定义
- 输出格式选择

## 最佳实践

1. **数据准备**: 确保数据格式正确，时间列为datetime类型
2. **算子顺序**: 先分析，后可视化，最后生成报告
3. **文件管理**: 使用有意义的文件名和路径
4. **性能优化**: 大数据集时考虑采样或分批处理

## 扩展功能

### 自定义图表类型
可以通过继承 `ChartGeneratorOperator` 添加新的图表类型。

### 自定义报告模板
可以通过修改 `HTMLReportGeneratorOperator` 的CSS样式创建自定义模板。

### 数据源集成
支持从多种数据源读取数据，包括CSV、Excel、数据库等。

## 故障排除

### 常见问题

1. **图表不显示**: 检查matplotlib后端设置
2. **PDF生成失败**: 确保安装了reportlab库
3. **中文显示问题**: 配置正确的字体文件

### 调试技巧

1. 使用 `workflow.debug = True` 启用调试模式
2. 检查中间结果文件
3. 查看算子执行日志

## 更新日志

### v1.0.0 (2024-01-XX)
- 新增数据分析算子
- 新增可视化算子
- 新增报告生成算子
- 支持多种图表类型
- 支持HTML/PDF报告输出