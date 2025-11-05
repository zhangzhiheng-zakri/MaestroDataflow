"""
MaestroDataflow 数字经济数据分析示例
演示如何使用新的数据分析、可视化和报告生成功能
使用真实的中国数字经济发展数据（2005-2023年）
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# 导入MaestroDataflow组件
from maestro.pipeline.pipeline import Pipeline
from maestro.utils.storage import FileStorage
from maestro.operators.analytics_ops import DataAnalysisOperator, DataSummaryOperator
from maestro.operators.visualization_ops import ChartGeneratorOperator, DashboardGeneratorOperator
from maestro.operators.report_ops import HTMLReportGeneratorOperator, ReportTemplateOperator
from maestro.operators.io_ops import SaveToFileOperator


def preprocess_digital_economy_data(df):
    """
    数字经济数据预处理和清洗函数
    
    Args:
        df: 原始数据框
        
    Returns:
        清洗后的数据框
    """
    print(f"开始数据预处理，原始数据形状: {df.shape}")
    
    # 创建数据副本
    df_cleaned = df.copy()
    
    # 1. 检查和处理缺失值
    print("1. 检查缺失值...")
    missing_info = df_cleaned.isnull().sum()
    if missing_info.sum() > 0:
        print(f"发现缺失值: \n{missing_info[missing_info > 0]}")
        
        # 对数值列使用前向填充和后向填充
        numeric_columns = df_cleaned.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if df_cleaned[col].isnull().sum() > 0:
                # 先前向填充，再后向填充，最后用0填充剩余的NaN
                df_cleaned[col] = df_cleaned[col].fillna(method='ffill').fillna(method='bfill').fillna(0)
                print(f"   - 填充列 '{col}' 的缺失值")
        
        # 对非数值列使用众数填充
        non_numeric_columns = df_cleaned.select_dtypes(exclude=[np.number]).columns
        for col in non_numeric_columns:
            if df_cleaned[col].isnull().sum() > 0:
                mode_value = df_cleaned[col].mode()
                if len(mode_value) > 0:
                    df_cleaned[col] = df_cleaned[col].fillna(mode_value[0])
                else:
                    df_cleaned[col] = df_cleaned[col].fillna('未知')
                print(f"   - 用众数填充列 '{col}' 的缺失值")
        
        # 最终检查：确保没有剩余的NaN值
        remaining_nulls = df_cleaned.isnull().sum().sum()
        if remaining_nulls > 0:
            print(f"   - 警告：仍有 {remaining_nulls} 个空值，用默认值填充")
            # 对剩余的数值列空值用0填充
            df_cleaned = df_cleaned.fillna(0)
    else:
        print("   - 未发现缺失值")
    
    # 2. 检查和处理重复行
    print("2. 检查重复行...")
    duplicates = df_cleaned.duplicated().sum()
    if duplicates > 0:
        print(f"   - 发现 {duplicates} 行重复数据，正在删除...")
        df_cleaned = df_cleaned.drop_duplicates()
    else:
        print("   - 未发现重复行")
    
    # 3. 数据类型优化和验证
    print("3. 数据类型验证和优化...")
    for col in df_cleaned.columns:
        if col != '指标':  # 跳过年份列
            # 确保数值列为数值类型
            if df_cleaned[col].dtype == 'object':
                try:
                    df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')
                    print(f"   - 转换列 '{col}' 为数值类型")
                    # 转换后可能产生新的NaN值，需要再次填充
                    if df_cleaned[col].isnull().sum() > 0:
                        df_cleaned[col] = df_cleaned[col].fillna(0)
                        print(f"   - 填充转换后产生的空值")
                except:
                    pass
    
    # 4. 数据范围验证
    print("4. 数据范围验证...")
    # 检查百分比数据是否在合理范围内
    percentage_columns = [col for col in df_cleaned.columns if '%' in col]
    for col in percentage_columns:
        if col in df_cleaned.columns:
            # 将超出合理范围的百分比数据标记为异常
            invalid_values = (df_cleaned[col] < 0) | (df_cleaned[col] > 100)
            if invalid_values.sum() > 0:
                print(f"   - 警告: 列 '{col}' 中有 {invalid_values.sum()} 个值超出0-100%范围")
    
    # 5. 确保年份列的连续性
    print("5. 验证时间序列连续性...")
    if '指标' in df_cleaned.columns:
        years = sorted(df_cleaned['指标'].unique())
        expected_years = list(range(min(years), max(years) + 1))
        missing_years = set(expected_years) - set(years)
        if missing_years:
            print(f"   - 警告: 缺少年份数据: {sorted(missing_years)}")
        else:
            print("   - 时间序列完整")
    
    print(f"数据预处理完成，最终数据形状: {df_cleaned.shape}")
    
    # 6. 最终空值检查
    final_nulls = df_cleaned.isnull().sum().sum()
    if final_nulls > 0:
        print(f"   - 最终检查：仍有 {final_nulls} 个空值，强制填充为0")
        df_cleaned = df_cleaned.fillna(0)
    else:
        print("   - 最终检查：数据清洗完成，无空值")
    
    return df_cleaned


def load_digital_economy_data():
    """加载真实的中国数字经济发展数据"""
    try:
        # 使用FileStorage加载Excel数据
        storage = FileStorage(input_file_path="../sample_data/中国数字经济发展数据（2005-2023年）.xlsx")
        storage.step()
        df = storage.read(output_type="dataframe")
        
        print(f"成功加载数字经济数据，共 {len(df)} 条记录")
        print(f"数据列: {list(df.columns)}")
        return df
        
    except Exception as e:
        print(f"加载数据失败: {e}")
        print("使用模拟数据作为备选...")
        return create_fallback_data()


def create_fallback_data():
    """创建备选的数字经济发展示例数据"""
    data = {
        '指标': [2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023],
        '数字经济规模(万亿元）': [2.6, 3.2, 4.0, 4.8, 5.4, 6.2, 7.8, 9.1, 10.5, 12.2, 14.1, 16.8, 19.6, 22.4, 25.8, 29.3, 33.2, 37.1, 41.5],
        '数字经济规模同比名义增长(%)': [15.2, 23.1, 25.0, 20.0, 12.5, 14.8, 25.8, 16.7, 15.4, 16.2, 15.6, 19.1, 16.7, 14.3, 15.2, 13.6, 13.3, 11.7, 11.9],
        '数字经济规模占GDP比重(%)': [14.2, 15.8, 17.1, 18.2, 18.8, 19.6, 21.4, 22.8, 24.1, 25.9, 27.5, 30.3, 32.9, 34.8, 36.2, 38.6, 39.8, 41.5, 42.8],
        '数字产业化规模(万亿元)': [1.1, 1.4, 1.8, 2.1, 2.3, 2.6, 3.2, 3.8, 4.2, 4.8, 5.4, 6.1, 6.8, 7.5, 8.2, 8.9, 9.6, 10.3, 11.1],
        '产业数字化规模(万亿元)': [1.5, 1.8, 2.2, 2.7, 3.1, 3.6, 4.6, 5.3, 6.3, 7.4, 8.7, 10.7, 12.8, 14.9, 17.6, 20.4, 23.6, 26.8, 30.4],
        '农业数字经济渗透率(%)': [2.1, 2.3, 2.6, 2.9, 3.2, 3.6, 4.1, 4.7, 5.3, 6.0, 6.8, 7.7, 8.8, 10.1, 11.5, 13.2, 15.1, 17.3, 19.8],
        '工业数字经济渗透率(%)': [12.8, 14.2, 15.8, 17.1, 18.3, 19.7, 21.5, 23.2, 24.9, 26.8, 28.9, 31.2, 33.7, 36.5, 39.6, 42.9, 46.5, 50.4, 54.6],
        '服务业数字经济渗透率(%)': [18.5, 20.3, 22.4, 24.2, 25.8, 27.6, 30.1, 32.5, 34.8, 37.4, 40.2, 43.3, 46.7, 50.4, 54.5, 58.9, 63.7, 68.9, 74.5]
    }
    
    return pd.DataFrame(data)


def run_digital_economy_analysis():
    """运行数字经济数据分析流程"""
    print("开始数字经济发展数据分析...")
    
    # 加载真实数据
    df = load_digital_economy_data()
    print(f"数据加载完成，共 {len(df)} 条记录")
    
    # 数据预处理和清洗
    print("\n🧹 执行数据预处理和清洗...")
    df_cleaned = preprocess_digital_economy_data(df)
    print(f"数据清洗完成，处理后数据形状: {df_cleaned.shape}")
    
    # 先保存清洗后的数据到文件
    os.makedirs('../output/digital_economy_analysis/data', exist_ok=True)
    df_cleaned.to_csv("../output/digital_economy_analysis/data/digital_economy_data.csv", index=False, encoding='utf-8')
    
    # 创建存储实例
    storage = FileStorage(
        input_file_path="../output/digital_economy_analysis/data/digital_economy_data.csv",
        cache_path="./cache"
    )
    
    # 创建工作流
    workflow = Pipeline(storage=storage)
    
    # 1. 数据分析
    print("\n1. 执行数据分析...")
    analysis_operator = DataAnalysisOperator(
        columns_to_analyze=['数字经济规模(万亿元）', '数字经济规模同比名义增长(%)', '数字经济规模占GDP比重(%)'],
        time_column='指标',
        include_growth_analysis=True
    )
    
    # 2. 数据摘要
    print("2. 生成数据摘要...")
    summary_operator = DataSummaryOperator(
        group_by_column=None
    )
    
    # 3. 生成图表
    print("3. 生成数据可视化图表...")
    
    # 趋势图
    trend_chart = ChartGeneratorOperator(
        chart_type='line',
        x_column='指标',
        y_columns=['数字经济规模(万亿元）', '数字经济规模占GDP比重(%)'],
        title='中国数字经济发展趋势',
        output_dir='../output/digital_economy_analysis/charts',
        output_filename='digital_economy_trend'
    )
    
    # 增长率柱状图
    growth_chart = ChartGeneratorOperator(
        chart_type='bar',
        x_column='指标',
        y_columns=['数字经济规模同比名义增长(%)'],
        title='数字经济年度增长率',
        output_dir='../output/digital_economy_analysis/charts',
        output_filename='digital_economy_growth'
    )
    
    # 渗透率对比图
    penetration_chart = ChartGeneratorOperator(
        chart_type='line',
        x_column='指标',
        y_columns=['农业数字经济渗透率(%)', '工业数字经济渗透率(%)', '服务业数字经济渗透率(%)'],
        title='各行业数字化渗透率对比',
        output_dir='../output/digital_economy_analysis/charts',
        output_filename='digitalization_penetration'
    )
    
    # 产业结构饼图（使用2023年数据）
    pie_chart = ChartGeneratorOperator(
        chart_type='pie',
        x_column='指标',
        y_columns=['数字经济规模(万亿元）', '数字产业化规模(万亿元)', '产业数字化规模(万亿元)'],
        title='数字经济结构分析',
        output_dir='../output/digital_economy_analysis/charts',
        output_filename='digital_economy_structure'
    )
    
    # 确保输出目录存在
    os.makedirs('../output/digital_economy_analysis/reports', exist_ok=True)
    os.makedirs('../output/digital_economy_analysis/charts', exist_ok=True)
    os.makedirs('../output/digital_economy_analysis/data', exist_ok=True)
    
    # 4. 生成仪表板
    print("4. 生成综合仪表板...")
    dashboard_operator = DashboardGeneratorOperator(
        dashboard_title='中国数字经济发展分析仪表板',
        output_file='../output/digital_economy_analysis/reports/digital_economy_dashboard.html',
        include_charts=['line', 'bar']
    )
    
    # 5. 生成HTML报告
    print("5. 生成分析报告...")
    html_report = HTMLReportGeneratorOperator(
        report_title='中国数字经济发展分析报告（2005-2023）',
        output_file='../output/digital_economy_analysis/reports/digital_economy_report.html',
        include_charts=True,
        include_data_table=True,
        template_style='modern'
    )
    
    # 6. 保存数据
    save_operator = SaveToFileOperator(
        output_path='../output/digital_economy_analysis/data/digital_economy_results.csv',
        format_type='csv'
    )
    
    # 构建工作流
    workflow.add_operator(analysis_operator, "analysis")
    workflow.add_operator(summary_operator, "summary")
    workflow.add_operator(trend_chart, "trend_chart")
    workflow.add_operator(growth_chart, "growth_chart")
    workflow.add_operator(penetration_chart, "penetration_chart")
    workflow.add_operator(pie_chart, "pie_chart")
    workflow.add_operator(dashboard_operator, "dashboard")
    workflow.add_operator(html_report, "html_report")
    workflow.add_operator(save_operator, "save_data")
    
    # 执行工作流
    print("\n开始执行数据分析工作流...")
    result = workflow.run(df)
    
    print("\n✅ 数字经济数据分析完成！")
    print("\n生成的文件:")
    print("- ../output/digital_economy_analysis/charts/digital_economy_trend.png - 数字经济发展趋势图")
    print("- ../output/digital_economy_analysis/charts/digital_economy_growth.png - 数字经济增长率图")
    print("- ../output/digital_economy_analysis/charts/digitalization_penetration.png - 各行业数字化渗透率图")
    print("- ../output/digital_economy_analysis/charts/digital_economy_structure.png - 数字经济结构图")
    print("- ../output/digital_economy_analysis/reports/digital_economy_dashboard.html - 综合仪表板")
    print("- ../output/digital_economy_analysis/reports/digital_economy_report.html - 完整分析报告")
    print("- ../output/digital_economy_analysis/data/digital_economy_results.csv - 处理后的数据")
    
    return result


def run_executive_summary():
    """生成执行摘要报告"""
    print("\n生成执行摘要报告...")
    
    df = load_digital_economy_data()  
    
    # 先保存数据到文件
    df.to_csv("../output/digital_economy_analysis/data/digital_economy_executive_data.csv", index=False, encoding='utf-8')
    
    # 创建存储实例
    storage = FileStorage(
        input_file_path="../output/digital_economy_analysis/data/digital_economy_executive_data.csv",
        cache_path="./cache"
    )
    
    # 使用报告模板生成执行摘要
    template_operator = ReportTemplateOperator(
        template_name='executive',
        output_format='html'
    )
    
    workflow = Pipeline(storage=storage)
    
    # 先进行基础分析
    analysis_operator = DataAnalysisOperator(
        columns_to_analyze=['数字经济规模(万亿元）', '数字经济规模占GDP比重(%)'],
        time_column='指标',
        include_growth_analysis=True
    )
    
    workflow.add_operator(analysis_operator, "analysis")
    workflow.add_operator(template_operator, "executive_report")
    
    result = workflow.run(df)
    print("✅ 执行摘要报告生成完成！")
    
    return result


def demonstrate_custom_analysis():
    """演示自定义分析功能"""
    print("\n演示自定义分析功能...")
    
    df = load_digital_economy_data()  
    
    # 对数据进行预处理，确保没有空值
    df = preprocess_digital_economy_data(df)
    
    # 自定义分析：重点关注近5年发展
    recent_df = df[df['指标'] >= 2019].copy()
    
    # 检查是否有足够的数据
    if len(recent_df) < 2:
        print("数据不足，使用全部数据进行分析...")
        recent_df = df.copy()
    
    # 计算复合增长率
    if len(recent_df) >= 2:
        start_value = recent_df.iloc[0]['数字经济规模(万亿元）']
        end_value = recent_df.iloc[-1]['数字经济规模(万亿元）']
        years = len(recent_df) - 1
        if years > 0 and start_value > 0:
            cagr = ((end_value / start_value) ** (1/years) - 1) * 100
            print(f"数字经济规模复合增长率: {cagr:.2f}%")
        else:
            print("无法计算复合增长率")
    else:
        print("数据不足，无法计算增长率")
    
    # 生成专门的近期趋势分析
    # 先保存数据到文件
    recent_df.to_csv("../output/digital_economy_analysis/data/recent_digital_data.csv", index=False, encoding='utf-8')
    
    storage = FileStorage(
        input_file_path="../output/digital_economy_analysis/data/recent_digital_data.csv",
        cache_path="./cache"
    )
    
    workflow = Pipeline(storage=storage)
    
    analysis_operator = DataAnalysisOperator(
        columns_to_analyze=['数字经济规模(万亿元）', '数字产业化规模(万亿元)', '产业数字化规模(万亿元)'],
        time_column='指标',
        include_growth_analysis=True
    )
    
    chart_operator = ChartGeneratorOperator(
        chart_type='line',
        x_column='指标',
        y_columns=['数字产业化规模(万亿元)', '产业数字化规模(万亿元)'],
        title='近5年数字经济结构变化',
        output_dir='../output/digital_economy_analysis/charts'
    )
    
    report_operator = HTMLReportGeneratorOperator(
        report_title='近5年数字经济发展专项分析',
        output_file='../output/digital_economy_analysis/reports/recent_analysis_report.html'
    )
    
    workflow.add_operator(analysis_operator, "analysis")
    workflow.add_operator(chart_operator, "chart")
    workflow.add_operator(report_operator, "report")
    
    result = workflow.run(recent_df)
    print("✅ 自定义分析完成！")
    
    return result


if __name__ == "__main__":
    print("🚀 MaestroDataflow 数字经济数据分析示例")
    print("=" * 50)
    
    try:
        # 运行完整分析
        result1 = run_digital_economy_analysis()
        
        # 生成执行摘要
        result2 = run_executive_summary()
        
        # 演示自定义分析
        result3 = demonstrate_custom_analysis()
        
        print("\n🎉 所有分析任务完成！")
        print("\n📊 MaestroDataflow 现在支持:")
        print("✓ 全面的数据统计分析")
        print("✓ 多种图表类型生成")
        print("✓ 交互式仪表板")
        print("✓ 专业的HTML/PDF报告")
        print("✓ 灵活的工作流编排")
        print("✓ 自定义分析模板")
        
    except Exception as e:
        print(f"❌ 执行过程中出现错误: {e}")
        print("请确保已安装所需依赖: matplotlib, plotly, pandas")