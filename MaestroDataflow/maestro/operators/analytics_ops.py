"""
数据分析算子，支持统计分析和报告生成功能。
"""
from __future__ import annotations
from typing import Dict, Any, List, Optional, Union
import pandas as pd
import numpy as np
from datetime import datetime
import json

from maestro.core import OperatorABC
from maestro.utils.storage import MaestroStorage


class DataAnalysisOperator(OperatorABC):
    """数据统计分析算子，生成详细的数据分析报告"""
    
    def __init__(
        self,
        columns_to_analyze: Optional[List[str]] = None,
        include_growth_analysis: bool = True,
        time_column: Optional[str] = None,
        output_format: str = "dict"  # "dict", "json", "text"
    ):
        """
        初始化数据分析算子
        
        Args:
            columns_to_analyze: 要分析的列名列表，None表示分析所有数值列
            include_growth_analysis: 是否包含增长率分析
            time_column: 时间列名，用于时间序列分析
            output_format: 输出格式
        """
        self.columns_to_analyze = columns_to_analyze
        self.include_growth_analysis = include_growth_analysis
        self.time_column = time_column
        self.output_format = output_format

    def run(self, storage: MaestroStorage, **kwargs) -> Dict[str, Any]:
        """执行数据分析"""
        # 读取数据
        if hasattr(storage, 'operator_step') and storage.operator_step == -1:
            storage = storage.step()
        
        df = storage.read(output_type="dataframe")
        
        # 生成分析报告
        analysis_report = self._generate_analysis_report(df)
        
        # 保存结果
        if self.output_format == "json":
            result_data = json.dumps(analysis_report, ensure_ascii=False, indent=2)
        elif self.output_format == "text":
            result_data = self._format_text_report(analysis_report)
        else:
            result_data = analysis_report
        
        # 将分析结果添加到DataFrame中
        result_df = df.copy()
        result_df.attrs['analysis_report'] = analysis_report
        
        path = storage.write(result_df)
        
        return {
            "path": path,
            "analysis_report": analysis_report,
            "analyzed_columns": len(self.columns_to_analyze) if self.columns_to_analyze else len(df.select_dtypes(include=[np.number]).columns),
            "total_records": len(df)
        }

    def _generate_analysis_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """生成详细的数据分析报告"""
        report = {
            "report_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "data_overview": self._get_data_overview(df),
            "statistical_analysis": {},
            "data_quality": self._assess_data_quality(df)
        }
        
        # 确定要分析的列
        if self.columns_to_analyze:
            numeric_columns = [col for col in self.columns_to_analyze if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
        else:
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # 对每个数值列进行统计分析
        for column in numeric_columns:
            if column == self.time_column:
                continue
                
            column_analysis = self._analyze_column(df, column)
            report["statistical_analysis"][column] = column_analysis
        
        return report

    def _get_data_overview(self, df: pd.DataFrame) -> Dict[str, Any]:
        """获取数据概览信息"""
        time_range = "未知"
        if self.time_column and self.time_column in df.columns:
            min_time = df[self.time_column].min()
            max_time = df[self.time_column].max()
            time_range = f"{min_time}-{max_time}"
        
        return {
            "time_range": time_range,
            "total_records": len(df),
            "total_columns": len(df.columns),
            "column_names": df.columns.tolist()
        }

    def _analyze_column(self, df: pd.DataFrame, column: str) -> Dict[str, Any]:
        """分析单个列的统计信息"""
        series = df[column].dropna()
        
        if len(series) == 0:
            return {"error": "列中没有有效数据"}
        
        analysis = {
            "max_value": float(series.max()) if not pd.isna(series.max()) else None,
            "min_value": float(series.min()) if not pd.isna(series.min()) else None,
            "mean_value": float(series.mean()) if not pd.isna(series.mean()) else None,
            "std_value": float(series.std()) if not pd.isna(series.std()) else None,
            "median_value": float(series.median()) if not pd.isna(series.median()) else None
        }
        
        # 计算增长率（如果有时间序列数据）
        if self.include_growth_analysis and len(series) > 1:
            first_value = series.iloc[0]
            last_value = series.iloc[-1]
            
            if first_value != 0 and not pd.isna(first_value) and not pd.isna(last_value):
                total_growth_rate = ((last_value - first_value) / first_value) * 100
                annual_growth_rate = ((last_value / first_value) ** (1 / (len(series) - 1)) - 1) * 100
                
                analysis.update({
                    "total_growth_rate": round(float(total_growth_rate), 1),
                    "annual_growth_rate": round(float(annual_growth_rate), 1)
                })
        
        return analysis

    def _assess_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """评估数据质量"""
        return {
            "total_records": len(df),
            "total_columns": len(df.columns),
            "missing_values": int(df.isnull().sum().sum()),
            "duplicate_records": int(df.duplicated().sum())
        }

    def _format_text_report(self, report: Dict[str, Any]) -> str:
        """将分析报告格式化为文本"""
        text_lines = []
        text_lines.append("数据分析报告")
        text_lines.append("=" * 60)
        text_lines.append(f"报告生成时间: {report['report_time']}")
        text_lines.append("")
        
        # 数据概览
        overview = report['data_overview']
        text_lines.append("一、数据概览")
        text_lines.append("-" * 30)
        text_lines.append(f"数据时间跨度: {overview['time_range']}")
        text_lines.append(f"数据点数量: {overview['total_records']}个")
        text_lines.append(f"指标数量: {overview['total_columns']}个")
        text_lines.append(f"列名: {', '.join(overview['column_names'])}")
        text_lines.append("")
        
        # 统计分析
        text_lines.append("二、主要发现")
        text_lines.append("-" * 30)
        
        for column, analysis in report['statistical_analysis'].items():
            if 'error' in analysis:
                continue
                
            text_lines.append(f"{column}:")
            text_lines.append(f"  最大值: {analysis['max_value']}")
            text_lines.append(f"  最小值: {analysis['min_value']}")
            text_lines.append(f"  平均值: {analysis['mean_value']:.2f}")
            text_lines.append(f"  标准差: {analysis['std_value']:.2f}")
            
            if 'total_growth_rate' in analysis:
                text_lines.append(f"  总增长率: {analysis['total_growth_rate']}%")
                text_lines.append(f"  年均增长率: {analysis['annual_growth_rate']}%")
            
            text_lines.append("")
        
        # 数据质量
        quality = report['data_quality']
        text_lines.append("三、数据质量评估")
        text_lines.append("-" * 30)
        text_lines.append(f"总记录数: {quality['total_records']}")
        text_lines.append(f"总列数: {quality['total_columns']}")
        text_lines.append(f"缺失值总数: {quality['missing_values']}")
        text_lines.append(f"重复记录数: {quality['duplicate_records']}")
        
        return "\n".join(text_lines)


class DataSummaryOperator(OperatorABC):
    """数据摘要算子，生成简洁的数据摘要"""
    
    def __init__(self, group_by_column: Optional[str] = None):
        """
        初始化数据摘要算子
        
        Args:
            group_by_column: 分组列名
        """
        self.group_by_column = group_by_column

    def run(self, storage: MaestroStorage, **kwargs) -> Dict[str, Any]:
        """执行数据摘要"""
        if hasattr(storage, 'operator_step') and storage.operator_step == -1:
            storage = storage.step()
        
        df = storage.read(output_type="dataframe")
        
        if self.group_by_column and self.group_by_column in df.columns:
            # 按指定列分组摘要
            summary_df = df.groupby(self.group_by_column).describe()
        else:
            # 整体摘要
            summary_df = df.describe()
        
        path = storage.write(summary_df)
        
        return {
            "path": path,
            "summary_shape": summary_df.shape,
            "grouped_by": self.group_by_column
        }