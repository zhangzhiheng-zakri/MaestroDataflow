"""
Intelligent data processing operators for MaestroDataflow.
Supports automatic data cleaning, annotation, feature engineering, and more.
"""

import re
import json
from typing import Dict, Any, List, Optional, Union, Callable, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

from ...core.operator import OperatorABC
from maestro.core.prompt import PromptABC, StandardPrompt, DIYPromptABC
from ...serving.llm_serving import LLMServingABC
from ...utils.storage import MaestroStorage


class AutoDataCleaner(OperatorABC):
    """
    自动数据清洗操作符。
    使用AI智能识别和处理数据质量问题。
    """

    ALLOWED_PROMPTS = (StandardPrompt, DIYPromptABC)

    def __init__(
        self,
        llm_serving: LLMServingABC,
        cleaning_strategies: Optional[List[str]] = None,
        confidence_threshold: float = 0.8,
        output_column_suffix: str = "_cleaned",
        generate_report: bool = True
    ):
        """
        初始化自动数据清洗操作符。

        Args:
            llm_serving: LLM服务实例
            cleaning_strategies: 清洗策略列表
            confidence_threshold: 置信度阈值
            output_column_suffix: 输出列后缀
            generate_report: 是否生成清洗报告
        """
        super().__init__()
        self.llm_serving = llm_serving
        self.cleaning_strategies = cleaning_strategies or [
            "remove_duplicates", "fix_typos", "standardize_format",
            "handle_missing", "detect_outliers"
        ]
        self.confidence_threshold = confidence_threshold
        self.output_column_suffix = output_column_suffix
        self.generate_report = generate_report

        # 设置清洗提示词
        self.cleaning_prompt = DIYPromptABC(
            "Analyze the following data and suggest cleaning operations:\n"
            "Data sample: {data_sample}\n"
            "Column info: {column_info}\n"
            "Please identify data quality issues and suggest specific cleaning actions. "
            "Return your response in JSON format with 'issues' and 'actions' fields."
        )

    def run(self, storage: MaestroStorage, **kwargs) -> Dict[str, Any]:
        """
        执行自动数据清洗。

        Args:
            storage: 存储实例
            **kwargs: 其他参数

        Returns:
            Dict[str, Any]: 清洗结果
        """
        self.log_operation_start(
            operation="auto_data_cleaning",
            strategies=self.cleaning_strategies
        )

        try:
            # 读取数据
            if hasattr(storage, 'operator_step') and storage.operator_step == -1:
                storage = storage.step()
            
            data = storage.read(output_type="dataframe")
            print(f"AutoDataCleaner读取的数据类型: {type(data)}")
            if hasattr(data, 'shape'):
                print(f"AutoDataCleaner读取的数据形状: {data.shape}")
            
            if isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                df = data.copy()

            print(f"转换后的DataFrame形状: {df.shape}")
            print(f"DataFrame列: {df.columns.tolist()}")

            # 检查DataFrame是否为空
            if df.empty:
                self.logger.warning("Received empty DataFrame, returning empty result")
                return {
                    "status": "success",
                    "original_shape": (0, 0),
                    "final_shape": (0, 0),
                    "cleaning_report": {
                        "original_shape": (0, 0),
                        "final_shape": (0, 0),
                        "operations_performed": [],
                        "issues_found": ["Empty DataFrame received"],
                        "statistics": {}
                    }
                }

            original_shape = df.shape
            cleaning_report = {
                "original_shape": original_shape,
                "operations_performed": [],
                "issues_found": [],
                "statistics": {}
            }

            # 对每列进行智能清洗
            for column in df.columns:
                if df[column].dtype == 'object':  # 主要处理文本列
                    column_report = self._clean_column_intelligently(df, column)
                    cleaning_report["operations_performed"].extend(column_report["operations"])
                    cleaning_report["issues_found"].extend(column_report["issues"])

            # 执行基础清洗策略
            if "remove_duplicates" in self.cleaning_strategies:
                duplicates_removed = self._remove_duplicates(df)
                cleaning_report["operations_performed"].append({
                    "operation": "remove_duplicates",
                    "rows_removed": duplicates_removed
                })

            if "handle_missing" in self.cleaning_strategies:
                missing_handled = self._handle_missing_values(df)
                cleaning_report["operations_performed"].append({
                    "operation": "handle_missing",
                    "details": missing_handled
                })

            if "detect_outliers" in self.cleaning_strategies:
                outliers_detected = self._detect_outliers(df)
                cleaning_report["operations_performed"].append({
                    "operation": "detect_outliers",
                    "outliers_found": outliers_detected
                })

            # 更新统计信息
            cleaning_report["final_shape"] = df.shape
            cleaning_report["statistics"] = {
                "rows_processed": original_shape[0],
                "columns_processed": original_shape[1],
                "rows_remaining": df.shape[0],
                "data_quality_score": self._calculate_quality_score(df)
            }

            # 保存清洗后的数据
            storage.write(df)

            # 保存清洗报告
            if self.generate_report:
                storage.write([cleaning_report])

            self.log_operation_end({
                "operation": "auto_data_cleaning",
                "original_rows": original_shape[0],
                "final_rows": df.shape[0],
                "operations_count": len(cleaning_report["operations_performed"])
            })

            return {
                "status": "success",
                "original_shape": original_shape,
                "final_shape": df.shape,
                "cleaning_report": cleaning_report
            }

        except Exception as e:
            return self.handle_error("auto_data_cleaning", e)

    def _clean_column_intelligently(self, df: pd.DataFrame, column: str) -> Dict[str, Any]:
        """
        智能清洗单个列。

        Args:
            df: 数据框
            column: 列名

        Returns:
            Dict[str, Any]: 清洗报告
        """
        report = {"operations": [], "issues": []}

        try:
            # 获取列的样本数据
            sample_data = df[column].dropna().head(10).tolist()
            column_info = {
                "name": column,
                "dtype": str(df[column].dtype),
                "null_count": df[column].isnull().sum(),
                "unique_count": df[column].nunique(),
                "sample_values": sample_data[:5]
            }

            # 使用LLM分析数据质量
            analysis_prompt = self.cleaning_prompt.format(
                data_sample=sample_data,
                column_info=json.dumps(column_info, ensure_ascii=False)
            )

            response = self.llm_serving.generate(
                prompt=analysis_prompt,
                max_tokens=500,
                temperature=0.1
            )

            # 解析LLM响应
            try:
                analysis_result = json.loads(response)
                issues = analysis_result.get("issues", [])
                actions = analysis_result.get("actions", [])

                report["issues"].extend(issues)

                # 执行建议的清洗操作
                for action in actions:
                    if action.get("type") == "standardize_format":
                        self._standardize_format(df, column, action.get("pattern"))
                        report["operations"].append({
                            "column": column,
                            "operation": "standardize_format",
                            "pattern": action.get("pattern")
                        })
                    elif action.get("type") == "fix_typos":
                        corrections = self._fix_typos(df, column, action.get("corrections", {}))
                        report["operations"].append({
                            "column": column,
                            "operation": "fix_typos",
                            "corrections_made": corrections
                        })

            except json.JSONDecodeError:
                self.logger.warning(f"Failed to parse LLM response for column {column}")

        except Exception as e:
            self.logger.error(f"Failed to clean column {column}: {e}")
            report["issues"].append(f"Error processing column {column}: {str(e)}")

        return report

    def _remove_duplicates(self, df: pd.DataFrame) -> int:
        """移除重复行。"""
        original_count = len(df)
        df.drop_duplicates(inplace=True)
        df.reset_index(drop=True, inplace=True)  # 重置索引
        return original_count - len(df)

    def _handle_missing_values(self, df: pd.DataFrame) -> Dict[str, Any]:
        """处理缺失值。"""
        missing_info = {}

        for column in df.columns:
            missing_count = df[column].isnull().sum()
            if missing_count > 0:
                if df[column].dtype in ['int64', 'float64']:
                    # 数值列用均值填充
                    df[column].fillna(df[column].mean(), inplace=True)
                    missing_info[column] = {"method": "mean", "count": missing_count}
                else:
                    # 文本列用众数填充
                    mode_value = df[column].mode()
                    if len(mode_value) > 0:
                        df[column].fillna(mode_value[0], inplace=True)
                        missing_info[column] = {"method": "mode", "count": missing_count}

        return missing_info

    def _detect_outliers(self, df: pd.DataFrame) -> Dict[str, Any]:
        """检测异常值。"""
        outliers_info = {}

        for column in df.select_dtypes(include=[np.number]).columns:
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]

            if len(outliers) > 0:
                outliers_info[column] = {
                    "count": len(outliers),
                    "percentage": len(outliers) / len(df) * 100,
                    "bounds": {"lower": lower_bound, "upper": upper_bound}
                }

        return outliers_info

    def _standardize_format(self, df: pd.DataFrame, column: str, pattern: Optional[str]):
        """标准化格式。"""
        if pattern and column in df.columns:
            # 简单的格式标准化示例
            if pattern == "lowercase":
                df[column] = df[column].astype(str).str.lower()
            elif pattern == "uppercase":
                df[column] = df[column].astype(str).str.upper()
            elif pattern == "title_case":
                df[column] = df[column].astype(str).str.title()

    def _fix_typos(self, df: pd.DataFrame, column: str, corrections: Dict[str, str]) -> int:
        """修复拼写错误。"""
        corrections_made = 0
        for wrong, correct in corrections.items():
            mask = df[column].astype(str).str.contains(wrong, case=False, na=False)
            corrections_made += mask.sum()
            df.loc[mask, column] = df.loc[mask, column].astype(str).str.replace(wrong, correct, case=False)
        return corrections_made

    def _calculate_quality_score(self, df: pd.DataFrame) -> float:
        """计算数据质量分数。"""
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = df.isnull().sum().sum()

        # 简单的质量分数计算
        completeness_score = 1 - (missing_cells / total_cells)

        # 可以添加更多质量维度的计算
        return round(completeness_score * 100, 2)


class SmartAnnotator(OperatorABC):
    """
    智能标注操作符。
    使用AI自动为数据添加标签和注释。
    """

    ALLOWED_PROMPTS = (StandardPrompt, DIYPromptABC)

    def __init__(
        self,
        llm_serving: LLMServingABC,
        annotation_type: str = "classification",
        target_column: str = "text",
        output_column: str = "annotation",
        categories: Optional[List[str]] = None,
        prompt: Optional[PromptABC] = None,
        batch_size: int = 10
    ):
        """
        初始化智能标注操作符。

        Args:
            llm_serving: LLM服务实例
            annotation_type: 标注类型 ("classification", "sentiment", "entity", "custom")
            target_column: 目标列名
            output_column: 输出列名
            categories: 分类类别（用于分类任务）
            prompt: 自定义提示词
            batch_size: 批处理大小
        """
        super().__init__()
        self.llm_serving = llm_serving
        self.annotation_type = annotation_type
        self.target_column = target_column
        self.output_column = output_column
        self.categories = categories
        self.batch_size = batch_size

        # 设置默认提示词
        if prompt is None:
            if annotation_type == "classification":
                categories_str = ", ".join(categories) if categories else "positive, negative, neutral"
                self.prompt = DIYPromptABC(
                    f"Classify the following text into one of these categories: {categories_str}\n"
                    "Text: {text}\n"
                    "Category:"
                )
            elif annotation_type == "sentiment":
                self.prompt = StandardPrompt("classify", categories="positive, negative, neutral")
            elif annotation_type == "entity":
                self.prompt = DIYPromptABC(
                    "Extract named entities from the following text:\n"
                    "Text: {text}\n"
                    "Entities (format: entity_type:entity_name):"
                )
            else:
                self.prompt = DIYPromptABC("Analyze the following text: {text}")
        else:
            self.prompt = prompt

        self.validate_prompts([self.prompt])

    def run(self, storage: MaestroStorage, **kwargs) -> Dict[str, Any]:
        """
        执行智能标注。

        Args:
            storage: 存储实例
            **kwargs: 其他参数

        Returns:
            Dict[str, Any]: 标注结果
        """
        self.log_operation_start(
            annotation_type=self.annotation_type,
            target_column=self.target_column
        )

        try:
            # 读取数据
            data = storage.read(output_type="dataframe")
            if isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                df = data.copy()

            if self.target_column not in df.columns:
                raise ValueError(f"Target column '{self.target_column}' not found in data")

            # 批量处理标注
            annotations = []
            total_rows = len(df)

            for i in range(0, total_rows, self.batch_size):
                batch_end = min(i + self.batch_size, total_rows)
                batch_df = df.iloc[i:batch_end]

                batch_annotations = self._annotate_batch(batch_df)
                annotations.extend(batch_annotations)

                self.logger.info(f"Processed {batch_end}/{total_rows} rows")

            # 添加标注结果到数据框
            df[self.output_column] = annotations

            # 生成标注统计
            annotation_stats = self._generate_annotation_stats(annotations)

            # 保存结果
            storage.write(df)

            self.log_operation_end({
                "operation": "smart_annotation",
                "annotated_rows": len(annotations),
                "annotation_stats": annotation_stats
            })

            return {
                "status": "success",
                "annotated_count": len(annotations),
                "annotation_stats": annotation_stats
            }

        except Exception as e:
            return self.handle_error("smart_annotation", e)

    def _annotate_batch(self, batch_df: pd.DataFrame) -> List[str]:
        """
        批量标注数据。

        Args:
            batch_df: 批次数据框

        Returns:
            List[str]: 标注结果列表
        """
        annotations = []

        for _, row in batch_df.iterrows():
            try:
                text = str(row[self.target_column])

                # 格式化提示词
                formatted_prompt = self.prompt.format(text=text)

                # 调用LLM进行标注
                response = self.llm_serving.generate(
                    prompt=formatted_prompt,
                    max_tokens=100,
                    temperature=0.1
                )

                annotation = response.strip()
                annotations.append(annotation)

            except Exception as e:
                self.logger.error(f"Failed to annotate text: {e}")
                annotations.append("ERROR")

        return annotations

    def _generate_annotation_stats(self, annotations: List[str]) -> Dict[str, Any]:
        """
        生成标注统计信息。

        Args:
            annotations: 标注结果列表

        Returns:
            Dict[str, Any]: 统计信息
        """
        annotation_counts = {}
        for annotation in annotations:
            annotation_counts[annotation] = annotation_counts.get(annotation, 0) + 1

        return {
            "total_annotations": len(annotations),
            "unique_annotations": len(annotation_counts),
            "annotation_distribution": annotation_counts,
            "most_common": max(annotation_counts.items(), key=lambda x: x[1]) if annotation_counts else None
        }


class FeatureEngineer(OperatorABC):
    """
    智能特征工程操作符。
    使用AI自动生成和选择特征。
    """

    ALLOWED_PROMPTS = (StandardPrompt, DIYPromptABC)

    def __init__(
        self,
        llm_serving: Optional[LLMServingABC] = None,
        feature_types: Optional[List[str]] = None,
        target_column: Optional[str] = None,
        max_features: int = 50,
        feature_selection_method: str = "correlation"
    ):
        """
        初始化特征工程操作符。

        Args:
            llm_serving: LLM服务实例（用于智能特征建议）
            feature_types: 特征类型列表
            target_column: 目标列（用于监督特征选择）
            max_features: 最大特征数量
            feature_selection_method: 特征选择方法
        """
        super().__init__()
        self.llm_serving = llm_serving
        self.feature_types = feature_types or [
            "statistical", "temporal", "text", "categorical", "interaction"
        ]
        self.target_column = target_column
        self.max_features = max_features
        self.feature_selection_method = feature_selection_method

    def run(self, storage: MaestroStorage, **kwargs) -> Dict[str, Any]:
        """
        执行特征工程。

        Args:
            storage: 存储实例
            **kwargs: 其他参数

        Returns:
            Dict[str, Any]: 特征工程结果
        """
        self.log_operation_start(
            feature_types=self.feature_types,
            max_features=self.max_features
        )

        try:
            # 读取数据
            data = storage.read(output_type="dataframe")
            if isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                df = data.copy()

            original_columns = df.columns.tolist()
            feature_report = {
                "original_features": len(original_columns),
                "generated_features": [],
                "selected_features": [],
                "feature_importance": {}
            }

            # 生成不同类型的特征
            if "statistical" in self.feature_types:
                self._generate_statistical_features(df, feature_report)

            if "temporal" in self.feature_types:
                self._generate_temporal_features(df, feature_report)

            if "text" in self.feature_types:
                self._generate_text_features(df, feature_report)

            if "categorical" in self.feature_types:
                self._generate_categorical_features(df, feature_report)

            if "interaction" in self.feature_types:
                self._generate_interaction_features(df, feature_report)

            # 特征选择
            if len(df.columns) > self.max_features:
                selected_features = self._select_features(df, feature_report)
                df = df[selected_features]

            # 更新报告
            feature_report["final_features"] = len(df.columns)
            feature_report["feature_names"] = df.columns.tolist()

            # 保存结果
            storage.write(df)

            # 保存特征报告
            storage.write([feature_report])

            self.log_operation_end({
                "operation": "feature_engineering",
                "original_features": len(original_columns),
                "final_features": len(df.columns),
                "generated_count": len(feature_report["generated_features"])
            })

            return {
                "status": "success",
                "original_feature_count": len(original_columns),
                "final_feature_count": len(df.columns),
                "feature_report": feature_report
            }

        except Exception as e:
            return self.handle_error("feature_engineering", e)

    def _generate_statistical_features(self, df: pd.DataFrame, report: Dict[str, Any]):
        """生成统计特征。"""
        numeric_columns = df.select_dtypes(include=[np.number]).columns

        for col in numeric_columns:
            # 滚动统计特征
            if len(df) > 5:
                df[f"{col}_rolling_mean_5"] = df[col].rolling(window=5, min_periods=1).mean()
                df[f"{col}_rolling_std_5"] = df[col].rolling(window=5, min_periods=1).std()
                report["generated_features"].extend([f"{col}_rolling_mean_5", f"{col}_rolling_std_5"])

            # 分位数特征
            df[f"{col}_percentile_rank"] = df[col].rank(pct=True)
            report["generated_features"].append(f"{col}_percentile_rank")

    def _generate_temporal_features(self, df: pd.DataFrame, report: Dict[str, Any]):
        """生成时间特征。"""
        date_columns = df.select_dtypes(include=['datetime64']).columns

        for col in date_columns:
            df[f"{col}_year"] = df[col].dt.year
            df[f"{col}_month"] = df[col].dt.month
            df[f"{col}_day"] = df[col].dt.day
            df[f"{col}_weekday"] = df[col].dt.weekday
            df[f"{col}_hour"] = df[col].dt.hour

            generated_features = [
                f"{col}_year", f"{col}_month", f"{col}_day",
                f"{col}_weekday", f"{col}_hour"
            ]
            report["generated_features"].extend(generated_features)

    def _generate_text_features(self, df: pd.DataFrame, report: Dict[str, Any]):
        """生成文本特征。"""
        text_columns = df.select_dtypes(include=['object']).columns

        for col in text_columns:
            # 基础文本特征
            df[f"{col}_length"] = df[col].astype(str).str.len()
            df[f"{col}_word_count"] = df[col].astype(str).str.split().str.len()
            df[f"{col}_char_count"] = df[col].astype(str).str.replace(' ', '').str.len()

            generated_features = [
                f"{col}_length", f"{col}_word_count", f"{col}_char_count"
            ]
            report["generated_features"].extend(generated_features)

    def _generate_categorical_features(self, df: pd.DataFrame, report: Dict[str, Any]):
        """生成分类特征。"""
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns

        for col in categorical_columns:
            # 频次编码
            value_counts = df[col].value_counts()
            df[f"{col}_frequency"] = df[col].map(value_counts)

            # 标签编码
            unique_values = df[col].unique()
            label_map = {val: idx for idx, val in enumerate(unique_values)}
            df[f"{col}_label_encoded"] = df[col].map(label_map)

            generated_features = [f"{col}_frequency", f"{col}_label_encoded"]
            report["generated_features"].extend(generated_features)

    def _generate_interaction_features(self, df: pd.DataFrame, report: Dict[str, Any]):
        """生成交互特征。"""
        numeric_columns = df.select_dtypes(include=[np.number]).columns

        # 限制交互特征数量以避免维度爆炸
        if len(numeric_columns) > 1:
            for i, col1 in enumerate(numeric_columns[:5]):  # 限制前5列
                for col2 in numeric_columns[i+1:6]:  # 避免过多组合
                    # 乘积特征
                    df[f"{col1}_x_{col2}"] = df[col1] * df[col2]
                    # 比值特征（避免除零）
                    df[f"{col1}_div_{col2}"] = df[col1] / (df[col2] + 1e-8)

                    generated_features = [f"{col1}_x_{col2}", f"{col1}_div_{col2}"]
                    report["generated_features"].extend(generated_features)

    def _select_features(self, df: pd.DataFrame, report: Dict[str, Any]) -> List[str]:
        """
        特征选择。

        Args:
            df: 数据框
            report: 特征报告

        Returns:
            List[str]: 选择的特征列表
        """
        if self.feature_selection_method == "correlation" and self.target_column:
            if self.target_column in df.columns:
                # 基于相关性的特征选择
                numeric_df = df.select_dtypes(include=[np.number])
                if self.target_column in numeric_df.columns:
                    correlations = numeric_df.corr()[self.target_column].abs().sort_values(ascending=False)
                    selected_features = correlations.head(self.max_features).index.tolist()

                    # 添加非数值列
                    non_numeric_cols = [col for col in df.columns if col not in numeric_df.columns]
                    selected_features.extend(non_numeric_cols[:self.max_features - len(selected_features)])

                    report["selected_features"] = selected_features
                    report["feature_importance"] = correlations.head(self.max_features).to_dict()

                    return selected_features

        # 默认选择前N个特征
        selected_features = df.columns.tolist()[:self.max_features]
        report["selected_features"] = selected_features

        return selected_features