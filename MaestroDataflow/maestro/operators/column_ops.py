"""
列名处理操作符 - 用于生成数据列名的意义和单位说明
"""
from __future__ import annotations
from typing import Dict, Any, List, Optional
import pandas as pd
import json

from maestro.core import OperatorABC
from maestro.utils.storage import MaestroStorage
from maestro.serving.llm_serving import APILLMServing, LocalLLMServing


class ColumnMeaningGeneratorOperator(OperatorABC):
    """
    使用AI生成数据列名的意义和单位说明
    基于process_column_name的prompt功能
    支持两种模式：
    1. 直接处理模式：直接分析列名生成意义
    2. 模板填充模式：从模板JSON读取并填充空的意义字段
    """
    
    def __init__(
        self, 
        dataset_description: str,
        max_columns_per_batch: int = 10,
        service: APILLMServing | LocalLLMServing | None = None,
        template_mode: bool = False,
        template_path: Optional[str] = None
    ):
        self.dataset_description = dataset_description
        self.max_columns_per_batch = max_columns_per_batch
        self.service = service
        self.template_mode = template_mode
        self.template_path = template_path
        
        # 更新后的专业prompt模板
        self.prompt_template = """你是一个拥有20年研究经验的顶尖数据分析专家，专门负责解释数据中的列名含义和单位，若列名中无单位，结合意义给出单位。

请根据列名，提供准确、专业的解释。要求：
1. 意义：详细说明该字段的含义、用途和计算方法
2. 单位：准确标注数据或意义中的计量单位（如：元、万元、%、个等）

请以JSON格式返回，格式如下：
{{
    "意义": "详细解释...",
    "单位": "单位名称"
}}

注意：
- 如果是百分比数据，单位写"%"
- 如果是年份数据，单位写"年"
- 如果是分数数据，单位写"分"
- 如果是金额数据，通常单位为"元"或"万元"
- 如果是数量数据，单位为"个"、"只"、"股"等
- 如果是比率数据，如增长率、回报率，单位写"%"，如果是倍率，单位写"倍"，若没有单位，写"没有单位"
- 解释要专业、准确、完整

需要解释的列名：
{column_names}

请直接返回JSON格式的结果，不要包含其他文字说明。"""

    def _generate_column_meanings(self, column_names: List[str], service) -> List[Dict[str, Any]]:
        """为一批列名生成意义解释"""
        column_names_str = "\n".join([f"- {name}" for name in column_names])
        
        prompt = self.prompt_template.format(
            column_names=column_names_str
        )
        
        try:
            response = service.generate(prompt)
            # 尝试解析JSON响应
            result = json.loads(response)
            
            # 处理新格式的JSON响应
            processed_results = []
            if isinstance(result, dict):
                # 单个列名的情况
                for i, name in enumerate(column_names):
                    processed_results.append({
                        "column_name": name,
                        "chinese_name": name,
                        "meaning": result.get("意义", "待人工补充说明"),
                        "unit": result.get("单位", "没有单位"),
                        "data_type": "未知",
                        "possible_values": "待分析"
                    })
            elif isinstance(result, list):
                # 多个列名的情况
                for i, (name, item) in enumerate(zip(column_names, result)):
                    if isinstance(item, dict):
                        processed_results.append({
                            "column_name": name,
                            "chinese_name": name,
                            "meaning": item.get("意义", "待人工补充说明"),
                            "unit": item.get("单位", "没有单位"),
                            "data_type": "未知",
                            "possible_values": "待分析"
                        })
                    else:
                        processed_results.append(self._create_fallback_meanings([name])[0])
            else:
                # 如果解析失败，创建基础结构
                return self._create_fallback_meanings(column_names)
                
            return processed_results
            
        except (json.JSONDecodeError, Exception) as e:
            print(f"警告：AI响应解析失败，使用默认格式: {e}")
            return self._create_fallback_meanings(column_names)
    
    def _create_fallback_meanings(self, column_names: List[str]) -> List[Dict[str, Any]]:
        """创建默认的列名解释结构"""
        return [
            {
                "column_name": name,
                "chinese_name": name,
                "meaning": "待人工补充说明",
                "unit": "没有单位",
                "data_type": "未知",
                "possible_values": "待分析"
            }
            for name in column_names
        ]
    
    def run(self, storage: MaestroStorage, **kwargs) -> Dict[str, Any]:
        """执行列名意义生成"""
        # 对于FileStorage，需要先调用step()来初始化处理步骤
        if hasattr(storage, 'operator_step') and storage.operator_step == -1:
            storage = storage.step()
        
        # 检查是否为模板模式
        if self.template_mode:
            return self._run_template_mode(storage, **kwargs)
        else:
            return self._run_direct_mode(storage, **kwargs)
    
    def _run_direct_mode(self, storage: MaestroStorage, **kwargs) -> Dict[str, Any]:
        """直接处理模式：直接分析列名生成意义"""
        # 读取数据获取列名
        df = storage.read(output_type="dataframe")
        column_names = df.columns.tolist()
        
        # 获取LLM服务
        service = self.service or kwargs.get("llm_service")
        if service is None:
            raise ValueError("ColumnMeaningGeneratorOperator 需要传入 llm_service 或在初始化时提供 service")
        
        # 分批处理列名
        all_meanings = []
        for i in range(0, len(column_names), self.max_columns_per_batch):
            batch_columns = column_names[i:i + self.max_columns_per_batch]
            batch_meanings = self._generate_column_meanings(batch_columns, service)
            all_meanings.extend(batch_meanings)
        
        # 创建列名意义字典
        column_meanings = {
            "dataset_description": self.dataset_description,
            "total_columns": len(column_names),
            "columns": all_meanings,
            "generated_by": "MaestroDataflow ColumnMeaningGeneratorOperator"
        }
        
        # 保存结果
        output_path = storage.write(column_meanings)
        
        return {
            "path": output_path,
            "total_columns": len(column_names),
            "processed_columns": len(all_meanings),
            "column_meanings": column_meanings
        }
    
    def _run_template_mode(self, storage: MaestroStorage, **kwargs) -> Dict[str, Any]:
        """模板填充模式：从模板JSON读取并填充空的意义字段"""
        # 获取模板路径
        template_path = kwargs.get('template_path', self.template_path)
        if not template_path:
            raise ValueError("模板模式需要提供 template_path")
        
        # 获取LLM服务
        service = self.service or kwargs.get("llm_service")
        if service is None:
            raise ValueError("ColumnMeaningGeneratorOperator 需要传入 llm_service 或在初始化时提供 service")
        
        # 读取模板
        template = self._load_template(template_path)
        
        # 获取需要填充的列名
        empty_columns = self._get_empty_columns(template)
        
        if not empty_columns:
            print("所有列名都已有意义说明，无需填充")
            # 转换为标准格式
            all_meanings = self._convert_template_to_meanings(template)
        else:
            # 分批填充意义
            filled_template = self._fill_template_meanings(template, empty_columns, service)
            # 转换为标准格式
            all_meanings = self._convert_template_to_meanings(filled_template)
        
        # 创建列名意义字典
        column_meanings = {
            "dataset_description": self.dataset_description,
            "total_columns": len(template),
            "columns": all_meanings,
            "generated_by": "MaestroDataflow ColumnMeaningGeneratorOperator (Template Mode)"
        }
        
        # 保存结果
        output_path = storage.write(column_meanings)
        
        return {
            "path": output_path,
            "total_columns": len(template),
            "processed_columns": len(empty_columns),
            "filled_columns": len(empty_columns),
            "column_meanings": column_meanings
        }
    
    def _load_template(self, template_path: str) -> Dict[str, Any]:
        """加载模板文件"""
        with open(template_path, 'r', encoding='utf-8') as f:
            template = json.load(f)
        return template
    
    def _get_empty_columns(self, template: Dict[str, Any]) -> List[str]:
        """获取需要填充意义的列名"""
        empty_columns = []
        for column, info in template.items():
            if not info.get('意义', '').strip():
                empty_columns.append(column)
        return empty_columns
    
    def _fill_template_meanings(self, template: Dict[str, Any], empty_columns: List[str], service) -> Dict[str, Any]:
        """批量填充模板中的列名意义"""
        filled_template = template.copy()
        
        # 分批处理
        for i in range(0, len(empty_columns), self.max_columns_per_batch):
            batch_columns = empty_columns[i:i + self.max_columns_per_batch]
            
            try:
                # 生成批次的意义
                batch_meanings = self._generate_column_meanings(batch_columns, service)
                
                # 更新模板
                for j, column in enumerate(batch_columns):
                    if j < len(batch_meanings):
                        filled_template[column]['意义'] = batch_meanings[j].get('meaning', '待人工补充说明')
                        filled_template[column]['单位'] = batch_meanings[j].get('unit', '没有单位')
                
            except Exception as e:
                print(f"处理批次时出错: {str(e)}")
                # 使用默认值填充
                for column in batch_columns:
                    filled_template[column]['意义'] = '待人工补充说明'
                    filled_template[column]['单位'] = '没有单位'
        
        return filled_template
    
    def _convert_template_to_meanings(self, template: Dict[str, Any]) -> List[Dict[str, Any]]:
        """将模板格式转换为标准的meanings格式"""
        meanings = []
        for column_name, info in template.items():
            meaning = {
                "column_name": column_name,
                "chinese_name": column_name,
                "meaning": info.get('意义', '待人工补充说明'),
                "unit": info.get('单位', '没有单位'),
                "data_type": info.get('数据类型', '未知'),
                "possible_values": info.get('可能值', '待分析')
            }
            meanings.append(meaning)
        return meanings


class ColumnMetadataExtractorOperator(OperatorABC):
    """
    从数据中提取列的元数据信息（统计信息、数据类型等）
    """
    
    def __init__(self, include_sample_values: bool = True, max_sample_values: int = 10):
        self.include_sample_values = include_sample_values
        self.max_sample_values = max_sample_values
    
    def run(self, storage: MaestroStorage, **kwargs) -> Dict[str, Any]:
        """提取列的元数据信息"""
        if hasattr(storage, 'operator_step') and storage.operator_step == -1:
            storage = storage.step()
            
        df = storage.read(output_type="dataframe")
        
        metadata = {
            "dataset_info": {
                "total_rows": len(df),
                "total_columns": len(df.columns),
                "memory_usage": df.memory_usage(deep=True).sum()
            },
            "columns": []
        }
        
        for col in df.columns:
            col_info = {
                "column_name": col,
                "data_type": str(df[col].dtype),
                "non_null_count": df[col].count(),
                "null_count": df[col].isnull().sum(),
                "null_percentage": (df[col].isnull().sum() / len(df)) * 100
            }
            
            # 数值型列的统计信息
            if df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                col_info.update({
                    "min_value": df[col].min(),
                    "max_value": df[col].max(),
                    "mean_value": df[col].mean(),
                    "std_value": df[col].std(),
                    "unique_count": df[col].nunique()
                })
            
            # 分类型列的信息
            elif df[col].dtype == 'object' or df[col].dtype.name == 'category':
                col_info.update({
                    "unique_count": df[col].nunique(),
                    "most_frequent": df[col].mode().iloc[0] if not df[col].mode().empty else None
                })
                
                if self.include_sample_values:
                    sample_values = df[col].dropna().unique()[:self.max_sample_values].tolist()
                    col_info["sample_values"] = sample_values
            
            metadata["columns"].append(col_info)
        
        # 保存元数据
        output_path = storage.write(metadata)
        
        return {
            "path": output_path,
            "metadata": metadata
        }