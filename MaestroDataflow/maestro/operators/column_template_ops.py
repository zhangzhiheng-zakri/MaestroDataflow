"""
列名模板生成器操作符
用于生成包含空意义字段的JSON模板，为后续的大模型分析做准备
"""

import json
import os
from typing import Dict, List, Any, Optional
import pandas as pd

from maestro.core.operator import OperatorABC
from maestro.utils.storage import FileStorage


class ColumnTemplateGeneratorOperator(OperatorABC):
    """
    列名模板生成器操作符
    
    功能：
    1. 从CSV文件中提取列名
    2. 生成包含空意义字段的JSON模板
    3. 支持自定义模板格式
    """
    
    def __init__(self, 
                 storage: FileStorage,
                 template_format: str = "standard",
                 output_filename: str = "column_template.json"):
        """
        初始化列名模板生成器
        
        Args:
            storage: 文件存储对象
            template_format: 模板格式，"standard"或"detailed"
            output_filename: 输出文件名
        """
        super().__init__()  # 不传递storage参数
        self.storage = storage
        self.template_format = template_format
        self.output_filename = output_filename
    
    def run(self, storage: FileStorage, **kwargs) -> Dict[str, Any]:
        """
        执行列名模板生成操作
        
        Args:
            storage: 存储对象
            **kwargs: 额外参数，可包含data参数
        
        Returns:
            操作结果字典
        """
        # 从kwargs中获取数据，如果没有则从storage读取
        data = kwargs.get('data')
        if data is None:
            # 从storage读取数据
            data = pd.read_csv(storage.input_file_path)
        
        return self.execute(data, **kwargs)
    
    def execute(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        执行列名模板生成
        
        Args:
            data: 输入数据框
            **kwargs: 其他参数
                - custom_columns: 自定义列名列表（可选）
        
        Returns:
            包含模板路径和列名信息的字典
        """
        try:
            # 获取列名
            custom_columns = kwargs.get('custom_columns', None)
            if custom_columns:
                columns = custom_columns
            else:
                columns = list(data.columns)
            
            # 生成模板
            template = self._generate_template(columns)
            
            # 保存模板
            template_path = self._save_template(template)
            
            return {
                'template_path': template_path,
                'total_columns': len(columns),
                'columns': columns,
                'template_format': self.template_format,
                'template': template
            }
            
        except Exception as e:
            self.logger.error(f"列名模板生成失败: {str(e)}")
            raise
    
    def _generate_template(self, columns: List[str]) -> Dict[str, Any]:
        """
        生成列名模板
        
        Args:
            columns: 列名列表
        
        Returns:
            模板字典
        """
        template = {}
        
        for column in columns:
            if self.template_format == "standard":
                template[column] = {
                    "意义": "",
                    "单位": "没有单位"
                }
            elif self.template_format == "detailed":
                template[column] = {
                    "意义": "",
                    "单位": "没有单位",
                    "数据类型": "",
                    "可能值": "",
                    "计算方法": "",
                    "使用场景": ""
                }
        
        return template
    
    def _save_template(self, template: Dict[str, Any]) -> str:
        """
        保存模板到文件
        
        Args:
            template: 模板字典
        
        Returns:
            保存的文件路径
        """
        # 确保输出目录存在
        output_dir = os.path.join(self.storage.cache_path, "templates")
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存文件
        template_path = os.path.join(output_dir, self.output_filename)
        with open(template_path, 'w', encoding='utf-8') as f:
            json.dump(template, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"列名模板已保存到: {template_path}")
        return template_path


class ColumnMeaningFillerOperator(OperatorABC):
    """
    列名意义填充器操作符
    
    功能：
    1. 读取模板JSON文件
    2. 使用大模型填充空的意义字段
    3. 生成完整的列名解释JSON
    """
    
    def __init__(self, 
                 storage: FileStorage,
                 llm_service,
                 template_path: Optional[str] = None,
                 output_filename: str = "column_meanings_filled.json",
                 batch_size: int = 5):
        """
        初始化列名意义填充器
        
        Args:
            storage: 文件存储对象
            llm_service: LLM服务对象
            template_path: 模板文件路径（可选）
            output_filename: 输出文件名
            batch_size: 批处理大小
        """
        super().__init__()  # 不传递storage参数
        self.storage = storage
        self.llm_service = llm_service
        self.template_path = template_path
        self.output_filename = output_filename
        self.batch_size = batch_size
        
        # 使用更新后的prompt模板
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

需要解释的列名：{column_names}

请直接返回JSON格式的结果，不要包含其他文字说明。"""
    
    def run(self, storage: FileStorage, **kwargs) -> Dict[str, Any]:
        """
        执行列名意义填充操作
        
        Args:
            storage: 存储对象
            **kwargs: 额外参数，可包含template_path等
        
        Returns:
            操作结果字典
        """
        # 从kwargs中获取模板路径
        template_path = kwargs.get('template_path', self.template_path)
        data = kwargs.get('data')
        
        return self.execute(data, template_path=template_path, **kwargs)
    
    def execute(self, data: Optional[pd.DataFrame] = None, **kwargs) -> Dict[str, Any]:
        """
        执行列名意义填充
        
        Args:
            data: 输入数据框（可选）
            **kwargs: 其他参数
                - template_path: 模板文件路径
        
        Returns:
            包含填充结果的字典
        """
        try:
            # 获取模板路径
            template_path = kwargs.get('template_path', self.template_path)
            if not template_path:
                raise ValueError("必须提供模板文件路径")
            
            # 读取模板
            template = self._load_template(template_path)
            
            # 获取需要填充的列名
            empty_columns = self._get_empty_columns(template)
            
            if not empty_columns:
                self.logger.info("所有列名都已有意义说明，无需填充")
                return {
                    'filled_path': template_path,
                    'total_columns': len(template),
                    'filled_columns': 0,
                    'template': template
                }
            
            # 批量填充意义
            filled_template = self._fill_meanings(template, empty_columns)
            
            # 保存填充后的结果
            filled_path = self._save_filled_template(filled_template)
            
            return {
                'filled_path': filled_path,
                'total_columns': len(template),
                'filled_columns': len(empty_columns),
                'empty_columns': empty_columns,
                'template': filled_template
            }
            
        except Exception as e:
            self.logger.error(f"列名意义填充失败: {str(e)}")
            raise
    
    def _load_template(self, template_path: str) -> Dict[str, Any]:
        """
        加载模板文件
        
        Args:
            template_path: 模板文件路径
        
        Returns:
            模板字典
        """
        with open(template_path, 'r', encoding='utf-8') as f:
            template = json.load(f)
        
        self.logger.info(f"已加载模板文件: {template_path}")
        return template
    
    def _get_empty_columns(self, template: Dict[str, Any]) -> List[str]:
        """
        获取需要填充意义的列名
        
        Args:
            template: 模板字典
        
        Returns:
            需要填充的列名列表
        """
        empty_columns = []
        for column, info in template.items():
            if not info.get('意义', '').strip():
                empty_columns.append(column)
        
        return empty_columns
    
    def _fill_meanings(self, template: Dict[str, Any], empty_columns: List[str]) -> Dict[str, Any]:
        """
        批量填充列名意义
        
        Args:
            template: 原始模板
            empty_columns: 需要填充的列名列表
        
        Returns:
            填充后的模板
        """
        filled_template = template.copy()
        
        # 分批处理
        for i in range(0, len(empty_columns), self.batch_size):
            batch_columns = empty_columns[i:i + self.batch_size]
            
            try:
                # 生成prompt
                prompt = self.prompt_template.format(
                    column_names='\n'.join([f"- {col}" for col in batch_columns])
                )
                
                # 调用LLM
                response = self.llm_service.generate(prompt)
                
                # 解析响应
                meanings = self._parse_llm_response(response, batch_columns)
                
                # 更新模板
                for j, column in enumerate(batch_columns):
                    if j < len(meanings):
                        filled_template[column]['意义'] = meanings[j].get('意义', '待人工补充说明')
                        filled_template[column]['单位'] = meanings[j].get('单位', '没有单位')
                
                self.logger.info(f"已处理批次 {i//self.batch_size + 1}: {len(batch_columns)} 个列名")
                
            except Exception as e:
                self.logger.error(f"处理批次 {i//self.batch_size + 1} 时出错: {str(e)}")
                # 使用默认值填充
                for column in batch_columns:
                    filled_template[column]['意义'] = '待人工补充说明'
                    filled_template[column]['单位'] = '没有单位'
        
        return filled_template
    
    def _parse_llm_response(self, response: str, column_names: List[str]) -> List[Dict[str, str]]:
        """
        解析LLM响应
        
        Args:
            response: LLM响应文本
            column_names: 列名列表
        
        Returns:
            解析后的意义列表
        """
        try:
            # 尝试解析JSON
            meanings = json.loads(response)
            
            if isinstance(meanings, list):
                return meanings
            elif isinstance(meanings, dict):
                return [meanings]
            else:
                raise ValueError("响应格式不正确")
                
        except Exception as e:
            self.logger.error(f"解析LLM响应失败: {str(e)}")
            # 返回默认值
            return [{"意义": "待人工补充说明", "单位": "没有单位"} for _ in column_names]
    
    def _save_filled_template(self, template: Dict[str, Any]) -> str:
        """
        保存填充后的模板
        
        Args:
            template: 填充后的模板
        
        Returns:
            保存的文件路径
        """
        # 确保输出目录存在
        output_dir = os.path.join(self.storage.cache_path, "filled")
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存文件
        filled_path = os.path.join(output_dir, self.output_filename)
        with open(filled_path, 'w', encoding='utf-8') as f:
            json.dump(template, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"填充后的模板已保存到: {filled_path}")
        return filled_path