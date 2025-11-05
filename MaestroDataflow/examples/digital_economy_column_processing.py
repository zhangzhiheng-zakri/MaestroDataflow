#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
中国数字经济发展数据列名处理工作流
处理 D:\MaestroDataflow\sample_data\中国数字经济发展数据（2005-2023年）.xlsx 文件的列名
"""

import os
import sys
import json
import pandas as pd
from pathlib import Path
from datetime import datetime

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from maestro.operators.data_column_process_ops import DataColumnProcessOperator
from maestro.operators.column_ops import ColumnMeaningGeneratorOperator
from maestro.serving.llm_serving import APILLMServing
from maestro.serving.enhanced_llm_serving import LocalLLMServing, EnhancedLLMServing


def create_real_llm_service():
    """创建真实的LLM服务，优先使用DeepSeek API"""
    
    # 首先尝试使用DeepSeek API
    try:
        print("🚀 使用DeepSeek API服务")
        api_serving = APILLMServing(
            api_url="https://api.deepseek.com/v1/chat/completions",
            api_key="sk-e987d89ccdbe46c6948112314096b038",
            model_name="deepseek-chat",
            max_tokens=1000,
            temperature=0.3,  # 降低温度以获得更稳定的结果
            api_type="openai"  # DeepSeek兼容OpenAI API格式
        )
        
        # 使用增强服务包装，启用缓存
        llm_service = EnhancedLLMServing(
            base_serving=api_serving,
            enable_cache=True,
            cache_ttl=3600  # 缓存1小时
        )
        return llm_service, "deepseek_api"
    except Exception as e:
        print(f"⚠️ DeepSeek API服务初始化失败: {e}")
    
    # 备选：检查环境变量中的OpenAI API密钥
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key and api_key != "sk-123456":  # 排除测试密钥
            print("✅ 找到OpenAI API密钥，使用API服务")
            api_serving = APILLMServing(
                api_url="https://api.openai.com/v1/chat/completions",
                api_key=api_key,
                model_name="gpt-3.5-turbo",
                max_tokens=1000,
                temperature=0.3,  # 降低温度以获得更稳定的结果
                api_type="openai"
            )
            
            # 使用增强服务包装，启用缓存
            llm_service = EnhancedLLMServing(
                base_serving=api_serving,
                enable_cache=True,
                cache_ttl=3600  # 缓存1小时
            )
            return llm_service, "openai_api"
        else:
            print("⚠️ 未找到有效的OPENAI_API_KEY环境变量")
    except Exception as e:
        print(f"⚠️ OpenAI API服务初始化失败: {e}")
    
    # 回退到本地模型
    try:
        print("🔄 尝试使用本地LLM模型...")
        llm_service = LocalLLMServing(
            model_name="microsoft/DialoGPT-medium",
            device="cpu",
            max_tokens=500,
            temperature=0.3
        )
        return llm_service, "local_model"
    except Exception as e:
        print(f"⚠️ 本地模型加载失败: {e}")
    
    # 如果都失败，返回None
    print("❌ 无法初始化任何LLM服务，将使用Mock服务")
    return None, "mock"


class RealLLMColumnProcessor:
    """使用真实LLM的列名处理器"""
    
    def __init__(self, llm_service, service_type):
        self.llm_service = llm_service
        self.service_type = service_type
        
    def generate_column_meaning(self, column_name, sample_data=None):
        """使用真实LLM生成列名含义"""
        if self.llm_service is None:
            return f"无法分析列名：{column_name}（LLM服务不可用）"
        
        # 构建提示词
        prompt = f"""你是一个拥有20年研究经验的顶尖数据分析专家，专门负责解释数据中的列名含义和单位，若列名中无单位，结合意义给出单位。

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
- {column_name}

示例数据：{sample_data if sample_data else "无"}

请直接返回JSON格式的结果，不要包含其他文字说明。"""
        
        try:
            response = self.llm_service.generate(prompt)
            return response.strip()
        except Exception as e:
            print(f"⚠️ LLM生成失败: {e}")
            return f"列名分析失败：{column_name}"
    
    def standardize_column_name(self, column_name):
        """使用真实LLM标准化列名"""
        if self.llm_service is None:
            # 回退到简单规则
            return column_name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('%', 'pct')
        
        prompt = f"""
请将以下中文列名转换为标准的英文列名：

原列名：{column_name}

要求：
1. 使用英文单词
2. 用下划线连接
3. 全部小写
4. 简洁明确
5. 符合数据库命名规范

只返回标准化后的列名，不要其他解释。
"""
        
        try:
            response = self.llm_service.generate(prompt)
            # 清理响应，只保留列名
            standardized = response.strip().lower()
            # 移除可能的标点符号和空格
            standardized = ''.join(c if c.isalnum() or c == '_' else '_' for c in standardized)
            # 移除连续的下划线
            while '__' in standardized:
                standardized = standardized.replace('__', '_')
            return standardized.strip('_')
        except Exception as e:
            print(f"⚠️ LLM标准化失败: {e}")
            return column_name.lower().replace(' ', '_')
    
    def generate_column_template(self, columns):
        """使用真实LLM生成列名模板"""
        template = {}
        
        for col in columns:
            if col.strip() == '' or 'Unnamed' in col:
                continue
                
            # 获取列的含义和标准名
            meaning = self.generate_column_meaning(col)
            standard_name = self.standardize_column_name(col)
            
            # 推断数据类型
            data_type = self._infer_data_type(col)
            category = self._infer_category(col)
            
            template[col] = {
                'english_name': standard_name,
                'standard_name': standard_name,
                'meaning': meaning,
                'data_type': data_type,
                'category': category
            }
        
        return template
    
    def _infer_data_type(self, column_name):
        """推断数据类型"""
        if '年' in column_name or '指标' in column_name:
            return 'integer'
        elif '%' in column_name or '率' in column_name or '比重' in column_name:
            return 'float'
        elif '规模' in column_name or '万亿' in column_name:
            return 'float'
        else:
            return 'string'
    
    def _infer_category(self, column_name):
        """推断列分类"""
        if '年' in column_name or '指标' in column_name:
            return 'time_dimension'
        elif '规模' in column_name:
            return 'economic_scale'
        elif '增长' in column_name:
            return 'growth_rate'
        elif '比重' in column_name:
            return 'economic_ratio'
        elif '渗透率' in column_name:
            return 'penetration_rate'
        else:
            return 'other'
class MockLLMService:
    """模拟LLM服务，为中国数字经济数据提供预定义的列名映射（备用方案）"""
    
    def __init__(self):
        # 预定义的中国数字经济数据列名映射
        self.column_mappings = {
            '指标': {
                'english_name': 'year_indicator',
                'standard_name': 'year',
                'meaning': '年份指标，表示数据对应的年份',
                'data_type': 'integer',
                'category': 'time_dimension'
            },
            '数字经济规模(万亿元）': {
                'english_name': 'digital_economy_scale_trillion_yuan',
                'standard_name': 'digital_economy_scale',
                'meaning': '数字经济总规模，单位为万亿元人民币',
                'data_type': 'float',
                'category': 'economic_scale'
            },
            '数字经济规模同比名义增长(%)': {
                'english_name': 'digital_economy_growth_rate_pct',
                'standard_name': 'digital_economy_growth_rate',
                'meaning': '数字经济规模同比名义增长率，以百分比表示',
                'data_type': 'float',
                'category': 'growth_rate'
            },
            '数字经济规模占GDP比重(%)': {
                'english_name': 'digital_economy_gdp_ratio_pct',
                'standard_name': 'digital_economy_gdp_ratio',
                'meaning': '数字经济规模占国内生产总值(GDP)的比重，以百分比表示',
                'data_type': 'float',
                'category': 'economic_ratio'
            },
            '数字产业化规模(万亿元)': {
                'english_name': 'digital_industrialization_scale_trillion_yuan',
                'standard_name': 'digital_industrialization_scale',
                'meaning': '数字产业化规模，即数字技术产业本身的规模，单位为万亿元',
                'data_type': 'float',
                'category': 'economic_scale'
            },
            '数字产业化规模同比名义增长(%)': {
                'english_name': 'digital_industrialization_growth_rate_pct',
                'standard_name': 'digital_industrialization_growth_rate',
                'meaning': '数字产业化规模同比名义增长率，以百分比表示',
                'data_type': 'float',
                'category': 'growth_rate'
            },
            '数字产业化规模占数字经济比重(%)': {
                'english_name': 'digital_industrialization_digital_economy_ratio_pct',
                'standard_name': 'digital_industrialization_ratio',
                'meaning': '数字产业化规模占数字经济总规模的比重，以百分比表示',
                'data_type': 'float',
                'category': 'economic_ratio'
            },
            '数字产业化规模占GDP比重(%)': {
                'english_name': 'digital_industrialization_gdp_ratio_pct',
                'standard_name': 'digital_industrialization_gdp_ratio',
                'meaning': '数字产业化规模占国内生产总值(GDP)的比重，以百分比表示',
                'data_type': 'float',
                'category': 'economic_ratio'
            },
            '产业数字化规模(万亿元)': {
                'english_name': 'industry_digitalization_scale_trillion_yuan',
                'standard_name': 'industry_digitalization_scale',
                'meaning': '产业数字化规模，即传统产业通过数字化转型产生的经济价值，单位为万亿元',
                'data_type': 'float',
                'category': 'economic_scale'
            },
            '产业数字化规模同比名义增长(%)': {
                'english_name': 'industry_digitalization_growth_rate_pct',
                'standard_name': 'industry_digitalization_growth_rate',
                'meaning': '产业数字化规模同比名义增长率，以百分比表示',
                'data_type': 'float',
                'category': 'growth_rate'
            },
            '产业数字化规模占数字经济比重(%)': {
                'english_name': 'industry_digitalization_digital_economy_ratio_pct',
                'standard_name': 'industry_digitalization_ratio',
                'meaning': '产业数字化规模占数字经济总规模的比重，以百分比表示',
                'data_type': 'float',
                'category': 'economic_ratio'
            },
            '产业数字化规模占GDP比重(%)': {
                'english_name': 'industry_digitalization_gdp_ratio_pct',
                'standard_name': 'industry_digitalization_gdp_ratio',
                'meaning': '产业数字化规模占国内生产总值(GDP)的比重，以百分比表示',
                'data_type': 'float',
                'category': 'economic_ratio'
            },
            '农业数字经济渗透率(%)': {
                'english_name': 'agriculture_digital_penetration_rate_pct',
                'standard_name': 'agriculture_digital_penetration',
                'meaning': '农业领域数字经济渗透率，反映数字技术在农业中的应用程度，以百分比表示',
                'data_type': 'float',
                'category': 'penetration_rate'
            },
            '工业数字经济渗透率(%)': {
                'english_name': 'industry_digital_penetration_rate_pct',
                'standard_name': 'industry_digital_penetration',
                'meaning': '工业领域数字经济渗透率，反映数字技术在工业中的应用程度，以百分比表示',
                'data_type': 'float',
                'category': 'penetration_rate'
            },
            '服务业数字经济渗透率(%)': {
                'english_name': 'service_digital_penetration_rate_pct',
                'standard_name': 'service_digital_penetration',
                'meaning': '服务业领域数字经济渗透率，反映数字技术在服务业中的应用程度，以百分比表示',
                'data_type': 'float',
                'category': 'penetration_rate'
            },
            'Unnamed: 15': {
                'english_name': 'empty_column',
                'standard_name': 'unused_column',
                'meaning': '空列，无实际数据内容，建议删除',
                'data_type': 'null',
                'category': 'unused'
            }
        }
    
    def generate_column_meaning(self, column_name, sample_data=None):
        """生成列名含义"""
        if column_name in self.column_mappings:
            return self.column_mappings[column_name]['meaning']
        return f"未知列名：{column_name}，需要进一步分析"
    
    def standardize_column_name(self, column_name):
        """标准化列名"""
        if column_name in self.column_mappings:
            return self.column_mappings[column_name]['standard_name']
        return column_name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('%', 'pct')
    
    def generate_column_template(self, columns):
        """生成列名模板"""
        template = {}
        for col in columns:
            if col in self.column_mappings:
                template[col] = self.column_mappings[col]
            else:
                template[col] = {
                    'english_name': col.lower().replace(' ', '_'),
                    'standard_name': col.lower().replace(' ', '_'),
                    'meaning': f"需要分析的列：{col}",
                    'data_type': 'unknown',
                    'category': 'unknown'
                }
        return template


def main():
    """主函数：执行中国数字经济数据列名处理工作流"""
    
    # 设置输出目录 - 使用项目根目录的output
    output_dir = project_root / "output" / "digital_economy_column_processing"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 数据文件路径
    data_file = project_root / "sample_data" / "中国数字经济发展数据（2005-2023年）.xlsx"
    
    print(f"开始处理中国数字经济发展数据列名...")
    print(f"数据文件: {data_file}")
    print(f"输出目录: {output_dir}")
    
    try:
        # 读取Excel文件
        df = pd.read_excel(data_file)
        print(f"数据形状: {df.shape}")
        print(f"列名: {df.columns.tolist()}")
        
        # 保存为CSV格式以便后续处理
        csv_file = output_dir / "digital_economy_data.csv"
        df.to_csv(csv_file, index=False, encoding='utf-8-sig')
        print(f"已保存CSV文件: {csv_file}")
        
        # 处理列名（去除空列）
        valid_columns = [col for col in df.columns if not col.startswith('Unnamed')]
        print(f"有效列名: {valid_columns}")
        
        # 初始化LLM服务（优先使用真实LLM）
        print("\n=== 初始化LLM服务 ===")
        llm_service, service_type = create_real_llm_service()
        
        if service_type == "mock":
            # 使用Mock服务作为备用
            print("🔄 使用Mock LLM服务作为备用方案")
            mock_service = MockLLMService()
            processor = RealLLMColumnProcessor(None, "mock")
            # 使用Mock服务的方法
            column_meanings = {}
            column_template = mock_service.generate_column_template(valid_columns)
            column_name_mapping = {col: mock_service.standardize_column_name(col) for col in valid_columns}
        else:
            # 使用真实LLM服务
            print(f"✅ 使用真实LLM服务: {service_type}")
            processor = RealLLMColumnProcessor(llm_service, service_type)
            
            # 生成列名含义
            print("📝 生成列名含义...")
            column_meanings = {}
            for col in valid_columns:
                print(f"  处理列: {col}")
                meaning = processor.generate_column_meaning(col, df[col].head(3).tolist() if col in df.columns else None)
                column_meanings[col] = meaning
            
            # 生成列名模板
            print("📋 生成列名模板...")
            column_template = processor.generate_column_template(valid_columns)
            
            # 生成列名映射
            print("🔄 生成标准化列名映射...")
            column_name_mapping = {}
            for col in valid_columns:
                standard_name = processor.standardize_column_name(col)
                column_name_mapping[col] = standard_name
        
        # 保存处理结果
        print("\n=== 保存处理结果 ===")
        
        # 保存列名含义
        meanings_file = output_dir / "column_meanings.json"
        with open(meanings_file, 'w', encoding='utf-8') as f:
            json.dump(column_meanings, f, ensure_ascii=False, indent=2)
        print(f"✅ 已保存列名含义: {meanings_file}")
        
        # 保存列名模板
        template_file = output_dir / "column_template.json"
        with open(template_file, 'w', encoding='utf-8') as f:
            json.dump(column_template, f, ensure_ascii=False, indent=2)
        print(f"✅ 已保存列名模板: {template_file}")
        
        # 保存列名映射
        mapping_file = output_dir / "column_name_mapping.json"
        with open(mapping_file, 'w', encoding='utf-8') as f:
            json.dump(column_name_mapping, f, ensure_ascii=False, indent=2)
        print(f"✅ 已保存列名映射: {mapping_file}")
        
        # 创建标准化数据
        print("🔄 创建标准化数据...")
        df_standardized = df.copy()
        df_standardized = df_standardized.rename(columns=column_name_mapping)
        
        # 保存标准化数据
        standardized_file = output_dir / "digital_economy_data_standardized.csv"
        df_standardized.to_csv(standardized_file, index=False, encoding='utf-8-sig')
        print(f"✅ 已保存标准化数据: {standardized_file}")
        
        # 生成处理报告
        report = {
            "processing_info": {
                "data_file": str(data_file),
                "processing_time": datetime.now().isoformat(),
                "llm_service_type": service_type,
                "data_shape": {
                    "rows": df.shape[0],
                    "columns": df.shape[1],
                    "valid_columns": len(valid_columns)
                }
            },
            "column_info": {
                "original_columns": df.columns.tolist(),
                "valid_columns": valid_columns,
                "standardized_columns": list(column_name_mapping.values())[:5]  # 只显示前5个
            }
        }
        
        report_file = output_dir / "processing_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"✅ 已保存处理报告: {report_file}")
        
        print(f"\n🎉 处理完成！使用的LLM服务类型: {service_type}")
        print(f"📁 所有结果已保存到: {output_dir}")
        
        return True
        
    except Exception as e:
        print(f"处理过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    if success:
        print("中国数字经济数据列名处理工作流执行成功！")
    else:
        print("中国数字经济数据列名处理工作流执行失败！")
        sys.exit(1)