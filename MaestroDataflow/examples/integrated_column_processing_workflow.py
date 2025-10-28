"""
整合的列名处理工作流程示例


完整的三步骤工作流程：
1. 第一步：生成包含空意义字段的JSON模板 (输出全部列名)
2. 第二步：进行数据处理 (数据清洗、存储到数据库)
3. 第三步：使用大模型填充JSON的意义字段

这个工作流程满足用户的需求：先输出全部列名，然后进行数据处理，最后进行大模型对JSON的填充
"""

import os
import sys
import json
import pandas as pd
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from maestro.utils.storage import FileStorage
from maestro.pipeline import Pipeline
from maestro.operators.column_template_ops import ColumnTemplateGeneratorOperator
from maestro.operators.column_ops import ColumnMeaningGeneratorOperator
from maestro.operators.data_column_process_ops import DataColumnProcessOperator
from maestro.operators.io_ops import LoadDataOperator


# 创建一个简单的模拟LLM服务
class MockLLMService:
    def generate(self, prompt: str, **kwargs) -> str:
        """模拟LLM响应，返回JSON格式的列名解释"""
        # 根据prompt中的列名返回相应的JSON格式响应
        
        # 用户行为数据相关列名的专业解释
        column_mappings = {
            "age": {"意义": "用户的年龄，表示从出生到当前时间的年数，是重要的人口统计学变量，用于分析不同年龄段用户的行为特征和偏好差异。", "单位": "年"},
            "income": {"意义": "用户的年收入，指用户一年内获得的总收入，包括工资、奖金、投资收益等，是衡量用户经济能力和消费潜力的重要指标。", "单位": "元"},
            "education_level": {"意义": "用户的教育水平，通常用数字编码表示不同的教育程度（如1=高中，2=本科，3=研究生），反映用户的知识背景和认知能力。", "单位": "等级"},
            "satisfaction_score": {"意义": "用户满意度评分，通过问卷调查或评价系统收集的用户对产品或服务的满意程度，通常采用李克特量表进行测量。", "单位": "分"},
            "city_code": {"意义": "城市代码，用简短的字母或数字组合标识用户所在的城市，便于进行地域分析和区域市场研究。", "单位": "代码"},
            "has_car": {"意义": "是否拥有汽车，二元变量表示用户是否拥有私家车（1表示有，0表示无），反映用户的经济状况和生活方式。", "单位": "布尔值"},
            "monthly_expense": {"意义": "月度支出，用户每月的平均消费支出金额，包括生活必需品、娱乐、交通等各项开支，用于分析消费行为模式。", "单位": "元"},
            "user_id": {"意义": "用户唯一标识符，用于区分不同用户的数字或字符串编码，确保数据记录的唯一性和可追溯性。", "单位": "标识符"},
            "registration_date": {"意义": "注册日期，用户首次注册账户或服务的日期，用于分析用户生命周期和留存情况。", "单位": "日期"},
            "last_login": {"意义": "最后登录时间，用户最近一次访问系统或使用服务的时间戳，用于评估用户活跃度和参与度。", "单位": "时间戳"}
        }
        
        # 从prompt中提取所有列名（查找"- "后面的列名）
        import re
        column_matches = re.findall(r'- (\w+)', prompt)
        
        if column_matches:
            # 按照列名顺序返回对应的结果
            results = []
            for column_name in column_matches:
                if column_name in column_mappings:
                    results.append(column_mappings[column_name])
                else:
                    results.append({"意义": "待人工补充说明", "单位": "没有单位"})
            
            # 如果只有一个列名，返回单个对象；多个列名返回数组
            if len(results) == 1:
                return json.dumps(results[0], ensure_ascii=False)
            else:
                return json.dumps(results, ensure_ascii=False)
        
        # 如果没有匹配的列名，返回默认响应
        return json.dumps({"意义": "待人工补充说明", "单位": "没有单位"}, ensure_ascii=False)


def create_user_behavior_data():
    """创建用户行为测试数据"""
    data = {
        'user_id': [f'U{str(i).zfill(4)}' for i in range(1, 21)],
        'age': [25, 30, 35, 28, 32, 29, 31, 27, 33, 26, 24, 36, 29, 31, 28, 34, 27, 30, 32, 25],
        'income': [50000, 60000, 70000, 55000, 65000, 58000, 62000, 53000, 68000, 51000,
                  48000, 72000, 59000, 63000, 56000, 69000, 52000, 61000, 66000, 49000],
        'education_level': [1, 2, 3, 2, 3, 2, 3, 1, 3, 2, 1, 3, 2, 3, 2, 3, 1, 2, 3, 1],
        'satisfaction_score': [4.2, 3.8, 4.5, 4.0, 4.3, 3.9, 4.1, 3.7, 4.4, 4.0,
                              3.6, 4.6, 3.9, 4.2, 4.0, 4.4, 3.8, 4.1, 4.3, 3.7],
        'city_code': ['BJ', 'SH', 'GZ', 'SZ', 'HZ', 'NJ', 'WH', 'CD', 'XA', 'QD',
                     'TJ', 'DL', 'SY', 'JN', 'ZZ', 'WX', 'SZ', 'FS', 'DG', 'ZH'],
        'has_car': [1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0],
        'monthly_expense': [3000, 3500, 4000, 3200, 3800, 3100, 3600, 2900, 3900, 3300,
                           2800, 4200, 3150, 3750, 3250, 4100, 2950, 3650, 3850, 2750],
        'registration_date': pd.date_range('2020-01-01', periods=20, freq='15D'),
        'last_login': pd.date_range('2024-01-01', periods=20, freq='2D')
    }
    
    df = pd.DataFrame(data)
    
    # 添加一些缺失值用于演示数据清洗
    df.loc[1, 'income'] = None
    df.loc[3, 'satisfaction_score'] = None
    df.loc[7, 'monthly_expense'] = None
    df.loc[12, 'age'] = None
    df.loc[15, 'education_level'] = None
    
    return df


def main():
    print("=== 整合的列名处理工作流程示例 ===")
    print("三步骤流程：1.生成列名模板 → 2.数据处理 → 3.大模型填充意义")
    
    # 创建测试数据
    print("\n📊 创建测试数据...")
    df = create_user_behavior_data()
    print(f"数据形状: {df.shape}")
    print(f"列名: {list(df.columns)}")
    
    # 设置输出目录 - 使用项目根目录的output
    output_dir = os.path.join(str(project_root), "output", "integrated_column_processing_workflow")
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存测试数据
    temp_csv_path = os.path.join(output_dir, "user_behavior_data.csv")
    df.to_csv(temp_csv_path, index=False, encoding='utf-8')
    
    # 创建存储对象
    storage = FileStorage(
        input_file_path=temp_csv_path,
        cache_path=output_dir,
        file_name_prefix="workflow"
    )
    
    print("\n" + "="*60)
    print("🔸 第一步：生成列名模板 (输出全部列名)")
    print("="*60)
    
    # 创建模板生成器
    template_generator = ColumnTemplateGeneratorOperator(
        storage=storage,
        template_format="standard",  # 使用标准格式（只包含意义和单位）
        output_filename="all_column_names_template.json"
    )
    
    # 执行模板生成
    template_result = template_generator.execute(df)
    
    print(f"✅ 列名模板生成完成")
    print(f"   - 模板路径: {template_result['template_path']}")
    print(f"   - 总列数: {template_result['total_columns']}")
    print(f"   - 模板格式: {template_result['template_format']}")
    
    # 显示生成的模板内容
    print("\n📄 生成的列名模板预览:")
    template_preview = dict(list(template_result['template'].items())[:5])  # 只显示前5个
    for column, info in template_preview.items():
        print(f"   {column}: {info}")
    print("   ...")
    
    print("\n" + "="*60)
    print("🔸 第二步：进行数据处理 (数据清洗、存储)")
    print("="*60)
    
    # 创建模拟LLM服务用于数据处理
    llm_service = MockLLMService()
    
    # 创建数据处理管道
    pipeline = Pipeline(storage=storage.step())  # 创建新的处理步骤
    
    # 数据列处理操作符
    data_process_op = DataColumnProcessOperator(
        dataset_name="用户行为分析数据",
        dataset_description="包含用户基本信息、收入、教育水平、满意度评分、城市代码、汽车拥有情况、月度支出、注册日期和最后登录时间的用户行为数据集，用于分析用户行为模式和特征。",
        db_connection_string=f"sqlite:///{output_dir}/user_behavior.db",
        service=llm_service
    )
    
    # 执行数据处理
    try:
        # 直接运行数据处理操作符
        data_process_result = data_process_op.run(storage.step(), data=df)
        
        print(f"✅ 数据处理完成")
        print(f"   - 输出路径: {data_process_result['output_path']}")
        print(f"   - 数据库表: {data_process_result['database_table']}")
        print(f"   - 最终数据形状: {data_process_result['final_data_shape']}")
        print(f"   - 处理的列数: {data_process_result['processed_columns']}")
        
        # 显示数据清洗统计
        if 'cleaning_stats' in data_process_result:
            stats = data_process_result['cleaning_stats']
            print(f"   - 清洗统计: 缺失值填充 {stats.get('missing_filled', 0)} 个")
        
    except Exception as e:
        print(f"❌ 数据处理失败: {str(e)}")
        print("   继续执行第三步...")
        data_process_result = {"output_path": "数据处理跳过"}
    
    print("\n" + "="*60)
    print("🔸 第三步：使用大模型填充JSON意义字段")
    print("="*60)
    
    # 使用修改后的ColumnMeaningGeneratorOperator进行模板填充
    meaning_generator = ColumnMeaningGeneratorOperator(
        dataset_description="用户行为分析数据集，包含用户的人口统计学信息、经济状况、满意度评价和行为数据",
        template_mode=True,  # 启用模板模式
        service=llm_service
    )
    
    # 执行意义填充
    meaning_result = meaning_generator.run(
        storage=storage.step(),  # 创建新的处理步骤
        template_path=template_result['template_path']
    )
    
    print(f"✅ 意义填充完成")
    print(f"   - 输出路径: {meaning_result['path']}")
    print(f"   - 总列数: {meaning_result['total_columns']}")
    print(f"   - 已填充列数: {meaning_result['filled_columns']}")
    
    # 保存最终的JSON文件
    final_json_path = os.path.join(output_dir, "final_column_meanings.json")
    
    # 从结果中提取并转换为目标格式
    final_template = {}
    for column_info in meaning_result['column_meanings']['columns']:
        column_name = column_info['column_name']
        final_template[column_name] = {
            "意义": column_info['meaning'],
            "单位": column_info['unit']
        }
    
    # 保存最终JSON
    with open(final_json_path, 'w', encoding='utf-8') as f:
        json.dump(final_template, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 最终JSON已保存: {final_json_path}")
    
    print("\n" + "="*60)
    print("📋 整合工作流程完成总结")
    print("="*60)
    
    print(f"\n🎯 三步骤处理结果:")
    print(f"   1️⃣ 列名模板: {template_result['template_path']}")
    print(f"   2️⃣ 数据处理: {data_process_result['output_path']}")
    print(f"   3️⃣ 最终JSON: {final_json_path}")
    
    print(f"\n📊 处理统计:")
    print(f"   - 总列数: {len(final_template)}")
    print(f"   - 数据行数: {df.shape[0]}")
    print(f"   - 输出目录: {output_dir}")
    
    # 显示处理结果对比
    print(f"\n🔍 处理前后对比（前3个列名）:")
    
    # 读取原始模板
    with open(template_result['template_path'], 'r', encoding='utf-8') as f:
        original_template = json.load(f)
    
    sample_columns = list(original_template.keys())[:3]
    
    for column in sample_columns:
        print(f"\n   列名: {column}")
        print(f"     第一步模板: {original_template[column]}")
        print(f"     第三步填充: {final_template[column]}")
    
    print(f"\n🎉 整合的三步骤工作流程执行完成！")
    print(f"   满足需求：先输出全部列名 → 数据处理 → 大模型填充JSON")


if __name__ == "__main__":
    main()