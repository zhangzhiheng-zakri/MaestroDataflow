"""
MaestroDataflow 综合AI工作流示例
展示完整的端到端AI数据处理流程，包括数据预处理、AI分析、知识库构建和智能问答
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import json
import os

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 导入MaestroDataflow组件
from maestro.pipeline.pipeline import Pipeline
from maestro.utils.storage import FileStorage
from maestro.serving.enhanced_llm_serving import EnhancedLLMServing, LocalLLMServing
from maestro.core.prompt import DIYPromptABC, StandardPrompt

# 导入AI操作符
from maestro.operators.ai_ops import (
    AutoDataCleaner, SmartAnnotator, FeatureEngineer,
    EmbeddingGenerator, KnowledgeBaseBuilder, RAGOperator, RAGRetriever,
    TextSummarizer, TextClassifier, PromptedGenerator
)


class ComprehensiveAIWorkflow:
    """
    综合AI工作流类
    实现完整的AI数据处理管道
    """

    def __init__(self, base_path="../output/comprehensive_ai_workflow"):
        """
        初始化综合AI工作流

        Args:
            base_path: 基础路径
        """
        self.base_path = base_path
        self.setup_environment()

    def setup_environment(self):
        """设置工作环境"""
        logger.info("🚀 设置综合AI工作流环境...")

        # 创建目录
        os.makedirs(self.base_path, exist_ok=True)
        os.makedirs(f"{self.base_path}/data", exist_ok=True)
        os.makedirs(f"{self.base_path}/results", exist_ok=True)
        os.makedirs(f"{self.base_path}/reports", exist_ok=True)

        # 创建一个空的示例数据文件，以便FileStorage可以初始化
        sample_data_path = f"{self.base_path}/data/sample_data.json"
        if not os.path.exists(sample_data_path):
            with open(sample_data_path, 'w', encoding='utf-8') as f:
                json.dump([], f)  # 创建空的JSON数组

        # 创建存储实例
        self.storage = FileStorage(
            input_file_path=sample_data_path,
            cache_path=f"{self.base_path}/cache",
            cache_type="csv",  # 设置为CSV格式
            enable_vector_storage=True,
            enable_model_cache=True,
            vector_db_config={"similarity_metric": "cosine"},
            model_cache_config={
                "cache_type": "hybrid",
                "cache_config": {
                    "memory": {"max_size": 100, "default_ttl": 3600},
                    "disk": {
                        "cache_dir": f"{self.base_path}/cache/model_cache",
                        "max_size_mb": 500,
                        "default_ttl": 86400
                    }
                }
            }
        )

        # 创建LLM服务
        try:
            self.llm_serving = LocalLLMServing(
                model_name="microsoft/DialoGPT-medium",
                device="cpu"
            )
            logger.info("✅ 使用本地LLM模型")
        except Exception as e:
            logger.warning(f"本地模型加载失败: {e}")
            from maestro.serving.llm_serving import APILLMServing
            self.llm_serving = APILLMServing(
                api_url="https://api.openai.com/v1/chat/completions",
                api_key="your-api-key-here",  # 请替换为实际的API密钥
                model_name="gpt-3.5-turbo",
                api_type="openai"
            )
            logger.info("✅ 使用API LLM服务")

        # 创建管道
        self.workflow = Pipeline("Comprehensive_AI_Pipeline")

        logger.info("✅ 环境设置完成")

    def generate_sample_data(self):
        """生成示例数据集"""
        logger.info("📊 生成示例数据集...")

        # 生成客户反馈数据
        np.random.seed(42)

        # 产品类别
        products = ["智能手机", "笔记本电脑", "平板电脑", "智能手表", "耳机"]

        # 生成反馈文本
        positive_comments = [
            "这个产品真的很棒，质量很好，推荐购买！",
            "使用体验非常好，功能强大，值得拥有。",
            "设计精美，性能出色，非常满意这次购买。",
            "质量超出预期，客服态度也很好，五星好评！",
            "产品功能齐全，使用简单，强烈推荐给大家。"
        ]

        negative_comments = [
            "产品质量一般，不太满意，有待改进。",
            "使用过程中遇到了一些问题，希望能够解决。",
            "价格偏高，性价比不是很好，不太推荐。",
            "产品有缺陷，客服处理不及时，比较失望。",
            "功能不如描述的那么好，有些夸大宣传。"
        ]

        neutral_comments = [
            "产品还可以，基本满足需求，中规中矩。",
            "使用感受一般，没有特别突出的地方。",
            "质量还行，但也没有什么惊喜，普通水平。",
            "产品功能基本够用，价格也算合理。",
            "整体体验还可以，有优点也有不足。"
        ]

        # 生成数据
        data_size = 200
        data = []

        for i in range(data_size):
            # 随机选择产品和情感
            product = np.random.choice(products)
            sentiment_type = np.random.choice(["positive", "negative", "neutral"], p=[0.5, 0.3, 0.2])

            if sentiment_type == "positive":
                comment = np.random.choice(positive_comments)
                rating = np.random.randint(4, 6)
            elif sentiment_type == "negative":
                comment = np.random.choice(negative_comments)
                rating = np.random.randint(1, 3)
            else:
                comment = np.random.choice(neutral_comments)
                rating = np.random.randint(3, 4)

            # 添加一些噪声数据
            if np.random.random() < 0.1:  # 10%的数据有问题
                if np.random.random() < 0.5:
                    comment = ""  # 空评论
                else:
                    rating = None  # 缺失评分

            # 生成时间戳
            days_ago = np.random.randint(0, 365)
            timestamp = datetime.now() - timedelta(days=days_ago)

            data.append({
                "id": f"review_{i+1:03d}",
                "product": product,
                "comment": comment,
                "rating": rating,
                "timestamp": timestamp,
                "user_id": f"user_{np.random.randint(1, 100):03d}",
                "purchase_amount": np.random.uniform(100, 5000),
                "is_verified": np.random.choice([True, False], p=[0.8, 0.2])
            })

        # 创建DataFrame并保存
        df = pd.DataFrame(data)
        print(f"生成的数据形状: {df.shape}")
        print(f"数据列: {df.columns.tolist()}")
        print(f"前5行数据:\n{df.head()}")
        
        # 直接保存到step 0，这样AutoDataCleaner就能读取到
        file_path = self.storage._get_cache_file_path(0)
        df.to_csv(file_path, index=False)
        print(f"数据已保存到: {file_path}")

        logger.info(f"✅ 生成了 {len(df)} 条客户反馈数据")
        return df

    def step1_data_cleaning(self):
        """步骤1: 数据清洗"""
        logger.info("🧹 步骤1: 执行数据清洗...")

        # 创建数据清洗器
        cleaner = AutoDataCleaner(
            llm_serving=self.llm_serving,
            cleaning_strategies=["remove_duplicates", "handle_missing", "standardize_format"],
            confidence_threshold=0.8,
            generate_report=True
        )

        # 执行清洗
        result = cleaner.run(
            self.storage,
            input_path="raw_customer_feedback",
            output_path="cleaned_feedback"
        )

        logger.info(f"✅ 数据清洗完成: {result['original_shape']} → {result['final_shape']}")
        return result

    def step2_intelligent_annotation(self):
        """步骤2: 智能标注"""
        logger.info("🏷️ 步骤2: 执行智能标注...")

        # 情感分析标注
        sentiment_annotator = SmartAnnotator(
            llm_serving=self.llm_serving,
            annotation_type="sentiment",
            target_column="comment",
            output_column="ai_sentiment",
            batch_size=5
        )

        result1 = sentiment_annotator.run(
            self.storage,
            input_path="cleaned_feedback",
            output_path="sentiment_annotated"
        )

        # 产品类别标注
        category_annotator = SmartAnnotator(
            llm_serving=self.llm_serving,
            annotation_type="classification",
            target_column="comment",
            output_column="comment_category",
            categories=["功能", "质量", "价格", "服务", "其他"],
            batch_size=5
        )

        result2 = category_annotator.run(
            self.storage,
            input_path="sentiment_annotated",
            output_path="fully_annotated"
        )

        logger.info(f"✅ 智能标注完成: 情感分析 + 类别分类")
        return result1, result2

    def step3_feature_engineering(self):
        """步骤3: 特征工程"""
        logger.info("⚙️ 步骤3: 执行特征工程...")

        # 创建特征工程器
        feature_engineer = FeatureEngineer(
            llm_serving=self.llm_serving,
            feature_types=["statistical", "temporal", "text", "categorical"],
            target_column="rating",
            max_features=30
        )

        result = feature_engineer.run(
            self.storage,
            input_path="fully_annotated",
            output_path="engineered_features"
        )

        logger.info(f"✅ 特征工程完成: {result['original_feature_count']} → {result['final_feature_count']} 特征")
        return result

    def step4_text_summarization(self):
        """步骤4: 文本摘要生成"""
        logger.info("📄 步骤4: 生成文本摘要...")

        # 按产品分组生成摘要
        data = self.storage.read()

        summaries = []
        for product in data['product'].unique():
            product_data = data[data['product'] == product]
            product_comments = product_data['comment'].dropna().tolist()

            if len(product_comments) > 0:
                # 合并评论
                combined_text = " ".join(product_comments[:10])  # 取前10条评论

                # 创建摘要器
                summarizer = TextSummarizer(
                    llm_serving=self.llm_serving,
                    input_column="text",
                    max_length=100
                )

                # 生成摘要
                temp_df = pd.DataFrame({"text": [combined_text]})
                self.storage.step().write(temp_df)

                summary_result = summarizer.run(
                    self.storage,
                    input_path=f"temp_text_{product}",
                    output_path=f"summary_{product}"
                )

                # 读取摘要结果
                summary_data = self.storage.read()
                summary_text = summary_data.iloc[0]['summary'] if len(summary_data) > 0 else "无法生成摘要"

                summaries.append({
                    "product": product,
                    "comment_count": len(product_comments),
                    "summary": summary_text,
                    "avg_rating": product_data['rating'].mean()
                })

        # 保存摘要结果
        summary_df = pd.DataFrame(summaries)
        self.storage.step().write(summary_df)

        logger.info(f"✅ 生成了 {len(summaries)} 个产品摘要")
        return summaries

    def step5_knowledge_base_construction(self):
        """步骤5: 知识库构建"""
        logger.info("🧠 步骤5: 构建知识库...")

        # 准备知识库文档
        data = self.storage.read()

        # 创建知识文档
        knowledge_docs = []
        for _, row in data.iterrows():
            if pd.notna(row['comment']) and row['comment'].strip():
                doc = f"产品: {row['product']}, 评论: {row['comment']}, 评分: {row['rating']}, 情感: {row.get('ai_sentiment', '未知')}"
                knowledge_docs.append({"document": doc, "source": row['id']})

        kb_df = pd.DataFrame(knowledge_docs)
        self.storage.step().write(kb_df)

        # 构建知识库
        kb_builder = KnowledgeBaseBuilder(
            chunk_size=200,
            chunk_overlap=50,
            text_column="document"
        )

        result = kb_builder.run(
            self.storage,
            input_path="knowledge_documents",
            output_path="knowledge_base"
        )

        logger.info(f"✅ 知识库构建完成: {result['total_chunks']} 个知识块")
        return result

    def step6_rag_system_setup(self):
        """步骤6: RAG系统设置"""
        logger.info("🔍 步骤6: 设置RAG问答系统...")

        # 创建RAG检索器
        retriever = RAGRetriever(
            top_k=5,
            similarity_threshold=0.3
        )

        # 创建RAG操作符
        rag_operator = RAGOperator(
            llm_serving=self.llm_serving,
            query_column="query",
            max_context_length=1000,
            include_sources=True
        )

        self.rag_operator = rag_operator
        self.retriever = retriever

        logger.info("✅ RAG问答系统设置完成")
        return rag_operator

    def step7_interactive_qa(self):
        """步骤7: 交互式问答"""
        logger.info("💬 步骤7: 交互式问答演示...")

        # 预设问题
        questions = [
            "用户对智能手机的评价如何？",
            "哪个产品的评分最高？",
            "用户主要关心什么问题？",
            "有哪些负面反馈？",
            "产品质量方面的评价怎么样？"
        ]

        qa_results = []

        for question in questions:
            logger.info(f"🤔 问题: {question}")

            # 准备查询
            query_df = pd.DataFrame({"query": [question]})
            self.storage.step().write(query_df)

            # 执行RAG查询
            result = self.rag_operator.run(
                self.storage,
                query_path="current_query",
                knowledge_base_path="knowledge_base",
                output_path="current_answer"
            )

            # 读取答案
            answer_data = self.storage.read()
            answer = answer_data.iloc[0]['response'] if len(answer_data) > 0 else "无法生成答案"

            qa_results.append({
                "question": question,
                "answer": answer,
                "timestamp": datetime.now()
            })

            logger.info(f"💡 答案: {answer[:100]}...")

        # 保存问答结果
        qa_df = pd.DataFrame(qa_results)
        self.storage.step().write(qa_df)

        logger.info(f"✅ 完成 {len(questions)} 个问题的问答")
        return qa_results

    def generate_comprehensive_report(self):
        """生成综合报告"""
        logger.info("📊 生成综合分析报告...")

        try:
            # 读取各阶段数据
            raw_data = self.storage.read()
            cleaned_data = self.storage.read()
            annotated_data = self.storage.read()
            engineered_data = self.storage.read()
            summaries = self.storage.read()
            qa_results = self.storage.read()

            # 生成统计报告
            report = {
                "workflow_summary": {
                    "execution_time": datetime.now().isoformat(),
                    "total_steps": 7,
                    "data_processing_pipeline": [
                        "数据清洗", "智能标注", "特征工程",
                        "文本摘要", "知识库构建", "RAG系统", "交互问答"
                    ]
                },
                "data_statistics": {
                    "raw_records": len(raw_data),
                    "cleaned_records": len(cleaned_data),
                    "annotated_records": len(annotated_data),
                    "final_features": len(engineered_data.columns),
                    "product_summaries": len(summaries),
                    "qa_pairs": len(qa_results)
                },
                "quality_metrics": {
                    "data_completeness": (1 - annotated_data.isnull().sum().sum() / (annotated_data.shape[0] * annotated_data.shape[1])) * 100,
                    "sentiment_distribution": annotated_data['ai_sentiment'].value_counts().to_dict() if 'ai_sentiment' in annotated_data.columns else {},
                    "average_rating": annotated_data['rating'].mean() if 'rating' in annotated_data.columns else 0,
                    "product_coverage": annotated_data['product'].nunique() if 'product' in annotated_data.columns else 0
                },
                "ai_capabilities_used": [
                    "自动数据清洗", "智能情感分析", "文本分类",
                    "特征工程", "文本摘要", "向量检索", "知识问答"
                ],
                "system_performance": {
                    "cache_stats": self.storage.get_cache_stats() if hasattr(self.storage, 'get_cache_stats') else {},
                    "vector_stats": self.storage.get_vector_stats() if hasattr(self.storage, 'get_vector_stats') else {}
                }
            }

            # 保存报告
            report_path = f"{self.base_path}/reports/comprehensive_report.json"
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2, default=str)

            logger.info(f"✅ 综合报告已保存到: {report_path}")
            return report

        except Exception as e:
            logger.error(f"生成报告时发生错误: {e}")
            return {"error": str(e)}

    def run_complete_workflow(self):
        """运行完整的工作流"""
        logger.info("🚀 开始执行综合AI工作流...")

        try:
            # 生成示例数据
            self.generate_sample_data()

            # 执行各个步骤
            step1_result = self.step1_data_cleaning()
            step2_result = self.step2_intelligent_annotation()
            step3_result = self.step3_feature_engineering()
            step4_result = self.step4_text_summarization()
            step5_result = self.step5_knowledge_base_construction()
            step6_result = self.step6_rag_system_setup()
            step7_result = self.step7_interactive_qa()

            # 生成综合报告
            final_report = self.generate_comprehensive_report()

            logger.info("🎉 综合AI工作流执行完成！")

            # 打印摘要
            print("\n" + "="*60)
            print("🎯 MaestroDataflow 综合AI工作流执行摘要")
            print("="*60)
            print(f"📊 处理数据量: {final_report.get('data_statistics', {}).get('raw_records', 0)} 条记录")
            print(f"🧹 数据清洗: 完成")
            print(f"🏷️ 智能标注: 完成")
            print(f"⚙️ 特征工程: 生成 {final_report.get('data_statistics', {}).get('final_features', 0)} 个特征")
            print(f"📄 文本摘要: 生成 {final_report.get('data_statistics', {}).get('product_summaries', 0)} 个产品摘要")
            print(f"🧠 知识库: 构建完成")
            print(f"💬 问答系统: 回答 {final_report.get('data_statistics', {}).get('qa_pairs', 0)} 个问题")
            print(f"📈 数据完整度: {final_report.get('quality_metrics', {}).get('data_completeness', 0):.1f}%")
            print(f"⭐ 平均评分: {final_report.get('quality_metrics', {}).get('average_rating', 0):.2f}")
            print("="*60)
            print(f"📁 结果保存在: {self.base_path}")
            print("="*60)

            return final_report

        except Exception as e:
            logger.error(f"工作流执行失败: {e}")
            raise


def main():
    """主函数"""
    print("🎯 MaestroDataflow 综合AI工作流演示")
    print("展示完整的端到端AI数据处理能力")
    print("="*60)

    try:
        # 创建并运行工作流
        workflow = ComprehensiveAIWorkflow()
        result = workflow.run_complete_workflow()

        print("\n✅ 演示成功完成！")
        print("🔍 请查看生成的报告和数据文件了解详细结果。")

    except Exception as e:
        print(f"❌ 演示失败: {e}")
        logger.error(f"主函数执行失败: {e}")


if __name__ == "__main__":
    main()