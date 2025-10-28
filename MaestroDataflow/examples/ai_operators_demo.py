"""
MaestroDataflow AI Operators Demo
演示各种AI操作符的使用方法和功能
"""

import pandas as pd
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 导入MaestroDataflow组件
from maestro.pipeline.pipeline import Pipeline
from maestro.utils.storage import FileStorage
from maestro.serving.enhanced_llm_serving import EnhancedLLMServing, LocalLLMServing
from maestro.serving.llm_serving import APILLMServing
from maestro.core.prompt import DIYPromptABC

# 导入AI操作符
from maestro.operators.ai_ops import (
    PromptedGenerator, TextSummarizer, TextClassifier,
    EmbeddingGenerator, SimilarityCalculator, TextMatcher,
    KnowledgeBaseBuilder, RAGRetriever, RAGOperator,
    ImageProcessor, AudioProcessor, VideoProcessor, MultimodalFusion,
    AutoDataCleaner, SmartAnnotator, FeatureEngineer
)


def setup_demo_environment():
    """设置演示环境"""
    print("🚀 设置MaestroDataflow AI操作符演示环境...")

    # 创建存储实例
    storage = FileStorage(
        input_file_path="../sample_data/employees.csv",
        cache_path="../output/ai_operators_demo/cache",
        file_name_prefix="demo_cache",
        cache_type="csv",
        enable_vector_storage=True,
        enable_model_cache=True
    )

    # 创建LLM服务实例（使用本地模型或API）
    try:
        # 尝试使用本地模型
        llm_serving = LocalLLMServing(
            model_name="microsoft/DialoGPT-medium",
            device="cpu"
        )
        print("✅ 使用本地LLM模型")
    except Exception as e:
        # 回退到API服务
        import os
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            print("⚠️ 未找到OPENAI_API_KEY环境变量，请设置后使用API服务")
            api_key = "demo-key-placeholder"  # 仅用于演示，实际使用时需要真实API密钥
        
        # 创建基础API服务
        base_serving = APILLMServing(
            api_url="https://api.openai.com/v1",
            api_key=api_key,
            model_name="gpt-3.5-turbo",
            api_type="openai"
        )
        
        # 创建增强服务
        llm_serving = EnhancedLLMServing(
            base_serving=base_serving,
            enable_cache=True
        )
        print("✅ 使用API LLM服务")

    return storage, llm_serving


def demo_text_generation_operators(storage, llm_serving):
    """演示文本生成操作符"""
    print("\n📝 === 文本生成操作符演示 ===")

    # 准备示例数据
    sample_texts = [
        "人工智能正在改变我们的世界，从自动驾驶汽车到智能助手，AI技术无处不在。",
        "机器学习是人工智能的一个重要分支，它使计算机能够从数据中学习和改进。",
        "深度学习使用神经网络来模拟人脑的工作方式，在图像识别和自然语言处理方面取得了突破性进展。"
    ]

    df = pd.DataFrame({"text": sample_texts})
    storage.write(df)

    # 1. 提示词生成器演示
    print("\n1️⃣ 提示词生成器演示")
    prompt_generator = PromptedGenerator(
        llm_serving=llm_serving,
        prompt=DIYPromptABC("请为以下文本生成一个创意标题：{text}"),
        input_column="text"
    )

    result = prompt_generator.run(storage, input_path="sample_texts", output_path="generated_titles")
    print(f"生成结果: 成功生成 {result['generated_count']} 个标题")

    # 2. 文本摘要器演示
    print("\n2️⃣ 文本摘要器演示")
    summarizer = TextSummarizer(
        llm_serving=llm_serving,
        input_column="text",
        max_length=50
    )

    result = summarizer.run(storage, input_path="sample_texts", output_path="summaries")
    print(f"摘要结果: 成功生成 {result['summarized_count']} 个摘要")

    # 3. 文本分类器演示
    print("\n3️⃣ 文本分类器演示")
    classifier = TextClassifier(
        llm_serving=llm_serving,
        input_column="text",
        categories=["技术", "科学", "教育"]
    )

    result = classifier.run(storage, input_path="sample_texts", output_path="classifications")
    print(f"分类结果: 成功分类 {result['classified_count']} 个文本")


def demo_embedding_operators(storage, llm_serving):
    """演示嵌入向量操作符"""
    print("\n🔍 === 嵌入向量操作符演示 ===")

    # 准备查询数据
    queries = [
        "什么是人工智能？",
        "机器学习如何工作？",
        "深度学习的应用领域"
    ]

    df_queries = pd.DataFrame({"query": queries})
    storage.write(df_queries)

    # 1. 嵌入生成器演示
    print("\n1️⃣ 嵌入生成器演示")
    embedding_generator = EmbeddingGenerator(
        input_column="query",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        device="cpu"
    )

    result = embedding_generator.run(storage.step())
    print(f"嵌入生成结果: 成功生成 {result['embedded_count']} 个嵌入向量")

    # 2. 相似度计算器演示
    print("\n2️⃣ 相似度计算器演示")
    similarity_calculator = SimilarityCalculator(
        embedding_column="embedding",
        reference_texts=["技术文档", "科学研究", "教育资料"],
        similarity_metric="cosine",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2"
    )

    result = similarity_calculator.run(storage.step())
    print(f"相似度计算结果: 成功计算 {result['calculated_count']} 个相似度")

    # 3. 文本匹配器演示
    print("\n3️⃣ 文本匹配器演示")
    reference_texts = [
        "人工智能是计算机科学的一个分支",
        "机器学习使用算法来分析数据",
        "深度学习在图像识别中很有用"
    ]

    text_matcher = TextMatcher(
        input_column="query",
        reference_texts=reference_texts,
        similarity_threshold=0.5
    )

    result = text_matcher.run(storage)
    print(f"文本匹配结果: 成功匹配 {result['matched_count']} 个文本")


def demo_rag_operators(storage, llm_serving):
    """演示RAG操作符"""
    print("\n🧠 === RAG操作符演示 ===")

    # 准备知识库文档
    documents = [
        "人工智能（AI）是指由机器展现出的智能，与人类和动物展现的自然智能形成对比。",
        "机器学习是人工智能的一个子集，它使用统计技术让计算机系统能够从数据中学习。",
        "深度学习是机器学习的一个子集，它使用多层神经网络来模拟人脑的工作方式。",
        "自然语言处理（NLP）是人工智能的一个分支，专注于计算机与人类语言之间的交互。",
        "计算机视觉是人工智能的一个领域，致力于让机器能够理解和解释视觉信息。"
    ]

    df_docs = pd.DataFrame({"document": documents})
    storage.write(df_docs)

    # 1. 知识库构建器演示
    print("\n1️⃣ 知识库构建器演示")
    kb_builder = KnowledgeBaseBuilder(
        text_column="text",
        chunk_size=200,
        chunk_overlap=50
    )

    result = kb_builder.run(storage, input_path="knowledge_documents", output_path="knowledge_base")
    print(f"知识库构建结果: 成功构建 {result['total_chunks']} 个知识块")

    # 2. RAG检索器演示
    print("\n2️⃣ RAG检索器演示")
    rag_retriever = RAGRetriever(
        query_column="query",
        top_k=3,
        similarity_threshold=0.3
    )

    # 准备查询
    query_data = pd.DataFrame({"query": ["什么是深度学习？"]})
    storage.write(query_data)

    result = rag_retriever.run(
        storage,
        query_path="rag_queries",
        knowledge_base_path="knowledge_base",
        output_path="retrieved_docs"
    )
    print(f"RAG检索结果: 成功检索 {result['total_retrieved']} 个文档")

    # 3. RAG操作符演示
    print("\n3️⃣ RAG操作符演示")
    rag_operator = RAGOperator(
        llm_serving=llm_serving,
        max_context_length=500,
        include_sources=True
    )

    result = rag_operator.run(storage.step())
    print(f"RAG生成结果: 成功处理 {result['successful_responses']} 个查询")


def demo_intelligent_processing_operators(storage, llm_serving):
    """演示智能数据处理操作符"""
    print("\n🤖 === 智能数据处理操作符演示 ===")

    # 准备需要清洗的数据
    dirty_data = pd.DataFrame({
        "name": ["张三", "李四", "王五", "张三", "赵六", None, "钱七"],
        "email": ["zhang@email.com", "li@email", "wang@email.com", "zhang@email.com", "zhao@email.com", "", "qian@email.com"],
        "age": [25, 30, None, 25, 35, 28, 40],
        "score": [85.5, 92.0, 78.5, 85.5, 88.0, 95.0, 72.0],
        "comment": ["很好的产品", "质量不错", "还可以", "很好的产品", "非常满意", "一般般", "需要改进"]
    })

    storage.write(dirty_data)

    # 1. 自动数据清洗器演示
    print("\n1️⃣ 自动数据清洗器演示")
    data_cleaner = AutoDataCleaner(
        llm_serving=llm_serving,
        cleaning_strategies=["remove_duplicates", "handle_missing", "standardize_format"],
        confidence_threshold=0.8
    )

    result = data_cleaner.run(storage, input_path="dirty_data", output_path="cleaned_data")
    print(f"数据清洗结果: 清洗了 {result['final_shape'][0]} 条记录")
    print(f"原始数据形状: {result['original_shape']}, 清洗后形状: {result['final_shape']}")

    # 2. 智能标注器演示
    print("\n2️⃣ 智能标注器演示")
    annotator = SmartAnnotator(
        llm_serving=llm_serving,
        annotation_type="sentiment",
        target_column="comment",
        output_column="sentiment"
    )

    result = annotator.run(storage, input_path="cleaned_data", output_path="annotated_data")
    print(f"智能标注结果: 标注了 {result['annotated_count']} 条记录")
    print(f"标注统计: {result.get('annotation_stats', {})}")

    # 3. 特征工程器演示
    print("\n3️⃣ 特征工程器演示")
    feature_engineer = FeatureEngineer(
        llm_serving=llm_serving,
        feature_types=["statistical", "text", "categorical"],
        max_features=20
    )

    result = feature_engineer.run(storage, input_path="annotated_data", output_path="engineered_features")
    print(f"特征工程结果: 生成了 {result['final_feature_count']} 个特征")
    print(f"原始特征数: {result['original_feature_count']}, 最终特征数: {result['final_feature_count']}")


def demo_multimodal_operators(storage, llm_serving):
    """演示多模态操作符"""
    print("\n🎨 === 多模态操作符演示 ===")

    # 多模态处理示例
    print("\n=== 多模态处理示例 ===")
    
    # 图像处理
    image_processor = ImageProcessor(
        llm_serving=llm_serving,
        task_type="describe",
        image_column="image_path",
        output_column="image_description"
    )
    result = image_processor.run(storage, input_path="sample_images.csv", output_path="image_descriptions.csv")
    print(f"图像处理完成，处理了 {result['processed_count']} 张图像")
    
    # 音频处理
    audio_processor = AudioProcessor(
        llm_serving=llm_serving,
        task_type="transcribe",
        audio_column="audio_path",
        output_column="transcription"
    )
    result = audio_processor.run(storage, input_path="sample_audios.csv", output_path="audio_transcriptions.csv")
    print(f"音频处理完成，处理了 {result['processed_count']} 个音频文件")
    
    # 视频处理
    video_processor = VideoProcessor(
        llm_serving=llm_serving,
        task_type="analyze",
        video_column="video_path",
        output_column="video_analysis"
    )
    result = video_processor.run(storage, input_path="sample_videos.csv", output_path="video_analyses.csv")
    print(f"视频处理完成，处理了 {result['processed_count']} 个视频文件")
    
    # 多模态融合
    multimodal_fusion = MultimodalFusion(
        llm_serving=llm_serving,
        modalities=["text", "image"],
        fusion_strategy="concatenate",
        output_column="multimodal_analysis"
    )
    result = multimodal_fusion.run(storage, input_path="multimodal_data.csv", output_path="multimodal_results.csv")
    print(f"多模态融合完成，处理了 {result['processed_count']} 条记录")


def demo_workflow_integration():
    """演示工作流集成"""
    print("\n🔄 === 工作流集成演示 ===")

    # 创建管道
    pipeline = Pipeline("AI_Processing_Pipeline")

    # 设置存储和LLM服务
    storage, llm_serving = setup_demo_environment()

    # 创建一个完整的AI处理管道
    print("创建AI处理管道...")

    # 步骤1: 数据清洗
    data_cleaner = AutoDataCleaner(llm_serving=llm_serving)

    # 步骤2: 文本分类
    classifier = TextClassifier(
        llm_serving=llm_serving,
        input_column="text",
        categories=["正面", "负面", "中性"]
    )

    # 步骤3: 特征工程
    feature_engineer = FeatureEngineer(llm_serving=llm_serving)

    print("✅ AI处理管道创建完成")
    print("管道包含: 数据清洗 → 文本分类 → 特征工程")


def main():
    """主演示函数"""
    print("🎯 MaestroDataflow AI操作符综合演示")
    print("=" * 50)

    try:
        # 设置环境
        storage, llm_serving = setup_demo_environment()

        # 运行各个演示
        demo_text_generation_operators(storage, llm_serving)
        demo_embedding_operators(storage, llm_serving)
        demo_rag_operators(storage, llm_serving)
        demo_intelligent_processing_operators(storage, llm_serving)
        demo_multimodal_operators(storage, llm_serving)
        demo_workflow_integration()

        print("\n🎉 === 演示完成 ===")
        print("所有AI操作符演示已成功运行！")
        print("请查看 ./demo_data 目录中的输出文件。")

        # 显示存储统计
        if hasattr(storage, 'get_cache_stats'):
            cache_stats = storage.get_cache_stats()
            print(f"\n📊 缓存统计: {cache_stats}")

        if hasattr(storage, 'get_vector_stats'):
            vector_stats = storage.get_vector_stats()
            print(f"📊 向量存储统计: {vector_stats}")

    except Exception as e:
        logger.error(f"演示过程中发生错误: {e}")
        print(f"❌ 演示失败: {e}")
        print("请检查配置和依赖项是否正确安装。")


if __name__ == "__main__":
    main()