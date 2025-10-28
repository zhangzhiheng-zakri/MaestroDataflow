"""
AI Operators for MaestroDataflow.
Provides advanced AI-powered data processing operators.
"""

from .text_generation import PromptedGenerator, TextSummarizer, TextClassifier
from .embedding import EmbeddingGenerator, SimilarityCalculator, TextMatcher
from .rag import KnowledgeBaseBuilder, RAGRetriever, RAGOperator
from .multimodal import ImageProcessor, AudioProcessor, VideoProcessor, MultimodalFusion
from .intelligent_processing import AutoDataCleaner, SmartAnnotator, FeatureEngineer

__all__ = [
    # Text Generation Operators
    "PromptedGenerator",
    "TextSummarizer",
    "TextClassifier",

    # Embedding Operators
    "EmbeddingGenerator",
    "SimilarityCalculator",
    "TextMatcher",

    # RAG Operators
    "KnowledgeBaseBuilder",
    "RAGRetriever",
    "RAGOperator",

    # Multimodal Operators
    "ImageProcessor",
    "AudioProcessor",
    "VideoProcessor",
    "MultimodalFusion",

    # Intelligent Processing Operators
    "AutoDataCleaner",
    "SmartAnnotator",
    "FeatureEngineer"
]