# MaestroDataflow AI升级架构设计

## 🎯 升级目标

将MaestroDataflow从传统数据处理框架升级为AI数据处理平台，在保持现有优势的基础上集成强大的AI能力。

## 📊 当前架构分析

### 现有核心组件
1. **OperatorABC**: 简洁的操作符抽象基类
2. **MaestroStorage**: 统一的存储抽象层
3. **FileStorage**: 多格式文件存储实现
4. **DBStorage**: 数据库存储实现
5. **Pipeline**: 数据处理管道系统
6. **LLMServingABC**: 已有的LLM服务抽象层

### 现有优势
- ✅ 统一的存储抽象层
- ✅ 灵活的操作符系统
- ✅ 完善的管道机制
- ✅ 多格式数据支持
- ✅ 已有LLM服务基础

## 🏗️ AI升级架构设计

### 第一阶段：核心AI基础设施

#### 1.1 扩展OperatorABC接口
```python
# maestro/core/operator.py (升级版)
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Type, Union
import logging

class OperatorABC(ABC):
    """
    AI增强的操作符抽象基类
    """
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.ALLOWED_PROMPTS = tuple()  # 允许的Prompt类型
        
    @abstractmethod
    def run(self, storage: MaestroStorage, **kwargs) -> Dict[str, Any]:
        """执行操作符"""
        pass
    
    def validate_prompts(self, prompts: List[Any]) -> bool:
        """验证Prompt类型"""
        if not self.ALLOWED_PROMPTS:
            return True
        return all(isinstance(p, self.ALLOWED_PROMPTS) for p in prompts)
```

#### 1.2 Prompt系统设计
```python
# maestro/core/prompt.py (新增)
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from functools import wraps

class PromptABC(ABC):
    """Prompt抽象基类"""
    
    @abstractmethod
    def format(self, **kwargs) -> str:
        """格式化Prompt"""
        pass
    
    @abstractmethod
    def validate_inputs(self, **kwargs) -> bool:
        """验证输入参数"""
        pass

class DIYPromptABC(PromptABC):
    """自定义Prompt抽象基类"""
    
    def __init__(self, template: str, required_vars: List[str]):
        self.template = template
        self.required_vars = required_vars
    
    def format(self, **kwargs) -> str:
        if not self.validate_inputs(**kwargs):
            raise ValueError("Missing required variables")
        return self.template.format(**kwargs)
    
    def validate_inputs(self, **kwargs) -> bool:
        return all(var in kwargs for var in self.required_vars)

def prompt_restrict(*allowed_prompt_types):
    """Prompt类型限制装饰器"""
    def decorator(cls):
        cls.ALLOWED_PROMPTS = allowed_prompt_types
        return cls
    return decorator
```

#### 1.3 增强LLM服务层
```python
# maestro/serving/llm_serving.py (增强版)
# 基于现有LLMServingABC，添加新功能

class EnhancedLLMServing(LLMServingABC):
    """增强的LLM服务实现"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_cache = {}  # Prompt缓存
        self.model_cache = {}   # 模型缓存
    
    def generate_with_prompt(self, prompt: PromptABC, **kwargs) -> str:
        """使用Prompt对象生成文本"""
        formatted_prompt = prompt.format(**kwargs)
        return self.generate(formatted_prompt)
    
    def batch_generate_with_prompts(self, prompts: List[PromptABC], 
                                   prompt_kwargs: List[Dict]) -> List[str]:
        """批量使用Prompt生成文本"""
        formatted_prompts = [
            prompt.format(**kwargs) 
            for prompt, kwargs in zip(prompts, prompt_kwargs)
        ]
        return self.batch_generate(formatted_prompts)
```

### 第二阶段：AI操作符生态

#### 2.1 文本生成操作符
```python
# maestro/operators/ai_ops/text_generation.py (新增)
from maestro.core import OperatorABC
from maestro.core.prompt import PromptABC, DIYPromptABC, prompt_restrict
from maestro.serving.llm_serving import LLMServingABC

@prompt_restrict(PromptABC, DIYPromptABC)
class PromptedGenerator(OperatorABC):
    """基于Prompt的文本生成操作符"""
    
    def __init__(self, llm_serving: LLMServingABC, 
                 system_prompt: str, 
                 input_column: str,
                 output_column: str = "generated_text"):
        super().__init__()
        self.llm_serving = llm_serving
        self.system_prompt = system_prompt
        self.input_column = input_column
        self.output_column = output_column
    
    def run(self, storage: MaestroStorage, **kwargs) -> Dict[str, Any]:
        # 读取数据
        if hasattr(storage, 'operator_step') and storage.operator_step == -1:
            storage = storage.step()
        
        df = storage.read(output_type="dataframe")
        
        # 生成文本
        generated_texts = []
        for _, row in df.iterrows():
            prompt = f"{self.system_prompt}\n\nInput: {row[self.input_column]}"
            generated_text = self.llm_serving.generate(prompt)
            generated_texts.append(generated_text)
        
        # 添加生成的文本列
        df[self.output_column] = generated_texts
        
        # 写入结果
        path = storage.write(df)
        return {"path": path, "generated_count": len(generated_texts)}
```

#### 2.2 向量化操作符
```python
# maestro/operators/ai_ops/embedding.py (新增)
import numpy as np
from typing import List, Union
from maestro.core import OperatorABC

class EmbeddingGenerator(OperatorABC):
    """文本向量化操作符"""
    
    def __init__(self, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 input_column: str = "text",
                 output_column: str = "embedding"):
        super().__init__()
        self.embedding_model = embedding_model
        self.input_column = input_column
        self.output_column = output_column
        self._model = None
    
    def _load_model(self):
        """延迟加载嵌入模型"""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.embedding_model)
            except ImportError:
                raise ImportError("Please install sentence-transformers: pip install sentence-transformers")
    
    def run(self, storage: MaestroStorage, **kwargs) -> Dict[str, Any]:
        if hasattr(storage, 'operator_step') and storage.operator_step == -1:
            storage = storage.step()
        
        self._load_model()
        df = storage.read(output_type="dataframe")
        
        # 生成向量
        texts = df[self.input_column].tolist()
        embeddings = self._model.encode(texts)
        
        # 添加向量列
        df[self.output_column] = embeddings.tolist()
        
        path = storage.write(df)
        return {"path": path, "embedding_dimension": embeddings.shape[1]}
```

#### 2.3 RAG操作符
```python
# maestro/operators/ai_ops/rag.py (新增)
from typing import List, Dict, Any, Optional
import numpy as np
from maestro.core import OperatorABC
from maestro.serving.llm_serving import LLMServingABC

class RAGOperator(OperatorABC):
    """检索增强生成操作符"""
    
    def __init__(self, 
                 llm_serving: LLMServingABC,
                 knowledge_base_storage: MaestroStorage,
                 query_column: str = "query",
                 context_column: str = "context",
                 answer_column: str = "answer",
                 top_k: int = 5):
        super().__init__()
        self.llm_serving = llm_serving
        self.knowledge_base_storage = knowledge_base_storage
        self.query_column = query_column
        self.context_column = context_column
        self.answer_column = answer_column
        self.top_k = top_k
    
    def _retrieve_context(self, query: str, query_embedding: List[float]) -> str:
        """检索相关上下文"""
        # 从知识库中检索最相关的文档
        kb_df = self.knowledge_base_storage.read(output_type="dataframe")
        
        # 计算相似度（简化实现）
        similarities = []
        for _, row in kb_df.iterrows():
            doc_embedding = np.array(row['embedding'])
            similarity = np.dot(query_embedding, doc_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
            )
            similarities.append((similarity, row['text']))
        
        # 获取top_k最相关的文档
        similarities.sort(key=lambda x: x[0], reverse=True)
        top_contexts = [ctx for _, ctx in similarities[:self.top_k]]
        
        return "\n\n".join(top_contexts)
    
    def run(self, storage: MaestroStorage, **kwargs) -> Dict[str, Any]:
        if hasattr(storage, 'operator_step') and storage.operator_step == -1:
            storage = storage.step()
        
        df = storage.read(output_type="dataframe")
        
        answers = []
        for _, row in df.iterrows():
            query = row[self.query_column]
            query_embedding = row.get('query_embedding', [])
            
            # 检索上下文
            context = self._retrieve_context(query, query_embedding)
            
            # 生成答案
            prompt = f"""基于以下上下文回答问题：

上下文：
{context}

问题：{query}

答案："""
            
            answer = self.llm_serving.generate(prompt)
            answers.append(answer)
        
        df[self.answer_column] = answers
        path = storage.write(df)
        
        return {"path": path, "answers_generated": len(answers)}
```

### 第三阶段：存储系统增强

#### 3.1 向量数据库支持
```python
# maestro/utils/vector_storage.py (新增)
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from maestro.utils.storage import MaestroStorage

class VectorStorage(MaestroStorage):
    """向量数据库存储实现"""
    
    def __init__(self, 
                 vector_db_path: str,
                 dimension: int,
                 index_type: str = "flat"):
        self.vector_db_path = vector_db_path
        self.dimension = dimension
        self.index_type = index_type
        self._index = None
        self._metadata = []
    
    def _build_index(self):
        """构建向量索引"""
        try:
            import faiss
            if self.index_type == "flat":
                self._index = faiss.IndexFlatIP(self.dimension)
            elif self.index_type == "ivf":
                quantizer = faiss.IndexFlatIP(self.dimension)
                self._index = faiss.IndexIVFFlat(quantizer, self.dimension, 100)
        except ImportError:
            raise ImportError("Please install faiss: pip install faiss-cpu")
    
    def add_vectors(self, vectors: np.ndarray, metadata: List[Dict]):
        """添加向量和元数据"""
        if self._index is None:
            self._build_index()
        
        self._index.add(vectors.astype('float32'))
        self._metadata.extend(metadata)
    
    def search(self, query_vector: np.ndarray, k: int = 5) -> List[Tuple[float, Dict]]:
        """搜索最相似的向量"""
        if self._index is None:
            return []
        
        scores, indices = self._index.search(query_vector.reshape(1, -1).astype('float32'), k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self._metadata):
                results.append((float(score), self._metadata[idx]))
        
        return results
```

#### 3.2 模型缓存系统
```python
# maestro/utils/model_cache.py (新增)
import os
import pickle
import hashlib
from typing import Any, Optional, Dict
from pathlib import Path

class ModelCache:
    """AI模型缓存系统"""
    
    def __init__(self, cache_dir: str = "./model_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def _get_cache_key(self, model_name: str, model_config: Dict) -> str:
        """生成缓存键"""
        config_str = str(sorted(model_config.items()))
        cache_input = f"{model_name}_{config_str}"
        return hashlib.md5(cache_input.encode()).hexdigest()
    
    def get(self, model_name: str, model_config: Dict) -> Optional[Any]:
        """获取缓存的模型"""
        cache_key = self._get_cache_key(model_name, model_config)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load cached model: {e}")
        
        return None
    
    def set(self, model_name: str, model_config: Dict, model: Any) -> None:
        """缓存模型"""
        cache_key = self._get_cache_key(model_name, model_config)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(model, f)
        except Exception as e:
            self.logger.warning(f"Failed to cache model: {e}")
```

### 第四阶段：高级AI功能

#### 4.1 多模态支持
```python
# maestro/operators/ai_ops/multimodal.py (新增)
from typing import Union, List, Dict, Any
import base64
from maestro.core import OperatorABC
from maestro.serving.llm_serving import LLMServingABC

class MultimodalProcessor(OperatorABC):
    """多模态数据处理操作符"""
    
    def __init__(self, 
                 llm_serving: LLMServingABC,
                 input_columns: Dict[str, str],  # {"text": "text_col", "image": "image_col"}
                 output_column: str = "multimodal_result"):
        super().__init__()
        self.llm_serving = llm_serving
        self.input_columns = input_columns
        self.output_column = output_column
    
    def _encode_image(self, image_path: str) -> str:
        """编码图像为base64"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def run(self, storage: MaestroStorage, **kwargs) -> Dict[str, Any]:
        if hasattr(storage, 'operator_step') and storage.operator_step == -1:
            storage = storage.step()
        
        df = storage.read(output_type="dataframe")
        results = []
        
        for _, row in df.iterrows():
            # 构建多模态输入
            multimodal_input = {}
            
            if "text" in self.input_columns:
                multimodal_input["text"] = row[self.input_columns["text"]]
            
            if "image" in self.input_columns:
                image_path = row[self.input_columns["image"]]
                multimodal_input["image"] = self._encode_image(image_path)
            
            # 处理多模态数据（这里需要支持多模态的LLM服务）
            result = self._process_multimodal(multimodal_input)
            results.append(result)
        
        df[self.output_column] = results
        path = storage.write(df)
        
        return {"path": path, "processed_count": len(results)}
    
    def _process_multimodal(self, multimodal_input: Dict[str, Any]) -> str:
        """处理多模态输入（需要具体实现）"""
        # 这里需要根据具体的多模态LLM服务来实现
        return "Multimodal processing result"
```

## 🔄 向后兼容性保证

### 兼容性策略
1. **接口兼容**: 现有的OperatorABC接口保持不变
2. **渐进升级**: AI功能作为可选扩展，不影响现有功能
3. **配置驱动**: 通过配置文件控制AI功能的启用
4. **依赖隔离**: AI相关依赖作为可选依赖安装

### 升级路径
```python
# 现有代码无需修改
from maestro.operators.basic_ops import FilterRowsOperator
from maestro.pipeline.pipeline import Pipeline

# 新增AI功能（可选使用）
from maestro.operators.ai_ops.text_generation import PromptedGenerator
from maestro.core.prompt import DIYPromptABC
```

## 📁 新增目录结构

```
maestro/
├── core/
│   ├── operator.py          # 扩展的OperatorABC
│   ├── prompt.py           # 新增：Prompt系统
│   └── processor.py        # 现有DataProcessor
├── operators/
│   ├── basic_ops.py        # 现有基础操作符
│   ├── ai_ops/             # 新增：AI操作符
│   │   ├── __init__.py
│   │   ├── text_generation.py
│   │   ├── embedding.py
│   │   ├── rag.py
│   │   └── multimodal.py
│   └── advanced_ai/        # 新增：高级AI功能
├── utils/
│   ├── storage.py          # 现有存储系统
│   ├── vector_storage.py   # 新增：向量数据库
│   └── model_cache.py      # 新增：模型缓存
├── serving/
│   ├── llm_serving.py      # 增强现有LLM服务
│   ├── openai_serving.py   # 具体实现
│   └── local_serving.py    # 本地模型服务
└── pipeline/
    ├── pipeline.py         # 现有管道系统
    └── ai_pipeline.py      # 新增：AI增强管道
```

## 🚀 实施计划

### 阶段一：核心基础设施（1-2周）
- [ ] 扩展OperatorABC接口
- [ ] 实现Prompt系统
- [ ] 增强LLM服务层
- [ ] 添加日志和错误处理

### 阶段二：AI操作符生态（2-3周）
- [ ] 实现文本生成操作符
- [ ] 实现向量化操作符
- [ ] 实现RAG操作符
- [ ] 添加单元测试

### 阶段三：存储系统增强（1-2周）
- [ ] 实现向量数据库支持
- [ ] 实现模型缓存系统
- [ ] 集成到现有存储架构

### 阶段四：高级功能（2-3周）
- [ ] 实现多模态支持
- [ ] 添加智能数据处理功能
- [ ] 完善文档和示例

## 📈 预期收益

1. **功能扩展**: 从数据处理扩展到AI数据处理
2. **生态完整**: 提供完整的AI操作符生态
3. **易用性**: 保持简洁的API设计
4. **可扩展性**: 模块化设计，易于添加新功能
5. **生产就绪**: 支持缓存、批处理、错误处理等生产特性

这个架构设计确保了MaestroDataflow能够平滑升级为AI数据处理平台，既保持了原有的优势，又获得了强大的AI能力。