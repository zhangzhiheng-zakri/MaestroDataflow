"""
LLM-related operators for text generation.
"""
from __future__ import annotations
from typing import Dict, Any, List
import pandas as pd

from maestro.core import OperatorABC
from maestro.utils.storage import FileStorage
from maestro.serving.llm_serving import APILLMServing, LocalLLMServing


class LLMGenerateOperator(OperatorABC):
    """使用LLM为指定列生成文本摘要/结果"""
    def __init__(
        self, 
        prompt_template: str, 
        input_columns: List[str], 
        output_column: str = "llm_output", 
        service: APILLMServing | LocalLLMServing | None = None
    ):
        self.prompt_template = prompt_template
        self.input_columns = input_columns
        self.output_column = output_column
        self.service = service

    def run(self, storage: FileStorage, **kwargs) -> Dict[str, Any]:
        df = storage.step().read(output_type="dataframe")
        service = self.service or kwargs.get("llm_service")
        if service is None:
            raise ValueError("LLMGenerateOperator 需要传入 llm_service 或在初始化时提供 service")

        def build_prompt(row: pd.Series) -> str:
            values = {col: row[col] for col in self.input_columns}
            return self.prompt_template.format(**values)

        prompts = [build_prompt(row) for _, row in df.iterrows()]
        # 为效率考虑，只对前N条进行演示，也可批量
        outputs = service.batch_generate(prompts)
        df_copy = df.copy()
        df_copy[self.output_column] = outputs
        path = storage.write(df_copy)
        return {"path": path, "generated_rows": len(df_copy)}