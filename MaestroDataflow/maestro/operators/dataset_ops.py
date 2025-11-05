import os
import json
import re
from typing import Optional, Dict, Any

import pandas as pd
from maestro.operators.column_ops import ColumnMeaningGeneratorOperator


class DatasetPackagingOperator:
    """
    将经过 MaestroDataflow 处理后的数据打包为 datasets 规范结构：
    - 创建数据集目录：output/datasets/<dataset_name>
    - 写入数据文件（CSV）：<normalized_data_filename>
    - 生成列名 JSON：all_column_name.json（含义与单位）
    - 生成 dataXXX.py：类继承 DatasetBase，配置 base_path/data_path/columns_path
    - 更新 output/datasets/__init__.py，导入并注册新类到 ALL_DATASETS

    用法示例：
        packer = DatasetPackagingOperator(dataset_name="示例数据集2025")
        result = packer.run(storage=storage, service=None, normalized_data_filename="示例数据集2025_标准化.csv")
    """

    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name.strip()
        if not self.dataset_name:
            raise ValueError("dataset_name 不能为空")

    @staticmethod
    def _slugify(text: str) -> str:
        # 生成 ASCII slug，用于类名与文件名后缀
        slug = re.sub(r"[^A-Za-z0-9]+", "", text)
        return slug or "Dataset"

    @staticmethod
    def _shorten_slug(slug: str) -> str:
        """将PascalCase或连续英文slug压缩为更短的简称。
        规则：
        - 若存在多个单词的PascalCase，取所有大写字母作为首字母缩写（≥3优先）。
        - 否则取前10个字符作为简称。
        - 保证只包含A-Za-z字符。
        """
        letters_only = re.sub(r"[^A-Za-z]", "", slug)
        if not letters_only:
            return "Dataset"
        acronym = "".join(c for c in letters_only if c.isupper())
        if len(acronym) >= 3:
            return acronym
        # 若不存在足够的大写分词，截断原slug前10个字符并规范化为首字母大写
        short = letters_only[:10]
        # 确保以大写开头以适配Pascal风格
        return (short[0].upper() + short[1:]) if short else "Dataset"

    def run(
        self,
        storage,
        service: Optional[Any] = None,
        normalized_data_filename: Optional[str] = None,
        output_root: str = "output/datasets",
        dataset_description: Optional[str] = None,
        meanings_mapping: Optional[Dict[str, Dict[str, str]]] = None,
        meanings_path: Optional[str] = None,
        slug_source: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        参数：
        - storage: 传入当前处理流程使用的 storage（需支持 step().read("dataframe") 返回 DataFrame）
        - service: 可选，若提供将用于自动生成列名意义（当前实现不强制调用）
        - normalized_data_filename: 可选，最终写出的 CSV 文件名；默认使用 <dataset_name>_标准化.csv
        - output_root: 输出根目录，默认 output/datasets
        返回：包含各生成文件路径与类名的字典
        """

        # 1) 读取已处理数据
        df = None
        try:
            df = storage.step().read("dataframe")
        except Exception:
            # 兼容直接传 DataFrame 的 storage 或其他读取接口
            if hasattr(storage, "dataframe") and isinstance(storage.dataframe, pd.DataFrame):
                df = storage.dataframe
        if df is None or not isinstance(df, pd.DataFrame):
            raise RuntimeError("无法从 storage 读取 DataFrame，请确保已完成数据处理并可读出 DataFrame")

        # 2) 路径与命名
        # 优先从参数或storage.input_file_path获取源文件名来生成英文简称
        base_name = None
        if slug_source:
            base_name = os.path.splitext(os.path.basename(slug_source))[0]
        elif hasattr(storage, "input_file_path") and isinstance(storage.input_file_path, str):
            base_name = os.path.splitext(os.path.basename(storage.input_file_path))[0]
        else:
            base_name = self.dataset_name

        ascii_slug = self._slugify(base_name)
        # 若去除非ASCII后没有字母（可能仅有数字），且提供了LLM服务，则尝试用LLM生成英文简称
        if not re.search(r"[A-Za-z]", ascii_slug) and service is not None:
            try:
                prompt = (
                    f"Generate a concise English abbreviation (letters only, PascalCase) for the dataset name '{base_name}'. "
                    f"Return ONLY the abbreviation without any explanations."
                )
                # 直接使用服务生成简称
                resp = service.generate(prompt)
                candidate = re.sub(r"[^A-Za-z]+", "", resp).strip()
                if candidate:
                    ascii_slug = candidate
            except Exception as e:
                # 保留现有ascii_slug作为回退
                print(f"警告：英文简称生成失败，使用回退。错误: {e}")
        short_slug = self._shorten_slug(ascii_slug)
        class_name = f"Dataset{short_slug}"
        class_file_name = f"Dataset{short_slug}.py"

        # 目录名为数据集xlsx同名
        base_dir_name = base_name
        dataset_dir = os.path.join(output_root, base_dir_name)
        os.makedirs(dataset_dir, exist_ok=True)

        # 数据文件名与源xlsx同名（扩展名改为csv）；如传入normalized_data_filename则使用传入值
        data_filename = normalized_data_filename or f"{base_dir_name}.csv"
        data_path = os.path.join(dataset_dir, data_filename)

        # 3) 写数据文件
        df.to_csv(data_path, index=False)

        # 4) 生成列名意义 JSON
        columns = list(map(str, df.columns))

        # 生成/接收列名意义映射（DatasetBase期望的JSON结构）
        # 结构示例：{"列名": {"意义": "...", "单位": "..."}, ...}
        def _list_to_mapping(items: list) -> Dict[str, Dict[str, str]]:
            mapping: Dict[str, Dict[str, str]] = {}
            for idx, item in enumerate(items):
                col_name = item.get("column_name") or item.get("name") or str(idx)
                mapping[col_name] = {
                    "意义": item.get("meaning", item.get("意义", "待人工补充说明")),
                    "单位": item.get("unit", item.get("单位", "没有单位")),
                }
            return mapping

        meanings: Dict[str, Dict[str, str]]

        # 优先使用外部传入的映射
        if meanings_mapping is not None and isinstance(meanings_mapping, dict):
            meanings = meanings_mapping
        # 其次从文件路径读取
        elif meanings_path is not None and os.path.exists(meanings_path):
            try:
                with open(meanings_path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict) and "columns" in loaded and isinstance(loaded["columns"], list):
                    meanings = _list_to_mapping(loaded["columns"])
                elif isinstance(loaded, list):
                    meanings = _list_to_mapping(loaded)
                elif isinstance(loaded, dict):
                    meanings = loaded  # 假定已是映射结构
                else:
                    meanings = {col: {"意义": "", "单位": ""} for col in columns}
            except Exception as e:
                print(f"警告：读取meanings_path失败，使用占位空模板。错误: {e}")
                meanings = {col: {"意义": "", "单位": ""} for col in columns}
        # 再次尝试通过LLM生成
        elif service is not None:
            try:
                # 使用ColumnMeaningGeneratorOperator通过LLM生成列名意义
                tmp_storage = storage.step()
                tmp_storage.write(df)
                generator = ColumnMeaningGeneratorOperator(
                    dataset_description=dataset_description or self.dataset_name,
                    service=service
                )
                # 直接传入清洗后的DataFrame，避免存储读取不一致
                meaning_result = generator.run(tmp_storage, df=df)
                items = meaning_result.get("column_meanings", {}).get("columns", [])
                meanings = _list_to_mapping(items)
            except Exception as e:
                # 失败回退为占位空值映射
                print(f"警告：LLM生成列名意义失败，使用占位空模板。错误: {e}")
                meanings = {col: {"意义": "", "单位": ""} for col in columns}
        else:
            # 无LLM服务则写入空模板（仍为映射结构）
            meanings = {col: {"意义": "", "单位": ""} for col in columns}

        meanings_path = os.path.join(dataset_dir, "all_column_name.json")
        with open(meanings_path, "w", encoding="utf-8") as f:
            json.dump(meanings, f, ensure_ascii=False, indent=2)

        # 5) 生成数据集信息描述（info）：若提供LLM服务则生成简洁中文摘要
        info_text = "由 MaestroDataflow 自动打包的数据集，包含示例配置。"
        if service is not None:
            try:
                # 提供基础元信息与少量样本行，提示生成简洁摘要（限制长度，避免过长）
                sample_rows = df.head(3).to_dict(orient="records")
                prompt = (
                    "请基于以下数据集信息，生成一段不超过120字的中文数据集简介，"
                    "总结数据内容与用途，不要逐行罗列样本：\n"
                    f"数据集名称：{self.dataset_name}\n"
                    f"列名：{columns}\n"
                    f"样本行（最多3行）：{json.dumps(sample_rows, ensure_ascii=False)}\n"
                )
                resp = service.generate(prompt)
                if isinstance(resp, str) and resp.strip():
                    info_text = resp.strip()
            except Exception as e:
                print(f"警告：LLM生成数据集简介失败，使用默认描述。错误: {e}")

        # 5) 生成类文件 DatasetXXX.py
        class_file_path = os.path.join(output_root, class_file_name)
        info_literal = json.dumps(info_text, ensure_ascii=False)
        class_source = f"""
from .DatasetBase import DatasetBase


class {class_name}(DatasetBase):
    name = "{self.dataset_name}"
    info = {info_literal}
    base_path = "./{base_dir_name}"
    data_path = "{data_filename}"
    columns_path = "all_column_name.json"

    def __init__(self):
        super().__init__()

if __name__ == "__main__":
    ds = {class_name}()
    print(ds)
    try:
        ds.load_data()
        print(ds.data.head())
    except Exception as e:
        print("加载数据失败：", e)
"""

        with open(class_file_path, "w", encoding="utf-8") as f:
            f.write(class_source.strip() + "\n")

        # 6) 更新 output/datasets/__init__.py
        init_path = os.path.join(output_root, "__init__.py")
        import_line = f"from .{os.path.splitext(class_file_name)[0]} import {class_name}\n"
        if os.path.exists(init_path):
            with open(init_path, "r", encoding="utf-8") as f:
                init_content = f.read()
            # 添加 import
            if import_line not in init_content:
                init_content = import_line + init_content
            # 添加到 ALL_DATASETS 列表
            if "ALL_DATASETS" in init_content:
                # 简单字符串替换：在列表末尾追加类名
                init_content = re.sub(
                    r"ALL_DATASETS\s*=\s*\[(.*?)\]",
                    lambda m: (
                        f"ALL_DATASETS = [{m.group(1).strip()}" + (
                            ", " if m.group(1).strip() else ""
                        ) + f"{class_name}]"
                    ),
                    init_content,
                    flags=re.DOTALL,
                )
            with open(init_path, "w", encoding="utf-8") as f:
                f.write(init_content)
        else:
            # 初始化一个 __init__.py
            with open(init_path, "w", encoding="utf-8") as f:
                f.write(
                    import_line
                    + "ALL_DATASETS = [" + class_name + "]\n\n"
                    + "def create_dataset(name):\n"
                    + "    for cls in ALL_DATASETS:\n"
                    + "        if cls.name == name:\n"
                    + "            return cls()\n"
                    + "    raise ValueError(f'Unknown dataset: {name}')\n"
                )

        return {
            "dataset_dir": dataset_dir,
            "data_path": data_path,
            "all_column_name": meanings_path,
            "class_file": class_file_path,
            "class_name": class_name,
        }