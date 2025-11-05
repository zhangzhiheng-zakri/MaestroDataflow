"""
整合示例：生产治理流程（清洗→入库→标准化打包并补充列意义）

运行方式：
    python -m examples.integrated_packaging_workflow

说明：
- 第一步：使用 DataColumnProcessOperator 清洗数据、入库（不在此阶段生成列意义）。
- 第二步：DatasetPackagingOperator 在打包阶段调用LLM补充列意义与单位，输出到 output/datasets/。
- 优点：有完整处理报告与数据库落地，更适合生产治理。
"""

import os
import json
import re
import pandas as pd
from maestro.utils.db_storage import DBStorage
from maestro.utils.storage import FileStorage
from maestro.operators.data_column_process_ops import DataColumnProcessOperator
from maestro.operators.dataset_ops import DatasetPackagingOperator
from maestro.serving.llm_serving import APILLMServing
from maestro.serving.enhanced_llm_serving import EnhancedLLMServing


def setup_llm_service():
    """使用 DeepSeek API 作为LLM服务。"""
    api_key = os.getenv("DEEPSEEK_API_KEY", "")
    if not api_key:
        print("⚠️ 未设置DEEPSEEK_API_KEY环境变量，将按演示继续但可能返回占位说明")
        api_key = "demo-key-placeholder"
    base_serving = APILLMServing(
        api_url="https://api.deepseek.com/v1/chat/completions",
        api_key=api_key,
        model_name="deepseek-chat",
        api_type="openai"
    )
    service = EnhancedLLMServing(base_serving=base_serving, enable_cache=True)
    print("✅ 使用DeepSeek API LLM服务")
    return service

 


def main():
    # 输入数据：优先自动从 input/ 目录查找 .xlsx/.csv 文件
    def find_input_file() -> str:
        # 仅读取 input/datasets/ 目录
        search_root = os.path.join(os.getcwd(), "input", "datasets")
        # 1) 允许通过环境变量指定，但必须位于 input/datasets/
        env_path = os.getenv("MAESTRO_INPUT_FILE")
        if env_path and os.path.exists(env_path):
            abs_env = os.path.abspath(env_path)
            abs_root = os.path.abspath(search_root)
            if abs_env.startswith(abs_root):
                print(f"🔎 使用环境变量指定的输入文件: {env_path}")
                return env_path
            else:
                print("⚠️ MAESTRO_INPUT_FILE 未位于 input/datasets/ 下，已忽略该设置")
        # 2) 递归搜索 input/datasets/ 目录优先 .xlsx, 其次 .csv
        candidates_xlsx = []
        candidates_csv = []
        if os.path.isdir(search_root):
            for root, _, files in os.walk(search_root):
                for name in files:
                    # 排除临时/隐藏文件（如 Excel 的 ~$ 前缀、. 开头等）
                    if name.startswith('~$') or name.startswith('.') or name.startswith('._'):
                        continue
                    lower = name.lower()
                    path = os.path.join(root, name)
                    if lower.endswith('.xlsx'):
                        candidates_xlsx.append(path)
                    elif lower.endswith('.csv'):
                        candidates_csv.append(path)
        # 3) 选择优先项
        if candidates_xlsx:
            chosen = sorted(candidates_xlsx)[0]
            print(f"🔎 自动检测到输入xlsx: {chosen}")
            return chosen
        if candidates_csv:
            chosen = sorted(candidates_csv)[0]
            print(f"🔎 自动检测到输入csv: {chosen}")
            return chosen
        # 4) 明确错误提示，仅限 input/datasets/
        raise FileNotFoundError(
            "未在 input/datasets/ 目录找到 .xlsx 或 .csv 文件。",
        )

    input_path = find_input_file()

    # 存储初始化
    storage = FileStorage(
        input_file_path=input_path,
        cache_path="./output/integrated_packaging/cache",
        file_name_prefix="integrated",
        cache_type="csv"
    )

    # LLM服务
    llm_service = setup_llm_service()

    # 第一步：整合处理（清洗→入库→LLM列意义）
    # 数据集名称改为源文件名（不含扩展名）
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    dataset_name = base_name
    dataset_description = f"{base_name} 数据集，自动打包并补充列意义与单位，适用于生产治理流程。"

    # 生成英文简称并构建类名，用于数据库表名
    def _slugify(text: str) -> str:
        slug = re.sub(r"[^A-Za-z0-9]+", "", text)
        return slug or "Dataset"

    def _shorten_slug(slug: str) -> str:
        letters_only = re.sub(r"[^A-Za-z]", "", slug)
        if not letters_only:
            return "Dataset"
        acronym = "".join(c for c in letters_only if c.isupper())
        if len(acronym) >= 3:
            return acronym
        short = letters_only[:10]
        return (short[0].upper() + short[1:]) if short else "Dataset"

    ascii_slug = _slugify(base_name)
    # 若去除非ASCII后没有字母（可能仅有数字或中文），尝试用LLM生成英文简称
    if not re.search(r"[A-Za-z]", ascii_slug) and llm_service is not None:
        try:
            prompt = (
                f"Generate a concise English abbreviation (letters only, PascalCase) for the dataset name '{base_name}'. "
                f"Return ONLY the abbreviation without any explanations."
            )
            resp = llm_service.generate(prompt)
            candidate = re.sub(r"[^A-Za-z]+", "", resp).strip()
            if candidate:
                ascii_slug = candidate
        except Exception as e:
            print(f"警告：英文简称生成失败，使用回退。错误: {e}")
    short_slug = _shorten_slug(ascii_slug)
    class_name_for_table = f"Dataset{short_slug}"
    db_path = "output/integrated_packaging/maestro_data.db"
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    db_conn = f"sqlite:///{db_path}"

    processor = DataColumnProcessOperator(
        dataset_name=dataset_name,
        dataset_description=dataset_description,
        db_connection_string=db_conn,
        table_name=class_name_for_table,
        service=llm_service
    )

    # 提前创建数据集目录，并输出原始列名JSON（不含意义与单位）
    output_root = "output/datasets"
    dataset_dir = os.path.join(output_root, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)
    try:
        # FileStorage需要先step()初始化处理步骤，step(0)读取原始输入文件
        raw_df = storage.step().read(output_type="dataframe")
        raw_columns = list(map(str, raw_df.columns))
        ori_meanings = {col: {"意义": "", "单位": ""} for col in raw_columns}
        ori_path = os.path.join(dataset_dir, "all_column_name_ori.json")
        with open(ori_path, "w", encoding="utf-8") as f:
            json.dump(ori_meanings, f, ensure_ascii=False, indent=2)
        print(f"📄 已输出原始列名JSON: {ori_path}")
    except Exception as e:
        print(f"⚠️ 读取原始数据列名失败，跳过输出 all_column_name_ori.json: {e}")

    # 尝试运行整合处理；如LLM不可用则回退占位说明
    try:
        process_result = processor.run(
            storage=storage,
            na_threshold=0.3,
            fill_method="median",
            llm_service=llm_service
        )

        # 不在清洗阶段生成列意义映射，改为在打包阶段调用LLM补充
        meanings_mapping = None
    except Exception as e:
        print(f"⚠️ 整合处理失败，将在打包阶段调用LLM补充列意义: {e}")
        meanings_mapping = None

    print("\n🔧 清洗与入库完成，准备标准化打包并补充列意义...")

    # 直接沿用当前处理上下文，让打包读取到最新缓存（避免找不到上一步生成的CSV）

    # 第二步：标准化打包（在打包阶段调用LLM补充列意义）
    # 优先从数据库读取清洗后的DataFrame（由处理阶段写入 'cleaned_data'）
    cleaned_df = None
    try:
        table_name = class_name_for_table
        db_reader = DBStorage(connection_string=db_conn, table_name=table_name)
        db_reader.step_count = 1  # 处理阶段写入使用了 step=1
        cleaned_df = db_reader.read(output_type="dataframe", key="cleaned_data")
        if isinstance(cleaned_df, pd.DataFrame) and not cleaned_df.empty:
            print(f"📦 已从数据库读取清洗后的DataFrame用于打包: 表 {table_name}, step 1, key 'cleaned_data'")
    except Exception as e:
        print(f"⚠️ 从数据库读取清洗后数据失败，将尝试使用缓存回退: {e}")

    # 回退：从缓存目录挑选最新的CSV（可能是中间产物，尽量作为备用）
    if cleaned_df is None or cleaned_df.empty:
        try:
            cache_dir = getattr(storage, "cache_path", None)
            prefix = getattr(storage, "file_name_prefix", None)
            if cache_dir and prefix and os.path.isdir(cache_dir):
                files = [f for f in os.listdir(cache_dir) if f.startswith(prefix+"_") and f.endswith(".csv")]
                def _suffix_num(name: str) -> int:
                    try:
                        base = os.path.splitext(name)[0]
                        return int(base.split("_")[-1])
                    except:
                        return -1
                files_sorted = sorted(files, key=_suffix_num, reverse=True)
                if files_sorted:
                    latest_path = os.path.join(cache_dir, files_sorted[0])
                    cleaned_df = pd.read_csv(latest_path)
                    print(f"📦 读取缓存DataFrame用于打包: {latest_path}")
        except Exception as e:
            print(f"⚠️ 读取缓存DataFrame失败，将使用原storage读取: {e}")

    # 构造一个最小存储包装，支持 step/read/write，供打包与LLM算子使用
    class _DFStorage:
        def __init__(self, df: pd.DataFrame):
            self._df = df
            self.operator_step = -1
        def step(self):
            self.operator_step += 1
            return self
        def write(self, data, **kwargs):
            if isinstance(data, pd.DataFrame):
                self._df = data
            return "memory://df"
        def read(self, output_type="dataframe", **kwargs):
            return self._df

    effective_storage = _DFStorage(cleaned_df) if isinstance(cleaned_df, pd.DataFrame) else storage

    packer = DatasetPackagingOperator(dataset_name=dataset_name)
    package_result = packer.run(
        storage=effective_storage,
        service=llm_service,
        output_root=output_root,
        dataset_description=dataset_description,
        meanings_mapping=meanings_mapping,
        # 传递源xlsx文件名用于生成英文简称
        slug_source=input_path
    )

    print("\n✅ 生产治理整合流程完成：")
    for k, v in package_result.items():
        print(f"- {k}: {v}")


if __name__ == "__main__":
    main()