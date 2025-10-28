"""
Multimodal AI operators for MaestroDataflow.
Supports image, audio, and video processing capabilities.
"""

import os
import base64
import io
from typing import Dict, Any, List, Optional, Union, Tuple
import pandas as pd
import numpy as np
import logging
try:
    from PIL import Image
except ImportError:
    Image = None
    
from ...core.operator import OperatorABC
from ...core.prompt import PromptABC, StandardPrompt, DIYPromptABC
from ...serving.llm_serving import LLMServingABC
from ...utils.storage import MaestroStorage


class ImageProcessor(OperatorABC):
    """
    图像处理操作符。
    支持图像分析、描述生成、OCR等功能。
    """

    ALLOWED_PROMPTS = (StandardPrompt, DIYPromptABC)

    def __init__(
        self,
        llm_serving: LLMServingABC,
        task_type: str = "describe",
        prompt: Optional[PromptABC] = None,
        image_column: str = "image_path",
        output_column: str = "image_analysis",
        max_image_size: Tuple[int, int] = (1024, 1024),
        supported_formats: Optional[List[str]] = None
    ):
        """
        初始化图像处理操作符。

        Args:
            llm_serving: LLM服务实例
            task_type: 任务类型 ("describe", "ocr", "classify", "custom")
            prompt: 自定义提示词
            image_column: 图像路径列名
            output_column: 输出列名
            max_image_size: 最大图像尺寸
            supported_formats: 支持的图像格式
        """
        super().__init__()
        self.llm_serving = llm_serving
        self.task_type = task_type
        self.image_column = image_column
        self.output_column = output_column
        self.max_image_size = max_image_size
        self.supported_formats = supported_formats or ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp']

        # 设置默认提示词
        if prompt is None:
            if task_type == "describe":
                self.prompt = StandardPrompt("describe_image")
            elif task_type == "ocr":
                self.prompt = StandardPrompt("extract_text")
            elif task_type == "classify":
                self.prompt = StandardPrompt("classify_image")
            else:
                self.prompt = DIYPromptABC("Analyze this image: {image}")
        else:
            self.prompt = prompt

        self.validate_prompts([self.prompt])

    def run(self, storage: MaestroStorage, **kwargs) -> Dict[str, Any]:
        """
        执行图像处理。

        Args:
            storage: 存储实例
            **kwargs: 其他参数

        Returns:
            Dict[str, Any]: 处理结果
        """
        self.log_operation_start(task_type=self.task_type)

        try:
            # 读取数据
            data = storage.read(output_type="dataframe")
            if isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                df = data

            if self.image_column not in df.columns:
                raise ValueError(f"Image column '{self.image_column}' not found in data")

            results = []

            for idx, row in df.iterrows():
                try:
                    image_path = row[self.image_column]

                    # 处理图像
                    analysis_result = self._process_image(image_path)

                    # 添加结果到行数据
                    result_row = row.to_dict()
                    result_row[self.output_column] = analysis_result
                    results.append(result_row)

                except Exception as e:
                    self.logger.error(f"Failed to process image at row {idx}: {e}")
                    result_row = row.to_dict()
                    result_row[self.output_column] = f"Error: {str(e)}"
                    results.append(result_row)

            # 保存结果
            output_data = pd.DataFrame(results)
            storage.write(output_data)

            self.log_operation_end({
                "operation": "image_processing",
                "processed_images": len(results)
            })

            return {
                "status": "success",
                "processed_count": len(results),
                "task_type": self.task_type
            }

        except Exception as e:
            return self.handle_error("image_processing", e)

    def _process_image(self, image_path: str) -> str:
        """
        处理单个图像。

        Args:
            image_path: 图像路径

        Returns:
            str: 分析结果
        """
        # 验证文件格式
        if not any(image_path.lower().endswith(fmt) for fmt in self.supported_formats):
            raise ValueError(f"Unsupported image format: {image_path}")

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        # 加载和预处理图像
        image = self._load_and_preprocess_image(image_path)

        # 将图像转换为base64
        image_base64 = self._image_to_base64(image)

        # 准备提示词
        if hasattr(self.prompt, 'format'):
            formatted_prompt = self.prompt.format(image=image_base64, image_path=image_path)
        else:
            formatted_prompt = str(self.prompt)

        # 调用LLM进行分析
        try:
            response = self.llm_serving.generate(
                prompt=formatted_prompt,
                max_tokens=500,
                temperature=0.1
            )
            return response.strip()
        except Exception as e:
            self.logger.error(f"LLM analysis failed for {image_path}: {e}")
            return f"Analysis failed: {str(e)}"

    def _load_and_preprocess_image(self, image_path: str) -> Image.Image:
        """
        加载和预处理图像。

        Args:
            image_path: 图像路径

        Returns:
            Image.Image: 预处理后的图像
        """
        try:
            # 加载图像
            image = Image.open(image_path)

            # 转换为RGB模式
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # 调整大小
            if image.size[0] > self.max_image_size[0] or image.size[1] > self.max_image_size[1]:
                image.thumbnail(self.max_image_size, Image.Resampling.LANCZOS)

            return image

        except Exception as e:
            raise ValueError(f"Failed to load image {image_path}: {e}")

    def _image_to_base64(self, image: Image.Image) -> str:
        """
        将图像转换为base64编码。

        Args:
            image: PIL图像对象

        Returns:
            str: base64编码的图像
        """
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=85)
        image_bytes = buffer.getvalue()
        return base64.b64encode(image_bytes).decode('utf-8')


class AudioProcessor(OperatorABC):
    """
    音频处理操作符。
    支持音频转录、分析等功能。
    """

    ALLOWED_PROMPTS = (StandardPrompt, DIYPromptABC)

    def __init__(
        self,
        llm_serving: Optional[LLMServingABC] = None,
        task_type: str = "transcribe",
        audio_column: str = "audio_path",
        output_column: str = "audio_analysis",
        supported_formats: Optional[List[str]] = None,
        max_duration: int = 300  # 最大5分钟
    ):
        """
        初始化音频处理操作符。

        Args:
            llm_serving: LLM服务实例（用于分析转录结果）
            task_type: 任务类型 ("transcribe", "analyze", "classify")
            audio_column: 音频路径列名
            output_column: 输出列名
            supported_formats: 支持的音频格式
            max_duration: 最大音频时长（秒）
        """
        super().__init__()
        self.llm_serving = llm_serving
        self.task_type = task_type
        self.audio_column = audio_column
        self.output_column = output_column
        self.max_duration = max_duration
        self.supported_formats = supported_formats or ['.wav', '.mp3', '.m4a', '.flac', '.ogg']

    def run(self, storage: MaestroStorage, **kwargs) -> Dict[str, Any]:
        """
        执行音频处理。

        Args:
            storage: 存储实例
            **kwargs: 其他参数

        Returns:
            Dict[str, Any]: 处理结果
        """
        self.log_operation_start(task_type=self.task_type)

        try:
            # 读取数据
            data = storage.read(output_type="dataframe")
            if isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                df = data

            if self.audio_column not in df.columns:
                raise ValueError(f"Audio column '{self.audio_column}' not found in data")

            results = []

            for idx, row in df.iterrows():
                try:
                    audio_path = row[self.audio_column]

                    # 处理音频
                    analysis_result = self._process_audio(audio_path)

                    # 添加结果到行数据
                    result_row = row.to_dict()
                    result_row[self.output_column] = analysis_result
                    results.append(result_row)

                except Exception as e:
                    self.logger.error(f"Failed to process audio at row {idx}: {e}")
                    result_row = row.to_dict()
                    result_row[self.output_column] = f"Error: {str(e)}"
                    results.append(result_row)

            # 保存结果
            output_data = pd.DataFrame(results)
            storage.write(output_data)

            self.log_operation_end({
                "operation": "audio_processing",
                "processed_audios": len(results)
            })

            return {
                "status": "success",
                "processed_count": len(results),
                "task_type": self.task_type
            }

        except Exception as e:
            return self.handle_error("audio_processing", e)

    def _process_audio(self, audio_path: str) -> str:
        """
        处理单个音频文件。

        Args:
            audio_path: 音频路径

        Returns:
            str: 处理结果
        """
        # 验证文件格式
        if not any(audio_path.lower().endswith(fmt) for fmt in self.supported_formats):
            raise ValueError(f"Unsupported audio format: {audio_path}")

        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # 这里应该集成实际的音频处理库（如whisper、speech_recognition等）
        # 由于依赖关系，这里提供一个模拟实现

        if self.task_type == "transcribe":
            # 模拟转录功能
            return f"[Transcription placeholder for {os.path.basename(audio_path)}]"
        elif self.task_type == "analyze":
            # 模拟分析功能
            return f"[Audio analysis placeholder for {os.path.basename(audio_path)}]"
        elif self.task_type == "classify":
            # 模拟分类功能
            return f"[Audio classification placeholder for {os.path.basename(audio_path)}]"
        else:
            return f"[Unknown task type: {self.task_type}]"


class VideoProcessor(OperatorABC):
    """
    视频处理操作符。
    支持视频分析、关键帧提取、内容描述等功能。
    """

    ALLOWED_PROMPTS = (StandardPrompt, DIYPromptABC)

    def __init__(
        self,
        llm_serving: LLMServingABC,
        task_type: str = "analyze",
        video_column: str = "video_path",
        output_column: str = "video_analysis",
        frame_interval: int = 30,  # 每30帧提取一帧
        max_frames: int = 10,
        supported_formats: Optional[List[str]] = None
    ):
        """
        初始化视频处理操作符。

        Args:
            llm_serving: LLM服务实例
            task_type: 任务类型 ("analyze", "summarize", "extract_frames")
            video_column: 视频路径列名
            output_column: 输出列名
            frame_interval: 帧提取间隔
            max_frames: 最大提取帧数
            supported_formats: 支持的视频格式
        """
        super().__init__()
        self.llm_serving = llm_serving
        self.task_type = task_type
        self.video_column = video_column
        self.output_column = output_column
        self.frame_interval = frame_interval
        self.max_frames = max_frames
        self.supported_formats = supported_formats or ['.mp4', '.avi', '.mov', '.mkv', '.webm']

    def run(self, storage: MaestroStorage, **kwargs) -> Dict[str, Any]:
        """
        执行视频处理。

        Args:
            storage: 存储实例
            **kwargs: 其他参数

        Returns:
            Dict[str, Any]: 处理结果
        """
        self.log_operation_start(task_type=self.task_type)

        try:
            # 读取数据
            data = storage.read(output_type="dataframe")
            if isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                df = data

            if self.video_column not in df.columns:
                raise ValueError(f"Video column '{self.video_column}' not found in data")

            results = []

            for idx, row in df.iterrows():
                try:
                    video_path = row[self.video_column]

                    # 处理视频
                    analysis_result = self._process_video(video_path)

                    # 添加结果到行数据
                    result_row = row.to_dict()
                    result_row[self.output_column] = analysis_result
                    results.append(result_row)

                except Exception as e:
                    self.logger.error(f"Failed to process video at row {idx}: {e}")
                    result_row = row.to_dict()
                    result_row[self.output_column] = f"Error: {str(e)}"
                    results.append(result_row)

            # 保存结果
            output_data = pd.DataFrame(results)
            storage.write(output_data)

            self.log_operation_end({
                "operation": "video_processing",
                "processed_videos": len(results)
            })

            return {
                "status": "success",
                "processed_count": len(results),
                "task_type": self.task_type
            }

        except Exception as e:
            return self.handle_error("video_processing", e)

    def _process_video(self, video_path: str) -> str:
        """
        处理单个视频文件。

        Args:
            video_path: 视频路径

        Returns:
            str: 处理结果
        """
        # 验证文件格式
        if not any(video_path.lower().endswith(fmt) for fmt in self.supported_formats):
            raise ValueError(f"Unsupported video format: {video_path}")

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # 这里应该集成实际的视频处理库（如opencv、ffmpeg等）
        # 由于依赖关系，这里提供一个模拟实现

        if self.task_type == "analyze":
            return f"[Video analysis placeholder for {os.path.basename(video_path)}]"
        elif self.task_type == "summarize":
            return f"[Video summary placeholder for {os.path.basename(video_path)}]"
        elif self.task_type == "extract_frames":
            return f"[Frame extraction placeholder for {os.path.basename(video_path)}]"
        else:
            return f"[Unknown task type: {self.task_type}]"


class MultimodalFusion(OperatorABC):
    """
    多模态融合操作符。
    结合文本、图像、音频等多种模态进行综合分析。
    """

    ALLOWED_PROMPTS = (StandardPrompt, DIYPromptABC)

    def __init__(
        self,
        llm_serving: LLMServingABC,
        modalities: Optional[List[str]] = None,
        fusion_strategy: str = "concatenate",
        prompt: Optional[PromptABC] = None,
        output_column: str = "multimodal_analysis"
    ):
        """
        初始化多模态融合操作符。

        Args:
            llm_serving: LLM服务实例
            modalities: 要融合的模态列表 ["text", "image", "audio"]
            fusion_strategy: 融合策略 ("concatenate", "weighted", "attention")
            prompt: 自定义提示词
            output_column: 输出列名
        """
        super().__init__()
        self.llm_serving = llm_serving
        self.modalities = modalities or ["text", "image"]
        self.fusion_strategy = fusion_strategy
        self.output_column = output_column

        # 设置默认提示词
        if prompt is None:
            self.prompt = DIYPromptABC(
                "Analyze the following multimodal content and provide a comprehensive summary:\n"
                "Text: {text}\n"
                "Image: {image}\n"
                "Audio: {audio}\n"
                "Please provide insights that consider all modalities."
            )
        else:
            self.prompt = prompt

        self.validate_prompts([self.prompt])

    def run(self, storage: MaestroStorage, **kwargs) -> Dict[str, Any]:
        """
        执行多模态融合分析。

        Args:
            storage: 存储实例
            **kwargs: 其他参数

        Returns:
            Dict[str, Any]: 处理结果
        """
        self.log_operation_start(
            modalities=self.modalities,
            fusion_strategy=self.fusion_strategy
        )

        try:
            # 读取数据
            data = storage.read(output_type="dataframe")
            if isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                df = data

            results = []

            for idx, row in df.iterrows():
                try:
                    # 提取各模态数据
                    modal_data = {}

                    for modality in self.modalities:
                        if modality == "text":
                            modal_data["text"] = row.get("text", "")
                        elif modality == "image":
                            image_path = row.get("image_path", "")
                            if image_path and os.path.exists(image_path):
                                modal_data["image"] = f"[Image: {os.path.basename(image_path)}]"
                            else:
                                modal_data["image"] = "[No image]"
                        elif modality == "audio":
                            audio_path = row.get("audio_path", "")
                            if audio_path and os.path.exists(audio_path):
                                modal_data["audio"] = f"[Audio: {os.path.basename(audio_path)}]"
                            else:
                                modal_data["audio"] = "[No audio]"

                    # 融合分析
                    analysis_result = self._fuse_and_analyze(modal_data)

                    # 添加结果到行数据
                    result_row = row.to_dict()
                    result_row[self.output_column] = analysis_result
                    results.append(result_row)

                except Exception as e:
                    self.logger.error(f"Failed to process multimodal data at row {idx}: {e}")
                    result_row = row.to_dict()
                    result_row[self.output_column] = f"Error: {str(e)}"
                    results.append(result_row)

            # 保存结果
            output_data = pd.DataFrame(results)
            storage.write(output_data)

            self.log_operation_end({
                "operation": "multimodal_fusion",
                "processed_items": len(results)
            })

            return {
                "status": "success",
                "processed_count": len(results),
                "results": results
            }

        except Exception as e:
            return self.handle_error("multimodal_fusion", e)

    def _fuse_and_analyze(self, modal_data: Dict[str, str]) -> str:
        """
        融合多模态数据并进行分析。

        Args:
            modal_data: 各模态数据

        Returns:
            str: 分析结果
        """
        try:
            # 格式化提示词
            formatted_prompt = self.prompt.format(**modal_data)

            # 调用LLM进行分析
            response = self.llm_serving.generate(
                prompt=formatted_prompt,
                max_tokens=800,
                temperature=0.2
            )

            return response.strip()

        except Exception as e:
            self.logger.error(f"Multimodal analysis failed: {e}")
            return f"Analysis failed: {str(e)}"