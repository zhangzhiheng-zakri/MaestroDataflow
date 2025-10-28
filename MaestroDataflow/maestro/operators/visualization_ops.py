"""
数据可视化算子，支持生成各种类型的图表。
"""
from __future__ import annotations
from typing import Dict, Any, List, Optional, Union, Tuple
import pandas as pd
import numpy as np
import os
import base64
from io import BytesIO

from maestro.core import OperatorABC
from maestro.utils.storage import MaestroStorage

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # 使用非交互式后端
    
    # 设置中文字体支持
    import matplotlib.font_manager as fm
    import platform
    import warnings
    
    # 根据操作系统选择合适的中文字体
    system = platform.system()
    if system == "Windows":
        chinese_fonts = ['Microsoft YaHei', 'SimHei', 'KaiTi', 'FangSong']
    elif system == "Darwin":  # macOS
        chinese_fonts = ['PingFang SC', 'Hiragino Sans GB', 'STHeiti']
    else:  # Linux
        chinese_fonts = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'DejaVu Sans']
    
    available_font = None
    
    # 获取系统所有可用字体
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    # 查找可用的中文字体
    for font_name in chinese_fonts:
        if font_name in available_fonts:
            available_font = font_name
            break
    
    # 如果没有找到预设字体，尝试查找任何包含中文的字体
    if not available_font:
        for font in available_fonts:
            if any(keyword in font.lower() for keyword in ['chinese', 'cjk', 'han', 'zh']):
                available_font = font
                break
    
    if available_font:
        plt.rcParams['font.sans-serif'] = [available_font]
        print(f"使用字体: {available_font}")
        # 禁用字体缺失警告
        warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib.font_manager')
    else:
        # 如果完全没有中文字体，使用默认字体
        plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'sans-serif']
        print("警告: 未找到中文字体，中文可能显示为方框")
        # 禁用字体缺失警告
        warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib.font_manager')
    
    plt.rcParams['axes.unicode_minus'] = False
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.offline import plot
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


class ChartGeneratorOperator(OperatorABC):
    """图表生成算子，支持生成多种类型的图表"""
    
    def __init__(
        self,
        chart_type: str = "line",
        x_column: Optional[str] = None,
        y_columns: Optional[List[str]] = None,
        title: Optional[str] = None,
        output_dir: str = "charts",
        output_filename: Optional[str] = None,  # 自定义输出文件名
        chart_format: str = "png",  # "png", "svg", "html"
        figsize: Tuple[int, int] = (12, 8),
        style: str = "default"
    ):
        """
        初始化图表生成算子
        
        Args:
            chart_type: 图表类型 ("line", "bar", "scatter", "pie", "heatmap", "box")
            x_column: X轴列名
            y_columns: Y轴列名列表
            title: 图表标题
            output_dir: 输出目录
            output_filename: 自定义输出文件名（不包含扩展名）
            chart_format: 图表格式
            figsize: 图表尺寸
            style: 图表样式
        """
        self.chart_type = chart_type
        self.x_column = x_column
        self.y_columns = y_columns or []
        self.title = title
        self.output_dir = output_dir
        self.output_filename = output_filename
        self.chart_format = chart_format
        self.figsize = figsize
        self.style = style
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)

    def run(self, storage: MaestroStorage, **kwargs) -> Dict[str, Any]:
        """执行图表生成"""
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("需要安装matplotlib: pip install matplotlib")
        
        # 读取数据
        if hasattr(storage, 'operator_step') and storage.operator_step == -1:
            storage = storage.step()
        
        df = storage.read(output_type="dataframe")
        
        # 生成图表
        chart_paths = []
        chart_data = {}
        
        if self.chart_format == "html" and PLOTLY_AVAILABLE:
            chart_paths, chart_data = self._generate_plotly_charts(df)
        else:
            chart_paths, chart_data = self._generate_matplotlib_charts(df)
        
        # 保存结果
        result_df = df.copy()
        result_df.attrs['chart_paths'] = chart_paths
        result_df.attrs['chart_data'] = chart_data
        
        path = storage.write(result_df)
        
        return {
            "path": path,
            "chart_paths": chart_paths,
            "chart_count": len(chart_paths),
            "chart_type": self.chart_type
        }

    def _generate_matplotlib_charts(self, df: pd.DataFrame) -> Tuple[List[str], Dict[str, Any]]:
        """使用matplotlib生成图表"""
        chart_paths = []
        chart_data = {}
        
        # 设置样式
        if SEABORN_AVAILABLE:
            sns.set_style(self.style if self.style != "default" else "whitegrid")
        
        # 确定要绘制的列
        if not self.y_columns:
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            if self.x_column in numeric_columns:
                numeric_columns.remove(self.x_column)
            self.y_columns = numeric_columns[:5]  # 最多绘制5个指标
        
        if self.chart_type == "line":
            chart_path = self._create_line_chart(df)
            chart_paths.append(chart_path)
        elif self.chart_type == "bar":
            chart_path = self._create_bar_chart(df)
            chart_paths.append(chart_path)
        elif self.chart_type == "scatter":
            chart_paths.extend(self._create_scatter_charts(df))
        elif self.chart_type == "pie":
            chart_paths.extend(self._create_pie_charts(df))
        elif self.chart_type == "heatmap":
            chart_path = self._create_heatmap(df)
            chart_paths.append(chart_path)
        elif self.chart_type == "box":
            chart_path = self._create_box_plot(df)
            chart_paths.append(chart_path)
        
        return chart_paths, chart_data

    def _create_line_chart(self, df: pd.DataFrame) -> str:
        """创建折线图"""
        plt.figure(figsize=self.figsize)
        
        for column in self.y_columns:
            if column in df.columns:
                if self.x_column and self.x_column in df.columns:
                    plt.plot(df[self.x_column], df[column], marker='o', label=column, linewidth=2)
                else:
                    plt.plot(df.index, df[column], marker='o', label=column, linewidth=2)
        
        # 使用配置的字体设置标题和标签
        plt.title(self.title or f"{', '.join(self.y_columns)} 趋势图", 
                 fontsize=16, fontweight='bold', fontproperties='Microsoft YaHei')
        plt.xlabel(self.x_column or "索引", fontsize=12, fontproperties='Microsoft YaHei')
        plt.ylabel("数值", fontsize=12, fontproperties='Microsoft YaHei')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', prop={'family': 'Microsoft YaHei'})
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 使用自定义文件名或默认文件名
        filename = self.output_filename or "line_chart"
        chart_path = os.path.join(self.output_dir, f"{filename}.{self.chart_format}")
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return chart_path

    def _create_bar_chart(self, df: pd.DataFrame) -> str:
        """创建柱状图"""
        plt.figure(figsize=self.figsize)
        
        x_data = df[self.x_column] if self.x_column and self.x_column in df.columns else df.index
        
        if len(self.y_columns) == 1:
            plt.bar(x_data, df[self.y_columns[0]], color='skyblue', alpha=0.8)
        else:
            width = 0.8 / len(self.y_columns)
            x_pos = range(len(x_data))
            
            for i, column in enumerate(self.y_columns):
                if column in df.columns:
                    offset = (i - len(self.y_columns)/2 + 0.5) * width
                    plt.bar([x + offset for x in x_pos], df[column], 
                           width=width, label=column, alpha=0.8)
        
        # 使用配置的字体设置标题和标签
        plt.title(self.title or f"{', '.join(self.y_columns)} 柱状图", 
                 fontsize=16, fontweight='bold', fontproperties='Microsoft YaHei')
        plt.xlabel(self.x_column or "索引", fontsize=12, fontproperties='Microsoft YaHei')
        plt.ylabel("数值", fontsize=12, fontproperties='Microsoft YaHei')
        
        if len(self.y_columns) > 1:
            plt.legend(prop={'family': 'Microsoft YaHei'})
        
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 使用自定义文件名或默认文件名
        filename = self.output_filename or "bar_chart"
        chart_path = os.path.join(self.output_dir, f"{filename}.{self.chart_format}")
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return chart_path

    def _create_scatter_plot(self, df: pd.DataFrame) -> str:
        """创建散点图"""
        plt.figure(figsize=self.figsize)
        
        if len(self.y_columns) >= 2:
            col1, col2 = self.y_columns[0], self.y_columns[1]
            if col1 in df.columns and col2 in df.columns:
                plt.scatter(df[col1], df[col2], alpha=0.6, s=50)
                
                # 使用配置的字体设置标题和标签
                plt.xlabel(col1, fontsize=12, fontproperties='Microsoft YaHei')
                plt.ylabel(col2, fontsize=12, fontproperties='Microsoft YaHei')
                plt.title(f"{col1} vs {col2} 散点图", fontsize=16, fontweight='bold', 
                         fontproperties='Microsoft YaHei')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
        
        chart_path = os.path.join(self.output_dir, f"scatter_plot.{self.chart_format}")
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return chart_path

    def _create_scatter_charts(self, df: pd.DataFrame) -> List[str]:
        """创建散点图"""
        chart_paths = []
        
        if len(self.y_columns) >= 2:
            for i in range(len(self.y_columns)):
                for j in range(i + 1, len(self.y_columns)):
                    col1, col2 = self.y_columns[i], self.y_columns[j]
                    if col1 in df.columns and col2 in df.columns:
                        plt.figure(figsize=self.figsize)
                        plt.scatter(df[col1], df[col2], alpha=0.7, s=60)
                        plt.xlabel(col1, fontsize=12)
                        plt.ylabel(col2, fontsize=12)
                        plt.title(f"{col1} vs {col2} 散点图", fontsize=16, fontweight='bold')
                        plt.grid(True, alpha=0.3)
                        plt.tight_layout()
                        
                        chart_path = os.path.join(self.output_dir, f"scatter_{col1}_{col2}.{self.chart_format}")
                        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
                        plt.close()
                        chart_paths.append(chart_path)
        
        return chart_paths

    def _create_pie_charts(self, df: pd.DataFrame) -> List[str]:
        """创建饼图 - 专门用于结构性数据展示"""
        chart_paths = []
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 检查是否有数字经济相关的结构数据
        if len(df) > 0:
            # 使用最新年份的数据
            latest_data = df.iloc[-1]
            
            # 创建数字经济结构饼图
            if all(col in df.columns for col in ['数字经济规模_万亿元', '数字产业化规模_万亿元']):
                plt.figure(figsize=(10, 8))
                
                digital_economy = latest_data['数字经济规模_万亿元']
                digital_industry = latest_data['数字产业化规模_万亿元']
                industry_digitalization = digital_economy - digital_industry
                
                values = [digital_industry, industry_digitalization]
                labels = ['数字产业化', '产业数字化']
                colors = ['#FF6B6B', '#4ECDC4']
                
                wedges, texts, autotexts = plt.pie(values, labels=labels, autopct='%1.1f%%', 
                                                  startangle=90, colors=colors, 
                                                  textprops={'fontsize': 12})
                
                # 美化文字
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontweight('bold')
                    autotext.set_fontsize(14)
                
                plt.title(f'{latest_data[self.x_column]}年数字经济结构分析\n(总规模: {digital_economy:.1f}万亿元)', 
                         fontsize=16, fontweight='bold', pad=20)
                plt.axis('equal')
                
                chart_path = os.path.join(self.output_dir, f"digital_economy_structure.{self.chart_format}")
                plt.savefig(chart_path, dpi=300, bbox_inches='tight')
                plt.close()
                chart_paths.append(chart_path)
            
            # 创建GDP占比饼图
            if '数字经济占GDP比重_%' in df.columns:
                plt.figure(figsize=(10, 8))
                
                digital_ratio = latest_data['数字经济占GDP比重_%']
                traditional_ratio = 100 - digital_ratio
                
                values = [digital_ratio, traditional_ratio]
                labels = ['数字经济', '传统经济']
                colors = ['#FF9F43', '#70A1FF']
                
                wedges, texts, autotexts = plt.pie(values, labels=labels, autopct='%1.1f%%', 
                                                  startangle=90, colors=colors,
                                                  textprops={'fontsize': 12})
                
                # 美化文字
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontweight('bold')
                    autotext.set_fontsize(14)
                
                plt.title(f'{latest_data[self.x_column]}年数字经济占GDP比重', 
                         fontsize=16, fontweight='bold', pad=20)
                plt.axis('equal')
                
                chart_path = os.path.join(self.output_dir, f"digital_economy_gdp_ratio.{self.chart_format}")
                plt.savefig(chart_path, dpi=300, bbox_inches='tight')
                plt.close()
                chart_paths.append(chart_path)
        
        return chart_paths

    def _create_pie_chart(self, df: pd.DataFrame, column: str) -> str:
        """创建饼图"""
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        plt.figure(figsize=self.figsize)
        
        # 获取数据并处理
        if column in df.columns:
            value_counts = df[column].value_counts()
            
            # 确保 values 是 list 类型，避免 numpy 类型错误
            values = [float(v) if not pd.isna(v) else 0.0 for v in value_counts.values]
            labels = list(value_counts.index)
            
            # 只显示前10个最大的值
            if len(values) > 10:
                values = values[:10]
                labels = labels[:10]
            
            # 创建饼图
            plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
            
            # 设置标题
            plt.title(f"{column} 分布饼图", fontsize=16, fontweight='bold')
            plt.axis('equal')
        
        chart_path = os.path.join(self.output_dir, f"pie_chart_{column}.{self.chart_format}")
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return chart_path

    def _create_heatmap(self, df: pd.DataFrame) -> str:
        """创建相关性热力图"""
        plt.figure(figsize=self.figsize)
        
        # 只选择数值列
        numeric_df = df.select_dtypes(include=[np.number])
        
        if not numeric_df.empty:
            correlation_matrix = numeric_df.corr()
            
            # 创建热力图
            im = plt.imshow(correlation_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
            
            # 设置坐标轴标签
            plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, 
                      rotation=45, ha='right', fontproperties='Microsoft YaHei')
            plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns,
                      fontproperties='Microsoft YaHei')
            
            # 添加颜色条
            plt.colorbar(im)
            
            # 使用配置的字体设置标题
            plt.title(self.title or "相关性热力图", fontsize=16, fontweight='bold',
                     fontproperties='Microsoft YaHei')
            plt.tight_layout()
        
        chart_path = os.path.join(self.output_dir, f"heatmap.{self.chart_format}")
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return chart_path

    def _create_box_plot(self, df: pd.DataFrame) -> str:
        """创建箱线图"""
        plt.figure(figsize=self.figsize)
        
        # 只选择数值列
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 0:
            data_to_plot = [df[col].dropna() for col in numeric_columns[:5]]  # 最多显示5列
            
            plt.boxplot(data_to_plot, labels=numeric_columns[:5])
            
            # 使用配置的字体设置标题和标签
            plt.title(self.title or "数据分布箱线图", fontsize=16, fontweight='bold',
                     fontproperties='Microsoft YaHei')
            plt.ylabel("数值", fontsize=12, fontproperties='Microsoft YaHei')
            plt.xticks(rotation=45, fontproperties='Microsoft YaHei')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
        
        chart_path = os.path.join(self.output_dir, f"box_plot.{self.chart_format}")
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return chart_path

    def _generate_plotly_charts(self, df: pd.DataFrame) -> Tuple[List[str], Dict[str, Any]]:
        """使用plotly生成交互式图表"""
        chart_paths = []
        chart_data = {}
        
        if self.chart_type == "line":
            fig = go.Figure()
            for column in self.y_columns:
                if column in df.columns:
                    if self.x_column and self.x_column in df.columns:
                        fig.add_trace(go.Scatter(x=df[self.x_column], y=df[column], 
                                               mode='lines+markers', name=column))
                    else:
                        fig.add_trace(go.Scatter(y=df[column], mode='lines+markers', name=column))
            
            fig.update_layout(title=self.title or f"{', '.join(self.y_columns)} 趋势图",
                            xaxis_title=self.x_column or "索引",
                            yaxis_title="数值")
        
        elif self.chart_type == "bar":
            fig = go.Figure()
            for column in self.y_columns:
                if column in df.columns:
                    if self.x_column and self.x_column in df.columns:
                        fig.add_trace(go.Bar(x=df[self.x_column], y=df[column], name=column))
                    else:
                        fig.add_trace(go.Bar(y=df[column], name=column))
            
            fig.update_layout(title=self.title or f"{', '.join(self.y_columns)} 柱状图")
        
        else:
            # 默认创建折线图
            fig = go.Figure()
            for column in self.y_columns:
                if column in df.columns:
                    fig.add_trace(go.Scatter(y=df[column], mode='lines+markers', name=column))
        
        # 保存HTML文件
        chart_path = os.path.join(self.output_dir, f"{self.chart_type}_chart.html")
        plot(fig, filename=chart_path, auto_open=False)
        chart_paths.append(chart_path)
        
        return chart_paths, chart_data


class DashboardGeneratorOperator(OperatorABC):
    """仪表板生成算子，创建包含多个图表的综合仪表板"""
    
    def __init__(
        self,
        dashboard_title: str = "数据分析仪表板",
        output_file: str = "dashboard.html",
        include_charts: List[str] = None
    ):
        """
        初始化仪表板生成算子
        
        Args:
            dashboard_title: 仪表板标题
            output_file: 输出文件名
            include_charts: 包含的图表类型列表
        """
        self.dashboard_title = dashboard_title
        self.output_file = output_file
        self.include_charts = include_charts or ["line", "bar", "heatmap"]

    def run(self, storage: MaestroStorage, **kwargs) -> Dict[str, Any]:
        """生成综合仪表板"""
        if hasattr(storage, 'operator_step') and storage.operator_step == -1:
            storage = storage.step()
        
        df = storage.read(output_type="dataframe")
        
        # 生成多个图表
        chart_generators = []
        for chart_type in self.include_charts:
            generator = ChartGeneratorOperator(
                chart_type=chart_type,
                chart_format="png",
                output_dir="dashboard_charts"
            )
            result = generator.run(storage)
            chart_generators.append((chart_type, result['chart_paths']))
        
        # 创建HTML仪表板
        dashboard_path = self._create_html_dashboard(chart_generators)
        
        path = storage.write(df)
        
        return {
            "path": path,
            "dashboard_path": dashboard_path,
            "chart_count": sum(len(paths) for _, paths in chart_generators)
        }

    def _create_html_dashboard(self, chart_generators: List[Tuple[str, List[str]]]) -> str:
        """创建HTML仪表板"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{self.dashboard_title}</title>
            <meta charset="utf-8">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ text-align: center; margin-bottom: 30px; }}
                .chart-section {{ margin-bottom: 40px; }}
                .chart-title {{ font-size: 18px; font-weight: bold; margin-bottom: 10px; }}
                .chart-container {{ text-align: center; }}
                .chart-container img {{ max-width: 100%; height: auto; margin: 10px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{self.dashboard_title}</h1>
                <p>生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
        """
        
        for chart_type, chart_paths in chart_generators:
            html_content += f"""
            <div class="chart-section">
                <div class="chart-title">{chart_type.upper()} 图表</div>
                <div class="chart-container">
            """
            
            for chart_path in chart_paths:
                # 将图片转换为base64编码
                try:
                    with open(chart_path, 'rb') as f:
                        img_data = base64.b64encode(f.read()).decode()
                    html_content += f'<img src="data:image/png;base64,{img_data}" alt="{chart_type} chart">'
                except Exception as e:
                    html_content += f'<p>无法加载图表: {chart_path}</p>'
            
            html_content += """
                </div>
            </div>
            """
        
        html_content += """
        </body>
        </html>
        """
        
        # 保存HTML文件
        dashboard_path = self.output_file
        with open(dashboard_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return dashboard_path