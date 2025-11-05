"""
报告生成算子，支持生成HTML和PDF格式的分析报告。
"""
from __future__ import annotations
from typing import Dict, Any, List, Optional, Union
import pandas as pd
import os
import base64
from datetime import datetime
from io import BytesIO

from maestro.core import OperatorABC
from maestro.utils.storage import MaestroStorage

try:
    import weasyprint
    WEASYPRINT_AVAILABLE = True
except Exception:
    # 在Windows等环境中，weasyprint可能因缺少系统DLL（如libgobject）在导入阶段抛出非ImportError异常
    WEASYPRINT_AVAILABLE = False

try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


class HTMLReportGeneratorOperator(OperatorABC):
    """HTML报告生成算子"""
    
    def __init__(
        self,
        report_title: str = "数据分析报告",
        output_file: str = "analysis_report.html",
        include_charts: bool = True,
        include_data_table: bool = True,
        template_style: str = "modern"
    ):
        """
        初始化HTML报告生成算子
        
        Args:
            report_title: 报告标题
            output_file: 输出文件名
            include_charts: 是否包含图表
            include_data_table: 是否包含数据表格
            template_style: 模板样式 ("modern", "classic", "minimal")
        """
        self.report_title = report_title
        self.output_file = output_file
        self.include_charts = include_charts
        self.include_data_table = include_data_table
        self.template_style = template_style

    def run(self, storage: MaestroStorage, **kwargs) -> Dict[str, Any]:
        """生成HTML报告"""
        if hasattr(storage, 'operator_step') and storage.operator_step == -1:
            storage = storage.step()
        
        df = storage.read(output_type="dataframe")
        
        # 获取分析报告数据
        analysis_report = getattr(df, 'attrs', {}).get('analysis_report', {})
        chart_paths = getattr(df, 'attrs', {}).get('chart_paths', [])
        
        # 生成HTML报告
        html_content = self._generate_html_report(df, analysis_report, chart_paths)
        
        # 保存HTML文件
        with open(self.output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        path = storage.write(df)
        
        return {
            "path": path,
            "report_path": self.output_file,
            "report_size": os.path.getsize(self.output_file) if os.path.exists(self.output_file) else 0
        }

    def _generate_html_report(self, df: pd.DataFrame, analysis_report: Dict[str, Any], chart_paths: List[str]) -> str:
        """生成HTML报告内容"""
        # 获取CSS样式
        css_style = self._get_css_style()
        
        html_content = f"""
        <!DOCTYPE html>
        <html lang="zh-CN">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{self.report_title}</title>
            <style>{css_style}</style>
        </head>
        <body>
            <div class="container">
                <header class="report-header">
                    <h1>{self.report_title}</h1>
                    <p class="report-meta">生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </header>
                
                <main class="report-content">
        """
        
        # 添加数据概览
        if analysis_report:
            html_content += self._generate_overview_section(analysis_report)
            html_content += self._generate_analysis_section(analysis_report)
        
        # 添加图表
        if self.include_charts and chart_paths:
            html_content += self._generate_charts_section(chart_paths)
        
        # 添加数据表格
        if self.include_data_table:
            html_content += self._generate_data_table_section(df)
        
        # 添加数据质量评估
        if analysis_report and 'data_quality' in analysis_report:
            html_content += self._generate_quality_section(analysis_report['data_quality'])
        
        html_content += """
                </main>
                
                <footer class="report-footer">
                    <p>本报告由 MaestroDataflow 自动生成</p>
                </footer>
            </div>
        </body>
        </html>
        """
        
        return html_content

    def _get_css_style(self) -> str:
        """获取CSS样式"""
        if self.template_style == "modern":
            return """
                * { margin: 0; padding: 0; box-sizing: border-box; }
                body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; line-height: 1.6; color: #333; background: #f5f5f5; }
                .container { max-width: 1200px; margin: 0 auto; background: white; box-shadow: 0 0 20px rgba(0,0,0,0.1); }
                .report-header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 2rem; text-align: center; }
                .report-header h1 { font-size: 2.5rem; margin-bottom: 0.5rem; }
                .report-meta { font-size: 1.1rem; opacity: 0.9; }
                .report-content { padding: 2rem; }
                .section { margin-bottom: 3rem; }
                .section-title { font-size: 1.8rem; color: #667eea; border-bottom: 3px solid #667eea; padding-bottom: 0.5rem; margin-bottom: 1.5rem; }
                .overview-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1.5rem; margin-bottom: 2rem; }
                .overview-card { background: #f8f9ff; border-left: 4px solid #667eea; padding: 1.5rem; border-radius: 8px; }
                .overview-card h3 { color: #667eea; margin-bottom: 0.5rem; }
                .metric-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1.5rem; }
                .metric-card { background: white; border: 1px solid #e0e0e0; border-radius: 8px; padding: 1.5rem; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                .metric-title { font-weight: bold; color: #333; margin-bottom: 1rem; font-size: 1.1rem; }
                .metric-value { display: flex; justify-content: space-between; margin-bottom: 0.5rem; }
                .metric-label { color: #666; }
                .metric-number { font-weight: bold; color: #667eea; }
                .chart-container { text-align: center; margin: 2rem 0; }
                .chart-container img { max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
                .data-table { width: 100%; border-collapse: collapse; margin-top: 1rem; }
                .data-table th, .data-table td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
                .data-table th { background: #667eea; color: white; font-weight: bold; }
                .data-table tr:nth-child(even) { background: #f8f9ff; }
                .quality-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; }
                .quality-item { background: #f0f8ff; padding: 1rem; border-radius: 6px; text-align: center; }
                .quality-number { font-size: 2rem; font-weight: bold; color: #667eea; }
                .quality-label { color: #666; margin-top: 0.5rem; }
                .report-footer { background: #333; color: white; text-align: center; padding: 1rem; }
            """
        elif self.template_style == "classic":
            return """
                body { font-family: Times, serif; line-height: 1.6; color: #333; margin: 2rem; }
                .container { max-width: 800px; margin: 0 auto; }
                .report-header { border-bottom: 2px solid #333; padding-bottom: 1rem; margin-bottom: 2rem; }
                .report-header h1 { font-size: 2rem; }
                .section-title { font-size: 1.5rem; margin: 2rem 0 1rem 0; border-bottom: 1px solid #666; }
                .metric-card { margin-bottom: 1.5rem; padding: 1rem; border: 1px solid #ccc; }
                .data-table { width: 100%; border-collapse: collapse; margin: 1rem 0; }
                .data-table th, .data-table td { padding: 8px; border: 1px solid #ccc; }
                .data-table th { background: #f0f0f0; }
            """
        else:  # minimal
            return """
                body { font-family: Arial, sans-serif; line-height: 1.5; color: #333; margin: 1rem; }
                .container { max-width: 900px; margin: 0 auto; }
                .section-title { font-size: 1.3rem; margin: 1.5rem 0 0.5rem 0; }
                .metric-card { margin-bottom: 1rem; }
                .data-table { width: 100%; border-collapse: collapse; }
                .data-table th, .data-table td { padding: 6px; border-bottom: 1px solid #ddd; }
                .data-table th { background: #f5f5f5; }
            """

    def _generate_overview_section(self, analysis_report: Dict[str, Any]) -> str:
        """生成概览部分"""
        overview = analysis_report.get('data_overview', {})
        
        return f"""
        <section class="section">
            <h2 class="section-title">数据概览</h2>
            <div class="overview-grid">
                <div class="overview-card">
                    <h3>时间范围</h3>
                    <p>{overview.get('time_range', '未知')}</p>
                </div>
                <div class="overview-card">
                    <h3>数据记录</h3>
                    <p>{overview.get('total_records', 0)} 条</p>
                </div>
                <div class="overview-card">
                    <h3>指标数量</h3>
                    <p>{overview.get('total_columns', 0)} 个</p>
                </div>
            </div>
        </section>
        """

    def _generate_analysis_section(self, analysis_report: Dict[str, Any]) -> str:
        """生成分析结果部分"""
        statistical_analysis = analysis_report.get('statistical_analysis', {})
        
        if not statistical_analysis:
            return ""
        
        html = """
        <section class="section">
            <h2 class="section-title">统计分析结果</h2>
            <div class="metric-grid">
        """
        
        for column, analysis in statistical_analysis.items():
            if 'error' in analysis:
                continue
            
            html += f"""
            <div class="metric-card">
                <div class="metric-title">{column}</div>
                <div class="metric-value">
                    <span class="metric-label">最大值:</span>
                    <span class="metric-number">{analysis.get('max_value', 'N/A')}</span>
                </div>
                <div class="metric-value">
                    <span class="metric-label">最小值:</span>
                    <span class="metric-number">{analysis.get('min_value', 'N/A')}</span>
                </div>
                <div class="metric-value">
                    <span class="metric-label">平均值:</span>
                    <span class="metric-number">{analysis.get('mean_value', 'N/A'):.2f}</span>
                </div>
                <div class="metric-value">
                    <span class="metric-label">标准差:</span>
                    <span class="metric-number">{analysis.get('std_value', 'N/A'):.2f}</span>
                </div>
            """
            
            if 'total_growth_rate' in analysis:
                html += f"""
                <div class="metric-value">
                    <span class="metric-label">总增长率:</span>
                    <span class="metric-number">{analysis['total_growth_rate']}%</span>
                </div>
                <div class="metric-value">
                    <span class="metric-label">年均增长率:</span>
                    <span class="metric-number">{analysis['annual_growth_rate']}%</span>
                </div>
                """
            
            html += "</div>"
        
        html += """
            </div>
        </section>
        """
        
        return html

    def _generate_charts_section(self, chart_paths: List[str]) -> str:
        """生成图表部分"""
        if not chart_paths:
            return ""
        
        html = """
        <section class="section">
            <h2 class="section-title">数据可视化</h2>
        """
        
        for chart_path in chart_paths:
            if os.path.exists(chart_path):
                try:
                    with open(chart_path, 'rb') as f:
                        img_data = base64.b64encode(f.read()).decode()
                    
                    chart_name = os.path.basename(chart_path).replace('.png', '').replace('_', ' ').title()
                    html += f"""
                    <div class="chart-container">
                        <h3>{chart_name}</h3>
                        <img src="data:image/png;base64,{img_data}" alt="{chart_name}">
                    </div>
                    """
                except Exception as e:
                    html += f'<p>无法加载图表: {chart_path}</p>'
        
        html += "</section>"
        return html

    def _generate_data_table_section(self, df: pd.DataFrame) -> str:
        """生成数据表格部分"""
        # 只显示前10行数据
        display_df = df.head(10)
        
        html = """
        <section class="section">
            <h2 class="section-title">数据预览</h2>
            <table class="data-table">
                <thead>
                    <tr>
        """
        
        for column in display_df.columns:
            html += f"<th>{column}</th>"
        
        html += """
                    </tr>
                </thead>
                <tbody>
        """
        
        for _, row in display_df.iterrows():
            html += "<tr>"
            for value in row:
                # 格式化数值
                if isinstance(value, (int, float)):
                    if isinstance(value, float):
                        formatted_value = f"{value:.2f}"
                    else:
                        formatted_value = str(value)
                else:
                    formatted_value = str(value)
                html += f"<td>{formatted_value}</td>"
            html += "</tr>"
        
        html += """
                </tbody>
            </table>
        """
        
        if len(df) > 10:
            html += f"<p><em>显示前10行，共{len(df)}行数据</em></p>"
        
        html += "</section>"
        return html

    def _generate_quality_section(self, quality_data: Dict[str, Any]) -> str:
        """生成数据质量部分"""
        return f"""
        <section class="section">
            <h2 class="section-title">数据质量评估</h2>
            <div class="quality-grid">
                <div class="quality-item">
                    <div class="quality-number">{quality_data.get('total_records', 0)}</div>
                    <div class="quality-label">总记录数</div>
                </div>
                <div class="quality-item">
                    <div class="quality-number">{quality_data.get('total_columns', 0)}</div>
                    <div class="quality-label">总列数</div>
                </div>
                <div class="quality-item">
                    <div class="quality-number">{quality_data.get('missing_values', 0)}</div>
                    <div class="quality-label">缺失值</div>
                </div>
                <div class="quality-item">
                    <div class="quality-number">{quality_data.get('duplicate_records', 0)}</div>
                    <div class="quality-label">重复记录</div>
                </div>
            </div>
        </section>
        """


class PDFReportGeneratorOperator(OperatorABC):
    """PDF报告生成算子"""
    
    def __init__(
        self,
        report_title: str = "数据分析报告",
        output_file: str = "analysis_report.pdf",
        page_size: str = "A4"
    ):
        """
        初始化PDF报告生成算子
        
        Args:
            report_title: 报告标题
            output_file: 输出文件名
            page_size: 页面大小
        """
        self.report_title = report_title
        self.output_file = output_file
        self.page_size = A4 if page_size == "A4" else letter

    def run(self, storage: MaestroStorage, **kwargs) -> Dict[str, Any]:
        """生成PDF报告"""
        if not REPORTLAB_AVAILABLE:
            raise ImportError("需要安装reportlab: pip install reportlab")
        
        if hasattr(storage, 'operator_step') and storage.operator_step == -1:
            storage = storage.step()
        
        df = storage.read(output_type="dataframe")
        
        # 获取分析报告数据
        analysis_report = getattr(df, 'attrs', {}).get('analysis_report', {})
        
        # 生成PDF报告
        self._generate_pdf_report(df, analysis_report)
        
        path = storage.write(df)
        
        return {
            "path": path,
            "report_path": self.output_file,
            "report_size": os.path.getsize(self.output_file) if os.path.exists(self.output_file) else 0
        }

    def _generate_pdf_report(self, df: pd.DataFrame, analysis_report: Dict[str, Any]):
        """生成PDF报告"""
        doc = SimpleDocTemplate(self.output_file, pagesize=self.page_size)
        styles = getSampleStyleSheet()
        story = []
        
        # 标题
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=1  # 居中
        )
        story.append(Paragraph(self.report_title, title_style))
        story.append(Spacer(1, 12))
        
        # 生成时间
        story.append(Paragraph(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
        story.append(Spacer(1, 20))
        
        # 数据概览
        if analysis_report and 'data_overview' in analysis_report:
            story.append(Paragraph("数据概览", styles['Heading2']))
            overview = analysis_report['data_overview']
            
            overview_data = [
                ['项目', '值'],
                ['时间范围', overview.get('time_range', '未知')],
                ['数据记录', f"{overview.get('total_records', 0)} 条"],
                ['指标数量', f"{overview.get('total_columns', 0)} 个"]
            ]
            
            overview_table = Table(overview_data)
            overview_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 14),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(overview_table)
            story.append(Spacer(1, 20))
        
        # 统计分析
        if analysis_report and 'statistical_analysis' in analysis_report:
            story.append(Paragraph("统计分析结果", styles['Heading2']))
            
            for column, analysis in analysis_report['statistical_analysis'].items():
                if 'error' in analysis:
                    continue
                
                story.append(Paragraph(f"{column}:", styles['Heading3']))
                
                analysis_data = [
                    ['指标', '数值'],
                    ['最大值', f"{analysis.get('max_value', 'N/A')}"],
                    ['最小值', f"{analysis.get('min_value', 'N/A')}"],
                    ['平均值', f"{analysis.get('mean_value', 'N/A'):.2f}"],
                    ['标准差', f"{analysis.get('std_value', 'N/A'):.2f}"]
                ]
                
                if 'total_growth_rate' in analysis:
                    analysis_data.extend([
                        ['总增长率', f"{analysis['total_growth_rate']}%"],
                        ['年均增长率', f"{analysis['annual_growth_rate']}%"]
                    ])
                
                analysis_table = Table(analysis_data)
                analysis_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                
                story.append(analysis_table)
                story.append(Spacer(1, 12))
        
        # 构建PDF
        doc.build(story)


class ReportTemplateOperator(OperatorABC):
    """报告模板算子，提供多种报告模板"""
    
    def __init__(
        self,
        template_name: str = "comprehensive",
        output_format: str = "html",
        custom_sections: Optional[List[str]] = None
    ):
        """
        初始化报告模板算子
        
        Args:
            template_name: 模板名称 ("comprehensive", "executive", "technical")
            output_format: 输出格式 ("html", "pdf")
            custom_sections: 自定义章节列表
        """
        self.template_name = template_name
        self.output_format = output_format
        self.custom_sections = custom_sections

    def run(self, storage: MaestroStorage, **kwargs) -> Dict[str, Any]:
        """生成模板化报告"""
        if hasattr(storage, 'operator_step') and storage.operator_step == -1:
            storage = storage.step()
        
        df = storage.read(output_type="dataframe")
        
        # 根据模板类型选择报告生成器
        if self.template_name == "comprehensive":
            report_title = "综合数据分析报告"
            include_charts = True
            include_data_table = True
        elif self.template_name == "executive":
            report_title = "执行摘要报告"
            include_charts = True
            include_data_table = False
        else:  # technical
            report_title = "技术分析报告"
            include_charts = True
            include_data_table = True
        
        if self.output_format == "html":
            generator = HTMLReportGeneratorOperator(
                report_title=report_title,
                include_charts=include_charts,
                include_data_table=include_data_table
            )
        else:
            generator = PDFReportGeneratorOperator(
                report_title=report_title
            )
        
        result = generator.run(storage)
        
        return result