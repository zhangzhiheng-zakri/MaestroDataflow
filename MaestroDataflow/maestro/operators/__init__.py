"""MaestroDataflow operators package.
"""

from .basic_ops import FilterRowsOperator, SelectColumnsOperator, MapRowsOperator
from .io_ops import LoadDataOperator, SaveDataOperator
from .llm_ops import LLMGenerateOperator
from .analytics_ops import DataAnalysisOperator, DataSummaryOperator
from .visualization_ops import ChartGeneratorOperator, DashboardGeneratorOperator
from .dataset_ops import DatasetPackagingOperator
# 可选导入报告相关操作符，避免在缺少系统依赖（如weasyprint的libgobject）时阻塞其他功能
try:
    from .report_ops import HTMLReportGeneratorOperator, PDFReportGeneratorOperator, ReportTemplateOperator
    _REPORT_OPS_AVAILABLE = True
except Exception:
    HTMLReportGeneratorOperator = None
    PDFReportGeneratorOperator = None
    ReportTemplateOperator = None
    _REPORT_OPS_AVAILABLE = False
from .column_ops import ColumnMeaningGeneratorOperator, ColumnMetadataExtractorOperator
from .data_column_process_ops import DataColumnProcessOperator, QuickDataColumnProcessOperator

__all__ = [
    "FilterRowsOperator",
    "SelectColumnsOperator", 
    "MapRowsOperator",
    "LoadDataOperator",
    "SaveDataOperator",
    "LLMGenerateOperator",
    "DataAnalysisOperator",
    "DataSummaryOperator",
    "ChartGeneratorOperator",
    "DashboardGeneratorOperator",
    "ColumnMeaningGeneratorOperator",
    "ColumnMetadataExtractorOperator",
    "DataColumnProcessOperator",
    "QuickDataColumnProcessOperator"
]

# 成功导入报告操作符时再暴露到__all__
if _REPORT_OPS_AVAILABLE:
    __all__ += [
        "HTMLReportGeneratorOperator",
        "PDFReportGeneratorOperator",
        "ReportTemplateOperator",
    ]

__all__.append("DatasetPackagingOperator")