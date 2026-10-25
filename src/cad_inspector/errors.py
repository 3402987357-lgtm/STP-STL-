"""我为项目提供统一、可读、可追踪的异常类型。"""


class InspectorError(Exception):
    """我使用这个异常表示所有可以向用户解释的项目错误。"""


class InputValidationError(InspectorError):
    """我使用这个异常表示输入文件或参数不满足要求。"""


class DependencyMissingError(InspectorError):
    """我使用这个异常表示某个可选能力缺少运行依赖。"""


class GeometryProcessingError(InspectorError):
    """我使用这个异常表示几何计算或格式解析失败。"""
