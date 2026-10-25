"""我用标准库和 NumPy 读取、写出 ASCII 与 Binary STL。

算法原理：STL 有两种互不兼容的格式，而且**不能靠文件头是 ASCII 就判定为 ASCII 格式**——
二进制 STL 的 80 字节头部是任意字节，完全可能以 "solid" 开头。我按信号强度排序判断：

1. 长度精确匹配（84 + 三角面数 × 50 == 文件长度）是最强的二进制信号。ASCII 文本不可能
   在偏移 80 处编码出与文件长度自洽的三角面数（那要求后三个字节全是 NUL）。这一步必须
   排在任何文本启发式之前，否则"头部以 solid 开头且含换行"的合法二进制 STL 会被误判成
   空 ASCII STL——这是本项目真实修过的一个 bug。
2. 头部前 80 字节含换行 -> 判为 ASCII（二进制头部几乎不会出现换行）。
3. 声明的三角面数超过文件长度能容纳的上限 -> 不可能是完整二进制。
4. 数据区含 NUL 字节 -> 二进制。

读取时按"坐标四舍五入到 1e-12"做顶点去重，让相邻三角面共享的顶点合并成同一索引；
写出统一用 ASCII 并走原子替换，避免中断留下半个文件。
"""

from __future__ import annotations

import re
import struct
from pathlib import Path

import numpy as np

from ..errors import GeometryProcessingError, InputValidationError
from .mesh import TriangleMesh


def load_stl(path: str | Path) -> TriangleMesh:
    """我自动判断 STL 类型，并把它统一成 TriangleMesh。"""

    path = Path(path)
    if not path.exists():
        raise InputValidationError(f"我找不到 STL 文件：{path}")
    if path.suffix.lower() != ".stl":
        raise InputValidationError(f"我只在这个函数中接受 .stl 文件：{path}")
    data = path.read_bytes()
    if not data:
        raise GeometryProcessingError(f"我发现 STL 文件为空：{path}")
    # 判据顺序不能调整：长度精确匹配必须先于任何文本启发式。
    if _has_plausible_binary_layout(data):
        return _parse_binary_stl(data, path)
    if _looks_like_truncated_binary_stl(data):
        return _parse_binary_stl(data, path)
    if _looks_like_ascii_stl(data):
        mesh = _parse_ascii_stl(data, path)
        if mesh is not None:
            return mesh
        raise GeometryProcessingError(
            f"我发现 ASCII STL 结构不完整或已截断：{path}"
        )
    if _looks_like_ascii_stl_header(data):
        raise GeometryProcessingError(
            f"我发现 ASCII STL 不包含任何三角面：{path}"
        )
    return _parse_binary_stl(data, path)


def save_stl(mesh: TriangleMesh, path: str | Path, solid_name: str = "cad_inspector") -> None:
    """我把网格保存为便于交换的 ASCII STL。"""

    # 延迟导入避免与 output 包形成循环依赖。
    from ..output.atomic import atomic_path

    path = Path(path)
    normals = mesh.triangle_normals
    # 走 atomic_path：写到临时文件再原子替换。
    # 直接覆盖会让导出中途失败时留下半截 STL，
    # 下游拿到的就是能打开但几何残缺的文件，比"文件不存在"更难发现。
    with atomic_path(path) as temporary:
        # 固定 ascii 编码与 \n 换行：STL 是交换格式，
        # 用本机默认编码会让中文 solid 名在别的机器上变成乱码。
        with temporary.open("w", encoding="ascii", newline="\n") as handle:
            handle.write(f"solid {solid_name}\n")
            for face_id, face in enumerate(mesh.faces):
                normal = normals[face_id]
                # %.9g 而不是 %f：STL 没有精度约定，%f 会把小件坐标
                # 全部写成 0.000000，%g 既省体积又保住有效位。
                handle.write(
                    "  facet normal "
                    f"{normal[0]:.9g} {normal[1]:.9g} {normal[2]:.9g}\n"
                )
                handle.write("    outer loop\n")
                for vertex_id in face:
                    point = mesh.vertices[int(vertex_id)]
                    handle.write(
                        f"      vertex {point[0]:.9g} {point[1]:.9g} "
                        f"{point[2]:.9g}\n"
                    )
                handle.write("    endloop\n")
                handle.write("  endfacet\n")
            handle.write(f"endsolid {solid_name}\n")


def _looks_like_ascii_stl(data: bytes) -> bool:
    """我判断文件是否是带 facet/vertex 结构的 ASCII STL。"""
    # 先看开头是不是 solid：二进制 STL 的前 80 字节是任意头信息，
    # 恰好可能以 "solid" 开头，所以单看开头不足以判定。
    prefix = data[:512].lstrip().lower()
    if not prefix.startswith(b"solid"):
        return False
    # 再要求正文中真的有 facet/vertex：只看头几个 KB 就够，
    # 全文件扫描在大模型上会白白多花几秒。
    text = data[: min(len(data), 8192)].decode("utf-8", errors="ignore").lower()
    return "facet" in text and "vertex" in text


def _looks_like_ascii_stl_header(data: bytes) -> bool:
    """我识别以 solid 开头但没有任何 facet 的 ASCII STL。

    这种文件通常是空模型导出，不应该被回退成二进制解析并报出“头部不完整”
    这类与事实不符的错误。
    """

    return data[:512].lstrip().lower().startswith(b"solid")


def _has_exact_binary_size(data: bytes) -> bool:
    """我优先识别长度严格匹配的 Binary STL，处理 header 以 solid 开头的情况。"""

    if len(data) < 84:
        return False
    triangle_count = struct.unpack_from("<I", data, 80)[0]
    return 84 + triangle_count * 50 == len(data)


def _has_plausible_binary_layout(data: bytes) -> bool:
    """我识别完整或截断的二进制 STL，避免被 solid 文本头部伪装。"""

    if len(data) < 84:
        return False
    # 长度精确匹配是最强的二进制信号：ASCII 文本不可能在偏移 80 处用可打印
    # 字节编码出与文件长度自洽的三角面数（那要求 b1..b3 全为 NUL）。所以这
    # 一步必须排在行尾启发式之前，否则"solid 开头且头里带换行"的合法二进制
    # STL 会被误判成空 ASCII STL。
    if _has_exact_binary_size(data):
        return True
    if b"\n" in data[:80] or b"\r" in data[:80]:
        return False
    triangle_count = struct.unpack_from("<I", data, 80)[0]
    max_count_for_size = max((len(data) - 84 + 49) // 50, 0)
    if triangle_count > max_count_for_size:
        return False
    return b"\x00" in data[84:]


def _looks_like_truncated_binary_stl(data: bytes) -> bool:
    """我识别带二进制零字节且声明了超出文件长度三角面数的损坏 STL。"""

    if len(data) < 84 or (
        b"\x00" not in data[:80] and b"\x00" not in data[84:]
    ):
        return False
    triangle_count = struct.unpack_from("<I", data, 80)[0]
    return triangle_count > 0 and 84 + triangle_count * 50 > len(data)


def _parse_ascii_stl(data: bytes, path: Path) -> TriangleMesh | None:
    """我解析 ASCII STL；结构不完整时返回 None 而不是抛出模糊异常。"""
    text = data.decode("utf-8", errors="ignore")
    # 我试过把下面这行改成流式 finditer 以省内存（2026-10-17），
    # 但同进程 A/B 实测：流式反而慢 1.8 倍（6.24s vs 3.49s），
    # 只省下 35MB（25 万面）/ 约 245MB（178 万面），
    # 而大件的峰值是 GB 级，这点省下来不足以改变它能不能跑。
    # 慢 1.8 倍却是每份文件都要付的。所以**改回 list**，理由留在这里，
    # 免得下一个人（或下一次的我）再试一遍同样的东西。
    facet_blocks = list(
        re.finditer(
            r"\bfacet\b(?P<body>.*?)\bendfacet\b",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
    )
    if not facet_blocks:
        return None
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    index_by_key: dict[tuple[float, float, float], int] = {}
    # 每个 facet 必须有 outer loop / endloop 与恰好三个 vertex，否则判为结构不完整。
    for block in facet_blocks:
        body = block.group("body")
        if not re.search(r"\bouter\s+loop\b", body, flags=re.IGNORECASE):
            return None
        if not re.search(r"\bendloop\b", body, flags=re.IGNORECASE):
            return None
        matches = re.findall(
            r"\bvertex\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)",
            body,
            flags=re.IGNORECASE,
        )
        if len(matches) != 3:
            return None
        face_indices: list[int] = []
        for raw in matches:
            point = tuple(float(value) for value in raw)
            key = tuple(round(value, 12) for value in point)
            if key not in index_by_key:
                index_by_key[key] = len(vertices)
                vertices.append(point)
            face_indices.append(index_by_key[key])
        faces.append(tuple(face_indices))
    return TriangleMesh(
        vertices=np.asarray(vertices, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int64),
        name=path.name,
        metadata={
            "format": "ascii_stl",
            "unit": "unknown",
            "source_unit": "unknown",
            "coordinate_unit": "unknown",
            "unit_scale_to_mm": None,
            "unit_normalization": "not_available",
        },
    )


def _parse_binary_stl(data: bytes, path: Path) -> TriangleMesh:
    """我解析 Binary STL；头部或长度不匹配时给出明确的字节数错误信息。"""
    if len(data) < 84:
        # 84 = 80 字节头 + 4 字节三角面计数。不足就说明文件被截断，
        # 继续解包会在 struct 层抛一个看不懂的错误。
        raise GeometryProcessingError(f"我发现 Binary STL 头部不完整：{path}")
    triangle_count = struct.unpack_from("<I", data, 80)[0]
    if triangle_count == 0:
        # 零面直接拒绝：空网格在后面所有判据里都会走退化分支，
        # 与其让它流下去产生一堆无意义的空结果，不如在这里说清楚。
        raise GeometryProcessingError(
            f"我发现 Binary STL 不包含任何三角面：{path}"
        )
    # 每个三角面固定 50 字节：12 个 float（法向 3 + 顶点 9）
    # 加 2 字节属性。用声明的面数反推文件长度是否吻合，
    # 这是发现"文件截断/拼接错误"最直接的一处校验。
    expected_size = 84 + triangle_count * 50
    if expected_size != len(data):
        raise GeometryProcessingError(
            f"我发现 Binary STL 长度不匹配：期望 {expected_size} 字节，实际 {len(data)} 字节。"
        )
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    # 按坐标去重合并顶点：STL 是每个面独立存三个顶点，
    # 不合并的话同一空间点会有很多份，邻接与边都无从谈起。
    index_by_key: dict[tuple[float, float, float], int] = {}
    offset = 84
    for _ in range(triangle_count):
        values = struct.unpack_from("<12fH", data, offset)
        offset += 50
        # 跳过前三个 float（面法向）：它通常是导出时算的，
        # 与顶点实际朝向可能不一致，一律按顶点重算，不采用文件里的值。
        triangle_vertices = (
            (float(values[3]), float(values[4]), float(values[5])),
            (float(values[6]), float(values[7]), float(values[8])),
            (float(values[9]), float(values[10]), float(values[11])),
        )
        face_indices: list[int] = []
        for point in triangle_vertices:
            # 坐标先 round 到 12 位再当键：同一个点在不同面里
            # 可能有 1e-16 级的差异，不 round 就合并不上，
            # 于是"共享顶点"永远判不出来。
            key = tuple(round(value, 12) for value in point)
            if key not in index_by_key:
                index_by_key[key] = len(vertices)
                vertices.append(point)
            face_indices.append(index_by_key[key])
        faces.append(tuple(face_indices))
    return TriangleMesh(
        vertices=np.asarray(vertices, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int64),
        name=path.name,
        metadata={
            "format": "binary_stl",
            # STL 格式本身不携带单位信息，如实标 unknown：
            # 猜测成 mm 会让后面所有毫米量级的阈值失去依据。
            "unit": "unknown",
            "source_unit": "unknown",
            "coordinate_unit": "unknown",
            "unit_scale_to_mm": None,
            "unit_normalization": "not_available",
        },
    )
