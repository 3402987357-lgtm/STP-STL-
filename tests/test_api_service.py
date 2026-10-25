"""我为 HTTP 服务区域写回归测试。

我只在这个文件里真正跑一次真实件（一次几十秒，可接受），其余用例都用
"不真跑检测"的方式覆盖契约。原因和批量测试一样：服务层要验的是**编排与
边界**，检测本身的正确性由 inspect_pair 自己的测试覆盖，在这里重复跑
几十次只会让门禁慢到没人愿意跑。
"""

from __future__ import annotations

import shutil
import tempfile
import time
from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi")
TestClient = pytest.importorskip("fastapi.testclient").TestClient

from cad_inspector.api.service import _resolve_within, create_app  # noqa: E402

# 我为什么按仓库根定位：这里原本写相对路径 Path("data")，
# 只在"当前工作目录恰好是仓库根"时才找得到，别处执行 pytest 会静默 skip。
ROOT = Path(__file__).resolve().parents[1]


def _reference_source() -> Path:
    """我返回甲方提供的真实参考件路径。

    单独抽出来是因为有个用例不走 `api_root` fixture（它要注入
    `_run_inspection`），但仍需要同一个真实件当参考件。
    """

    return (
        ROOT
        / "data"
        / "provided"
        / "三维模型破损区域智能修补"
        / "测试集"
        / "6.stl"
    )


@pytest.fixture()
def api_root(tmp_path: Path) -> Path:
    """我准备一个只允许读取自己的临时根目录。"""

    root = tmp_path / "allowed"
    root.mkdir()
    source = _reference_source()
    if not source.exists():
        pytest.skip("甲方提供的真实件不存在。")
    shutil.copy(source, root / "6.stl")
    shutil.copy(source, root / "ref.stl")
    return root


def test我方健康检查报告允许根目录(api_root: Path, tmp_path: Path) -> None:
    client = TestClient(create_app(api_root, output_root=tmp_path / "out"))

    body = client.get("/health").json()

    assert body["status"] == "OK"
    assert body["allowed_root"] == str(api_root.resolve())
    assert body["job_count"] == 0


def test我方拒绝根目录之外的路径(api_root: Path, tmp_path: Path) -> None:
    """越界路径是 400 而不是 500：前者是"你传错了"，后者看着像"服务坏了"。

    我为什么要钉这条：这个接口能读本机文件，不限制根目录的话一个
    `../../` 就能把任意文件当模型读进来。
    """

    client = TestClient(create_app(api_root, output_root=tmp_path / "out"))

    for bad in ("../../Windows/win.ini", "/Windows/win.ini", "..\\..\\etc/passwd"):
        response = client.post(
            "/inspect", json={"candidate": bad, "reference": "6.stl"}
        )
        assert response.status_code == 400, bad
        assert "根目录之外" in response.json()["detail"]


def test我方找不到文件时返回404(api_root: Path, tmp_path: Path) -> None:
    client = TestClient(create_app(api_root, output_root=tmp_path / "out"))

    response = client.post(
        "/inspect", json={"candidate": "nope.stl", "reference": "6.stl"}
    )

    assert response.status_code == 404
    assert "我找不到候选件文件" in response.json()["detail"]


def test我方查询不存在的任务返回404(api_root: Path, tmp_path: Path) -> None:
    client = TestClient(create_app(api_root, output_root=tmp_path / "out"))

    assert client.get("/jobs/nope").status_code == 404


def test我方提交后立刻返回任务句柄而不是阻塞等结果(
    api_root: Path, tmp_path: Path
) -> None:
    """同步接口会把线程占死几十秒，所以必须是"提交 -> 拿 id -> 查"。"""

    client = TestClient(create_app(api_root, output_root=tmp_path / "out"))

    response = client.post(
        "/inspect", json={"candidate": "6.stl", "reference": "ref.stl"}
    )

    assert response.status_code == 200
    body = response.json()
    # 提交时任务已在跑（或已跑完），但一定带着 job_id 与状态。
    assert body["job_id"]
    assert body["status"] in {"RUNNING", "DONE"}
    assert body["candidate"].endswith("6.stl")
    assert body["reference"].endswith("ref.stl")


def test我方任务最终能拿到检测结果(api_root: Path, tmp_path: Path) -> None:
    """端到端跑一次真实件，确认结果结构与单件 CLI 一致。"""

    client = TestClient(create_app(api_root, output_root=tmp_path / "out"))
    job_id = client.post(
        "/inspect", json={"candidate": "6.stl", "reference": "ref.stl"}
    ).json()["job_id"]

    # 检测是 CPU 密集活，轮询等待而不是固定 sleep。
    # 每次轮询之间必须让出一点时间：紧密空转会把 CPU 全占在轮询上，
    # 反倒拖慢线程池里那个真正干活的检测线程。
    for _ in range(300):
        job = client.get(f"/jobs/{job_id}").json()
        if job["status"] in {"DONE", "FAILED"}:
            break
        time.sleep(0.5)

    assert job["status"] == "DONE", job.get("error")
    assert job["error"] is None
    result = job["result"]
    assert result is not None
    # 与单件 CLI 同一套 schema，便于用同一套审计脚本检查。
    assert result["schema_version"] == "missing-region-detection-1.3"
    assert "regions" in result
    assert "warnings" in result


def test我方失败任务保留错误与堆栈(api_root: Path, tmp_path: Path) -> None:
    """失败必须留痕：静默失败比明确失败危险得多。"""

    (api_root / "bad.stl").write_bytes(b"not a real stl at all")
    client = TestClient(create_app(api_root, output_root=tmp_path / "out"))
    job_id = client.post(
        "/inspect", json={"candidate": "bad.stl", "reference": "6.stl"}
    ).json()["job_id"]

    # 预算要和真实耗时匹配：单件真实检测实测约 6 秒，
    # 但线程池只有 2 个 worker，与另一个端到端用例并发时要排队，
    # 原来 60*0.2=12 秒的预算在门禁机器上不够，会留假失败。
    for _ in range(300):
        job = client.get(f"/jobs/{job_id}").json()
        if job["status"] in {"DONE", "FAILED"}:
            break
        time.sleep(0.5)

    # 必须**先断言终态**再看 error：轮询超时退出时 job 还是上一轮的
    # RUNNING 快照，error 自然是 None。直接断言 error 会得到一个
    # 看起来像"服务没留痕"的假失败，真凶是等待没等到。
    #
    # 我为什么把三条断言合成一条、并把整个 job 打出来（2026-10-15 订正）：
    # 这里曾经报过 "assert None"，我此前一直读成 "status=FAILED 但
    # error=None"。**那是错的** —— 原来的断言在 break 之后没有先查 status，
    # 所以它证明不了 status 是什么。真相是轮询预算 12 秒不够（线程池
    # max_workers=2 要排队），job 仍是 RUNNING 快照，error 自然为 None。
    # 现在失败时一次性给出 status / error / traceback / 完整 job，
    # 下次撞上就能当场定性，不用再靠重跑碰运气。
    assert job["status"] == "FAILED", f"任务未进入终态，最后一次状态：{job}"
    assert (
        job["error"] and job["traceback"]
    ), f"我进了终态却没留痕：status={job['status']} error={job['error']!r} traceback={job['traceback']!r} job={job}"
    # traceback 必须完整保留，否则事后没法定位到底哪一步炸的。
    assert job["traceback"]


def test我方失败任务先留痕后置终态() -> None:
    """我钉住"先留痕、后置终态"这个顺序，以及"置终态必须自带兜底"。

    为什么顺序必须是这样：任务对象在工作线程与 HTTP 轮询线程之间共享。
    若先 `status = "FAILED"`、后 `error = ...`，就存在一个真实窗口——
    调用方看到"已失败"却拿到 error=None，也就是"静默失败"。
    反过来写：只要能看到 FAILED，error 和 traceback 必然已就位。
    我扫源码文本来钉顺序，因为这种竞态靠跑是跑不出来的
    （窗口极小，只在机器负载高时才偶尔命中）。

    **关于那条 flaky，我要订正一个流传了几轮的错误推断（2026-10-15）**：
    本文件上面那条"保留错误与堆栈"曾报 `assert None`，我此前一直把它
    读成"status=FAILED 但 error=None"，并据此认为存在第二条逃逸路径。
    **那是错的。**原始断言在 break 之后**没有先查 status**，
    所以它证明不了 status 是什么。真相（开发记录 16189 行早有记载）：
    轮询预算只有 12 秒，线程池 max_workers=2 且要与另一个端到端用例排队，
    预算内没跑完时 job 仍是上一轮的 **RUNNING 快照**，error 自然是 None。
    **真凶是"等待没等到"，不是"服务没留痕"**——这是一次假失败。
    预算已放宽到 300x0.5=150 秒，且先断言终态，所以那类假失败
    现在会明说"卡在 RUNNING"，不会再伪装成"服务没留痕"。

    那我为什么**仍然保留** `_mark_failed()` 兜底和这条用例：
    顺序正确只能保证"我自己写的路径不会漏"，但兜底能把"任何未知路径
    绕过来"的最坏后果从 None 降级成一条说明文案。这是廉价的防御，
    不是因为证实了有第二条路径（我 12 次 x 4000 次轮询 0 命中、
    6 种异常形态穷举也 0 次静默失败）。**兜底留着，但别再拿它当"还有
    一条未知路径"的证据去追。**
    """

    import inspect

    from cad_inspector.api import service

    source = inspect.getsource(service)
    lines = [
        stripped
        for stripped in (raw.strip() for raw in source.splitlines())
        if stripped
    ]

    def index_of(fragment: str, *, start: int) -> int:
        for offset in range(start, len(lines)):
            if fragment in lines[offset]:
                return offset
        raise AssertionError(f"我没找到 {fragment!r}，服务实现可能改了")

    # 兜底必须存在：置终态的那一步自己要保证 error/traceback 不为空，
    # 不能只靠调用方记得先赋值。这是我被第二次 flaky 逼出来的加固。
    fallback_at = index_of("def _mark_failed", start=0)
    assert index_of("if not job.error", start=fallback_at) > fallback_at, (
        "置终态时必须保证 error 非空：否则调用方会拿到 error=None，"
        "也就是静默失败。"
    )
    assert index_of("if not job.traceback", start=fallback_at) > fallback_at, (
        "置终态时必须保证 traceback 非空：否则事后没法定位哪一步炸的。"
    )

    # 不变式一：兜底函数内部，留痕补齐必须排在置终态之前。
    # （它定义在 _task 之前，所以这里的下标都从 fallback_at 往后找。）
    final_status_at = index_of('job.status = "FAILED"', start=fallback_at)
    fallback_error_at = index_of("if not job.error", start=fallback_at)
    fallback_traceback_at = index_of("if not job.traceback", start=fallback_at)
    assert fallback_error_at < final_status_at, (
        "兜底里也要先补 error 再置终态：反了就会让调用方看到 FAILED 却拿到 None。"
    )
    assert fallback_traceback_at < final_status_at, "traceback 兜底同理。"

    # 不变式二：两个 except 分支都必须走 _mark_failed()，
    # 不允许任何一处图省事直接 job.status = "FAILED"。
    task_at = index_of("def _task", start=0)
    cursor = task_at
    checked = 0
    for _ in range(2):
        error_at = index_of("job.error = ", start=cursor)
        traceback_at = index_of("job.traceback = ", start=cursor)
        mark_at = index_of("_mark_failed()", start=cursor)
        assert error_at < mark_at, (
            "我先写 error 再置终态，顺序不能反：反了就会让调用方"
            f"看到 FAILED 却拿到 error=None（第 {mark_at} 行 vs {error_at} 行）"
        )
        assert traceback_at < mark_at, "traceback 同样必须在终态之前写完，理由同上。"
        checked += 1
        cursor = mark_at + 1
        if cursor >= len(lines):
            break
    assert checked == 2, f"两个 except 分支都得走 _mark_failed()，实际只找到 {checked} 处"


def test我方留痕自己炸了也要有终态() -> None:
    """我钉住"任务永远卡在 RUNNING"这条最难排查的失败。

    某次排查里我发现：**异常处理器里抛出的新异常，不会被同级的
    except 接住**。所以若留痕那一步自己炸了（例如某个异常的
    ``__str__`` 返回非字符串，Python 会抛 ``TypeError``），
    异常会从 ``except InspectorError`` 块里直接冒出 ``_task``，
    executor 只把它吞进 Future —— 任务**永远停在 RUNNING**，
    调用方轮询到超时也等不到终态。

    这比"失败了但说不清原因"更糟：连失败都没报出来。
    修法是在里外各包一层 try，让最外层兜住这种情况。
    这里用"__str__ 返回 None"精确复现那条路径。
    """

    from cad_inspector.api import service
    from cad_inspector.errors import InspectorError

    class BrokenMessageError(InspectorError):
        def __str__(self) -> str:  # type: ignore[invalid-return-type]
            return None  # 返回非字符串，逼 Python 抛 TypeError

    def boom(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise BrokenMessageError()

    original = service._run_inspection
    service._run_inspection = boom  # type: ignore[assignment]
    try:
        root = Path(tempfile.mkdtemp())
        (root / "bad.stl").write_bytes(b"not a real stl at all")
        shutil.copy(_reference_source(), root / "6.stl")
        client = TestClient(
            create_app(root, output_root=Path(tempfile.mkdtemp()) / "out")
        )
        job_id = client.post(
            "/inspect", json={"candidate": "bad.stl", "reference": "6.stl"}
        ).json()["job_id"]
        for _ in range(200):
            job = client.get(f"/jobs/{job_id}").json()
            if job["status"] in {"DONE", "FAILED"}:
                break
            time.sleep(0.2)
    finally:
        service._run_inspection = original  # type: ignore[assignment]

    assert job["status"] == "FAILED", f"任务不该卡在非终态：{job}"
    assert job["error"], f"留痕炸了也要留下原因：{job}"


def test我方拒绝不存在的允许根目录(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="allowed_root"):
        create_app(tmp_path / "nope")


def test我方解析同根下相似前缀不被误判(api_root: Path) -> None:
    """`is_relative_to` 而不是 startswith：后者会把 root_b 当成 root 的子目录。"""

    (api_root / "sub").mkdir()
    allowed = _resolve_within(api_root, "sub/../6.stl")
    assert allowed == (api_root / "6.stl").resolve()
