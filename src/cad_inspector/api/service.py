"""我提供缺失区域检测的 HTTP 服务。

我为什么不用"一个同步接口直接返回结果"：单件检测在真实数据上要几十秒，
同步接口会把 worker 线程占死，第二个请求只能排队，而且 HTTP 客户端常在
60 秒超时。所以我做成**提交任务 -> 拿 job_id -> 轮询/取结果**两步，
检测本身放到线程池里跑（它是 OCCT/numpy 的 CPU 密集活，`run_in_executor`
不会阻塞事件循环）。

我守住的几条边界：

- **路径必须落在允许的根目录内**。这个服务能读本机文件，若不限制根目录，
  一个 `../../` 就能把任意文件当模型读进来。我在解析后做 `resolve()` 再
  比对前缀，而不是只做字符串包含判断——后者会被 `/a/b_c` 绕过 `/a/b`。
- **任务结果落盘**，不只留在内存。服务重启后已完成的结论必须还能取到，
  否则甲方没法复核。
- **失败任务的 traceback 完整保留**，和 ``batch_inspect`` 的契约一致：
  静默失败比明确失败危险得多。
"""

from __future__ import annotations

import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator

from ..errors import InspectorError


def _build_request_model():
    """我定义并返回请求模型类。

    两个约束在这里是**互相拉扯**的，我把取舍写清楚：

    1. pydantic 是随 fastapi 一起的可选依赖，不能在模块顶层无条件
       `import pydantic`，否则它变成必需依赖；
    2. 但请求模型类**又不能定义在 create_app 函数体内部**——FastAPI
       无法解析局部类的注解，会把请求体当成**查询参数**去要，
       实际表现是"必填字段缺失"的 422（loc 为 ['query', ...]），
       而不是任何显式报错，极难排查。

    我的解法：类定义写在这个函数里，但把返回值赋给**模块级**名字
    `InspectRequest`，这样注解指向的是一个模块级可解析的对象。
    """

    from pydantic import BaseModel, Field

    class InspectRequest(BaseModel):
        """我接收一次检测请求的入参。"""

        candidate: str = Field(..., description="候选件相对 allowed_root 的路径。")
        reference: str = Field(..., description="参考件相对 allowed_root 的路径。")

    return InspectRequest


# 模块级名字：FastAPI 靠它解析 /inspect 的请求体注解。
# 缺省为 None，表示"pydantic 未安装"，此时 create_app 会给出明确提示。
InspectRequest = None


@dataclass
class Job:
    """我记录一个检测任务的状态与结果。"""

    job_id: str
    candidate: str
    reference: str
    status: str = "PENDING"
    result: dict[str, Any] | None = None
    error: str | None = None
    traceback: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """我把任务状态转成可序列化的字典。"""

        return {
            "job_id": self.job_id,
            "candidate": self.candidate,
            "reference": self.reference,
            "status": self.status,
            "result": self.result,
            "error": self.error,
            "traceback": self.traceback,
        }


@dataclass
class JobStore:
    """我在内存里保存任务，并把已完成的结果写盘。

    我为什么还要写盘：只放内存的话服务一重启，甲方就取不到已完成的
    结论，也没法事后复核。写盘路径与单件 CLI 的输出布局一致，
    便于用同一套审计脚本检查。
    """

    output_root: Path
    jobs: dict[str, Job] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """我确保输出根目录存在。"""

        self.output_root.mkdir(parents=True, exist_ok=True)


def _dump_silence_forensics(job: "Job", output_root: Path) -> None:
    """我把"静默失败"触发瞬间的现场写盘，供事后取证。

        我为什么**仍然保留**这个函数（2026-10-15 订正）：我曾以为存在
    "status=FAILED 但 error=None" 的现象并查了很久，证否了三个假说
    （写入顺序窗口 / job_id 冲突 / __str__ 返回 None），
    12x4000 次高频轮询也没抓到。**真相是当初列为第四个假说的
    "等待预算不够" —— 它才是真凶**：轮询预算 12 秒不足时 job 仍是
    RUNNING 快照，error 合法地为 None（详见 tests/test_api_service.py）。
    也就是说那个现象**从来没有发生过**，是我把"看起来像"当成了"是"。
    但取证仍然留着：兜底一旦被触发，说明有我没预料到的路径走了进来，
    那时我要的是现场物证，而不是再猜一轮。

    取证自己绝不能再制造失败：写盘的任何异常我都吞掉。
    """

    import datetime
    import json as _json
    import threading

    try:
        folder = output_root / "_silence_forensics"
        folder.mkdir(parents=True, exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        payload = {
            "触发时间": stamp,
            "job": job.to_dict(),
            "线程": threading.current_thread().name,
            "活跃线程数": threading.active_count(),
            "当时调用栈": traceback.format_stack()[:-1],
        }
        target = folder / f"{stamp}-{job.job_id}.json"
        target.write_text(
            _json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:  # noqa: BLE001 - 取证失败不该影响主流程
        pass


def _resolve_within(root: Path, raw: str) -> Path:
    """我把传入路径解析，并强制它落在 root 之内。

    为什么必须先 resolve 再比前缀：不做 resolve 的话，`a/../../etc/passwd`
    这种字符串照样能通过"看起来在 root 下"的检查。
    我用 `is_relative_to` 而不是 `str.startswith`：后者会把
    `root_b` 当成 `root` 的子目录放过。
    """

    root_resolved = Path(root).resolve()
    candidate = (root_resolved / raw).resolve() if not Path(raw).is_absolute() else Path(raw).resolve()
    if not candidate.is_relative_to(root_resolved):
        raise ValueError(
            f"我拒绝访问根目录之外的路径：{raw}。请把它放到 {root_resolved} 下。"
        )
    return candidate


def _run_inspection(
    candidate: str,
    reference: str,
    output_dir: str,
) -> dict[str, Any]:
    """我执行一次检测并返回结果字典。

    我在函数内部导入 pipeline，而不是在模块顶层导入：这样"没装 FastAPI"
    或"只想用 CLI"的场景不会被服务模块的导入拖累，也和本项目
    "可选依赖延迟到使用时"的一贯做法一致。
    """

    from ..pipeline import inspect_pair

    result = inspect_pair(candidate, reference, output_dir)
    return result.to_dict()


def create_app(
    allowed_root: str | Path,
    output_root: str | Path | None = None,
    max_workers: int = 2,
):
    """我构造 FastAPI 应用。

    allowed_root 是我能读取模型的唯一根目录，这是安全边界而不是配置开关。
    max_workers 默认 2：检测是 CPU 密集活，开太多线程只会让它们互相抢
    CPU，单个任务反而更慢。
    """

    try:
        from fastapi import FastAPI, HTTPException
    except ImportError as exc:  # pragma: no cover - 依赖缺失时的显式提示
        raise RuntimeError(
            "我无法启动 HTTP 服务，因为当前环境没有安装 fastapi。"
            "可执行：python -m pip install fastapi uvicorn"
        ) from exc
    global InspectRequest
    if InspectRequest is None:
        InspectRequest = _build_request_model()

    root = Path(allowed_root).resolve()
    if not root.is_dir():
        raise ValueError(f"我要求 allowed_root 是已存在的目录，当前为 {root}。")
    store = JobStore(
        output_root=Path(output_root) if output_root else root / "_api_output"
    )
    executor = ThreadPoolExecutor(max_workers=max_workers)

    @asynccontextmanager
    async def _lifespan(_app: Any) -> AsyncIterator[None]:
        """我在应用关闭时收掉线程池，而不是让它一直挂着。

        我为什么要加这个：`create_app` 每建一个应用就起一个
        ``ThreadPoolExecutor``，但原先**没有任何地方关掉它**。
        测试里每个用例都新建一个 app，跑完全量就是十几个线程池常驻；
        服务反复重启时这些线程也会一直堆着占资源。
        我在查那个偶发失败时量到过本机可用内存只剩 0.48 GB，
        这种"每个 app 都留一批线程"的泄漏会让它雪上加霜。

        我用 lifespan 而不是 ``@app.on_event("shutdown")``：后者在新版
        FastAPI/Starlette 上已废弃，会刷 DeprecationWarning 污染测试输出。
        """

        try:
            yield
        finally:
            executor.shutdown(wait=False)

    app = FastAPI(
        title="我方缺失区域检测服务",
        description="我按参考件与候选件比对，报出缺失与多余区域。",
        lifespan=_lifespan,
    )
    def _submit(candidate: str, reference: str) -> Job:
        """我校验路径、建任务、丢进线程池，立刻返回任务句柄。"""

        # 越界路径要变成 400 而不是任由 ValueError 冒成 500：
        # 500 在调用方看来是"服务坏了"，而真相是"你传了个不该传的路径"，
        # 这两种失败的处理方式完全不同。
        try:
            candidate_path = _resolve_within(root, candidate)
            reference_path = _resolve_within(root, reference)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        for label, path in (("候选件", candidate_path), ("参考件", reference_path)):
            if not path.is_file():
                raise HTTPException(
                    status_code=404, detail=f"我找不到{label}文件：{path}"
                )
        job_id = uuid.uuid4().hex
        job = Job(
            job_id=job_id,
            candidate=str(candidate_path),
            reference=str(reference_path),
            status="RUNNING",
        )
        store.jobs[job_id] = job
        job_dir = store.output_root / job_id

        def _mark_failed() -> None:
            """我把任务置成 FAILED，并且**保证此刻 error/traceback 非空**。

            我为什么要单独收口这一步：门禁里报过一次 "assert None"，
            我此前一直读成 "status=FAILED 但 error=None"，**那是错的**
            （2026-10-15 订正）：那条断言在轮询 break 后没有先查 status，
            证明不了 status 是什么；真相是轮询预算不足、job 仍是 RUNNING
            快照。详见 tests/test_api_service.py 里的说明。
            查明之后我**仍然保留**这层兜底：顺序正确只保证我自己写的
            路径不漏，兜底能把任何未知路径的最坏后果从 None 降级成
            一条说明文案。这是廉价防御，不是"确有第二条路径"的证据
            （12x4000 次轮询 0 命中、6 种异常形态穷举 0 次静默失败）。

            兜底而不是断言：这是服务端后台线程，抛异常只会让任务永远卡在
            RUNNING（那比"失败了但说不清原因"更难排查）。
            """

            if not job.error:
                # 取证必须在**改写之前**：我要留的是"原始现场"（当时 error
                # 到底是 None 还是空串），不是被我兜底之后的文案。
                _dump_silence_forensics(job, store.output_root)
                job.error = (
                    "我方任务进入了 FAILED，但错误原因没有被记录下来"
                    "（兜底文案，说明有路径绕过了正常赋值，请查服务日志）。"
                )
            if not job.traceback:
                job.traceback = traceback.format_exc()
            job.status = "FAILED"

        def _task() -> None:
            """我在线程里跑检测，并把终态写回任务对象。"""

            # 我为什么在里外各包一层 try：**异常处理器里抛出的新异常，
            # 不会被同级的 except 接住**（Python 只交给更外层的 try）。
            # 我实测过：某个异常的 __str__ 返回非字符串时，
            # `job.error = str(error)` 会抛 TypeError，而它从
            # `except InspectorError` 块里冒出去，**同级的
            # except Exception 接不到**，于是异常逃出 _task，
            # executor 只把它吞进 Future —— 任务永远卡在 RUNNING，
            # 调用方轮询到超时也等不到终态。这比"失败了但说不清原因"
            # 更糟：连失败都没报出来。
            # 外层这层 try 就是为这种"留痕自己炸了"兜底。
            try:
                try:
                    job.result = _run_inspection(
                        str(candidate_path), str(reference_path), str(job_dir)
                    )
                    job.status = "DONE"
                except InspectorError as error:
                    # 我方自己的异常带中文说明，直接透传给调用方，
                    # 不包装成笼统的 500，否则看不出到底哪一步不合法。
                    job.error = str(error)
                    job.traceback = traceback.format_exc()
                    # 我为什么**先写 error/traceback、后写 status**（顺序很关键）：
                    # 任务对象在线程间共享，HTTP 轮询线程随时可能读它。
                    # 若先置 status="FAILED" 再赋 error，就存在一个真实窗口——
                    # 调用方看到"已失败"却拿到 error=None，
                    # 也就是"静默失败"，而这正是我要消灭的东西。
                    # 门禁里那条 test我方失败任务保留错误与堆栈
                    # 就是踩到这个窗口红的（"assert None"）。
                    # 反过来写：只要能看到 FAILED，error 和 traceback 必然已就位。
                    _mark_failed()
                except Exception as error:  # noqa: BLE001 - 我要的是"任何失败都留痕"
                    job.error = f"{type(error).__name__}: {error}"
                    job.traceback = traceback.format_exc()
                    # 同上：先留痕、后置终态。
                    _mark_failed()
            except BaseException as error:  # noqa: BLE001 - 留痕自己炸了也要有终态
                job.error = (
                    f"我方在记录失败原因时就炸了：{type(error).__name__}: {error}"
                )
                job.traceback = traceback.format_exc()
                _mark_failed()
                raise

        executor.submit(_task)
        return job

    @app.get("/health")
    def health() -> dict[str, Any]:
        """我返回服务健康状态与当前任务数。"""

        return {
            "status": "OK",
            "allowed_root": str(root),
            "job_count": len(store.jobs),
        }

    @app.post("/inspect")
    def inspect(payload: "InspectRequest") -> dict[str, Any]:
        """我提交一次检测，返回 job_id 供后续查询结果。

        形参名为什么不能叫 request：Starlette 会把名为 request 的形参
        当成原始 Request 对象注入，请求体于是不再被解析成模型，
        实际表现为"必填字段缺失"的 422 而不是任何显式报错。
        这类失败最难查，所以我把形参改名并在注释里写清原因。
        """

        return _submit(payload.candidate, payload.reference).to_dict()

    @app.get("/jobs/{job_id}")
    def get_job(job_id: str) -> dict[str, Any]:
        """我查询一个任务的状态与结果。"""

        job = store.jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"我找不到任务 {job_id}。")
        return job.to_dict()

    return app
