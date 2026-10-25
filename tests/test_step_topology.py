from __future__ import annotations

from pathlib import Path

import pytest

from cad_inspector.io.step_topology import StepTopologyReport
from cad_inspector.io.step_topology import inspect_step_topology


def test我方_step拓扑报告可以稳定序列化() -> None:
    report = StepTopologyReport(
        status="COMPLETE",
        file_path=r"D:\demo\sample.stp",
        unit="mm",
        backend="OCP",
        shape_is_null=False,
        shape_is_valid=True,
        entity_counts={
            "SOLID": 1,
            "SHELL": 1,
            "FACE": 2,
            "WIRE": 2,
            "EDGE": 6,
            "VERTEX": 4,
        },
        face_ids=[1, 2],
        elapsed_seconds=0.1234567,
    )
    payload = report.to_dict()
    assert payload["schema_version"] == "step-topology-1.2"
    assert payload["entity_counts"]["FACE"] == 2
    assert payload["edge_topology"] == {}
    assert payload["elapsed_seconds"] == 0.123457


def test我方真实盒体拓扑报告记录开放边和非流形边(tmp_path: Path) -> None:
    pytest.importorskip("OCP")
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.IFSelect import IFSelect_ReturnStatus
    from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer

    step_path = tmp_path / "box.step"
    writer = STEPControl_Writer()
    shape = BRepPrimAPI_MakeBox(10.0, 20.0, 30.0).Solid()
    assert (
        writer.Transfer(shape, STEPControl_AsIs)
        == IFSelect_ReturnStatus.IFSelect_RetDone
    )
    assert writer.Write(str(step_path)) == IFSelect_ReturnStatus.IFSelect_RetDone

    report = inspect_step_topology(step_path)

    assert report.status == "COMPLETE"
    assert report.edge_topology["mapped_edge_count"] == 12
    assert report.edge_topology["unique_edge_count"] == 12
    assert report.edge_topology["face_edge_occurrence_count"] == 24
    assert report.edge_topology["boundary_edge_count"] == 0
    assert report.edge_topology["non_manifold_edge_count"] == 0
    assert report.edge_topology["manifold_edge_count"] == 12
    assert report.edge_topology["zero_face_edge_count"] == 0
    assert report.edge_topology["topology_index_method"] == "face_local_map"
    assert report.to_dict()["schema_version"] == "step-topology-1.2"
