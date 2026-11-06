"""Checks the RTMO ONNX wrapper's I/O handling with a tiny stand-in graph."""

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")
from onnx import TensorProto, helper, numpy_helper  # noqa: E402

from rallyiq.perception.players.pose import RTMOOnnx, merge_detections  # noqa: E402
from rallyiq.perception.players.models import PersonDetection  # noqa: E402


def _fake_rtmo(path):
    dets = np.array([[[100, 100, 200, 300, 0.9], [0, 0, 10, 10, 0.1]]], np.float32)
    kps = np.zeros((1, 2, 17, 3), np.float32)
    kps[0, 0, :, 0], kps[0, 0, :, 1], kps[0, 0, :, 2] = 150, 250, 0.8
    nodes = [
        helper.make_node("Constant", [], ["dets"], value=numpy_helper.from_array(dets)),
        helper.make_node("Constant", [], ["keypoints"], value=numpy_helper.from_array(kps)),
    ]
    graph = helper.make_graph(
        nodes, "fake_rtmo",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 640, 640])],
        [helper.make_tensor_value_info("dets", TensorProto.FLOAT, [1, 2, 5]),
         helper.make_tensor_value_info("keypoints", TensorProto.FLOAT, [1, 2, 17, 3])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.save(model, path)


def test_rtmo_rescales_to_original_pixels(tmp_path):
    p = tmp_path / "fake.onnx"
    _fake_rtmo(str(p))
    est = RTMOOnnx(str(p), providers=["CPUExecutionProvider"])
    img = np.zeros((720, 1280, 3), np.uint8)  # letterbox ratio 0.5
    dets = est(img)
    assert len(dets) == 1  # low-score row dropped
    np.testing.assert_allclose(dets[0].bbox, [200, 200, 400, 600])
    np.testing.assert_allclose(dets[0].keypoints[0], [300, 500, 0.8], rtol=1e-6)


def test_merge_detections_nms():
    a = PersonDetection([0, 0, 10, 20], np.zeros((17, 3)), 0.9)
    b = PersonDetection([1, 0, 11, 20], np.zeros((17, 3)), 0.5)
    c = PersonDetection([50, 0, 60, 20], np.zeros((17, 3)), 0.4)
    assert [d.score for d in merge_detections([b, a, c])] == [0.9, 0.4]
