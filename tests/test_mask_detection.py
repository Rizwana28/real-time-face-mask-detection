import numpy as np
import pytest

from mask_detection import draw_detections, parse_yolo_outputs


def test_parse_yolo_outputs_scales_boxes_and_filters_confidence():
    outputs = [np.array([[0.5, 0.5, 0.4, 0.2, 0.9, 0.8, 0.1]], dtype=np.float32)]
    detections = parse_yolo_outputs(outputs, (100, 200, 3), ["mask", "no_mask"], 0.5, 0.4)
    assert len(detections) == 1
    assert detections[0].label == "mask"
    assert detections[0].confidence == pytest.approx(0.72)
    assert detections[0].box == (60, 40, 80, 20)


def test_parse_yolo_outputs_returns_empty_for_low_confidence():
    outputs = [np.array([[0.5, 0.5, 0.4, 0.2, 0.2, 0.8, 0.1]], dtype=np.float32)]
    assert parse_yolo_outputs(outputs, (100, 200, 3), ["mask", "no_mask"], 0.5, 0.4) == []


def test_draw_detections_does_not_mutate_input():
    frame = np.zeros((40, 60, 3), dtype=np.uint8)
    from mask_detection import Detection

    rendered = draw_detections(frame, [Detection("without_mask", 0.9, (-10, -5, 30, 20))])
    assert np.array_equal(frame, np.zeros((40, 60, 3), dtype=np.uint8))
    assert rendered.shape == frame.shape
    assert rendered.any()
