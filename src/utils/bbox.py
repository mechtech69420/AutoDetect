def xyxy_to_xywh(bbox: list) -> list:
    """Convert [xmin, ymin, xmax, ymax] to [x, y, width, height]"""
    xmin, ymin, xmax, ymax = bbox
    return [xmin, ymin, xmax - xmin, ymax - ymin]
