"""用 QPainter 自绘的矢量图标与 Logo（不依赖图标库/emoji 字体，高分屏清晰）。"""
from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QIcon, QLinearGradient, QPainter, QPainterPath, QPen, QPixmap

_S = 24.0  # 图标设计坐标系 24×24


def _pen(color: str, w: float = 1.8) -> QPen:
    p = QPen(QColor(color), w)
    p.setCapStyle(Qt.RoundCap)
    p.setJoinStyle(Qt.RoundJoin)
    return p


def _data(p: QPainter):
    p.drawRoundedRect(QRectF(3.5, 4.5, 17, 15), 2.5, 2.5)
    p.drawLine(QPointF(3.5, 10), QPointF(20.5, 10))
    p.drawLine(QPointF(3.5, 15), QPointF(20.5, 15))
    p.drawLine(QPointF(9.5, 10), QPointF(9.5, 19.5))


def _preprocess(p: QPainter):
    for y, x in ((6.5, 8), (12, 16), (17.5, 10)):
        p.drawLine(QPointF(4, y), QPointF(20, y))
        p.setBrush(p.pen().color())
        p.drawEllipse(QPointF(x, y), 2.2, 2.2)
        p.setBrush(Qt.NoBrush)


def _visualize(p: QPainter):
    p.drawLine(QPointF(3.5, 20), QPointF(20.5, 20))
    for x, h in ((6, 7), (11, 12), (16, 9)):
        p.drawRoundedRect(QRectF(x - 1.6, 20 - h, 3.2, h - 1), 1, 1)


def _stats(p: QPainter):
    path = QPainterPath(QPointF(3, 19))
    path.cubicTo(8, 19, 8, 5, 12, 5)
    path.cubicTo(16, 5, 16, 19, 21, 19)
    p.drawPath(path)
    p.drawLine(QPointF(3, 20.5), QPointF(21, 20.5))


def _model(p: QPainter):
    pts = [QPointF(6, 6.5), QPointF(6, 17.5), QPointF(18, 12)]
    for a, b in ((0, 2), (1, 2), (0, 1)):
        p.drawLine(pts[a], pts[b])
    p.setBrush(p.pen().color())
    for q in pts:
        p.drawEllipse(q, 2.4, 2.4)
    p.setBrush(Qt.NoBrush)


def _cluster(p: QPainter):
    for c in (QPointF(7.5, 8), QPointF(16.5, 8), QPointF(12, 16.5)):
        p.drawEllipse(c, 3.6, 3.6)
    p.setBrush(p.pen().color())
    for c in (QPointF(7.5, 8), QPointF(16.5, 8), QPointF(12, 16.5)):
        p.drawEllipse(c, 1, 1)
    p.setBrush(Qt.NoBrush)


def _predict(p: QPainter):
    p.drawEllipse(QPointF(12, 12), 7.5, 7.5)
    p.setBrush(p.pen().color())
    p.drawEllipse(QPointF(12, 12), 1.8, 1.8)
    p.setBrush(Qt.NoBrush)
    for a, b in (((12, 1.5), (12, 5)), ((12, 19), (12, 22.5)), ((1.5, 12), (5, 12)), ((19, 12), (22.5, 12))):
        p.drawLine(QPointF(*a), QPointF(*b))


def _image(p: QPainter):
    p.drawRoundedRect(QRectF(3.5, 4.5, 17, 15), 2.5, 2.5)
    path = QPainterPath(QPointF(4.5, 16.5))
    path.lineTo(9, 11.5)
    path.lineTo(13, 15.5)
    path.lineTo(15.5, 13)
    path.lineTo(19.5, 17)
    p.drawPath(path)
    p.drawEllipse(QPointF(15.8, 8.8), 1.4, 1.4)


def _ai(p: QPainter):
    def star(cx, cy, r):
        path = QPainterPath(QPointF(cx, cy - r))
        k = r * 0.28
        path.cubicTo(cx, cy - k, cx + k, cy, cx + r, cy)
        path.cubicTo(cx + k, cy, cx, cy + k, cx, cy + r)
        path.cubicTo(cx, cy + k, cx - k, cy, cx - r, cy)
        path.cubicTo(cx - k, cy, cx, cy - k, cx, cy - r)
        return path
    p.drawPath(star(10.5, 13.5, 7.5))
    p.drawPath(star(18.5, 5.5, 3.6))


def _report(p: QPainter):
    path = QPainterPath(QPointF(6.5, 3.5))
    path.lineTo(14.5, 3.5)
    path.lineTo(19.5, 8.5)
    path.lineTo(19.5, 20.5)
    path.lineTo(6.5, 20.5)
    path.closeSubpath()
    p.drawPath(path)
    p.drawLine(QPointF(14.5, 3.5), QPointF(14.5, 8.5))
    p.drawLine(QPointF(14.5, 8.5), QPointF(19.5, 8.5))
    p.drawLine(QPointF(9.5, 13), QPointF(16.5, 13))
    p.drawLine(QPointF(9.5, 16.5), QPointF(16.5, 16.5))


_DRAW = {"data": _data, "preprocess": _preprocess, "visualize": _visualize, "stats": _stats, "model": _model,
         "cluster": _cluster, "predict": _predict, "image": _image, "ai": _ai, "report": _report}


def _render(name: str, color: str, px: int = 22, scale: int = 2) -> QPixmap:
    pm = QPixmap(px * scale, px * scale)
    pm.setDevicePixelRatio(scale)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    p.scale(px / _S, px / _S)
    p.setPen(_pen(color))
    p.setBrush(Qt.NoBrush)
    _DRAW[name](p)
    p.end()
    return pm


def nav_icon(name: str, normal: str = "#8FA3C9", selected: str = "#FFFFFF") -> QIcon:
    ic = QIcon()
    n, s = _render(name, normal), _render(name, selected)
    ic.addPixmap(n, QIcon.Normal, QIcon.Off)
    ic.addPixmap(s, QIcon.Selected, QIcon.Off)
    ic.addPixmap(s, QIcon.Active, QIcon.Off)
    return ic


def logo_pixmap(px: int = 38, scale: int = 2) -> QPixmap:
    """圆角渐变方块 + 心电图折线。"""
    pm = QPixmap(px * scale, px * scale)
    pm.setDevicePixelRatio(scale)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    g = QLinearGradient(0, 0, px, px)
    g.setColorAt(0, QColor("#5B8CFF"))
    g.setColorAt(1, QColor("#19B5A5"))
    p.setBrush(g)
    p.setPen(Qt.NoPen)
    p.drawRoundedRect(QRectF(0, 0, px, px), px * 0.27, px * 0.27)
    p.setPen(_pen("#FFFFFF", px * 0.075))
    k = px / 24.0
    path = QPainterPath(QPointF(3 * k, 13 * k))
    for x, y in ((8 * k, 13 * k), (10.5 * k, 6.5 * k), (13.5 * k, 18 * k), (16 * k, 11 * k), (21 * k, 11 * k)):
        path.lineTo(x, y)
    p.drawPath(path)
    p.end()
    return pm


def arrow_pixmap(direction: str, color: str = "#6B778C", px: int = 12, scale: int = 2) -> QPixmap:
    """下拉/微调框的小箭头（direction: up | down）。"""
    pm = QPixmap(px * scale, px * scale)
    pm.setDevicePixelRatio(scale)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    p.setPen(_pen(color, px * 0.14))
    m, t, b = px * 0.28, px * 0.38, px * 0.62
    if direction == "down":
        path = QPainterPath(QPointF(m, t))
        path.lineTo(px / 2, b)
        path.lineTo(px - m, t)
    else:
        path = QPainterPath(QPointF(m, b))
        path.lineTo(px / 2, t)
        path.lineTo(px - m, b)
    p.drawPath(path)
    p.end()
    return pm
