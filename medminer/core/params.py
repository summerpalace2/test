"""通用参数描述：界面按它自动生成参数表单（算法参数、预处理步骤参数共用）。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ParamSpec:
    name: str
    label: str
    kind: str  # int | float | choice | bool | text | columns | column
    default: Any = None
    min: float | None = None
    max: float | None = None
    step: float | None = None
    choices: list[Any] = field(default_factory=list)  # choice 的取值
    help: str = ""
    grid: list[Any] = field(default_factory=list)  # 调参时的候选值
    nullable: bool = False  # int/float 允许 None（界面用“不限”表示）

    def coerce(self, value: Any) -> Any:
        if value is None:
            return None
        if self.kind == "int":
            return int(value)
        if self.kind == "float":
            return float(value)
        if self.kind == "bool":
            return bool(value)
        return value


def defaults(specs: list[ParamSpec]) -> dict[str, Any]:
    return {s.name: s.default for s in specs}
