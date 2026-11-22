"""Area units. India mostly uses acres (and locally cents / guntha)."""

from __future__ import annotations

M2_PER_UNIT = {
    "ha": 10_000.0,
    "acre": 4_046.8564224,
    "cent": 40.468564224,  # 1 acre = 100 cents
    "guntha": 101.17141056,  # 1 acre = 40 guntha
}
LABEL = {
    "ha": ("hectare", "hectares"),
    "acre": ("acre", "acres"),
    "cent": ("cent", "cents"),
    "guntha": ("guntha", "guntha"),
}


def convert_area(m2: float, unit: str) -> float:
    if unit not in M2_PER_UNIT:
        raise ValueError(f"Unknown unit '{unit}'. Use one of {list(M2_PER_UNIT)}")
    return m2 / M2_PER_UNIT[unit]


def format_area(m2: float, unit: str = "acre", digits: int = 2) -> str:
    v = convert_area(m2, unit)
    one, many = LABEL[unit]
    return f"{v:.{digits}f} {one if round(v, digits) == 1 else many}"


def all_units(m2: float) -> dict[str, float]:
    return {u: round(convert_area(m2, u), 4) for u in M2_PER_UNIT}
