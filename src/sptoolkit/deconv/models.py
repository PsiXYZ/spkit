from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ParamSpec:
    value: float
    minimum: float | None = None
    maximum: float | None = None
    vary: bool = True


@dataclass
class PeakSpec:
    kind: str
    center: ParamSpec
    amplitude: ParamSpec
    sigma: ParamSpec
    extras: dict[str, ParamSpec] = field(default_factory=dict)

    def all_params(self) -> dict[str, ParamSpec]:
        params = {
            "center": self.center,
            "amplitude": self.amplitude,
            "sigma": self.sigma,
        }
        params.update(self.extras)
        return params

