from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScenarioSpec:
    paper_name: str
    code_name: str
    file_stem: str
    module_name: str


SCENARIOS = {
    s.paper_name: s
    for s in [
        ScenarioSpec("navigation", "dispersion", "dispersion", "custom_dispersion"),
        ScenarioSpec("dispersion", "dispersion_vmas", "dispersion_vmas", "custom_dispersion_vmas"),
        ScenarioSpec("reverse_transport", "reverse_transport", "reverse_transport",
                     "custom_reverse_transport"),
        ScenarioSpec("pressure_plate", "pressure_plate", "pressure_plate", "custom_pressure_plate"),
        ScenarioSpec("football", "football", "football", "custom_football"),
        ScenarioSpec("wind_flocking", "wind_flocking_position", "wind_flocking_position",
                     "custom_wind_flocking"),
    ]
}

CODE_NAME_TO_PAPER = {s.code_name: s.paper_name for s in SCENARIOS.values()}


def get_scenario(paper_name: str) -> ScenarioSpec:
    try:
        return SCENARIOS[paper_name]
    except KeyError:
        raise KeyError(
            f"unknown environment {paper_name!r}; expected one of {sorted(SCENARIOS)}"
        ) from None

