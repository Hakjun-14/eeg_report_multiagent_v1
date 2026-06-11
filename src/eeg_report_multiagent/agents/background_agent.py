from __future__ import annotations

from typing import Dict, List


class BackgroundAgent:
    """Rule-based bounded tool selector for background/activity analysis."""

    BASE_TOOLS = [
        "psd_power_spectrum_summary",
        "posterior_dominant_rhythm_spectral_v2",
        "background_organization_proxy",
        "background_unavailable_slot_status",
        "bandpower_summary",
        "amplitude_summary",
    ]

    def select_tools(self, scout_summary: Dict[str, float]) -> List[str]:
        tools = list(self.BASE_TOOLS)
        if scout_summary.get("global_slowing_hint", 0.0) >= 1.0:
            tools.append("slowing_score")
        else:
            tools.append("slowing_score")
        tools.append("beta_excess_score")
        return tools
