from __future__ import annotations

from typing import Dict, List


class EventAgent:
    """Rule-based bounded tool selector for epileptiform/event analysis."""

    def select_tools(self, scout_summary: Dict[str, float]) -> List[str]:
        tools = ["transient_candidate_score"]
        if scout_summary.get("event_density_hint", 0.0) > 0.01:
            tools.extend(
                [
                    "burst_train_duration_estimate",
                    "channel_spread_laterality_summary",
                    "event_peak_topography_localizer",
                    "focality_bifrontal_summary",
                    "event_type_separation_classifier",
                ]
            )
        if scout_summary.get("enable_local_encoder", False):
            tools.append("morphology_feature_encoder")
        return tools
