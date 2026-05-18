from __future__ import annotations

from typing import Dict, List, Tuple

from eeg_report_multiagent.schemas.finding import Finding


def finding_key(finding: Finding) -> Tuple[str, str]:
    return (finding.finding_type, finding.assertion.value)


def compare_findings(pred: List[Finding], gt: List[Finding]) -> Dict[str, object]:
    pred_set = {finding_key(f) for f in pred}
    gt_set = {finding_key(f) for f in gt}

    return {
        "matched": sorted(pred_set & gt_set),
        "missing": sorted(gt_set - pred_set),
        "extra": sorted(pred_set - gt_set),
        "pred_count": len(pred_set),
        "gt_count": len(gt_set),
    }
