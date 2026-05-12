from __future__ import annotations

from eeg_report_multiagent.schemas.evidence import EvidenceBoard
from eeg_report_multiagent.schemas.report import ClaimSupportLabel, VerificationRecord


class ClaimVerifier:
    def verify(self, board: EvidenceBoard) -> list[VerificationRecord]:
        finding_ids = {f.finding_id for f in board.findings}
        out: list[VerificationRecord] = []

        for claim in board.claims:
            if not claim.linked_finding_ids:
                out.append(
                    VerificationRecord(
                        claim_id=claim.claim_id,
                        support_label=ClaimSupportLabel.MISSING,
                        evidence_finding_ids=[],
                        reason="claim has no linked_finding_ids",
                    )
                )
                continue

            missing = [fid for fid in claim.linked_finding_ids if fid not in finding_ids]
            if missing:
                out.append(
                    VerificationRecord(
                        claim_id=claim.claim_id,
                        support_label=ClaimSupportLabel.UNSUPPORTED,
                        evidence_finding_ids=[fid for fid in claim.linked_finding_ids if fid in finding_ids],
                        reason=f"missing evidence ids: {missing}",
                    )
                )
            else:
                out.append(
                    VerificationRecord(
                        claim_id=claim.claim_id,
                        support_label=ClaimSupportLabel.SUPPORTED,
                        evidence_finding_ids=list(claim.linked_finding_ids),
                        reason="all linked findings exist in evidence board",
                    )
                )
        return out
