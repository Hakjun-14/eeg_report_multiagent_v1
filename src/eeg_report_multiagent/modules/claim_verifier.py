from __future__ import annotations

from eeg_report_multiagent.schemas.evidence import EvidenceBoard
from eeg_report_multiagent.schemas.report import ClaimSupportLabel, VerificationRecord


class ClaimVerifier:
    def verify(self, board: EvidenceBoard) -> list[VerificationRecord]:
        shared_board = board.ensure_shared_evidence_board()
        evidence_ids = {item.evidence_id for item in shared_board.evidence_items}
        out: list[VerificationRecord] = []

        for claim in board.claims:
            linked_evidence_ids = shared_board.claim_evidence_links.get(claim.claim_id, [])
            if not linked_evidence_ids:
                out.append(
                    VerificationRecord(
                        claim_id=claim.claim_id,
                        support_label=ClaimSupportLabel.MISSING,
                        evidence_ids=[],
                        reason="claim has no linked evidence_ids",
                    )
                )
                continue

            missing = [eid for eid in linked_evidence_ids if eid not in evidence_ids]
            if missing:
                out.append(
                    VerificationRecord(
                        claim_id=claim.claim_id,
                        support_label=ClaimSupportLabel.UNSUPPORTED,
                        evidence_ids=[eid for eid in linked_evidence_ids if eid in evidence_ids],
                        reason=f"missing evidence ids: {missing}",
                    )
                )
            else:
                out.append(
                    VerificationRecord(
                        claim_id=claim.claim_id,
                        support_label=ClaimSupportLabel.SUPPORTED,
                        evidence_ids=list(linked_evidence_ids),
                        reason="all linked evidence ids exist in shared evidence board",
                    )
                )
        return out
