"""Data models for the Person Integrity Auditor.

Defines issue types, severity scoring, and result containers for
the 5-phase audit pipeline (dedup, wikidata, factcheck, coherence, scoring).
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class AuditIssueType(str, Enum):
    """Categories of data quality issues, each with a base severity score."""

    DUPLICATE_ENTRY = "duplicate_entry"                # base: 50
    MERGED_IDENTITY = "merged_identity"                # base: 80
    WRONG_PERSON = "wrong_person_entirely"             # base: 90
    BIO_INACCURACY = "bio_inaccuracy"                  # base: 30
    BIO_CONTRADICTION = "bio_contradiction"            # base: 40
    EXTERNAL_CONTRADICTION = "external_contradiction"  # base: 60
    WRONG_CATEGORY = "wrong_category"                  # base: 20
    MISSING_CRITICAL_INFO = "missing_critical_info"    # base: 15
    UNGROUNDED_CLAIM = "ungrounded_claim"              # base: 25
    STALE_DATA = "stale_data"                          # base: 10
    WRONG_IMAGE = "wrong_image"                        # base: 35


# Base severity scores for each issue type
ISSUE_BASE_SCORES: dict[AuditIssueType, int] = {
    AuditIssueType.WRONG_PERSON: 90,
    AuditIssueType.MERGED_IDENTITY: 80,
    AuditIssueType.EXTERNAL_CONTRADICTION: 60,
    AuditIssueType.DUPLICATE_ENTRY: 50,
    AuditIssueType.BIO_CONTRADICTION: 40,
    AuditIssueType.WRONG_IMAGE: 35,
    AuditIssueType.BIO_INACCURACY: 30,
    AuditIssueType.UNGROUNDED_CLAIM: 25,
    AuditIssueType.WRONG_CATEGORY: 20,
    AuditIssueType.MISSING_CRITICAL_INFO: 15,
    AuditIssueType.STALE_DATA: 10,
}


class SeverityDimensions(BaseModel):
    """Multi-dimensional severity assessment."""

    impact: int = Field(ge=1, le=5, description="How visible/important is this person?")
    certainty: int = Field(ge=1, le=5, description="How confident are we in this finding?")
    scope: int = Field(ge=1, le=5, description="How many records/facts are affected?")
    legal_risk: int = Field(ge=1, le=5, description="Defamation/legal exposure risk")
    factual_gravity: int = Field(ge=1, le=5, description="How serious is the factual error?")


class AuditEvidence(BaseModel):
    """A piece of evidence supporting an audit finding."""

    type: str  # "document", "wikidata", "wikipedia", "co_occurrence", "embedding"
    id: str | None = None
    title: str | None = None
    snippet: str
    relevance: float = Field(ge=0.0, le=1.0, default=0.5)


class AuditIssue(BaseModel):
    """A single data quality issue found during audit."""

    person_id: str
    person_name: str
    issue_type: AuditIssueType
    severity: int = Field(ge=0, le=100, default=0)
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    title: str
    details: str
    evidence: list[AuditEvidence] = Field(default_factory=list)
    wikidata_qid: str | None = None
    phase: str  # "dedup", "wikidata", "factcheck", "coherence"
    dimensions: SeverityDimensions = Field(
        default_factory=lambda: SeverityDimensions(
            impact=3, certainty=3, scope=1, legal_risk=2, factual_gravity=3
        )
    )

    def calculate_severity(self) -> int:
        """Calculate composite severity from base score, dimensions, and confidence."""
        base = ISSUE_BASE_SCORES.get(self.issue_type, 25)
        dim = self.dimensions
        multiplier = (
            dim.impact * 0.25
            + dim.certainty * 0.30
            + dim.scope * 0.15
            + dim.legal_risk * 0.20
            + dim.factual_gravity * 0.10
        ) / 5.0
        self.severity = min(100, max(0, round(base * multiplier * self.confidence)))
        return self.severity


class ClaimVerification(BaseModel):
    """Result of verifying a single atomic claim from a bio."""

    claim: str
    claim_type: str  # "biographical", "relational", "temporal", "legal", "professional"
    verdict: str  # "SUPPORTED", "CONTRADICTED", "UNVERIFIABLE", "PARTIALLY_SUPPORTED"
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[AuditEvidence] = Field(default_factory=list)
    reasoning: str = ""


class WikidataMatch(BaseModel):
    """Wikidata cross-reference result for a person."""

    qid: str
    label: str
    description: str | None = None
    birth_date: str | None = None
    death_date: str | None = None
    occupations: list[str] = Field(default_factory=list)
    nationality: str | None = None
    wikipedia_summary: str | None = None
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)


class PersonAuditResult(BaseModel):
    """Complete audit result for a single person."""

    person_id: str
    person_name: str
    person_slug: str
    issues: list[AuditIssue] = Field(default_factory=list)
    wikidata_match: WikidataMatch | None = None
    claim_verifications: list[ClaimVerification] = Field(default_factory=list)
    max_severity: int = 0
    phase_timings: dict[str, float] = Field(default_factory=dict)

    def compute_max_severity(self) -> int:
        """Set max_severity to highest issue severity."""
        self.max_severity = max((i.severity for i in self.issues), default=0)
        return self.max_severity


class AuditRunSummary(BaseModel):
    """Summary of a complete audit run."""

    run_id: str
    started_at: str
    finished_at: str | None = None
    persons_scanned: int = 0
    issues_found: int = 0
    critical_count: int = 0  # severity >= 70
    high_count: int = 0      # severity >= 40
    medium_count: int = 0    # severity >= 20
    low_count: int = 0       # severity < 20
    total_cost_cents: int = 0
    phases_completed: list[str] = Field(default_factory=list)
    results: list[PersonAuditResult] = Field(default_factory=list, exclude=True)

    def tally(self, results: list[PersonAuditResult], critical: int = 70, high: int = 40, medium: int = 20) -> None:
        """Compute summary stats from results."""
        self.results = results
        self.persons_scanned = len(results)
        all_issues = [i for r in results for i in r.issues]
        self.issues_found = len(all_issues)
        self.critical_count = sum(1 for i in all_issues if i.severity >= critical)
        self.high_count = sum(1 for i in all_issues if high <= i.severity < critical)
        self.medium_count = sum(1 for i in all_issues if medium <= i.severity < high)
        self.low_count = sum(1 for i in all_issues if i.severity < medium)
