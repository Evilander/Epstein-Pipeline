"""Person Integrity Auditor — 5-phase data quality pipeline.

Scans all person records against the Neon database, Wikidata, and Wikipedia
to detect inaccuracies, duplicates, merged identities, and ungrounded claims.

Phases:
1. Dedup — detect duplicate/merged person entries (rapidfuzz + embeddings)
2. Wikidata — cross-reference against structured knowledge base
3. Fact-Check — decompose bios into atomic claims, verify against documents
4. Coherence — detect merged identities via document sampling
5. Score — calculate severity, store issues, create ai_leads

Usage:
    epstein-pipeline audit-persons
    epstein-pipeline audit-persons --phases dedup,wikidata --limit 50
    epstein-pipeline audit-persons --person jeffrey-epstein --dry-run
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone

import httpx
from rapidfuzz import fuzz
from rich.progress import Progress, TextColumn, BarColumn, MofNCompleteColumn

from epstein_pipeline.config import Settings
from epstein_pipeline.models.audit import (
    AuditEvidence,
    AuditIssue,
    AuditIssueType,
    AuditRunSummary,
    ClaimVerification,
    PersonAuditResult,
    SeverityDimensions,
    WikidataMatch,
)
from epstein_pipeline.state import ProcessingState

logger = logging.getLogger(__name__)

# Wikidata property IDs
WD_BIRTH = "P569"
WD_DEATH = "P570"
WD_OCCUPATION = "P106"
WD_CITIZENSHIP = "P27"
WD_EDUCATED_AT = "P69"
WD_POSITION = "P39"
WD_INSTANCE_OF = "P31"

# Category mapping: Wikidata occupations -> our categories
OCCUPATION_TO_CATEGORY = {
    "politician": "politician",
    "businessperson": "business",
    "entrepreneur": "business",
    "investor": "business",
    "hedge fund manager": "business",
    "financier": "business",
    "banker": "business",
    "lawyer": "legal",
    "attorney": "legal",
    "judge": "legal",
    "prosecutor": "legal",
    "professor": "academic",
    "scientist": "academic",
    "researcher": "academic",
    "physicist": "academic",
    "mathematician": "academic",
    "neuroscientist": "academic",
    "actor": "celebrity",
    "model": "celebrity",
    "singer": "celebrity",
    "film director": "celebrity",
    "socialite": "socialite",
    "prince": "royalty",
    "princess": "royalty",
    "duke": "royalty",
    "military officer": "military-intelligence",
    "intelligence officer": "military-intelligence",
    "spy": "military-intelligence",
}


class PersonIntegrityAuditor:
    """5-phase person data quality auditor.

    Connects to Neon Postgres, Anthropic API, Voyage AI, Cohere,
    and Wikidata/Wikipedia for comprehensive cross-referencing.
    """

    ALL_PHASES = ("dedup", "wikidata", "factcheck", "coherence", "score")

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.state = ProcessingState()
        self._neon = None
        self._anthropic = None
        self._voyage = None
        self._cohere_client = None
        self._http = None
        self._total_tokens = 0
        self._total_cost_cents = 0

    # ── Lazy clients ────────────────────────────────────────────────────

    def _get_neon(self):
        if self._neon is None:
            import psycopg
            self._neon = psycopg.connect(self.settings.neon_database_url)
        return self._neon

    def _get_anthropic(self):
        if self._anthropic is None:
            import anthropic
            api_key = self.settings.auditor_anthropic_api_key
            if not api_key:
                raise ValueError("EPSTEIN_AUDITOR_ANTHROPIC_API_KEY required")
            self._anthropic = anthropic.Anthropic(api_key=api_key)
        return self._anthropic

    def _get_voyage(self):
        if self._voyage is None:
            import voyageai
            api_key = self.settings.auditor_voyage_api_key
            if not api_key:
                raise ValueError("EPSTEIN_AUDITOR_VOYAGE_API_KEY required")
            self._voyage = voyageai.Client(api_key=api_key)
        return self._voyage

    def _get_cohere(self):
        if self._cohere_client is None:
            import cohere
            api_key = self.settings.auditor_cohere_api_key
            if not api_key:
                logger.warning("No Cohere API key — reranking disabled")
                return None
            self._cohere_client = cohere.ClientV2(api_key=api_key)
        return self._cohere_client

    def _get_http(self) -> httpx.Client:
        if self._http is None:
            self._http = httpx.Client(
                timeout=30.0,
                headers={"User-Agent": "EpsteinPipeline/1.0 (j.tyler.eveland@gmail.com)"},
            )
        return self._http

    # ── Main entry point ────────────────────────────────────────────────

    async def run(
        self,
        phases: list[str] | None = None,
        person_ids: list[str] | None = None,
        limit: int | None = None,
        resume: bool = True,
        dry_run: bool = False,
        min_severity: int = 0,
    ) -> AuditRunSummary:
        """Run the audit pipeline."""
        run_id = f"audit-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        phases = phases or list(self.ALL_PHASES)
        logger.info("Audit run %s — phases: %s, limit: %s, resume: %s", run_id, phases, limit, resume)

        summary = AuditRunSummary(
            run_id=run_id,
            started_at=datetime.now(timezone.utc).isoformat(),
        )

        # Fetch all persons from Neon
        persons = self._fetch_persons(person_ids, limit)
        logger.info("Loaded %d persons from Neon", len(persons))

        all_issues: list[AuditIssue] = []
        results: list[PersonAuditResult] = []

        # Phase 1: Dedup
        if "dedup" in phases:
            t0 = time.time()
            logger.info("Phase 1: Dedup scan...")
            issues = self._phase_dedup(persons, resume)
            all_issues.extend(issues)
            summary.phases_completed.append("dedup")
            logger.info("Phase 1 complete: %d issues in %.1fs", len(issues), time.time() - t0)

        # Phase 2: Wikidata
        if "wikidata" in phases:
            t0 = time.time()
            logger.info("Phase 2: Wikidata cross-reference...")
            issues = self._phase_wikidata(persons, resume)
            all_issues.extend(issues)
            summary.phases_completed.append("wikidata")
            logger.info("Phase 2 complete: %d issues in %.1fs", len(issues), time.time() - t0)

        # Phase 3: Fact-check
        if "factcheck" in phases:
            t0 = time.time()
            logger.info("Phase 3: Bio fact-check...")
            issues = self._phase_fact_check(persons, resume)
            all_issues.extend(issues)
            summary.phases_completed.append("factcheck")
            logger.info("Phase 3 complete: %d issues in %.1fs", len(issues), time.time() - t0)

        # Phase 4: Coherence
        if "coherence" in phases:
            t0 = time.time()
            logger.info("Phase 4: Identity coherence...")
            issues = self._phase_coherence(persons, resume)
            all_issues.extend(issues)
            summary.phases_completed.append("coherence")
            logger.info("Phase 4 complete: %d issues in %.1fs", len(issues), time.time() - t0)

        # Calculate severity for all issues
        for issue in all_issues:
            issue.calculate_severity()

        # Filter by min_severity
        all_issues = [i for i in all_issues if i.severity >= min_severity]

        # Group into per-person results
        by_person: dict[str, list[AuditIssue]] = {}
        for issue in all_issues:
            by_person.setdefault(issue.person_id, []).append(issue)

        for p in persons:
            pid = p["id"]
            result = PersonAuditResult(
                person_id=pid,
                person_name=p["name"],
                person_slug=p["slug"],
                issues=by_person.get(pid, []),
            )
            result.compute_max_severity()
            results.append(result)

        # Phase 5: Score + store
        if "score" in phases and not dry_run:
            t0 = time.time()
            logger.info("Phase 5: Storing results and creating leads...")
            self._phase_score(run_id, all_issues)
            summary.phases_completed.append("score")
            logger.info("Phase 5 complete in %.1fs", time.time() - t0)

        summary.finished_at = datetime.now(timezone.utc).isoformat()
        summary.total_cost_cents = self._total_cost_cents
        summary.tally(results)

        return summary

    # ── Phase 1: Dedup ──────────────────────────────────────────────────

    def _phase_dedup(self, persons: list[dict], resume: bool) -> list[AuditIssue]:
        """Detect duplicate person entries using name similarity."""
        issues: list[AuditIssue] = []
        threshold = self.settings.auditor_name_fuzzy_threshold
        checked = set()

        with Progress(TextColumn("[progress.description]{task.description}"),
                      BarColumn(), MofNCompleteColumn()) as progress:
            task = progress.add_task("Dedup scan", total=len(persons))

            for i, p1 in enumerate(persons):
                progress.advance(task)
                if resume and self.state.is_processed(p1["id"], "audit_dedup"):
                    continue

                for p2 in persons[i + 1:]:
                    pair_key = tuple(sorted([p1["id"], p2["id"]]))
                    if pair_key in checked:
                        continue
                    checked.add(pair_key)

                    # Name similarity
                    name_sim = fuzz.ratio(p1["name"].lower(), p2["name"].lower())
                    if name_sim < threshold:
                        # Also check aliases
                        alias_match = False
                        for a1 in (p1.get("aliases") or []):
                            if fuzz.ratio(a1.lower(), p2["name"].lower()) >= threshold:
                                alias_match = True
                                break
                        for a2 in (p2.get("aliases") or []):
                            if fuzz.ratio(p1["name"].lower(), a2.lower()) >= threshold:
                                alias_match = True
                                break
                        if not alias_match:
                            continue

                    # Potential duplicate found
                    issues.append(AuditIssue(
                        person_id=p1["id"],
                        person_name=p1["name"],
                        issue_type=AuditIssueType.DUPLICATE_ENTRY,
                        confidence=name_sim / 100.0,
                        title=f"Possible duplicate: {p1['name']} ↔ {p2['name']}",
                        details=f"Name similarity: {name_sim}%. IDs: {p1['id']} and {p2['id']}. "
                                f"Categories: {p1.get('category', '?')} / {p2.get('category', '?')}.",
                        evidence=[AuditEvidence(
                            type="person",
                            id=p2["id"],
                            snippet=f"{p2['name']} ({p2.get('category', '?')}): {(p2.get('shortBio') or '')[:100]}",
                            relevance=name_sim / 100.0,
                        )],
                        phase="dedup",
                        dimensions=SeverityDimensions(
                            impact=3, certainty=max(1, round(name_sim / 20)),
                            scope=2, legal_risk=3, factual_gravity=4,
                        ),
                    ))

                self.state.mark_processed(p1["id"], "audit_dedup")

        return issues

    # ── Phase 2: Wikidata ───────────────────────────────────────────────

    def _phase_wikidata(self, persons: list[dict], resume: bool) -> list[AuditIssue]:
        """Cross-reference persons against Wikidata and Wikipedia."""
        issues: list[AuditIssue] = []
        http = self._get_http()
        rate_limit = self.settings.auditor_wikidata_rate_limit

        with Progress(TextColumn("[progress.description]{task.description}"),
                      BarColumn(), MofNCompleteColumn()) as progress:
            task = progress.add_task("Wikidata check", total=len(persons))

            for person in persons:
                progress.advance(task)
                pid = person["id"]

                if resume and self.state.is_processed(pid, "audit_wikidata"):
                    continue

                name = person["name"]
                category = person.get("category", "")
                bio = person.get("shortBio", "") or ""

                # Search Wikidata
                try:
                    wd = self._query_wikidata(http, name)
                except Exception as e:
                    logger.debug("Wikidata error for %s: %s", name, e)
                    wd = None

                if wd:
                    # Check category mismatch
                    if wd.occupations and category:
                        wd_cats = set()
                        for occ in wd.occupations:
                            occ_lower = occ.lower()
                            for kw, cat in OCCUPATION_TO_CATEGORY.items():
                                if kw in occ_lower:
                                    wd_cats.add(cat)
                        if wd_cats and category not in wd_cats:
                            issues.append(AuditIssue(
                                person_id=pid,
                                person_name=name,
                                issue_type=AuditIssueType.WRONG_CATEGORY,
                                confidence=wd.confidence,
                                title=f"Category mismatch: DB says '{category}', Wikidata says '{', '.join(wd.occupations)}'",
                                details=f"Wikidata QID: {wd.qid}. Our category: {category}. "
                                        f"Wikidata occupations: {', '.join(wd.occupations)}. "
                                        f"Suggested categories: {', '.join(wd_cats)}.",
                                evidence=[AuditEvidence(
                                    type="wikidata", id=wd.qid,
                                    snippet=f"Wikidata: {wd.label} — {wd.description or 'no description'}",
                                    relevance=wd.confidence,
                                )],
                                wikidata_qid=wd.qid,
                                phase="wikidata",
                                dimensions=SeverityDimensions(
                                    impact=3, certainty=4, scope=1, legal_risk=1, factual_gravity=2,
                                ),
                            ))

                    # Check if person is deceased but bio doesn't mention it
                    if wd.death_date and "deceased" not in bio.lower() and "died" not in bio.lower():
                        issues.append(AuditIssue(
                            person_id=pid,
                            person_name=name,
                            issue_type=AuditIssueType.STALE_DATA,
                            confidence=wd.confidence * 0.8,
                            title=f"Person deceased ({wd.death_date}) but bio doesn't note it",
                            details=f"Wikidata shows death date: {wd.death_date}. Bio: {bio[:200]}",
                            evidence=[AuditEvidence(
                                type="wikidata", id=wd.qid,
                                snippet=f"Death date: {wd.death_date}",
                                relevance=0.9,
                            )],
                            wikidata_qid=wd.qid,
                            phase="wikidata",
                            dimensions=SeverityDimensions(
                                impact=2, certainty=4, scope=1, legal_risk=1, factual_gravity=1,
                            ),
                        ))

                    # Get Wikipedia summary for deeper comparison
                    try:
                        wiki_summary = self._query_wikipedia(http, name)
                    except Exception:
                        wiki_summary = None

                    if wiki_summary and bio:
                        # Use LLM to compare Wikipedia vs our bio
                        comparison_issues = self._compare_bios(pid, name, bio, wiki_summary, wd.qid)
                        issues.extend(comparison_issues)

                self.state.mark_processed(pid, "audit_wikidata")
                time.sleep(rate_limit)

        return issues

    def _query_wikidata(self, http: httpx.Client, name: str) -> WikidataMatch | None:
        """Search Wikidata for a person and fetch structured claims."""
        # Search
        search_url = "https://www.wikidata.org/w/api.php"
        resp = http.get(search_url, params={
            "action": "wbsearchentities",
            "search": name,
            "language": "en",
            "type": "item",
            "limit": 5,
            "format": "json",
        })
        resp.raise_for_status()
        results = resp.json().get("search", [])

        if not results:
            return None

        # Find best match (human entity)
        qid = None
        best_label = ""
        best_desc = ""
        for r in results:
            qid = r["id"]
            best_label = r.get("label", "")
            best_desc = r.get("description", "")
            # Prefer items described as human-related
            desc_lower = (best_desc or "").lower()
            if any(kw in desc_lower for kw in ("politician", "business", "lawyer", "actor", "model",
                                                 "socialite", "scientist", "academic", "prince",
                                                 "investor", "financier", "journalist")):
                break

        if not qid:
            return None

        # Fetch entity claims
        entity_url = "https://www.wikidata.org/w/api.php"
        resp = http.get(entity_url, params={
            "action": "wbgetentities",
            "ids": qid,
            "props": "claims|descriptions|labels",
            "languages": "en",
            "format": "json",
        })
        resp.raise_for_status()
        entity = resp.json().get("entities", {}).get(qid, {})
        claims = entity.get("claims", {})

        # Extract structured properties
        birth_date = self._extract_date(claims.get(WD_BIRTH, []))
        death_date = self._extract_date(claims.get(WD_DEATH, []))
        occupations = self._extract_labels(claims.get(WD_OCCUPATION, []), http)

        # Check if it's actually a human (P31 = Q5)
        instance_of = claims.get(WD_INSTANCE_OF, [])
        is_human = any(
            c.get("mainsnak", {}).get("datavalue", {}).get("value", {}).get("id") == "Q5"
            for c in instance_of
        )
        confidence = 0.9 if is_human else 0.5

        # Name similarity check
        name_sim = fuzz.ratio(name.lower(), best_label.lower())
        confidence *= min(1.0, name_sim / 90.0)

        return WikidataMatch(
            qid=qid,
            label=best_label,
            description=best_desc,
            birth_date=birth_date,
            death_date=death_date,
            occupations=occupations,
            confidence=round(confidence, 2),
        )

    def _extract_date(self, claims: list[dict]) -> str | None:
        """Extract date from Wikidata time claims."""
        for c in claims:
            try:
                time_val = c["mainsnak"]["datavalue"]["value"]["time"]
                # Format: +1953-01-20T00:00:00Z -> 1953-01-20
                return time_val.lstrip("+").split("T")[0]
            except (KeyError, IndexError):
                continue
        return None

    def _extract_labels(self, claims: list[dict], http: httpx.Client) -> list[str]:
        """Extract human-readable labels from Wikidata entity claims."""
        labels = []
        qids = []
        for c in claims:
            try:
                qid = c["mainsnak"]["datavalue"]["value"]["id"]
                qids.append(qid)
            except (KeyError, IndexError):
                continue

        if not qids:
            return labels

        # Batch fetch labels
        resp = http.get("https://www.wikidata.org/w/api.php", params={
            "action": "wbgetentities",
            "ids": "|".join(qids[:10]),
            "props": "labels",
            "languages": "en",
            "format": "json",
        })
        resp.raise_for_status()
        entities = resp.json().get("entities", {})
        for qid in qids[:10]:
            label = entities.get(qid, {}).get("labels", {}).get("en", {}).get("value")
            if label:
                labels.append(label)

        return labels

    def _query_wikipedia(self, http: httpx.Client, name: str) -> str | None:
        """Fetch Wikipedia summary for a person."""
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{name.replace(' ', '_')}"
        try:
            resp = http.get(url)
            if resp.status_code == 200:
                data = resp.json()
                return data.get("extract", "")
        except Exception:
            pass
        return None

    def _compare_bios(self, pid: str, name: str, our_bio: str, wiki_summary: str, qid: str) -> list[AuditIssue]:
        """Use Claude to compare our bio against Wikipedia for contradictions."""
        issues = []
        try:
            client = self._get_anthropic()
            resp = client.messages.create(
                model=self.settings.auditor_anthropic_model,
                max_tokens=500,
                messages=[{
                    "role": "user",
                    "content": f"""Compare these two descriptions of {name} and identify any factual contradictions.

OUR DATABASE BIO: {our_bio[:500]}

WIKIPEDIA: {wiki_summary[:800]}

If there are contradictions (wrong profession, wrong dates, wrong nationality, conflated with different person), output JSON:
{{"contradictions": [{{"our_claim": "...", "wikipedia_says": "...", "severity": "high|medium|low"}}]}}

If no contradictions found, output: {{"contradictions": []}}

Output ONLY valid JSON, nothing else."""
                }],
            )
            self._track_cost(resp.usage, self.settings.auditor_anthropic_model)

            text = resp.content[0].text.strip()
            data = json.loads(text)
            for c in data.get("contradictions", []):
                sev_map = {"high": 4, "medium": 3, "low": 2}
                issues.append(AuditIssue(
                    person_id=pid,
                    person_name=name,
                    issue_type=AuditIssueType.EXTERNAL_CONTRADICTION,
                    confidence=0.75,
                    title=f"Bio contradicts Wikipedia: {c.get('our_claim', '')[:60]}",
                    details=f"Our bio says: {c.get('our_claim', '')}. "
                            f"Wikipedia says: {c.get('wikipedia_says', '')}.",
                    evidence=[
                        AuditEvidence(type="wikipedia", snippet=c.get("wikipedia_says", ""), relevance=0.8),
                        AuditEvidence(type="wikidata", id=qid, snippet=f"QID: {qid}", relevance=0.9),
                    ],
                    wikidata_qid=qid,
                    phase="wikidata",
                    dimensions=SeverityDimensions(
                        impact=4, certainty=sev_map.get(c.get("severity", "medium"), 3),
                        scope=1, legal_risk=3, factual_gravity=sev_map.get(c.get("severity", "medium"), 3),
                    ),
                ))
        except Exception as e:
            logger.debug("Bio comparison error for %s: %s", name, e)

        return issues

    # ── Phase 3: Bio Fact-Check ─────────────────────────────────────────

    def _phase_fact_check(self, persons: list[dict], resume: bool) -> list[AuditIssue]:
        """Decompose bios into atomic claims and verify against documents."""
        issues: list[AuditIssue] = []
        conn = self._get_neon()
        client = self._get_anthropic()

        with Progress(TextColumn("[progress.description]{task.description}"),
                      BarColumn(), MofNCompleteColumn()) as progress:
            task = progress.add_task("Fact-checking bios", total=len(persons))

            for person in persons:
                progress.advance(task)
                pid = person["id"]

                if resume and self.state.is_processed(pid, "audit_factcheck"):
                    continue

                bio = (person.get("shortBio") or "") + " " + (person.get("description") or "")
                bio = bio.strip()
                if len(bio) < 30:
                    self.state.mark_processed(pid, "audit_factcheck")
                    continue

                # Step 1: Decompose bio into atomic claims
                claims = self._decompose_bio(client, person["name"], bio)
                if not claims:
                    self.state.mark_processed(pid, "audit_factcheck")
                    continue

                # Step 2: For each claim, retrieve evidence and verify
                for claim in claims[:self.settings.auditor_max_claims_per_person]:
                    if not claim.get("verifiable", True):
                        continue

                    # Retrieve document evidence
                    evidence = self._retrieve_evidence(conn, claim["claim"], pid)

                    # Verify claim against evidence
                    verification = self._verify_claim(client, person["name"], claim, evidence)
                    if verification and verification.verdict == "CONTRADICTED":
                        issues.append(AuditIssue(
                            person_id=pid,
                            person_name=person["name"],
                            issue_type=AuditIssueType.BIO_CONTRADICTION,
                            confidence=verification.confidence,
                            title=f"Bio claim contradicted by documents: {claim['claim'][:60]}",
                            details=f"Claim: {claim['claim']}. Verdict: CONTRADICTED. "
                                    f"Reasoning: {verification.reasoning}",
                            evidence=verification.evidence,
                            phase="factcheck",
                            dimensions=SeverityDimensions(
                                impact=3, certainty=round(verification.confidence * 5),
                                scope=1, legal_risk=3, factual_gravity=3,
                            ),
                        ))
                    elif verification and verification.verdict == "UNVERIFIABLE":
                        # Only flag as ungrounded if confidence is high
                        if verification.confidence > 0.6:
                            issues.append(AuditIssue(
                                person_id=pid,
                                person_name=person["name"],
                                issue_type=AuditIssueType.UNGROUNDED_CLAIM,
                                confidence=verification.confidence * 0.7,
                                title=f"Ungrounded claim: {claim['claim'][:60]}",
                                details=f"Claim: {claim['claim']}. No supporting evidence found in documents.",
                                evidence=verification.evidence,
                                phase="factcheck",
                                dimensions=SeverityDimensions(
                                    impact=2, certainty=3, scope=1, legal_risk=2, factual_gravity=2,
                                ),
                            ))

                self.state.mark_processed(pid, "audit_factcheck")

        return issues

    def _decompose_bio(self, client, name: str, bio: str) -> list[dict]:
        """Use Claude to decompose a bio into atomic facts."""
        try:
            resp = client.messages.create(
                model=self.settings.auditor_fast_model,
                max_tokens=800,
                messages=[{
                    "role": "user",
                    "content": f"""Extract atomic factual claims from this person's bio. Each claim should be independently verifiable.

Person: {name}
Bio: {bio[:600]}

Output JSON array:
[{{"claim": "...", "type": "biographical|relational|temporal|legal|professional", "verifiable": true/false}}]

Only include factual claims, not opinions or vague descriptions. Max 10 claims. Output ONLY valid JSON."""
                }],
            )
            self._track_cost(resp.usage, self.settings.auditor_fast_model)
            return json.loads(resp.content[0].text.strip())
        except Exception as e:
            logger.debug("Bio decomposition error for %s: %s", name, e)
            return []

    def _retrieve_evidence(self, conn, claim: str, person_id: str) -> list[AuditEvidence]:
        """Retrieve document evidence for a claim using FTS."""
        evidence = []
        try:
            # Search documents linked to this person
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT d.id, d.title, d.summary
                    FROM document_persons dp
                    JOIN documents d ON d.id = dp.doc_id
                    WHERE dp.person_id = %s
                    AND d.tsv @@ plainto_tsquery('english', %s)
                    LIMIT %s
                """, (person_id, claim, self.settings.auditor_max_doc_chunks))

                for row in cur.fetchall():
                    evidence.append(AuditEvidence(
                        type="document",
                        id=row[0],
                        title=row[1],
                        snippet=(row[2] or "")[:200],
                        relevance=0.7,
                    ))

                # Also search OCR text if few results
                if len(evidence) < 2:
                    cur.execute("""
                        SELECT ot."docId", substring(ot.text, 1, 200) as excerpt
                        FROM ocr_text ot
                        WHERE ot."docId" IN (
                            SELECT dp.doc_id FROM document_persons dp WHERE dp.person_id = %s
                        )
                        AND ot.tsv @@ plainto_tsquery('english', %s)
                        LIMIT %s
                    """, (person_id, claim, 3))

                    for row in cur.fetchall():
                        evidence.append(AuditEvidence(
                            type="document",
                            id=row[0],
                            snippet=row[1] or "",
                            relevance=0.6,
                        ))

        except Exception as e:
            logger.debug("Evidence retrieval error: %s", e)

        return evidence

    def _verify_claim(self, client, name: str, claim: dict, evidence: list[AuditEvidence]) -> ClaimVerification | None:
        """Use Claude to verify a claim against document evidence."""
        evidence_text = "\n".join(
            f"- [{e.id or '?'}] {e.snippet}" for e in evidence
        ) or "No document evidence found."

        try:
            resp = client.messages.create(
                model=self.settings.auditor_anthropic_model,
                max_tokens=400,
                messages=[{
                    "role": "user",
                    "content": f"""Verify this claim about {name} against the provided document evidence.

CLAIM: {claim['claim']}

DOCUMENT EVIDENCE:
{evidence_text}

Classify the claim as:
- SUPPORTED: Evidence directly supports this claim
- CONTRADICTED: Evidence directly contradicts this claim
- PARTIALLY_SUPPORTED: Some aspects confirmed, others not
- UNVERIFIABLE: No relevant evidence found

Output JSON: {{"verdict": "...", "confidence": 0.0-1.0, "reasoning": "brief explanation"}}
Output ONLY valid JSON."""
                }],
            )
            self._track_cost(resp.usage, self.settings.auditor_anthropic_model)

            data = json.loads(resp.content[0].text.strip())
            return ClaimVerification(
                claim=claim["claim"],
                claim_type=claim.get("type", "biographical"),
                verdict=data["verdict"],
                confidence=data.get("confidence", 0.5),
                evidence=evidence,
                reasoning=data.get("reasoning", ""),
            )
        except Exception as e:
            logger.debug("Claim verification error: %s", e)
            return None

    # ── Phase 4: Identity Coherence ─────────────────────────────────────

    def _phase_coherence(self, persons: list[dict], resume: bool) -> list[AuditIssue]:
        """Detect merged identities by sampling linked documents."""
        issues: list[AuditIssue] = []
        conn = self._get_neon()
        client = self._get_anthropic()

        # Only check persons with significant document counts
        with Progress(TextColumn("[progress.description]{task.description}"),
                      BarColumn(), MofNCompleteColumn()) as progress:
            task = progress.add_task("Coherence check", total=len(persons))

            for person in persons:
                progress.advance(task)
                pid = person["id"]

                if resume and self.state.is_processed(pid, "audit_coherence"):
                    continue

                # Get doc count
                with conn.cursor() as cur:
                    cur.execute("SELECT COUNT(*)::int FROM document_persons WHERE person_id = %s", (pid,))
                    doc_count = cur.fetchone()[0]

                if doc_count < 10:
                    self.state.mark_processed(pid, "audit_coherence")
                    continue

                # Sample document titles
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT d.title, d.summary, d.category, d.source
                        FROM document_persons dp
                        JOIN documents d ON d.id = dp.doc_id
                        WHERE dp.person_id = %s AND d.title IS NOT NULL
                        ORDER BY RANDOM()
                        LIMIT 15
                    """, (pid,))
                    docs = cur.fetchall()

                if len(docs) < 5:
                    self.state.mark_processed(pid, "audit_coherence")
                    continue

                doc_list = "\n".join(
                    f"- [{d[3]}] {d[0]}: {(d[1] or '')[:100]}" for d in docs
                )

                # Ask Claude if this looks like one person
                try:
                    resp = client.messages.create(
                        model=self.settings.auditor_anthropic_model,
                        max_tokens=400,
                        messages=[{
                            "role": "user",
                            "content": f"""Review these documents linked to "{person['name']}" ({person.get('category', '?')}).
Bio: {(person.get('shortBio') or '')[:200]}

Documents (sample of {doc_count}):
{doc_list}

Question: Do these documents consistently reference the SAME person, or is there evidence that this record might be conflating TWO OR MORE different people with the same or similar name?

Output JSON: {{"coherent": true/false, "confidence": 0.0-1.0, "reasoning": "brief explanation", "possible_identities": ["identity1 description", "identity2 description"] if not coherent}}
Output ONLY valid JSON."""
                        }],
                    )
                    self._track_cost(resp.usage, self.settings.auditor_anthropic_model)

                    data = json.loads(resp.content[0].text.strip())
                    if not data.get("coherent", True):
                        identities = data.get("possible_identities", [])
                        issues.append(AuditIssue(
                            person_id=pid,
                            person_name=person["name"],
                            issue_type=AuditIssueType.MERGED_IDENTITY,
                            confidence=data.get("confidence", 0.6),
                            title=f"Possible merged identity: {person['name']} may be {len(identities)} people",
                            details=f"Reasoning: {data.get('reasoning', '')}. "
                                    f"Possible identities: {'; '.join(identities)}. "
                                    f"Based on sample of {len(docs)} of {doc_count} linked documents.",
                            evidence=[AuditEvidence(
                                type="co_occurrence",
                                snippet=f"Sampled {len(docs)} of {doc_count} documents",
                                relevance=0.8,
                            )],
                            phase="coherence",
                            dimensions=SeverityDimensions(
                                impact=4, certainty=round(data.get("confidence", 0.6) * 5),
                                scope=4, legal_risk=4, factual_gravity=5,
                            ),
                        ))

                except Exception as e:
                    logger.debug("Coherence check error for %s: %s", person["name"], e)

                self.state.mark_processed(pid, "audit_coherence")

        return issues

    # ── Phase 5: Score + Store ──────────────────────────────────────────

    def _phase_score(self, run_id: str, issues: list[AuditIssue]) -> None:
        """Store issues in Neon and create ai_leads for critical/high."""
        conn = self._get_neon()
        critical_threshold = self.settings.auditor_severity_critical
        high_threshold = self.settings.auditor_severity_high

        with conn.cursor() as cur:
            # Create audit run record
            cur.execute("""
                INSERT INTO person_audit_runs (id, started_at, issues_found,
                    critical_count, high_count, medium_count, low_count)
                VALUES (%s, NOW(), %s, %s, %s, %s, %s)
            """, (
                run_id,
                len(issues),
                sum(1 for i in issues if i.severity >= critical_threshold),
                sum(1 for i in issues if high_threshold <= i.severity < critical_threshold),
                sum(1 for i in issues if self.settings.auditor_severity_medium <= i.severity < high_threshold),
                sum(1 for i in issues if i.severity < self.settings.auditor_severity_medium),
            ))

            # Store each issue
            for issue in issues:
                cur.execute("""
                    INSERT INTO person_audit_issues
                        (run_id, person_id, issue_type, severity, confidence,
                         title, details, evidence, wikidata_qid, phase, dimensions)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    run_id, issue.person_id, issue.issue_type.value,
                    issue.severity, issue.confidence,
                    issue.title, issue.details,
                    json.dumps([e.model_dump() for e in issue.evidence]),
                    issue.wikidata_qid, issue.phase,
                    json.dumps(issue.dimensions.model_dump()),
                ))

                # Create ai_leads for critical and high issues
                if issue.severity >= high_threshold:
                    priority = min(0.95, issue.severity / 100.0)
                    lead_type = "site_issue"
                    cur.execute("""
                        INSERT INTO ai_leads
                            (id, agent, lead_type, status, priority,
                             title, summary, evidence, entity_refs, created_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
                    """, (
                        f"audit-{uuid.uuid4().hex[:12]}",
                        "auditor",
                        lead_type,
                        "pending",
                        priority,
                        issue.title,
                        issue.details,
                        json.dumps([e.model_dump() for e in issue.evidence]),
                        json.dumps([{"type": "person", "id": issue.person_id, "name": issue.person_name}]),
                    ))

        conn.commit()
        logger.info("Stored %d issues, created %d leads",
                     len(issues), sum(1 for i in issues if i.severity >= high_threshold))

    # ── Helpers ─────────────────────────────────────────────────────────

    def _fetch_persons(self, person_ids: list[str] | None, limit: int | None) -> list[dict]:
        """Fetch persons from Neon."""
        conn = self._get_neon()
        with conn.cursor() as cur:
            if person_ids:
                cur.execute("""
                    SELECT id, slug, name, aliases, category, "shortBio", description,
                           "blackBookEntry", "imageUrl"
                    FROM persons WHERE id = ANY(%s) ORDER BY id
                """, (person_ids,))
            else:
                query = """
                    SELECT id, slug, name, aliases, category, "shortBio", description,
                           "blackBookEntry", "imageUrl"
                    FROM persons ORDER BY id
                """
                if limit:
                    query += f" LIMIT {int(limit)}"
                cur.execute(query)

            columns = [desc[0] for desc in cur.description]
            return [dict(zip(columns, row)) for row in cur.fetchall()]

    def _track_cost(self, usage, model: str) -> None:
        """Track API costs."""
        input_tokens = getattr(usage, "input_tokens", 0)
        output_tokens = getattr(usage, "output_tokens", 0)
        self._total_tokens += input_tokens + output_tokens

        # Cost per million tokens
        costs = {
            "claude-sonnet-4-6": (3.0, 15.0),
            "claude-haiku-4-5-20251001": (0.8, 4.0),
            "claude-opus-4-6": (15.0, 75.0),
        }
        in_rate, out_rate = costs.get(model, (3.0, 15.0))
        cost = (input_tokens * in_rate + output_tokens * out_rate) / 1_000_000
        if self.settings.auditor_use_batch_api:
            cost *= 0.5
        self._total_cost_cents += round(cost * 100)

    def close(self) -> None:
        """Clean up connections."""
        if self._neon:
            self._neon.close()
        if self._http:
            self._http.close()
        self.state.close()
