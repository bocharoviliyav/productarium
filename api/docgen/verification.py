"""Verification pipeline for the docgen flows (Wave D).

Deterministic post-generation checks + LLM judge + diff-regeneration +
provenance, shared by the codebase (deepagents) and spec (LangGraph) flows.
Every stage is non-fatal by design EXCEPT the "all sections empty" guard
(surfaced by the callers through the pre-existing
``_raise_if_all_sections_unavailable`` contract).

Stages
------
(a) **Guard** — deterministic checks over generated content:
    - secret/token detection + in-place masking (always applied; the masked
      text is what gets persisted, so secrets never reach the UI or the
      memory index);
    - mermaid validity stays in the flows via the pre-existing
      ``api.formats.mermaid.run_repair_loop`` (this module only records the
      stats into the provenance report);
    - section-structure check (all expected sections present / non-empty).
(b) **Citations** — file-path citations (``src/api.py`` / ``src/api.py:42-58``)
    are extracted from the generated text and resolved against the actual
    repository file set. When the file set is real (the repo clone), the
    citation GUARD additionally rewrites the persisted text: unresolvable
    citations are removed and impossible line spans stripped
    (``api.docgen.citation_guard``, port of the fork's validators). Flows
    without a real file set (databases) keep the warning-only behaviour.
(c) **LLM judge** — factual-consistency scoring of a section against the
    source evidence, via the ``judge`` task factory (``api.llm``). Failure or
    an "inconsistent" verdict degrades to a warning recorded in provenance;
    content is kept (documented policy: judge flags, it does not block).
(d) **Diff regeneration** — when docs already exist, per-section source
    fingerprints (hash of the files the section was based on) decide which
    sections must be regenerated; unchanged sections are reused verbatim.
(e) **Provenance** — per-section metadata (model, prompt file, source files,
    citations, judge verdict, regen status) stored additively inside
    ``pages[page_id]["provenance"]`` (backwards compatible: the frontend
    viewer reads only the legacy keys).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from api.utils.fs import open_read_nofollow

logger = logging.getLogger(__name__)

# Statuses used across the pipeline.
CHECK_OK = "ok"
CHECK_WARN = "warn"
CHECK_FAIL = "fail"

# Replacement used for masked secrets. The original value NEVER survives
# masking, so the persisted docs and the memory index are secret-free.
SECRET_MASK = "***REDACTED***"


# --------------------------------------------------------------------------- #
# (a) Guard: secrets
# --------------------------------------------------------------------------- #
# Deterministic secret patterns. Each entry: (compiled regex, label).
# The patterns deliberately target well-known token SHAPES / inline usage
# forms that a documentation model may leak from real config files it read,
# not arbitrary long words (to keep the false-positive rate low).
_SECRET_PATTERNS: Tuple[Tuple[re.Pattern, str], ...] = (
    # Well-known token shapes (GitHub/GitLab/AWS/Google/slack/xAI prefixes).
    (re.compile(r"\bgh[pousr]_[A-Za-z0-9]{16,}\b"), "github_token"),
    (re.compile(r"\bglpat-[A-Za-z0-9_\-]{16,}\b"), "gitlab_token"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "aws_access_key"),
    (re.compile(r"\bAIza[0-9A-Za-z_\-]{30,}\b"), "google_api_key"),
    (re.compile(r"\bxox[baprs]-[A-Za-z0-9\-]{10,}\b"), "slack_token"),
    (re.compile(r"\bgsk_[A-Za-z0-9]{20,}\b"), "xai_api_key"),
    # OpenAI-style keys (sk-...), but not the project's own "not-needed"
    # placeholder.
    (re.compile(r"\bsk-(?!needed\b)[A-Za-z0-9_\-]{16,}\b"), "openai_style_key"),
    # Bearer tokens in URLs / header examples.
    (re.compile(r"(?i)\bbearer\s+[A-Za-z0-9_\-\.=/+]{16,}\b"), "bearer_token"),
    # Credentials embedded in URLs (https://user:pass@host).
    (
        re.compile(r"(?i)\b(https?|ftp)://[^\s/:]+:[^\s@/]{4,}@"),
        "url_credentials",
    ),
)

# Secret-ish KEY names for assignment-style leaks (JSON / YAML / ENV).
_SECRET_KEY_NAMES = (
    r"api[_-]?key|apikey|secret(?:[_-]?key)?|token|password|passwd|pwd|"
    r"access[_-]?key|private[_-]?key|client[_-]?secret|auth[_-]?token"
)
# ``key: value`` / ``key = value`` with an OPTIONAL quoted key (JSON) and a
# quoted (JSON/YAML) OR bare (YAML/ENV) value. The key stays visible; only
# the value is masked. Catches the review gaps: unquoted YAML
# (``password: Hunter2XXX``), JSON objects (``"password": "..."``) and hex
# secrets after a secret-ish key (``secret_key: 3f9a...``).
_ASSIGNMENT_RE = re.compile(
    r"(?i)(?P<keep>[\"']?\b(?:" + _SECRET_KEY_NAMES + r")\b[\"']?\s*[:=]\s*)"
    r"(?:(?P<q>['\"])(?P<qval>[^\s'\"]{8,})(?P=q)|(?P<bval>[^\s'\"`,;]{8,}))"
)

# Env-like assignments (KEY=value without quotes) are only masked when the
# value is long enough AND not an obvious placeholder/interpolation. Hash /
# signature-style variable names are included (``HEX_HASH=...``) — masking a
# benign hash value is an acceptable trade-off for a documentation guard.
_ENV_SECRET_RE = re.compile(
    r"(?i)\b([A-Z0-9_]*(?:API_KEY|APIKEY|SECRET|TOKEN|PASSWORD|PASSWD|PWD|"
    r"HASH|SIGNATURE|SIGNING)[A-Z0-9_]*)\b\s*=\s*([^\s'\"`,;]{8,})"
)
_ENV_PLACEHOLD_VALUES = frozenset({
    "not-needed", "not_needed", "changeme", "changeme!", "placeholder",
    "your-api-key", "your_api_key", "xxx", "xxxx", "example", "dummy",
    "test", "secret", "password", "${...}", "<your-key>", "your-token-here",
})


def _is_non_secret_value(value: str) -> bool:
    """True for values that must NOT be masked (placeholders, booleans,
    digits, empty or already-masked text)."""
    v = (value or "").strip()
    if not v:
        return True
    if SECRET_MASK in v:
        return True
    if v.lower() in _ENV_PLACEHOLD_VALUES or v.startswith("${"):
        return True
    if v.lower() in ("true", "false", "none", "null", "yes", "no"):
        return True
    if v.isdigit():
        return True
    return False


def mask_secrets(text: str) -> Tuple[str, List[str]]:
    """Mask secret-looking substrings in ``text``.

    Returns ``(masked_text, findings)`` where each finding is a short label
    (``"github_token"`` etc.; the matched value is deliberately NOT included
    in the finding so logs/reports stay secret-free). A value is masked at
    most once: later passes skip anything already containing the mask, so a
    token caught by its prefix (``token: ghp_...``) is counted exactly once.
    """
    if not text:
        return text, []
    findings: List[str] = []

    def _sub(match: "re.Match", label: str) -> str:
        findings.append(label)
        return SECRET_MASK

    # 1) well-known token shapes first (most specific: ``token: ghp_...`` is
    #    labelled as a github token, not a generic assignment).
    masked = text
    for pattern, label in _SECRET_PATTERNS:
        masked = pattern.sub(lambda m, _label=label: _sub(m, _label), masked)

    # 2) assignments with a secret-ish key (JSON/YAML/ENV, quoted or bare).
    def _sub_assignment(match: "re.Match") -> str:
        value = match.group("qval") if match.group("q") else match.group("bval")
        if _is_non_secret_value(value or ""):
            return match.group(0)
        findings.append("secret_assignment")
        return match.group("keep") + SECRET_MASK

    masked = _ASSIGNMENT_RE.sub(_sub_assignment, masked)

    # 3) env-style: mask only the VALUE, keep the variable name visible (it
    # is useful, non-secret documentation context).
    def _sub_env(match: "re.Match") -> str:
        if _is_non_secret_value(match.group(2)):
            return match.group(0)
        findings.append("env_secret")
        return f"{match.group(1)}={SECRET_MASK}"

    masked = _ENV_SECRET_RE.sub(_sub_env, masked)
    return masked, findings


# --------------------------------------------------------------------------- #
# (a1) Guard: DSN credentials (database connection strings)
# --------------------------------------------------------------------------- #
# ``scheme://[user[:password]@]host[:port][/name][?params]`` — only the
# password (and any secret-looking query param VALUE) is masked; scheme,
# user, host, port, database name and non-secret params stay visible because
# they are useful, non-secret documentation context.
_DSN_SCHEME_RE = re.compile(
    r"^(?P<scheme>[A-Za-z][A-Za-z0-9+.\-]*://)(?P<rest>.+)$",
    re.DOTALL,
)

# Non-URL (key/value) DSNs (libpq ``host=... password=... dbname=...``):
# the generic ``mask_secrets`` assignment rules require 8+ char values and
# env-style UPPERCASE names, so a SHORT lowercase ``password=hunter2`` token
# would slip through — this targeted rule closes that gap for the DSN path
# only (idempotent: re-masking an already-masked value is a no-op).
_KV_DSN_RE = re.compile(
    r"(?i)\b(password|passwd|pwd|secret|token|passphrase)\s*=\s*([^\s;]+)"
)


def _mask_kv_dsn(text: str) -> str:
    """Mask bare ``password=...``-style tokens in a key/value DSN."""
    return _KV_DSN_RE.sub(lambda m: f"{m.group(1)}={SECRET_MASK}", text)


def _mask_url_dsn_userinfo(text: str) -> Optional[str]:
    """Mask the password inside a URL-style DSN's userinfo — RIGHT-to-left.

    The userinfo necessarily ends at the LAST ``@`` (an RFC 3986 authority
    cannot contain a raw one), so passwords containing ``@`` or ``/`` are
    captured whole. A left-to-right match would cut the userinfo at the
    first ``@`` and leak the password tail — or fail to match at all when
    the password contains ``/`` and mask NOTHING (review #4, HIGH).
    Returns ``None`` when the text is not a URL-style DSN with userinfo.
    """
    m = _DSN_SCHEME_RE.match(text)
    if m is None or "@" not in m.group("rest"):
        return None
    creds_raw, _, tail = m.group("rest").rpartition("@")
    user, sep, password = creds_raw.partition(":")
    if sep and password and SECRET_MASK not in password:
        creds = f"{user}:{SECRET_MASK}"
    elif "@" in creds_raw and ":" not in creds_raw:
        # Multiple '@' with no password separator: ambiguous userinfo —
        # mask the whole credentials block instead of guessing a split.
        creds = SECRET_MASK
    else:
        creds = creds_raw
    # Mask only the REMAINDER (host/db/query params) so the generic
    # url_credentials pattern cannot re-mask (and mangle) our own
    # already-masked credentials — keeps the function idempotent.
    tail, _findings = mask_secrets(tail)
    # Query-param passwords (``?password=hunter2``) shorter than the
    # generic 8-char assignment floor get the targeted kv rule too.
    tail = _mask_kv_dsn(tail)
    return f"{m.group('scheme')}{creds}@{tail}"


def mask_dsn(dsn: str) -> str:
    """Mask the secret part of a connection DSN; keep the useful context.

    - ``postgres://app:hunter2@db:5432/prod`` →
      ``postgres://app:***REDACTED***@db:5432/prod``
    - passwords containing ``@`` or ``/`` are masked whole (the userinfo is
      split at the LAST ``@``)
    - credentials without a password keep the user (``oracle://app@host``)
    - non-URL DSNs (key/value strings) go through :func:`mask_secrets`
    - the result is additionally run through :func:`mask_secrets` so tokens
      in query params (``?password=…``) cannot survive either.

    Never raises: any parsing surprise degrades to full ``mask_secrets``.
    """
    text = (dsn or "").strip()
    if not text:
        return ""
    try:
        masked_url = _mask_url_dsn_userinfo(text)
        if masked_url is not None:
            return masked_url
        masked, _findings = mask_secrets(text)
        return _mask_kv_dsn(masked)
    except Exception:  # pragma: no cover - regex cannot realistically raise
        masked, _findings = mask_secrets(text)
        return _mask_kv_dsn(masked)


# --------------------------------------------------------------------------- #
# (b) Citations
# --------------------------------------------------------------------------- #
# File-path citation: inline code (`path`) or plain path[:line[-line]].
# Paths must look like repo-relative paths (contain / or a known source ext)
# to avoid matching every camel-case word.
_PATH_EXT_RE = re.compile(
    r"\.(py|js|ts|tsx|jsx|java|go|rs|cs|rb|php|kt|swift|c|cpp|h|hpp|sql|sh|"
    r"bash|zsh|yml|yaml|toml|json|ini|cfg|conf|env|md|txt|html|css|scss|vue|"
    r"svelte|proto|gradle|xml|dockerfile|makefile|lock)$",
    re.IGNORECASE,
)
# `path` or path with optional :line / :line-line span suffix. The colon
# stays OUTSIDE the span group so line_span is the bare "42-58" form.
_CITATION_RE = re.compile(
    r"`(?P<path>[A-Za-z0-9_\-./@]+(?:\.[A-Za-z0-9]+)?)(?::(?P<span>\d+(?:-\d+)?))?`"
)
_BARE_SPAN_RE = re.compile(
    r"(?P<path>(?:[A-Za-z0-9_\-.]+/){1,}[A-Za-z0-9_\-.@]+"
    r"|[\w\-\.]+\.(?:py|js|ts|tsx|json|yaml|yml|toml|md|sql|sh|env)):"
    r"(?P<span>\d+(?:-\d+)?)"
)


@dataclass
class Citation:
    """One file-path citation extracted from generated docs."""

    path: str
    line_span: Optional[str] = None

    def key(self) -> str:
        return self.path + (f":{self.line_span}" if self.line_span else "")


def extract_citations(text: str) -> List[Citation]:
    """Extract file-path citations from markdown text (deduplicated, in order).

    Recognizes:
    - inline-code paths: `` `src/api.py` ``, `` `src/api.py:42-58` ``
    - bare paths with line spans: ``src/api.py:42-58`` (outside code fences)
    """
    if not text:
        return []
    # Strip fenced code blocks (code samples routinely contain path-like
    # strings that are not citations).
    no_fences = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    seen: Dict[str, Citation] = {}
    for m in _CITATION_RE.finditer(no_fences):
        path = m.group("path").strip("/")
        if not path or path.startswith(("http://", "https://", "ftp://")):
            continue
        if "/" not in path and not _PATH_EXT_RE.search(path):
            continue
        cite = Citation(path=path, line_span=m.group("span") or None)
        seen.setdefault(cite.key(), cite)
    for m in _BARE_SPAN_RE.finditer(no_fences):
        path = m.group("path").strip("/")
        if path.startswith(("http://", "https://")):
            continue
        cite = Citation(path=path, line_span=m.group("span") or None)
        seen.setdefault(cite.key(), cite)
    return list(seen.values())


def normalize_repo_path(path: str) -> str:
    """Normalize a repo-relative path for comparison.

    Strips ``./`` segments, duplicate slashes, and resolves ``..`` segments
    (never above the root — leftover ``..`` at the top is dropped).
    """
    parts: List[str] = []
    for seg in path.replace("\\", "/").split("/"):
        if seg in ("", "."):
            continue
        if seg == "..":
            if parts:
                parts.pop()
            continue
        parts.append(seg)
    return "/".join(parts)


def check_citations(
    citations: Sequence[Citation],
    repo_files: Sequence[str],
) -> Tuple[List[Citation], List[Citation]]:
    """Split citations into (resolved, unresolved) against ``repo_files``."""
    normalized = {normalize_repo_path(p) for p in repo_files or []}
    resolved: List[Citation] = []
    unresolved: List[Citation] = []
    for cite in citations or []:
        if normalize_repo_path(cite.path) in normalized:
            resolved.append(cite)
        else:
            unresolved.append(cite)
    return resolved, unresolved


# --------------------------------------------------------------------------- #
# (a2) Guard: section structure
# --------------------------------------------------------------------------- #
@dataclass
class StructureCheck:
    """Result of the deterministic section-structure check."""

    missing: List[str] = field(default_factory=list)
    empty: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.missing and not self.empty


def check_section_structure(
    sections: Dict[str, str],
    expected_section_ids: Sequence[str],
    placeholder: str = "",
) -> StructureCheck:
    """Check that every expected section exists and has real content.

    A section equal to ``placeholder`` (the "temporarily unavailable" marker)
    counts as empty.
    """
    check = StructureCheck()
    for sid in expected_section_ids or []:
        content = (sections or {}).get(sid)
        if content is None:
            check.missing.append(sid)
        elif not str(content).strip() or (
            placeholder and str(content).strip() == placeholder
        ):
            check.empty.append(sid)
    return check


# --------------------------------------------------------------------------- #
# (d) Diff regeneration: fingerprints over the section's source files
# --------------------------------------------------------------------------- #
def hash_text(text: str) -> str:
    """Stable sha256 of a text (used for prompts and fingerprints)."""
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def hash_file(path: str) -> Optional[str]:
    """sha256 of a file's bytes; None when unreadable.

    ``ValueError`` (e.g. an embedded null byte in an LLM-controlled path)
    is caught alongside ``OSError`` so hashing can never crash the caller.
    """
    try:
        # O_NOFOLLOW: the confinement check in compute_file_hashes resolved
        # symlinks already — refuse one swapped onto the final component.
        with open_read_nofollow(path, binary=True) as f:
            return hashlib.sha256(f.read()).hexdigest()
    except (OSError, ValueError) as e:
        logger.debug("could not hash %r: %s", path, e)
        return None


def compute_file_hashes(repo_dir: str, rel_paths: Sequence[str]) -> Dict[str, Optional[str]]:
    """Hash the given repo-relative files. Missing files hash to None.

    Paths are realpath-RESOLVED before the confinement check, so a symlink
    planted inside the clone cannot make the fingerprint include the hash
    of a file outside ``repo_dir``.
    """
    repo_abs = os.path.realpath(repo_dir)
    out: Dict[str, Optional[str]] = {}
    for rel in rel_paths or []:
        full = os.path.realpath(os.path.normpath(os.path.join(repo_dir, rel)))
        # Confinement: stay inside repo_dir (after symlink resolution).
        try:
            inside = os.path.commonpath([repo_abs, full]) == repo_abs
        except ValueError:
            inside = False
        out[normalize_repo_path(rel)] = hash_file(full) if inside else None
    return out


def section_fingerprint(
    section_id: str,
    file_hashes: Dict[str, Optional[str]],
    *,
    prompt_hash: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    tree_hash: Optional[str] = None,
) -> str:
    """Fingerprint a section's SOURCE MATERIAL (not its output).

    Combines the per-file content hashes (sorted deterministically), the
    section prompt hash, the model name, the LLM endpoint (``base_url``) and
    the whole-repo ``tree_hash`` — so reuse only happens when the repo TREE
    is unchanged too (a breaking edit in a file the agent never opened still
    forces regeneration) and a model move across endpoints invalidates the
    old fingerprints. Two generation runs with the same inputs produce the
    same fingerprint → the section can be reused instead of regenerated
    (diff regeneration). NOTE: the payload format changed in Wave D review
    fixes — pre-change fingerprints intentionally mismatch (one full
    regeneration after deploy).
    """
    payload = {
        "section_id": section_id,
        "files": {p: (h or "missing") for p, h in sorted(file_hashes.items())},
        "prompt": prompt_hash or "",
        "model": model or "",
        "base_url": base_url or "",
        "tree": tree_hash or "",
    }
    return hash_text(json.dumps(payload, sort_keys=True, ensure_ascii=False))


def get_stored_provenance(page: Any) -> Dict[str, Any]:
    """Read the provenance dict off a page object/dict ({} when absent)."""
    if isinstance(page, dict):
        prov = page.get("provenance")
    else:
        prov = getattr(page, "provenance", None)
    return prov if isinstance(prov, dict) else {}


@dataclass
class RegenPlan:
    """Diff-regeneration plan for an existing doc set.

    ``reuse`` maps section_id → previous content for sections whose stored
    fingerprint still matches (source files unchanged); ``regenerate`` lists
    the section ids that must be produced anew.
    """

    reuse: Dict[str, str] = field(default_factory=dict)
    regenerate: List[str] = field(default_factory=list)
    reasons: Dict[str, str] = field(default_factory=dict)


def plan_regeneration(
    old_pages: Optional[Dict[str, Any]],
    old_sections: Dict[str, str],
    repo_dir: str,
    expected_section_ids: Sequence[str],
    *,
    prompt_hashes: Optional[Dict[str, str]] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    tree_hash: Optional[str] = None,
) -> RegenPlan:
    """Decide which sections can be reused vs must be regenerated.

    A section is reused when the OLD pages carry provenance with (a) a
    fingerprint, (b) the list of source files it was based on, AND the
    freshly computed fingerprint over those files (+ prompt + model)
    matches. Anything else regenerates. First generation (no old pages /
    no provenance) regenerates everything.
    """
    plan = RegenPlan()
    prompt_hashes = prompt_hashes or {}
    old_pages = old_pages if isinstance(old_pages, dict) else {}
    for sid in expected_section_ids or []:
        page = old_pages.get(f"page_{sid}")
        prov = get_stored_provenance(page)
        old_fp = prov.get("fingerprint")
        source_files = prov.get("source_files") or []
        old_content = (old_sections or {}).get(sid)
        if (
            old_fp
            and isinstance(source_files, list)
            and source_files
            and isinstance(old_content, str)
            and old_content.strip()
        ):
            file_hashes = compute_file_hashes(repo_dir, source_files)
            new_fp = section_fingerprint(
                sid, file_hashes,
                prompt_hash=prompt_hashes.get(sid),
                model=model,
                base_url=base_url,
                tree_hash=tree_hash,
            )
            if new_fp == old_fp:
                plan.reuse[sid] = old_content
                plan.reasons[sid] = "source-unchanged"
                continue
            plan.reasons[sid] = "source-changed"
        else:
            plan.reasons[sid] = "no-provenance" if old_pages else "first-run"
        plan.regenerate.append(sid)
    return plan


def diff_sections(
    old_sections: Dict[str, str],
    new_sections: Dict[str, str],
) -> Dict[str, str]:
    """Per-section diff status: unchanged / changed / added."""
    out: Dict[str, str] = {}
    for sid, new in (new_sections or {}).items():
        old = (old_sections or {}).get(sid)
        if old is None:
            out[sid] = "added"
        elif (old or "").strip() == (new or "").strip():
            out[sid] = "unchanged"
        else:
            out[sid] = "changed"
    return out


# --------------------------------------------------------------------------- #
# (c) LLM judge
# --------------------------------------------------------------------------- #
# Prompt body lives in refs/prompts/docgen_judge.md (EN). Inline fallback
# keeps the JSON contract usable if the file is missing.
_JUDGE_PROMPT_FALLBACK = (
    "You are a strict documentation verifier. Compare the DRAFT documentation "
    "section against the SOURCE evidence and decide whether the draft is "
    "factually consistent with the source.\n\n"
    "Rules:\n"
    "- Only flag claims that CONTRADICT the source or are clearly NOT "
    "supported by it (fabricated identifiers, paths, endpoints, versions).\n"
    "- Missing detail is NOT an inconsistency.\n"
    "- Respond with ONLY a JSON object:\n"
    '{"consistent": true|false, "issues": ["short issue description", ...]}\n\n'
    "<source_evidence>\n{source_evidence}\n</source_evidence>\n\n"
    "<draft_section>\n{draft_section}\n</draft_section>"
)


@dataclass
class JudgeVerdict:
    """Outcome of the LLM judge call."""

    verdict: str  # consistent | inconsistent | skipped
    issues: List[str] = field(default_factory=list)


# Env kill-switch for the LLM judge stage, read at CALL time (not import
# time) so ops toggles and test monkeypatching take effect without a process
# restart. The judge itself is non-fatal either way.
def judge_enabled() -> bool:
    """True unless ``DOCGEN_JUDGE_ENABLED`` disables the judge stage."""
    return (os.environ.get("DOCGEN_JUDGE_ENABLED", "true") or "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _resolve_judge_model(
    model: Optional[str],
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Resolve (model, base_url, api_key) for the judge task from admin config."""
    try:
        from api.config.abstraction import get_task_config

        cfg = get_task_config("judge") or {}
        return model or cfg.get("model"), cfg.get("base_url"), cfg.get("api_key")
    except Exception as e:  # pragma: no cover - settings store is import-safe
        logger.debug("get_task_config(judge) failed; using defaults: %s", e)
        return model, None, None


def _build_judge_llm(
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Any:
    """Build the judge task's generator through the central LLM factory."""
    from api.llm import GenerateLLM

    return GenerateLLM(model=model, base_url=base_url, api_key=api_key)


def _parse_judge_json(text: str) -> Optional[Dict[str, Any]]:
    """Parse the judge's JSON verdict (tolerant of fences / stray prose)."""
    if not text:
        return None
    t = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", t, re.DOTALL)
    if fence:
        t = fence.group(1)
    else:
        brace = re.search(r"\{.*\}", t, re.DOTALL)
        if brace:
            t = brace.group(0)
    try:
        data = json.loads(t)
    except (ValueError, TypeError):
        return None
    return data if isinstance(data, dict) else None


async def judge_section(
    section_id: str,
    draft: str,
    source_evidence: str,
    *,
    model: Optional[str] = None,
) -> JudgeVerdict:
    """Judge one section's factual consistency against the source evidence.

    Policy (documented): the judge NEVER blocks generation. Failures (no LLM,
    timeout, unparseable output) degrade to ``skipped``; an ``inconsistent``
    verdict is recorded with its issues for the provenance report so
    reviewers can inspect it in the UI.
    """
    from api.utils.llm_helpers import cap as _cap

    if not draft or not draft.strip():
        return JudgeVerdict(verdict="skipped", issues=["empty draft"])
    try:
        from api.prompts import load_prompt_file

        template = load_prompt_file("docgen_judge.md", _JUDGE_PROMPT_FALLBACK)
    except Exception:  # pragma: no cover - import-safe
        template = _JUDGE_PROMPT_FALLBACK

    prompt = template.replace("{source_evidence}", _cap(source_evidence or "", 24_000))
    prompt = prompt.replace("{draft_section}", _cap(draft, 24_000))

    r_model, r_base_url, r_api_key = _resolve_judge_model(model)
    try:
        llm = _build_judge_llm(r_model, base_url=r_base_url, api_key=r_api_key)
    except Exception as e:
        logger.warning("Judge LLM could not be built for section %s: %s", section_id, e)
        return JudgeVerdict(verdict="skipped", issues=["judge llm unavailable"])

    # One judge client per call — always close it (the docgen pool must not
    # leak an httpx connection per judged section).
    try:
        raw = await llm.generate(prompt)
    except Exception as e:  # pragma: no cover - depends on live LLM
        logger.warning("Judge call failed for section %s: %s", section_id, e)
        return JudgeVerdict(verdict="skipped", issues=["judge call failed"])
    finally:
        from api.docgen._common import _safe_aclose

        await _safe_aclose(llm)

    data = _parse_judge_json(raw)
    if data is None:
        logger.warning("Judge output unparseable for section %s.", section_id)
        return JudgeVerdict(verdict="skipped", issues=["judge output unparseable"])
    # Strict bool parsing: a JSON "false" STRING must not be truthy.
    raw_consistent = data.get("consistent", True)
    if isinstance(raw_consistent, bool):
        consistent = raw_consistent
    elif isinstance(raw_consistent, str):
        consistent = raw_consistent.strip().lower() == "true"
    elif isinstance(raw_consistent, int) and raw_consistent in (0, 1):
        consistent = bool(raw_consistent)
    else:
        consistent = True  # missing/weird → fail-open (judge flags, never blocks)
    raw_issues = data.get("issues") or []
    issues = (
        [str(i)[:300] for i in raw_issues][:20]
        if isinstance(raw_issues, list)
        else []
    )
    return JudgeVerdict(
        verdict="consistent" if consistent else "inconsistent",
        issues=issues,
    )


# --------------------------------------------------------------------------- #
# (e) Provenance + the orchestrating pipeline
# --------------------------------------------------------------------------- #
def build_section_provenance(
    section_id: str,
    *,
    model: Optional[str],
    prompt_file: str,
    source_files: Sequence[str],
    fingerprint: Optional[str],
    citations: Dict[str, List[str]],
    judge: Optional[JudgeVerdict],
    regen: str,
    mermaid_stats: Optional[Dict[str, int]] = None,
    secrets_masked: int = 0,
    ungrounded: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Assemble the JSON-safe provenance payload stored in the page dict."""
    prov: Dict[str, Any] = {
        "section_id": section_id,
        "generated_at": _utc_now_iso(),
        "generator": "deepagents" if regen != "legacy-fallback" else "standard-llm",
        "model": model or "",
        "prompt_file": prompt_file,
        "source_files": sorted({normalize_repo_path(p) for p in source_files or []}),
        "fingerprint": fingerprint,
        "regen": regen,
        "citations": {
            "resolved": list(citations.get("resolved", [])),
            "unresolved": list(citations.get("unresolved", [])),
        },
        "secrets_masked": int(secrets_masked or 0),
    }
    # Additive guard outcome (absent for flows/sections that ran no guard).
    if citations.get("removed"):
        prov["citations"]["removed"] = list(citations["removed"])
    if citations.get("fixed"):
        prov["citations"]["fixed"] = list(citations["fixed"])
    if judge is not None:
        prov["judge"] = {"verdict": judge.verdict, "issues": judge.issues}
    if mermaid_stats:
        prov["mermaid"] = dict(mermaid_stats)
    if ungrounded:
        prov["corroborate"] = {"removed": list(ungrounded)}
    return prov


def _utc_now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class SectionVerification:
    """Everything the verification pipeline learned about one section."""

    section_id: str
    masked_content: str
    secrets_masked: int
    citations_resolved: List[str] = field(default_factory=list)
    citations_unresolved: List[str] = field(default_factory=list)
    # Citation guard outcome (removed tokens / stripped-span notes).
    citations_removed: List[str] = field(default_factory=list)
    citations_fixed: List[str] = field(default_factory=list)
    # Corroborate guard outcome: ungrounded identifiers whose sentences were
    # dropped from the persisted prose (3.1).
    corroborate_removed: List[str] = field(default_factory=list)
    judge: Optional[JudgeVerdict] = None
    fingerprint: Optional[str] = None
    warnings: List[str] = field(default_factory=list)


async def verify_section(
    section_id: str,
    content: str,
    *,
    repo_dir: str,
    repo_files: Sequence[str],
    source_files: Sequence[str],
    model: Optional[str] = None,
    prompt_hash: Optional[str] = None,
    run_judge: bool = True,
    judge_evidence: Optional[str] = None,
    base_url: Optional[str] = None,
    tree_hash: Optional[str] = None,
    grounding: Optional[Set[str]] = None,
) -> SectionVerification:
    """Run the full verification pipeline over ONE generated section.

    Stages: secret masking → citation guard → corroborate filter → citation
    extraction/resolution → fingerprint → (optional) LLM judge. Every stage
    degrades with a warning; the masked content is always returned so the
    caller persists the safe version.

    ``grounding`` (3.1) is the entity-typed identifier evidence; sentences
    naming code-like identifiers absent from it are dropped BEFORE the
    citations/judge so all stages see the same persisted prose. Falsy
    grounding skips the stage entirely.
    """
    masked, findings = mask_secrets(content or "")
    result = SectionVerification(
        section_id=section_id,
        masked_content=masked,
        secrets_masked=len(findings),
    )
    if findings:
        result.warnings.append(
            f"masked {len(findings)} secret-like value(s) "
            f"({', '.join(sorted(set(findings)))})"
        )

    # Citation guard (codebase flows with a real file set): rewrite the
    # persisted text instead of only warning — unresolvable citations are
    # removed, impossible line spans stripped. Imported lazily: the guard
    # module imports this module's citation regexes.
    if repo_files:
        try:
            from api.docgen.citation_guard import (
                guard_citations,
                line_counts_for_citations,
            )

            allowed = {normalize_repo_path(p) for p in repo_files}
            pre_cited = extract_citations(masked)
            pre_resolved, _ = check_citations(pre_cited, repo_files)
            masked, guard_report = guard_citations(
                masked, allowed, line_counts_for_citations(repo_dir, pre_resolved)
            )
            result.masked_content = masked
            result.citations_removed = guard_report.removed
            result.citations_fixed = guard_report.fixed
            if guard_report.removed:
                result.warnings.append(
                    f"citation guard: removed {len(guard_report.removed)} "
                    "unresolvable citation(s)"
                )
            if guard_report.fixed:
                result.warnings.append(
                    f"citation guard: stripped invalid line span(s) in "
                    f"{len(guard_report.fixed)} citation(s)"
                )
        except Exception as e:  # pragma: no cover - guard must never break gen
            logger.warning("Citation guard failed for section %s: %s", section_id, e)

    # Corroborate filter (3.1): drop sentences that name code-like
    # identifiers absent from the grounding evidence. Fence-safe; runs after
    # the citation guard so the persisted text, the extracted citations and
    # the judge all see the same filtered prose. Lazy import: the module
    # pulls in the citation-guard module (which imports this one).
    if grounding:
        try:
            from api.docgen.corroborate import filter_ungrounded_prose

            masked, corrob = filter_ungrounded_prose(masked, grounding)
            result.masked_content = masked
            if corrob.emptied:
                result.warnings.append(
                    "corroborate: filter would empty the section; "
                    "original kept"
                )
            elif corrob.touched:
                result.corroborate_removed = list(corrob.removed_identifiers)
                result.warnings.append(
                    f"corroborate: dropped {corrob.sentences_removed} "
                    "sentence(s) naming ungrounded identifier(s): "
                    f"{', '.join(corrob.removed_identifiers[:10])}"
                )
        except Exception as e:  # pragma: no cover - guard must never break gen
            logger.warning("Corroborate filter failed for section %s: %s", section_id, e)

    citations = extract_citations(masked)
    resolved, unresolved = check_citations(citations, repo_files)
    result.citations_resolved = [c.key() for c in resolved]
    result.citations_unresolved = [c.key() for c in unresolved]
    if unresolved:
        result.warnings.append(
            f"{len(unresolved)} citation(s) not found in the repository"
        )

    file_hashes = compute_file_hashes(repo_dir, source_files)
    result.fingerprint = section_fingerprint(
        section_id, file_hashes, prompt_hash=prompt_hash, model=model,
        base_url=base_url, tree_hash=tree_hash,
    )

    if run_judge:
        evidence = judge_evidence if judge_evidence is not None else ""
        verdict = await judge_section(section_id, masked, evidence, model=model)
        result.judge = verdict
        if verdict.verdict == "skipped":
            result.warnings.append(
                "judge skipped: " + "; ".join(verdict.issues[:3])
            )
        elif verdict.verdict == "inconsistent":
            result.warnings.append(
                "judge flagged inconsistencies: " + "; ".join(verdict.issues[:3])
            )
    return result


def attach_provenance(
    pages: Dict[str, Any],
    section_id: str,
    provenance: Dict[str, Any],
) -> None:
    """Store the provenance payload on the section's page (additive)."""
    page = pages.get(f"page_{section_id}")
    if isinstance(page, dict):
        page["provenance"] = provenance


__all__ = [
    "CHECK_FAIL",
    "CHECK_OK",
    "CHECK_WARN",
    "Citation",
    "JudgeVerdict",
    "RegenPlan",
    "SECRET_MASK",
    "SectionVerification",
    "StructureCheck",
    "attach_provenance",
    "build_section_provenance",
    "check_citations",
    "check_section_structure",
    "compute_file_hashes",
    "diff_sections",
    "extract_citations",
    "get_stored_provenance",
    "hash_file",
    "hash_text",
    "judge_enabled",
    "judge_section",
    "mask_dsn",
    "mask_secrets",
    "normalize_repo_path",
    "plan_regeneration",
    "section_fingerprint",
    "verify_section",
]
