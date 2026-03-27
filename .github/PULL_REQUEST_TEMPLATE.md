<!--
AI: Remove all HTML comments when filling this template. Keep only visible content.

PR Title → squash commit subject (50 chars, imperative)
Format: type(scope): description
Types: feat | fix | docs | refactor | test | chore | perf

Scope: A noun describing a section of the codebase (per conventionalcommits.org).
  ✓ feat(kv-cache): add TQ3 nibble packing
  ✓ fix(quantizer): correct rotation matrix dtype
  ✓ perf(triton): fuse TQ4 decompress into FA kernel
  ✗ feat(042-feature): ...  ← NOT spec/issue numbers (breaks release-please)

Breaking: add ! after scope → feat(kv-cache)!: remove deprecated method
-->
<!--
Why this change? Problem solved? Contrast with previous behavior.
e.g., "Incremental dequantization only decompressed keys. This adds value cache support to cut overhead from 3.36x to 1.78x."
-->

<!--
What changed? 2-4 bullets, imperative mood.
e.g., - Add incremental value dequantization in CompressedDynamicCache
      - Update benchmark harness to report per-step overhead
-->
-

<!-- How to verify: command, manual steps, or "CI only" -->
Test: `uv run pytest -v`

<!--
Git trailers (one per line):
  Closes #123
  BREAKING CHANGE: remove deprecated foo() method
  Co-authored-by: Name <email>
-->
Closes #

<!--
═══════════════════════════════════════════════════════════════════════════════
MULTI-COMMIT PRs (release-please)
═══════════════════════════════════════════════════════════════════════════════
When a PR contains multiple logical changes that would normally be separate
commits, add additional conventional commit blocks as FOOTERS at the bottom
of the body (above this checklist section). Release-please parses these to
generate proper changelog entries.

Format: blank line, then type(scope): description, then details

Example PR body structure:
─────────────────────────────────────────────────────────────────────────────
Primary change description (associated with PR title).

- Bullet points for primary change

Test: `uv run pytest -v`

Closes #123

feat(triton)!: replace standalone Q@K^T kernel with full FA fusion

BREAKING CHANGE: `fused_qk_attention` module removed

docs(architecture): update module DAG for triton restructure

- Remove fused_qk_attention from dependency graph
─────────────────────────────────────────────────────────────────────────────

The PR title becomes the first changelog entry. Each footer block (starting
with a conventional commit type) becomes an additional entry.

Ref: https://github.com/googleapis/release-please#what-if-my-pr-contains-multiple-fixes-or-features
-->

---

## PR Review

### Checklist
- [ ] Self-reviewed my code
- [ ] Tests pass (`uv run pytest`)
- [ ] Lint passes (`uv run ruff check .`)
- [ ] Breaking changes use `!` in title and `BREAKING CHANGE:` in body

### Review Focus
<!-- Where should reviewers concentrate? Known limitations? -->

### Related
<!-- Other PRs, issues, ADRs for context -->
