# Personal Changelog

A running record of **my personal work on this fork** of
[`google-deepmind/simply`](https://github.com/google-deepmind/simply). Some of
this work is local-only and may never be pushed upstream, so this file is the
source of truth for what I have changed and why.

The format is loosely based on [Keep a Changelog](https://keepachangelog.com/).
Entries are grouped by date, newest first. For deep experiment narratives (env
details, GPU runbooks, findings) see [`experiments_note.md`](experiments_note.md);
this file is the concise "what changed" index.

Change types: **Added**, **Changed**, **Fixed**, **Removed**, **Infra** (local
environment / tooling, not committed to the repo), **Docs**.

## [Unreleased]

### Added
- Personal `CHANGELOG.md` to track fork work (incl. local-only changes).
  (`138cb96`)

### Changed
- Synced fork with upstream: merged `upstream/main` (reworked ragged paged
  attention — KV write-back clamp removed, host-RAM prefix cache
  PrefixNode/ChunkTile/ChunkHolder, new page batcher, much expanded serving
  tests; plus dependency bumps — aiohttp, cryptography, keras, gitpython,
  urllib3, pytest, pillow). Clean merge, no conflicts; `uv lock --check` passed.
  (merge `a17ab86`)
- Synced fork with upstream: merged `upstream/main` (Bash-tool disabled-timeout
  fix + dependency bumps — werkzeug, pyasn1, requests, pygments). Clean merge,
  no conflicts; `uv lock --check` passed. (merge `ddf895a`)

### Infra
- Set up local **amplio** agent harness (under `amplio/`): installed Go 1.26.2 +
  Node 22 toolchains to `~/toolchains/`, built the binary, and pointed it at the
  GitHub Copilot OpenAI-compatible endpoint (`~/.amplio/config.toml`; secrets via
  env vars). Exposed it remotely via a Cloudflare quick tunnel. *(Local only — not
  committed.)*

## 2026-08-17

### Changed
- Synced fork with upstream: merged `upstream/main` (up to v0.3.9 — KV prefix
  cache, quantization, ARC-AGI-2 / SWE-bench Pro data) into `main`. Resolved
  conflicts in `.gitignore` (kept both Jupyter + gRPC-stub ignores) and
  regenerated `uv.lock`. (merge `d03011b`)

## 2026-06-18

### Added
- 2026-06-18 experiment log: GPU upgrade runbook + contribution survey; recorded
  GPU setup completion and test results. (`b75ddc2`, `f6a9c8f`)

### Changed
- Corrected experiment-log dates to local timezone (UTC-7). (`dba80b8`)

### Removed
- Dropped the post-reboot checklist from `experiments_note.md`. (`6dd55ba`,
  `de5af82`)

## 2026-06-14

### Added
- 2026-06-14 experiment logs: upstream rebase, onboarding notes, PR #29; local
  env build + GPU driver upgrade runbook. (`45a2cb9`, `61ba68f`)
- Personal codebase onboarding notes (`codebase_notes.md`). (`ef8109a`)
- Scratch notebook for codebase experimentation. (`e0be05b`)
- `[agent]` extra dependencies to `uv.lock`. (`7f6d643`)

### Changed
- Ignore Jupyter checkpoint directories (`.gitignore`). (`cd37f7f`)

## 2026-06-12

### Added
- Experiment notes for the 2026-04-16 setup + GPU debugging session. (`4c8ee1d`)
- Date/step index with anchor navigation in `experiments_note.md`. (`5eb561f`)
- `flops2e16_tfm15m_imdb` config + experiment-note updates. (`1d5bd8f`)
- 2026-06-12 fork-sync log; reordered experiment notes newest-first. (`862f08d`)

### Fixed
- Removed duplicate `tensorflow-datasets` entry in `uv.lock`. (`8264e31`)

### Removed
- Empty 2026-04-17 experiment log entry. (`feba36c`)
