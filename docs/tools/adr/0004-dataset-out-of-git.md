# ADR-TOOLS-0004: Dataset stays out of git; Zenodo fetch, not Git LFS

**Status:** Accepted
**Date:** 2026-08-26
**Topic:** tooling
**Supersedes:** none
**Superseded-by:** none
**Related:** ADR-TOOLS-0001

## Context

Mommert industrial smoke plumes live on Zenodo record 4250706 (~6.4 GB, tens of
thousands of GeoTIFFs). GitHub LFS free quota is 10 GiB storage and 10 GiB
bandwidth per month. Calibration binaries already stay out of band.

## Decision

Do not store the training corpus in git or Git LFS. Ignore `data/raw/` and
`data/processed/`. Track a checksummed manifest and a fetch script that downloads
Zenodo 4250706. Tiny golden tensors for the acceptance gate may live in git under
`packages/tools/tests/fixtures/`. Default CI does not download the corpus.

## Consequences

- Clones stay small.
- Collaborators run the fetch script once for local training.
- Nightly or manual jobs may use a subset flag.

## Alternatives considered

- Git LFS — quota risk and 21k-file anti-pattern; duplicates a DOI archive.
- GitHub Releases — 2 GiB file cap; no DOI.
- git-annex — extra tooling with no GitHub native support.
