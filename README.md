# RSGA

Reciprocal Space Gated Attention.

This repository contains the core RSGA modules used to introduce a reciprocal-space long-range correction into a short-range local model.

## Files

- `k_frequencies_triclinic.py`
  Builds the triclinic reciprocal-space mode set used by RSGA. It constructs reciprocal vectors from the cell, applies Ewald-like Gaussian damping, prunes modes by cumulative spectral mass, and can return integer reciprocal indices `n` directly so downstream phase encoding can use the invariant identity `r · k(n) = 2π (f · n)`.
- `rsga.py`
  Defines `ReciprocalSpaceGatedAttention`, the main long-range attention block. It takes scalar node channels from a short-range backbone, builds query/key/value projections, applies reciprocal-space phase encoding using wrapped fractional coordinates, accumulates a chunked linear-attention field over reciprocal modes, weights the result with the Ewald spectrum, and returns both the long-range update and a node-wise short-range/long-range mixing gate.

## High-Level Method

At a high level, RSGA works as follows:

1. Start from a short-range message-passing representation produced by a local model.
2. Convert Cartesian coordinates into fractional coordinates for the current periodic cell.
3. Build a triclinic reciprocal grid together with physically motivated spectral weights.
4. Encode reciprocal phases through `2π (f · n)` so the phase basis stays invariant to the specific cell representation.
5. Perform gated linear attention in reciprocal space to aggregate a global long-range field.
6. Mix the long-range update back into the short-range backbone.

The goal is to preserve the strong local inductive bias of a short-range model while providing an explicit periodic long-range correction channel.

## Relation To `GSLab2025/MACE_RSGA`

Using the RSGA methodology, we modified the MACE codebase in [GSLab2025/MACE_RSGA](https://github.com/GSLab2025/MACE_RSGA) to provide a long-range correction on top of the MACE framework.

In that repository:

- `mace/mace/modules/k_frequencies_triclinic.py` and `mace/mace/modules/rsga.py` contain the reciprocal-grid helper and the reciprocal-space gated attention block.
- `mace/mace/modules/models.py` defines the `MACERSGA` model that integrates RSGA into the MACE interaction stack.
- `mace/mace/modules/__init__.py` exports `MACERSGA`.
- `mace/mace/tools/model_script_utils.py` registers the model keyword `MACERSGA`.
- `mace/mace/tools/scripts_utils.py` includes config extraction and optimizer wiring for the `rsga` modules.

## Pointer Example

For a concrete implementation of how a short-range MACE model was modified using this methodology, see [GSLab2025/MACE_RSGA](https://github.com/GSLab2025/MACE_RSGA).

The high-level integration pattern there is:

1. add the reciprocal-space helper modules from this repository into the MACE module tree
2. define a model class that augments the local MACE interaction stack with RSGA blocks
3. export and register that model in the MACE model-construction utilities
4. invoke the modified code through the MACE model keyword `MACERSGA`

Example entry point in the modified MACE stack:

```bash
mace_run_train --model=MACERSGA ...
```

This repository focuses on the reusable RSGA method itself, while `GSLab2025/MACE_RSGA` shows one concrete end-to-end integration into a modified MACE codebase.
