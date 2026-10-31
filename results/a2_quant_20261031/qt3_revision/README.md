# QT3 revised audit evidence

Current result: [summary.json](summary.json). Both historical audit attempts remain available locally in `attempt01/` and `attempt02/`; [evidence_index.json](evidence_index.json) binds their exact bytes. Attempt01 GPTQ receipts pass. Its RTN flags came from CPU direct division for scale; attempt02 reconstructs the original CUDA scalar-reciprocal arithmetic and passes. Both are CPU-only reads of saved weights.

All 56 selected per-model receipts are also retained in this directory. Their SHA locks and original B200 paths are in the summary. Large per-matrix files and pre-revision report backups are excluded from Git/runner source sync, not deleted. Original quantization results remain under `../artifacts/` and `../batches/b01/`.

[Current validity for all 160 NLL files](../validation_revision.json) supersedes the original artifacts' historical unverified flags. QT6 remains a documented overlap, accepted by the user as a minor deviation. Scientific measurements and calibration-sensitivity flags are unchanged.
