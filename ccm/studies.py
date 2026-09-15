"""Explicit protocol versions; historical pilot defaults remain unchanged."""
from .contracts import Budget, PILOT, require

PILOT_STUDY = "pilot12"
FOLLOWUP = "pilot12-shallow-followup"
SCALEUP = "scaleup28-12b-v3"
STUDIES = (PILOT_STUDY, FOLLOWUP, SCALEUP)
SCALEUP_BUDGET = Budget(common_steps=38147, continue_steps=7629)
SCALEUP_STAGE1 = ("shallow", "isolated", "contextual", "shuffled", "grad")
SCALEUP_STAGE2 = ("base", "contextual", "isolated", "shuffled", "shallow", "grad")


def study_of(corpus):
    return corpus.meta.get("study", PILOT_STUDY)


def validate_training_study(args, corpus):
    study = getattr(args, "study", PILOT_STUDY)
    require(study in STUDIES, "Unknown study")
    if study == SCALEUP:
        require(study_of(corpus) == SCALEUP, "28L requires explicit extended manifests and fresh holdout")
        require(args.arm != "delta", "Depth Delta is excluded from the 28L study")
        require(args.engineering or corpus.budget == SCALEUP_BUDGET, "Wrong 28L budget")
    else:
        require(study_of(corpus) == PILOT_STUDY, "Historical study cannot use 28L data")
        require(args.engineering or corpus.budget == PILOT, "Wrong historical budget")
        if study == FOLLOWUP:
            require(args.phase == "stage2" and args.arm == "shallow", "Follow-up is Stage-2 Shallow only")
        else:
            require(args.phase != "stage2" or args.arm != "shallow", "Shallow is Stage-1 diagnostic only in pilot12")
    return study


def require_vocabulary(corpus, vocab):
    expected = corpus.meta.get("historical_manifest_hash", corpus.meta["manifest_hash"])
    require(vocab.metadata["corpus_hash"] == expected, "Vocabulary corpus mismatch")
    if study_of(corpus) == SCALEUP:
        require(len(vocab.keys) == corpus.budget.slots, "Wrong scale-up memory capacity")
        require(vocab.hash == corpus.meta["historical_vocabulary_hash"], "28L must reuse exact historical vocabulary")
        require(vocab.mapping_hash == corpus.meta["ordered_mapping_hash"], "28L ordered key/slot hash differs")


def open_corpus(path, verify=True):
    from pathlib import Path
    from .contracts import read_json
    from .data import Corpus
    if read_json(Path(path)/"manifest.json").get("study") == SCALEUP:
        from .scaleup_data import ScaleupCorpus
        corpus = ScaleupCorpus(path, verify=verify)
        done = read_json(Path(path)/"complete.json")
        require(done.get("success") is True and done.get("manifest_hash") == corpus.meta["manifest_hash"],
                "Scale-up preparation has not completed validation")
        return corpus
    return Corpus(path, verify=verify)
