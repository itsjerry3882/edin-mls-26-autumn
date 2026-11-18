"""Generate answer candidates using the audio-to-answer baseline."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import argparse
import json
from src.asr_adapter import ASRAdapter
from src.context_builder import SimpleContextBuilder
from src.data import CourseData
from src.encoder import LocalEncoder
from src.generator import ELMGenerator
from src.rag import RAGPipeline
from src.retriever import ExactDenseRetriever
from contextlib import nullcontext

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", choices=["practice", "dev"], required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=None, help="Diagnostic subset only")
    ap.add_argument("--experiment", action="store_true", help="Use budgeted excerpt context selection from the top 20 candidates")
    ap.add_argument("--baseline-candidates", type=Path, default=None, help="Existing baseline JSONL used to reuse ASR transcripts")
    ap.add_argument("--diagnostics-output", type=Path, default=None, help="Optional JSONL file for context diagnostics")
    args = ap.parse_args()
    if args.k < 1 or (args.limit is not None and args.limit < 1):
        ap.error("--k and --limit must be positive")
    if args.output.exists():
        ap.error("Output exists; choose a new filename to preserve prior results")
    if args.experiment and args.k != 5:
        ap.error(
            "This experimental builder uses a fixed top-5 baseline. "
            "Use --k 5."
        )
    if args.experiment and args.baseline_candidates is None:
        ap.error(
            "--experiment requires --baseline-candidates "
            "to guarantee matching transcripts"
        )
    if args.baseline_candidates is not None:
        if not args.baseline_candidates.is_file():
            ap.error("Baseline candidates file does not exist")

    if args.diagnostics_output is not None:
        if args.diagnostics_output == args.output:
            ap.error("Diagnostics and candidates must use different paths")

        if args.diagnostics_output.exists():
            ap.error("Diagnostics output already exists")
    baseline_transcripts = {}

    if args.baseline_candidates is not None:
        with args.baseline_candidates.open(
            "r", encoding="utf-8"
        ) as f:
    
            for line_number, line in enumerate(f, start=1):
                if not line.strip():
                    continue

                row = json.loads(line)

                query_id = row["query_id"]
                transcript = row["transcript"]

                if row.get("split") != args.split:
                    raise ValueError(
                        f"Incorrect split at line {line_number}: "
                        f"{row.get('split')}"
                    )

                if query_id in baseline_transcripts:
                    raise ValueError(
                        f"Duplicate baseline query ID: {query_id}"
                    )

                if not isinstance(transcript, str):
                    raise ValueError(
                        f"Invalid transcript for {query_id}"
                    )

                baseline_transcripts[query_id] = transcript

        print(
            f"Loaded {len(baseline_transcripts)} "
            "baseline transcripts"
        )
    
    asr = ASRAdapter()
    asr.prepare()
    data_dir = ROOT / "data/rag_corpus"
    data = CourseData(data_dir)
    pipeline = RAGPipeline(asr, LocalEncoder(ROOT / "models/enc_a", device=args.device),
        ExactDenseRetriever(data_dir / "embeddings/corpus_enc_a.npy", data_dir / "embeddings/corpus_enc_a_ids.json"),
        data, SimpleContextBuilder(), ELMGenerator())
    items = data.audio_items(args.split)
    if args.limit:
        items = items[:args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_file = nullcontext(None)

    if args.diagnostics_output is not None:
        args.diagnostics_output.parent.mkdir(
            parents=True, exist_ok=True
        )

        diagnostics_file = args.diagnostics_output.open(
            "x", encoding="utf-8"
        )
    with (args.output.open("x", encoding="utf-8") as f, diagnostics_file as diag_f):
        for item in items:
            query_id = item["query_id"]

            if args.baseline_candidates is not None:
                if query_id not in baseline_transcripts:
                    raise ValueError(
                        f"Missing baseline transcript for {query_id}"
                    )

                transcript_override = baseline_transcripts[query_id]
            else:
                transcript_override = None

            result = pipeline.answer_audio(
                data_dir / item["audio_file"],
                k=args.k,
                experiment=args.experiment,
                transcript_override=transcript_override,
            )
            row = {"query_id": item["query_id"], "split": args.split, "audio_file": item["audio_file"],
                   "transcript": result["transcript"], "answer": result["answer"],
                   "retrieved_record_ids": result["retrieval"]["record_ids"], "timing": result["timing"]}
            if diag_f is not None:
                diagnostic_row = {
                    "query_id": item["query_id"],
                    "transcript": result["transcript"],
                    "retrieved_record_ids": (
                        result["retrieval"]["record_ids"]
                    ),
                    "context_stats": result.get("context_stats"),
                    "prompt": result["prompt"],
                    "answer": result["answer"],
                }

                diag_f.write(
                    json.dumps(diagnostic_row, ensure_ascii=False)
                    + "\n"
                )
                diag_f.flush()
            if args.experiment:
                row["context_stats"] = result["context_stats"]
                row["context_record_ids"] = (
                    result["context_stats"]["selected_record_ids"]
                )
                row["experiment"] = "budgeted_excerpt_top20_keep3"
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            print(item["query_id"])

if __name__ == "__main__":
    main()
