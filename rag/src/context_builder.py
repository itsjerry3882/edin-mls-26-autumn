from __future__ import annotations
import re
from time import perf_counter

BUDGET_UNIT = "words"
STOP_WORDS = set(
    "a an and are as at be by can could did do does for from how in is "
    "it of on or that the this to was were what when where which who "
    "why will with would".split()
)


def evidence_size(text: str) -> int:
    return len(text.split())


def render_evidence(texts: list[str]) -> str:
    # Matches SimpleContextBuilder's formatting.
    return "\n\n".join(
        f"[Document {i}]\n{text}"
        for i, text in enumerate(texts, start=1)
    )


def content_words(text: str) -> set[str]:
    return set(re.findall(r"\b\w+\b", text.casefold())) - STOP_WORDS


def excerpt_options(text: str, query_words: set[str]) -> list[str]:
    sentences = [
        s for s in re.split(r"(?<=[.!?])\s+|\n+", text.strip()) if s
    ]
    if not sentences:
        return []

    scores = [len(query_words & content_words(s)) for s in sentences]
    best = max(range(len(sentences)), key=scores.__getitem__)
    if scores[best] == 0:
        return []

    window = " ".join(sentences[max(0, best - 1):best + 2])
    return list(dict.fromkeys([window, sentences[best]]))


class SimpleContextBuilder:
    """Intentionally simple baseline; advanced context strategies are left open."""

    def build(self, transcript: str, record_ids: list[str], data) -> dict:
        contexts = [data.record_text(rid) for rid in record_ids]

        evidence = "\n\n".join(
            f"[Document {i}]\n{text}"
            for i, text in enumerate(contexts, start=1)
        )

        prompt = (
            "Answer the question using the retrieved evidence below. "
            "If the evidence is insufficient, say so.\n\n"
            f"Question:\n{transcript}\n\n"
            f"Evidence:\n{evidence}"
        )

        return {
            "contexts": contexts,
            "prompt": prompt,
        }


class BudgetedExcerptContextBuilder:
    def build(self, transcript: str, record_ids: list[str], data) -> dict:
        started = perf_counter()
        candidates = record_ids[:20]
        baseline_ids = candidates[:5]
        baseline = SimpleContextBuilder().build(transcript, baseline_ids, data)
        baseline_evidence = render_evidence(baseline["contexts"])
        budget = evidence_size(baseline_evidence)

        selected_ids = baseline_ids
        selected_ranks = list(range(1, len(baseline_ids) + 1))
        result = baseline

        if candidates:
            n_full = min(3, len(baseline_ids))

            texts = baseline["contexts"][:n_full]
            ids = candidates[:n_full]
            ranks = list(range(1, n_full + 1))
            query_words = content_words(transcript)

            for rank, rid in enumerate(candidates[n_full:], start=n_full + 1):
                # Reuse text already fetched by the baseline builder.
                text = (
                    baseline["contexts"][rank - 1]
                    if rank <= len(baseline_ids)
                    else data.record_text(rid)
                )
                for excerpt in excerpt_options(text, query_words):
                    trial = render_evidence(texts + [excerpt])
                    if evidence_size(trial) <= budget:
                        texts.append(excerpt)
                        ids.append(rid)
                        ranks.append(rank)
                        break

            # If nothing from below rank 5 fits, keep the baseline prompt.
            if any(rank > 5 for rank in ranks):
                result = {
                    "contexts": texts,
                    "prompt": (
                        "Answer the question using the retrieved evidence below. "
                        "If the evidence is insufficient, say so.\n\n"
                        f"Question:\n{transcript}\n\n"
                        f"Evidence:\n{render_evidence(texts)}"
                    ),
                }
                selected_ids, selected_ranks = ids, ranks

        return {
            **result,
            "context_stats": {
                "budget_unit": BUDGET_UNIT,
                "baseline_evidence_size": budget,
                "used_evidence_size": evidence_size(
                    render_evidence(result["contexts"])
                ),
                "selected_record_ids": selected_ids,
                "selected_ranks": selected_ranks,
                "used_deeper_evidence": any(r > 5 for r in selected_ranks),
                "build_ms": 1000 * (perf_counter() - started),
            },
        }
