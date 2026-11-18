from __future__ import annotations

import time


class RAGPipeline:
    def __init__(
        self,
        asr,
        encoder,
        retriever,
        data,
        context_builder,
        generator,
    ):
        self.asr = asr
        self.encoder = encoder
        self.retriever = retriever
        self.data = data
        self.context_builder = context_builder
        self.generator = generator

    def answer_audio(self, audio_path, k: int = 5, experiment: bool = False, transcript_override: str | None = None):
        total_start = time.perf_counter()

    # 1. ASR
        t0 = time.perf_counter()
        if transcript_override is not None:
            transcript = transcript_override
            asr_ms = 0.0
        else:
            transcript = self.asr.transcribe(audio_path)
            asr_ms = (time.perf_counter() - t0) * 1000.0

    # 2. Query encoding
        t0 = time.perf_counter()
        query_vector = self.encoder.encode(transcript)
        embedding_ms = (time.perf_counter() - t0) * 1000.0

    # 3. Retrieval
    # Baseline: retrieve top k (normally 5)
    # Experiment: retrieve top 20 candidates
        retrieval_k = 20 if experiment else k

        retrieval = self.retriever.retrieve(
            query_vector,
            k=retrieval_k,
        )

    # 4. Context construction
        if experiment:
            from .context_builder import BudgetedExcerptContextBuilder
            builder = BudgetedExcerptContextBuilder()
        else:
            builder = self.context_builder

        built = builder.build(
            transcript,
            retrieval["record_ids"],
            self.data,
        )

    # 5. Answer generation
        generation = self.generator.generate(built["prompt"])

        total_ms = (time.perf_counter() - total_start) * 1000.0

        return {
            "transcript": transcript,
            "retrieval": retrieval,
            "contexts": built["contexts"],
            "answer": generation["answer"],
            "context_stats": built.get("context_stats"),
            "timing": {
                "asr_ms": asr_ms,
                "embedding_ms": embedding_ms,
                "retrieval_ms": retrieval["latency_ms"],
                "generation_ms": generation["latency_ms"],
                "end_to_end_ms": total_ms,
            },
            "prompt": built["prompt"]
        }
