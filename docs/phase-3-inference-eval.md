# Phase 3 inference eval: recorded results (issue #32)

Test: `tests/test_memory_inference_eval.py::test_the_real_proposer_neither_over_extracts_nor_hallucinates`
(opt-in, `-m eval`; needs provider credentials). Provider at the time: Azure (`LLM_PROVIDER=azure`), temperature 0.
Five cases (three where nothing durable is said, one stated organisation fact, one stated threshold), three proposer
calls per case per test run. Properties only: no proposal where nothing durable was said, every quote found in the
allowed source text, every number in a proposal present in the input, preference keys inside the vocabulary,
caps respected, plainly stated facts and thresholds not missed.

| Run | Prompt | Result |
| --- | --- | --- |
| 1 | first draft | 2 failed: over-extraction. The model restated the run records (figures from the previous run) as a fact. |
| 2 | told not to restate run records | 5 passed |
| 3 | same | 1 failed: over-extraction on the one-off request ("just for this report, keep it short"). |
| 4-6 | added "an instruction about this one report is not a fact" | 5 passed on each of three consecutive runs |

The prompt in `analyst_crew/memory/infer.py` is the one from runs 4-6. Model output is not deterministic, so a
single green run is not a guarantee; the code-side grounding checks (quoted evidence must appear in the user's words,
numbers must appear in the evidence, vocabulary and range validation, per-run caps) hold regardless of the model.
