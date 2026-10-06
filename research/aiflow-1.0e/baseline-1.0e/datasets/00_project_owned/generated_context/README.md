# Project-owned generated candidate-validity contexts

Last verified: 2026-08-20

`candidate_validity_coverage_v1.jsonl.gz` contains 12,000 generated formula
contexts for 15 sparse homograph tokens: `0/O/o/\mathcal{O}/\circ`,
`1/|/l//\mathbb{1}`, and `x/\times/X/\mathcal{X}/\chi`.

- 800 unique contexts per token
- frozen 372-token vocabulary only
- exact direct/CROHME evaluation-sequence overlap: 0
- SHA-256: `69b38d30e4a432fa2fb41ddd4ef2ececcab888992c378ef7904f9dea35f4f79c`
- audit: `candidate_validity_coverage_v1_audit.json`

These are synthetic token-context examples, not handwriting samples. They may
diagnose and train candidate validity, but they do not prove transfer to unseen
human formulae. The r2/r3 experiments showed that mixing them into direct
fine-tuning reduced writer-OOF accuracy, so they are not part of the selected
runtime checkpoint.

