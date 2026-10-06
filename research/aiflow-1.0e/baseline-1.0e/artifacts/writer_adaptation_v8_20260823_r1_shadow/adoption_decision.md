# v8 adoption decision

- Base execution receipt SHA-256: `59a49af476b43085b2d7ec060165e35b8b7d0064478eccae52f2d406469fbc47`
- Final decision: `AUTO_REJECTED`
- Legacy aggregate improved, but `writer_004` regressed by 3.23%p Top-1 and 12.50%p formula exact.
- Legacy paired glyphs: 8 improved, 1 regressed, 9 changed; exact McNemar p=0.0391. Seven legacy writers are insufficient for commercial generalization and the per-writer safety gate failed.
- Known replay: 5 improved, 2 regressed, 7 changed; exact McNemar p=0.4531. Writer-level regression count is 0, but this is known replay and not promotion evidence.
- The replay dataset was not reopened for this strengthened decision.
- Product checkpoint, HWR, runtime, CROHME, MathWriting and v7 remain unchanged.
