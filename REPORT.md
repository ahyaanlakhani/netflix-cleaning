# Hit Rate@10, hidden final week
Cutoff date: **2026-11-10** (training = events before it; hidden week = 10-16 Nov 2026). Scored members: **266**.
A hit = the member's newest hidden title is in their top 10 picks.

| | Hit Rate@10 |
|---|---|
| Two-tower model (PyTorch, 64-d towers, 5 rounds) | **53.4%** (142/266) |
| Popular top 10 (same list for everyone) | 21.4% (57/266) |

Reported round: 5 (final). Per round on the hidden week: r1 56.8%, r2 51.5%, r3 53.4%, r4 53.0%, r5 53.4%

Title tower input = catalog description (TF-IDF) + type/genre/maturity, so zero-play titles get vectors: TT-0008, TT-0060, TT-0097, TT-0117, TT-0119, TT-0143, TT-0170, TT-0194, TT-0195, TT-0239, TT-0262, TT-0288, TT-0300
(TT-0300 vector norm 1.00; appears in 0 members' top 10).
Fixed hyperparameters, not tuned on the hidden week; a 3-9 Nov validation week gave model 52.5% vs popular 20.1%.
W&B run: https://forge.coreweave.com/wandb/ahyaanlakhani/netflix-next-title/runs/dtoxiivl
Files: model/twotower.pt (weights), model/twotower_vectors.npz (member_vectors, title_vectors, ids), model/title_features.npy.
