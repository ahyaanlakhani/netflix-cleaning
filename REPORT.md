# Hit Rate@10, hidden final week
Cutoff date: **2026-11-10** (training = events before it; hidden week = 10-16 Nov 2026). Scored members: **266**.
A hit = the member's newest hidden title is in their top 10 picks. Picks (model and popular) exclude every title the member played in training, so rewatch targets cannot hit.

| | Hit Rate@10 |
|---|---|
| Two-tower model (PyTorch, 64-d towers, 5 rounds) | **47.4%** (126/266) |
| Popular top 10 (most-played titles the member has not played) | 15.4% (41/266) |

Reported round: 5 (final). Per round on the hidden week: r1 46.2%, r2 45.1%, r3 45.9%, r4 44.4%, r5 47.4%

Title tower input = catalog description (TF-IDF) + type/genre/maturity, so zero-play titles get vectors: TT-0008, TT-0060, TT-0097, TT-0117, TT-0119, TT-0143, TT-0170, TT-0194, TT-0195, TT-0239, TT-0262, TT-0288, TT-0300
(TT-0300 vector norm 1.00; appears in 0 members' top 10).
Fixed hyperparameters, not tuned on the hidden week; a 3-9 Nov validation week gave model 44.4% vs popular 15.4%.
W&B run: https://forge.coreweave.com/wandb/ahyaanlakhani/netflix-next-title/runs/rx93q5z1
Files: model/twotower.pt (weights), model/twotower_vectors.npz (member_vectors, title_vectors, ids), model/title_features.npy.
