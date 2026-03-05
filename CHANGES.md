# Changes

## 2026-03-05
- Overrode `sh_degree=0`, `densify_grad_thresh=0.0002`, `use_absgrad=False` in `SeaSplatfactoModelConfig` to match SeaSplat's original 3DGS optimization defaults
- Fixed critical bugs in `losses.py`: `torch.Tensor([0, 0])` → `[0.0]` (root cause of shape-[2] tensor issue), wrong `vector_norm` dim, tuple-as-single-arg to MSE, wrong intensity loss formula
- Fixed `SmoothDepthLoss` missing edge-aware weighting, `DarkChannelPriorLossV3` wrong beta, `AlphaBackgroundLoss` missing `self.mse` + naming inconsistency
- Renamed abbreviated/ambiguous variables in `sea_splatfacto_model.py` (config fields, state vars, locals) and consolidated redundant bs/at counter pairs into single `medium_*` set


- Organized `reference/seasplat/` directory: deleted 22 superfluous files (infrastructure, data loading, metrics, etc. handled by nerfstudio), keeping only core SeaSplat method files
- Created `reference/seasplat/notes.md` with paper-to-code mapping, training flow, loss inventory, and key config defaults
