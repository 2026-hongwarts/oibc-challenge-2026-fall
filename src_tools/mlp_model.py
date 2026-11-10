"""MLP 보조 모델 (PyTorch, GPU) — train_site_model.py 의 --model mlp.

근거(11/10 조사, reference/외부_대회_조사.md D1): 같은 수치예보 표 데이터에서 신경망은 GBM 과 비슷하거나 조금 못하지만,
다른 계열이라 앙상블에 작은 비중(1/6)으로 넣으면 0 ~ −2% 기대. TabReD(시간순 분할)에서 MLP 는 시드 평균일 때 GBM 급.
레시피(조사 권장 최소판):
  - 입력: 같은 피처 표. 분위수 변환(정규 분포, **학습 행으로만 맞춤**) → 빈칸은 0(변환 뒤 중앙값) + 빈칸 표시 열.
  - 출력: sigmoid × 1.2 (발전율 0 ~ 1.2). 학습 행은 이미 태양 고도 > 0 만.
  - 손실: L1(가중치가 있으면 가중 L1). 구조: 512-512-512, SiLU, 드롭아웃 0.1.
  - AdamW(학습률 2e-3, 가중 감쇠 1e-5), 배치 4096, OneCycle, 에폭 고정(조기 종료 없음 → 검증 해를 보지 않음).
  - 시드: train_site_model 의 seed 를 그대로(torch·numpy). 결과는 GPU 비결정성 때문에 실행마다 아주 조금 다를 수 있다.
"""
from __future__ import annotations

import numpy as np

MLP_PARAMS = dict(hidden=512, layers=3, dropout=0.1, lr=2e-3, weight_decay=1e-5, batch=4096, epochs=20,
                  n_quantiles=1000, device="cuda", emb_dim=16)
# 범주형 열(--cat site_id 등, pandas category)은 분위수 변환 대신 임베딩(emb_dim 차원)으로 넣는다(조사 권장: 발전소 임베딩).


class MLPRegressor:
    def __init__(self, seed: int = 0, objective: str = "l1", **params):
        self.p = {**MLP_PARAMS, **params}
        self.seed = seed
        self.objective = objective

    def _split(self, X):
        cats = [c for c in X.columns if str(X[c].dtype) == "category"]
        C = np.stack([X[c].cat.codes.to_numpy().astype("int64") + 1 for c in cats], axis=1) if cats else None  # 0 = 빈칸
        return X.drop(columns=cats), C, [len(X[c].cat.categories) + 1 for c in cats]

    def _prep(self, X, fit: bool):
        from sklearn.preprocessing import QuantileTransformer
        A = X.to_numpy(dtype="float32", copy=True)
        if fit:
            self.nan_cols = np.flatnonzero(np.isnan(A).any(axis=0))
            self.qt = QuantileTransformer(n_quantiles=self.p["n_quantiles"], output_distribution="normal",
                                          subsample=200_000, random_state=self.seed)
            self.qt.fit(A)
        Z = self.qt.transform(A).astype("float32")
        miss = np.isnan(A[:, self.nan_cols]).astype("float32")
        Z = np.nan_to_num(Z, nan=0.0)
        return np.concatenate([Z, miss], axis=1)

    def fit(self, X, y, sample_weight=None):
        import torch
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        dev = torch.device(self.p["device"] if torch.cuda.is_available() else "cpu")
        Xn, C, sizes = self._split(X)
        Z = torch.from_numpy(self._prep(Xn, True)).to(dev)
        Ct = None if C is None else torch.from_numpy(C).to(dev)
        t = torch.from_numpy(np.asarray(y, dtype="float32")).to(dev)
        w = None if sample_weight is None else torch.from_numpy(np.asarray(sample_weight, dtype="float32")).to(dev)
        e = self.p["emb_dim"]
        self.embs = torch.nn.ModuleList([torch.nn.Embedding(n, e) for n in sizes]).to(dev)
        d, h = Z.shape[1] + e * len(sizes), self.p["hidden"]
        mods = []
        for i in range(self.p["layers"]):
            mods += [torch.nn.Linear(d if i == 0 else h, h), torch.nn.SiLU(), torch.nn.Dropout(self.p["dropout"])]
        mods += [torch.nn.Linear(h, 1)]
        self.net = torch.nn.Sequential(*mods).to(dev)
        opt = torch.optim.AdamW(list(self.net.parameters()) + list(self.embs.parameters()),
                                lr=self.p["lr"], weight_decay=self.p["weight_decay"])
        n, bs, ep = len(Z), self.p["batch"], self.p["epochs"]
        steps = (n + bs - 1) // bs
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=self.p["lr"], total_steps=steps * ep)
        g = torch.Generator(device=dev).manual_seed(self.seed)
        self.net.train()
        for _ in range(ep):
            perm = torch.randperm(n, device=dev, generator=g)
            for k in range(steps):
                idx = perm[k * bs:(k + 1) * bs]
                out = torch.sigmoid(self.net(self._inp(Z[idx], None if Ct is None else Ct[idx])).squeeze(1)) * 1.2
                err = (out - t[idx]).abs() if self.objective == "l1" else (out - t[idx]) ** 2
                loss = err.mean() if w is None else (err * w[idx]).sum() / w[idx].sum()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()
        if Ct is not None:                 # 학습에 없던 범주(처음 보는 발전소)는 학습된 임베딩의 평균으로 — 무작위 초깃값 방지
            with torch.no_grad():
                for i, emb in enumerate(self.embs):
                    seen = torch.zeros(emb.num_embeddings, dtype=torch.bool, device=dev)
                    seen[Ct[:, i].unique()] = True
                    if seen.any() and (~seen).any():
                        emb.weight[~seen] = emb.weight[seen].mean(0)
        self.dev = dev
        del Z, t, w, Ct
        torch.cuda.empty_cache()
        return self

    def _inp(self, z, c):
        import torch
        if c is None or not len(self.embs):
            return z
        return torch.cat([z] + [emb(c[:, i]) for i, emb in enumerate(self.embs)], dim=1)

    def predict(self, X):
        import torch
        Xn, C, _ = self._split(X)
        Z = self._prep(Xn, False)
        self.net.eval()
        out = []
        with torch.no_grad():
            for k in range(0, len(Z), 65536):
                zb = torch.from_numpy(Z[k:k + 65536]).to(self.dev)
                cb = None if C is None else torch.from_numpy(C[k:k + 65536]).to(self.dev)
                out.append((torch.sigmoid(self.net(self._inp(zb, cb)).squeeze(1)) * 1.2).float().cpu().numpy())
        return np.concatenate(out)
