"""Small deterministic inference models; importing this file imports no Torch.

These are new R24 synthetic workloads, not trained accuracy benchmarks or the
frozen R22 DLRM source. Every Linear is bias-free and receives a 2-D input.
"""
from types import SimpleNamespace

MODEL_NAMES = ("recommendation", "branch_mlp", "branch_cnn", "transformer")


def build_workload(torch, name, seed=20261029):
    if name not in MODEL_NAMES:
        raise ValueError("unknown workload: " + name)
    if torch.get_default_dtype() != torch.float32:
        raise ValueError("workloads require the unchanged FP32 default dtype")
    nn = torch.nn

    class Recommendation(nn.Module):
        def __init__(self):
            super().__init__()
            self.embeddings = nn.ModuleList([nn.EmbeddingBag(1024, 16, mode="sum", sparse=True) for _ in range(26)])
            self.bottom = nn.Sequential(nn.Linear(13, 64, bias=False), nn.ReLU(),
                nn.Linear(64, 32, bias=False), nn.ReLU(), nn.Linear(32, 16, bias=False))
            self.top = nn.Sequential(nn.Linear(367, 64, bias=False), nn.ReLU(),
                nn.Linear(64, 32, bias=False), nn.ReLU(), nn.Linear(32, 1, bias=False), nn.Sigmoid())
            triangles = torch.tril_indices(27, 27, offset=-1)
            self.register_buffer("li", triangles[0])
            self.register_buffer("lj", triangles[1])

        def forward(self, dense, offsets, indices):
            x = self.bottom(dense)
            features = [x.unsqueeze(1)]
            features.extend(module(index, offset).unsqueeze(1)
                for module, index, offset in zip(self.embeddings, indices, offsets))
            packed = torch.cat(features, dim=1)
            interactions = torch.bmm(packed, packed.transpose(1, 2))[:, self.li, self.lj]
            return self.top(torch.cat((x, interactions), dim=1))

    class BranchMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.left = nn.Sequential(nn.Linear(256, 128, bias=False), nn.ReLU(),
                nn.Linear(128, 128, bias=False), nn.ReLU())
            self.right = nn.Sequential(nn.Linear(256, 128, bias=False), nn.ReLU(),
                nn.Linear(128, 128, bias=False), nn.ReLU())
            self.head = nn.Linear(128, 16, bias=False)

        def forward(self, x):
            return self.head(self.left(x) + self.right(x))

    class BranchCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.left = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1, bias=False), nn.ReLU(),
                nn.Conv2d(8, 8, 3, padding=1, bias=False), nn.ReLU())
            self.right = nn.Conv2d(3, 8, 1, bias=False)
            self.pool = nn.AvgPool2d(2)
            self.head = nn.Sequential(nn.Linear(4096, 64, bias=False), nn.ReLU(),
                nn.Linear(64, 10, bias=False))

        def forward(self, x):
            merged = torch.cat((self.left(x), self.right(x)), dim=1)
            return self.head(self.pool(merged).flatten(1))

    class Transformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.qkv = nn.Linear(64, 192, bias=False)
            self.projection = nn.Linear(64, 64, bias=False)
            self.ff1 = nn.Linear(64, 128, bias=False)
            self.ff2 = nn.Linear(128, 64, bias=False)
            self.norm1, self.norm2 = nn.LayerNorm(64), nn.LayerNorm(64)

        def forward(self, x):
            batch, length, _ = x.shape
            q, k, v = self.qkv(x.reshape(-1, 64)).reshape(batch, length, 192).chunk(3, dim=-1)
            def heads(value):
                return value.reshape(batch, length, 4, 16).transpose(1, 2).reshape(batch * 4, length, 16)
            q, k, v = heads(q), heads(k), heads(v)
            scores = torch.bmm(q, k.transpose(1, 2)) * 0.25
            attention = torch.bmm(torch.softmax(scores, dim=-1), v)
            attention = attention.reshape(batch, 4, length, 16).transpose(1, 2).reshape(-1, 64)
            x = self.norm1(x + self.projection(attention).reshape(batch, length, 64))
            feedforward = self.ff2(torch.relu(self.ff1(x.reshape(-1, 64)))).reshape(batch, length, 64)
            return self.norm2(x + feedforward)

    # Only the CPU generator is seeded; do not initialize or seed an accelerator.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        if name == "recommendation":
            model = Recommendation()
            inputs = (torch.randn(128, 13), tuple(torch.arange(0, 256, 2, dtype=torch.int64) for _ in range(26)),
                tuple(torch.randint(1024, (256,), dtype=torch.int64) for _ in range(26)))
            core = {"embedding_bag": 26, "mm": 6, "bmm": 1}
        elif name == "branch_mlp":
            model, inputs, core = BranchMLP(), (torch.randn(32, 256),), {"mm": 5}
        elif name == "branch_cnn":
            model, inputs, core = BranchCNN(), (torch.randn(4, 3, 32, 32),), {"conv": 3, "mm": 2}
        else:
            model, inputs = Transformer(), (torch.randn(4, 16, 64),)
            core = {"mm": 4, "bmm": 2, "softmax": 1, "layer_norm": 2}
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    return SimpleNamespace(name=name, model=model, inputs=inputs, required_core=core, seed=seed,
        scope="R24 deterministic synthetic FP32 inference; no training/accuracy claim")
