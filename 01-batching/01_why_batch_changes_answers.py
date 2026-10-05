"""
Why the same question can get a different answer depending on batch size.
Runs locally with numpy only. No API key needed.

Three steps:
  1. Floating-point math isn't associative: the order you add numbers changes the result.
  2. A kernel that picks its reduction strategy by batch size gives the same row
     different numbers in different batches (the Thinking Machines finding).
  3. Those tiny differences only flip the model's answer when the top choices are
     close, i.e. when the prompt is ambiguous.
"""
import numpy as np

rng = np.random.default_rng(0)

# --- 1. Order matters --------------------------------------------------------
print("1) Floating point: (a + b) + c vs a + (b + c)")
a, b, c = np.float32(0.1), np.float32(1e8), np.float32(-1e8)
print(f"   (0.1 + 1e8) - 1e8 = {(a + b) + c}")
print(f"   0.1 + (1e8 - 1e8) = {a + (b + c)}")

x = rng.standard_normal(4096).astype(np.float32)
s_forward = np.float32(0)
for v in x:
    s_forward += v
s_backward = np.float32(0)
for v in x[::-1]:
    s_backward += v
print(f"   same 4096 numbers, summed forwards vs backwards differ by {abs(s_forward - s_backward):.2e}\n")

# --- 2. A batch-dependent kernel ---------------------------------------------
# Real GPU kernels split the work differently depending on how many requests
# share the batch (e.g. split-K matmul when the batch is small). We imitate that:
# small batches split the dot product into chunks and add the partial sums.
def dot_rows(batch, w):
    n = batch.shape[0]
    chunks = 1 if n >= 64 else 8          # strategy depends on batch size
    out = np.zeros((n, w.shape[1]), dtype=np.float32)
    for part in np.array_split(np.arange(batch.shape[1]), chunks):
        out += batch[:, part] @ w[part]   # partial sums added in a different order
    return out

hidden, vocab = 4096, 1000
w = (rng.standard_normal((hidden, vocab)) / np.sqrt(hidden)).astype(np.float32)
my_query = rng.standard_normal((1, hidden)).astype(np.float32)
others = rng.standard_normal((127, hidden)).astype(np.float32)

alone = dot_rows(my_query, w)[0]                                # quiet server
busy = dot_rows(np.vstack([my_query, others]), w)[0]            # busy server
print("2) Same query, quiet server (batch of 1) vs busy server (batch of 128)")
print(f"   max difference in the model's scores: {np.abs(alone - busy).max():.2e}")
print(f"   identical? {np.array_equal(alone, busy)}\n")

# --- 3. When does that tiny difference change the answer? --------------------
# Greedy decoding (temperature 0) picks the highest-scoring next token.
# Add noise the size of step 2 and count how often the pick flips.
noise = float(np.abs(alone - busy).max())
def flip_rate(margin, trials=20000):
    flips = 0
    for _ in range(trials):
        top1, top2 = 0.0, -margin                  # two best candidate tokens
        e1, e2 = rng.uniform(-noise, noise, 2)
        flips += (top2 + e2) > (top1 + e1)
    return flips / trials

print("3) How often batch noise flips the chosen token, by how close the top two choices are")
for label, margin in [("clear prompt   (big margin)", 1.0),
                      ("somewhat clear (margin 1e-5)", 1e-5),
                      ("ambiguous      (near tie)", noise / 4)]:
    print(f"   {label}: {flip_rate(margin):6.1%} of runs pick a different token")
print("\n   One flipped token changes everything generated after it.")
