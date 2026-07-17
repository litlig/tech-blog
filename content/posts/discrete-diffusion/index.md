---
title: "Language generation with discrete diffusion"
date: 2026-07-17
draft: false
categories: ["learning"]
tags: ["diffusion", "discrete-diffusion", "transformer", "notebook"]
summary: "Building a discrete diffusion transformer (DiT) from scratch and training it to generate Shakespeare-like text, one denoising jump at a time."
---

{{< katex >}}

> The code below is trimmed to the essentials; the full, runnable version lives in the original notebook:
> [**github.com/litlig/notebooks/discrete_diffusion.ipynb**](https://github.com/litlig/notebooks/blob/main/discrete_diffusion.ipynb)
>
> <a href="https://colab.research.google.com/github/litlig/notebooks/blob/main/discrete_diffusion.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

Diffusion models are largely used in image and video generation. They operate in a high-dimensional *continuous* space, where we convert noise to an image by following a denoising process. Text generation, on the other hand, is discrete in nature. For a text sequence, we tokenize it and get a vector of token indices. The numerical difference between tokens has no natural meaning — a smaller gap between two token ids does not mean the tokens are semantically closer.

Discrete diffusion is designed for this discrete setting. Here we use the Shakespeare dataset from Andrej Karpathy's LLM series, build a discrete DiT model from scratch, and see if it can generate Shakespeare-like text.

We work at the character level, so the "vocabulary" is just the set of distinct characters in the corpus:

```python
vocab = sorted(list(set(text)))
vocab_size = len(vocab)  # 65

stoi = { ch:i for i,ch in enumerate(vocab) }
itos = { i:ch for i,ch in enumerate(vocab) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])
```

## CTMC and the rate matrix

To recall (see the earlier [2-D diffusion post]({{< ref "posts/diffusion-2d-notebook" >}}) for the continuous case in full), the continuous diffusion model is done by flow matching, where we define a vector field \(u_t\) that tells the direction and velocity a noised image \(x_t\) should move at time \(t\). Since we do not know the distribution of the image manifold, \(u_t\) is not tractable. However, if we know the final state (a sample image), \(u_t\) can be computed analytically — one solution is to linearly connect the noised point \(x_t\) and the final state \(z\). What we actually need is not the vector field for a given \(z\), but the *average* vector field over all \(z\) that follow the posterior distribution \(p(z\,|\,x_t)\). In neural net training the loss is always an average over samples, so by training a network to minimize the average loss against the conditional vector field, we also recover the marginal vector field we need.

For a text sequence, the transition from noise to a meaningful sequence is a series of *jumps* across discrete states. Instead of a vector field, we define a **rate matrix** \(Q_t(y\,|\,x)\), the rate of jumping from state \(x\) to state \(y\) at time \(t\). If we are at state \(x\) at time \(t\), then over a very small interval \(h\) the probability we jump to \(y\) (with \(y \neq x\)) is \(h\, Q_t(y\,|\,x)\). This is a continuous-time Markov chain (CTMC).

The number of states is exponential in the sequence length — \(V^d\), where \(V\) is the vocab size and \(d\) is the sequence length. To keep \(Q\) manageable, we set the rate to zero whenever more than one position changes. For \(X_t = x\), a rate matrix \(Q_t(v, j)\) then defines the rate of changing position \(j\) to token \(v\), and we can sample with an Euler approximation:

```python
def euler_step(x_t, rates, h):
    # delta_{v,x}: (B, d, V) one-hot at the current token
    delta = F.one_hot(x_t, num_classes=rates.size(-1)).to(rates.dtype)

    off_diag = h * rates * (1.0 - delta)          # h*q(v) for v != x, else 0
    stay = (1.0 - off_diag.sum(-1, keepdim=True)) # 1 - h*sum_{v!=x} q(v)

    probs = off_diag + delta * stay
    return torch.distributions.Categorical(probs=probs).sample()

def sample(model, n_seq, n_steps, noise_gen):
  ts = torch.linspace(0.0, 1.0, n_steps + 1, device=device)
  x = noise_gen.gen(1, n_seq)
  for i in range(n_steps):
    s, t = ts[i], ts[i + 1]
    rate_mtx = model(x, s)     # (n_seq, n_vocab)
    x = euler_step(x, rate_mtx, t - s)
  return x
```

Before training anything, we can sanity-check the machinery with a mock model that returns random rates. Starting from uniform noise and running the sampler produces exactly what you'd expect — noise:

```python
class MockModel(nn.Module):
  def forward(self, x, t):
    logits = torch.randn(x.shape[-1], vocab_size, device=device)
    return F.softmax(logits, dim=-1) - F.one_hot(x, num_classes=vocab_size).to(device)
```

```text
nwAlQgrsSo 'bmm,DjUmT?hkdwWpui:&N
iVXMEEqDm'zkh
3FfL?EK,oAvGkdLAK!x,cX.bG
```

## Factorized mixture path

Given an initial noise distribution \(p_{\mathrm{init}}\) and a final data distribution \(p_{\mathrm{data}}\), a discrete probability path is a family \(p_t\) with \(p_0 \sim p_{\mathrm{init}}\) and \(p_1 \sim p_{\mathrm{data}}\).

The most commonly used discrete probability path is the **factorized mixture path**, which defines the conditional path as:

\[p_t(x\,|\,z) = \prod_{j=1}^d \big[(1 - \kappa_t)\, p_{\mathrm{init}}^{(j)}(x_j) + \kappa_t\, \delta_{z_j}(x_j)\big]\]

where \(\kappa_t\) is the noise schedule. Each token position is treated independently. The rate matrix conditioned on \(z\) is:

\[Q^z_t(i, v\,|\,x_i) = \frac{\dot{\kappa}_t}{1 - \kappa_t}\big(\delta_{z_i}(v) - \delta_{x_i}(v)\big)\]

and the marginal rate matrix is:

\[Q_t(i, v\,|\,x_i) = \sum_z Q^z_t(i, v\,|\,x_i)\, p(z\,|\,x) = \frac{\dot{\kappa}_t}{1 - \kappa_t}\big(p(z_j = v\,|\,x) - \delta_{x_i}(v)\big)\]

So the problem reduces to a categorization: predict \(z\) given \(x\) and \(t\). The rate matrix is just a reparameterization of that prediction.

## The model: a discrete DiT

The network is a transformer that takes noised tokens \(x_t\) and a scalar time \(t\), and outputs logits over the vocabulary at each position — its job is to predict the clean tokens \(z\). Time is injected through **AdaLN-Zero** conditioning, as in the DiT architecture: a sinusoidal time embedding is passed through an MLP to produce a conditioning vector, which modulates each transformer block. The final projection is zero-initialized so each block starts as an identity map, which stabilizes training at initialization.

```python
class DiscreteDiffusionTransformer(nn.Module):
    """
    Input:  x_t (B, L) token ids, t (B,)
    Output: logits (B, L, vocab_size) predicting the original tokens
    """
    def forward(self, x_t, t, padding_mask=None):
        B, L = x_t.shape
        x = self.token_emb(x_t) + self.pos_emb[:, :L, :]
        c = self.time_mlp(t)  # (B, cond_dim) conditioning vector shared across layers
        for block in self.blocks:
            x = block(x, c, key_padding_mask=padding_mask)
        return self.final(x, c)
```

## Training to predict z

Training follows the factorized mixture path directly. For each batch we sample a time \(t\), set the noise level \(\kappa_t\) (here the schedule is simply \(\kappa_t = t\)), and construct \(x_t\) by keeping each clean token with probability \(\kappa_t\) and replacing it with uniform noise otherwise. The network then predicts the clean sequence \(z\), and the loss is a plain cross-entropy over positions:

```python
def schedule(t):
  return t

def get_batch():
  t = torch.rand(n_sample, device=device)
  kappa = schedule(t)
  seq_idx = torch.randint(0, len(text) - n_seq + 1, (n_sample,), device=device).unsqueeze(1) \
            + torch.arange(n_seq, device=device)
  z = data[seq_idx]                                                    # (sample, seq)
  masks = torch.bernoulli(kappa.unsqueeze(1).expand(-1, n_seq)).long() # keep clean where 1
  noise = noise_gen.gen(n_sample, n_seq)
  x = masks * z + (1 - masks) * noise
  return x, t, z

def loss_fn(z_pred, z):
  return F.cross_entropy(z_pred.view(-1, vocab_size), z.view(-1))
```

## Sampling with the trained model

At sampling time we turn the model's clean-token prediction back into a rate matrix using the marginal formula above, then take an Euler step:

```python
@torch.no_grad()
def sample(model, n_seq, n_steps, noise_gen):
  model.eval()
  ts = torch.linspace(0.0, 1.0, n_steps + 1, device=device)
  x = noise_gen.gen(1, n_seq)
  for i in range(n_steps):
    s, t = ts[i], ts[i + 1]
    logits = model(x, s.expand(x.shape[0]))
    rate_mtx = (F.softmax(logits, dim=-1) - F.one_hot(x, num_classes=vocab_size).to(device)) / (1 - s)
    x = euler_step(x, rate_mtx, t - s)
  return x
```

After a short training run, the samples are no longer random noise — the model has picked up Shakespeare's shape: capitalized speaker names, line breaks, and vaguely English words.

```text
d did neyce?

HONRY VIA: sutur selon'sl mave,
Myoury abenly tuke
```

It is far from coherent — this is a tiny model trained for a few thousand steps on characters — but the discrete diffusion process clearly works: starting from pure noise, a sequence of denoising jumps lands us in something that looks like a play.
