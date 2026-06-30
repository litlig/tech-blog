---
title: "Train a diffusion model from scratch"
author: ["Strayforge"]
date: 2026-06-28
tags: ["diffusion", "flow-matching"]
categories: ["learning"]
draft: true
math: true
---

{{< katex >}}

When I first read the DDPM paper, I was both fascinated and confused by the idea that, by adding and estimating the noise in an image sample, we can build a model and a sampling procedure that generates aesthetic images from noise. Isn't the noise just white noise sampled from a Gaussian? We already know its distribution, so why would we want to estimate it? And why add noise again when, during sampling, we are trying to remove it?

If you share the same confusion, read on. We'll unpack how a diffusion model works and train one from scratch on a simple 2-D example — and by the end, we'll have answered all three questions.


## Why use a diffusion process {#why-use-a-diffusion-process}

Most machine learning problems are essentially estimating a target distribution with a model (a neural network). Image classification estimates the distribution over labels given an image, \\(p(y \mid x)\\). Next-token prediction in an LLM estimates the distribution over the next token given the preceding context, \\(p(x\_t \mid x\_{<t})\\).

For images, we can likewise say that all images are sampled from a distribution. A 256×256 image with 3 color channels is drawn from a distribution over a 3x256x256-dimensional space. We could model this distribution directly by training a network with a negative log-likelihood loss, but the normalizing constant over such a high-dimensional space is intractable, so the likelihood cannot be computed or optimized directly.

Instead, researchers take another path: design a process that gradually converts noise into images. This sidesteps the intractable likelihood, because we only need to learn the local direction of the transformation at each step rather than the full distribution. Each step then becomes a simple regression target the network can learn directly.


## Transform noise into an image {#transform-noise-into-an-image}

If we already know the image we want to transform into, we can simply interpolate linearly between the noise and that image.

{{< figure src="noise-to-image.png" caption="<span class=\"figure-number\">Figure 1: </span>Linearly interpolating from a noise sample to a target image." >}}

Mathematically, we can frame it as \\(x\_t = \beta\_t x\_0 + \alpha\_t z\\).

\\(t\\) respresents the time and is within [0,1], \\(t=0\\) is the inital state which is pure noise and \\(t=1\\) is the final state which is a image z. \\(\alpha\_t\\) and \\(\beta\_t\\) are schedules; for linear interpolation, \\(\beta\_t = 1 - t\\) and \\(\alpha\_t = t\\). We can define a vector field (also refered to as velocity field) as the direction and speed \\(x\_t\\) should move at a given time, which is the derivative of \\(x\_t\\) with respect to \\(t\\). In this simple case, for a given \\(t\\) the particle is just moving along the straight line toward \\(z\\).

What we actually want is for \\(x\_t\\) to approximate the image distribution at \\(t = 1\\). If the image space contains only a dog image and a cat image, we want sampling to generate dog images half the time and cat images the other half. So instead of moving the particle toward one destination, the vector field should be the average of the velocities from all images, \\(\mathbb{E}\_{z \sim p(z \mid x\_t)}[v\_t \mid z]\\).

Flow matching training:

```text
for each training step:
    z  ~ data distribution        # a real image
    x0 ~ N(0, I)                  # pure noise
    t  ~ Uniform(0, 1)
    xt = (1 - t)·x0 + t·z         # a point on the noise→data path
    v_target = z - x0             # conditional velocity
    loss = || f(xt, t) - v_target ||²
    backprop and update f
```


## flow matching, unpacked {#flow-matching-unpacked}

So what does the trained field actually do? In practice it moves each noise point toward the _nearest_ data point — even though the training data contains no such bias, since every \\(x\_0\\), target \\(z\\), and time \\(t\\) is drawn independently. The reason hides in the average: the estimated field at \\((x\_t, t)\\) is the mean velocity over all \\(z\\), weighted by \\(p(z \mid x\_t)\\).

At \\(t = 0\\), \\(x\_t\\) is pure noise and \\(p(z \mid x\_t)\\) is just the prior, so the field points toward the center of probability mass. As \\(t\\) grows, \\(x\_t\\) begins to encode which \\(z\\) it came from: \\(p(z \mid x\_t)\\) concentrates on the closer targets, and the particle is pulled toward them. The straight-line pull toward a single image is never trained in directly — it emerges in aggregate, from this shifting average.


## diffusion — adding randomness to the otherwise deterministic path {#diffusion-adding-randomness-to-the-otherwise-deterministic-path}

With a network trained using the steps above, we can estimate the vector field, and the sampling path becomes deterministic once the initial noise sample is drawn.

```text
x  ~ N(0, I)
dt = 1 / steps
for step = 0 … steps-1:
    t = step / steps
    x = x + f(x, t)·dt            # follow the estimated velocity
return x
```

There is little correction after the initial sample \\(x\_0\\). Because the whole trajectory is fixed by that first draw, any error in the estimated vector field accumulates along the path and pushes the sample off the data manifold, yielding a poor image. To add self-correction, noise is injected at every sampling step. The randomness lets the path re-explore and re-converge toward high-probability regions, so errors are corrected rather than compounded.

{{< figure src="sde-vs-ode.png" caption="<span class=\"figure-number\">Figure 2: </span>Deterministic ODE sampling vs. stochastic SDE sampling with added noise; the SDE samples land more tightly on the data." >}}


## flow matching and score matching {#flow-matching-and-score-matching}

Let \\(p(x\_t)\\) be the probability density of \\(x\_t\\). The vector field is the instantaneous velocity of a particle, while the score is the gradient of the log-probability, \\(\nabla\_{x\_t} \log p(x\_t)\\). When the noise is Gaussian, i.e. \\(x\_t = \beta\_t x\_0 + \alpha\_t z\\) with \\(x\_0\\) Gaussian, both the vector field and the score are linear combinations of \\(x\_t\\) and \\(z\\), and are therefore interchangeable.


## back to our three questions {#back-to-our-three-questions}

We opened with three confusions. Here are the answers.

-   _Isn't the noise just a Gaussian we already know?_ Yes — and that's the point. The Gaussian is only the starting line at \\(t = 0\\); what we learn is not the noise but the velocity field that transports that known Gaussian into the unknown data distribution.
-   _Then why estimate anything?_ Because the hard part isn't the noise, it's the destination. The network learns, at every point and time, the average direction toward the data — in effect, where the probability mass lives.
-   _Why add noise back when sampling?_ A fully deterministic path locks in every error from the first draw. Re-injecting noise at each step lets the trajectory correct itself and re-converge onto the data manifold — a self-correcting path instead of a fixed one.
