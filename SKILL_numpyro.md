---
name: numpyro-nuts
description: >
  Fit a NumPyro probabilistic model using NUTS sampling with empirical
  initialization. Use when the user asks to fit, sample, run inference,
  or do Bayesian modelling with NumPyro. Also triggers when the user
  mentions MCMC, posterior, prior, or probabilistic model in the context
  of JAX/NumPyro code.
---

# NumPyro NUTS Fitting

## Standard Settings
- **Sampler**: NUTS
- **Warmup steps**: 500
- **Samples**: 500
- **Chains**: 1
- **Initialization**: empirical values derived from the data

## Initialization Strategy
ALWAYS initialize chain parameters using rough empirical estimates
computed directly from the data BEFORE defining the kernel. This
reduces warmup time and avoids pathological starting points.

Common empirical initializations:
- **Mean-like parameters**: use `jnp.mean(data)`
- **Scale-like parameters**: use `jnp.std(data)`
- **Rate parameters**: use `1.0 / jnp.mean(data)`
- **Probability parameters**: use `jnp.mean(data > threshold)`
- **Mixture weights**: use observed class frequencies

Pass these as a dict to `init_to_value(values={...})`.

## Workflow

1. Compute empirical estimates from data
2. Define the NumPyro model with weakly informative priors
3. Comment each prior with its rationale
4. Initialize kernel with empirical values via `init_to_value`
5. Run MCMC with standard settings
6. Always call `mcmc.print_summary()` after sampling
7. Check `r_hat` values — flag any > 1.01 as a convergence warning

## Standard Template

```python
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, init_to_value
import jax
import jax.numpy as jnp

# ── 1. Empirical initialization from data ────────────────────────────
init_values = {
    "mu":    jnp.mean(data),
    "sigma": jnp.std(data),
    # Add / remove params to match your model
}

# ── 2. Model definition ───────────────────────────────────────────────
def model(data):
    # Weakly informative priors — document rationale for each
    mu    = numpyro.sample("mu",    dist.Normal(init_values["mu"], 10.0))
    sigma = numpyro.sample("sigma", dist.HalfNormal(init_values["sigma"] * 2))

    with numpyro.plate("obs", len(data)):
        numpyro.sample("y", dist.Normal(mu, sigma), obs=data)

# ── 3. Sampler setup ──────────────────────────────────────────────────
rng_key = jax.random.PRNGKey(0)

kernel = NUTS(model, init_strategy=init_to_value(values=init_values))

mcmc = MCMC(
    kernel,
    num_warmup=500,
    num_samples=500,
    num_chains=1,
    progress_bar=True,
)

# ── 4. Run and summarize ──────────────────────────────────────────────
mcmc.run(rng_key, data)
mcmc.print_summary()

# ── 5. Convergence check ──────────────────────────────────────────────
samples = mcmc.get_samples()
# Warn if any r_hat > 1.01
```

## Gotchas

- `init_to_value` requires ALL sampled sites to have an initial value,
  or it falls back silently — double-check your dict covers every
  `numpyro.sample` call in the model.
- `jax.random.PRNGKey(0)` is deterministic; use a different seed or
  `jax.random.PRNGKey(int(time.time()))` if you need varied runs.
- `HalfNormal` for scale parameters — NEVER use `Normal` for sigma
  (it allows negative values).
- With 1 chain, `r_hat` is undefined — convergence must be assessed
  via trace plots. Suggest switching to `num_chains=4` when
  convergence is in doubt.
- If warmup is consuming more than ~30% of total samples, the empirical
  init values are likely on a poor scale — revisit the initialization.

