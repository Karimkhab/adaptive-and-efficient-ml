"""PPO with a tanh Gaussian actor, GAE and separate timeout bootstrapping.

The actor receives only TaxiObservation.as_vector(), never TaxiState.
"""
import json
import hashlib
import io
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax


def control_repeat(config):
    """Inference may close the loop faster; training time scale stays explicit."""
    repeat=config.get('inference_repeat',config['repeat'])
    if not isinstance(repeat,int) or isinstance(repeat,bool) or repeat<1:
        raise ValueError('Control repeat must be a positive integer')
    return repeat


def init_network(seed, obs_dim=70, hidden=64):
    rng = np.random.default_rng(seed)

    def layer(n, m, scale):
        a = rng.normal(size=(max(n, m), min(n, m)))
        q, r = np.linalg.qr(a)
        q *= np.sign(np.diag(r))
        if n < m:
            q = q.T
        return {"w": jnp.array(q[:n, :m] * scale, dtype=jnp.float32),
                "b": jnp.zeros(m, dtype=jnp.float32)}

    def net(out, scale):
        return [layer(obs_dim, hidden, math.sqrt(2)),
                layer(hidden, hidden, math.sqrt(2)), layer(hidden, out, scale)]

    return {"actor": net(2, .01), "critic": net(1, 1.),
            "log_std": jnp.full((2,), -.5, dtype=jnp.float32)}


def mlp(layers, x):
    for layer in layers[:-1]:
        x = jnp.tanh(x @ layer["w"] + layer["b"])
    return x @ layers[-1]["w"] + layers[-1]["b"]


def forward(params, obs):
    # Bound a single outlying distance-to-path without online statistics.
    obs = obs.at[..., 1:3].multiply(20.)
    obs = jnp.clip(obs, -5., 5.)
    return mlp(params["actor"], obs), mlp(params["critic"], obs)[..., 0]


def log_prob(params, mean, latent):
    log_std = jnp.clip(params["log_std"], -5., 1.)
    gaussian = -.5 * (((latent - mean) / jnp.exp(log_std))**2
                       + 2 * log_std + math.log(2 * math.pi))
    # Stable log(1 - tanh(z)^2). It cancels in PPO ratios, but keeps
    # logged action densities those of the actual bounded policy.
    jacobian = 2 * (math.log(2.) - latent - jax.nn.softplus(-2 * latent))
    return jnp.sum(gaussian - jacobian, axis=-1)


def sample(params, obs, key):
    mean, value = forward(params, obs)
    latent = mean + jnp.exp(jnp.clip(params["log_std"], -5., 1.)) * jax.random.normal(key, mean.shape, dtype=mean.dtype)
    return jnp.tanh(latent), latent, log_prob(params, mean, latent), value


def advantages(reward, value, next_value, terminal, done, gamma=.99, lam=.95):
    """Bootstrap time limits, but never propagate GAE across an autoreset."""
    delta = reward + gamma * (1 - terminal) * next_value - value

    def body(carry, xs):
        d, end = xs
        carry = d + gamma * lam * (1 - end) * carry
        return carry, carry

    _, adv = jax.lax.scan(body, jnp.zeros_like(value[0]), (delta, done), reverse=True)
    return adv, adv + value


def ppo_loss(params, batch, clip=.2, entropy_coef=.001):
    obs, latent, old_logp, old_value, advantage, target = batch
    mean, value = forward(params, obs)
    new_logp = log_prob(params, mean, latent)
    ratio = jnp.exp(new_logp - old_logp)
    surrogate = jnp.minimum(ratio * advantage, jnp.clip(ratio, 1 - clip, 1 + clip) * advantage)
    actor_loss = -surrogate.mean()
    critic_loss = .5 * jnp.square(value - target).mean()
    # Entropy of the latent Normal is a documented exploration proxy;
    # this is not an estimator of post-tanh entropy.
    entropy = jnp.sum(jnp.clip(params["log_std"], -5., 1.) + .5 * math.log(2 * math.pi * math.e))
    loss = actor_loss + .5 * critic_loss - entropy_coef * entropy
    metrics = jnp.array([loss, actor_loss, critic_loss, entropy,
                         ((ratio - 1) - (new_logp - old_logp)).mean(),
                         (jnp.abs(ratio - 1) > clip).mean()])
    return loss, metrics


def make_update(learning_rate=3e-4, epochs=4, minibatches=4, entropy_coef=.001):
    optimizer = optax.chain(optax.clip_by_global_norm(.5), optax.adam(learning_rate, eps=1e-5))

    @jax.jit
    def update(params, opt_state, batch, key):
        n = batch[0].shape[0]

        def epoch(carry, _):
            params, opt_state, key = carry
            key, shuffle = jax.random.split(key)
            idx = jax.random.permutation(shuffle, n).reshape(minibatches, n // minibatches)

            def minibatch(carry, ids):
                params, opt_state = carry
                piece = tuple(x[ids] for x in batch)
                (_, metrics), grads = jax.value_and_grad(ppo_loss, has_aux=True)(
                    params, piece, entropy_coef=entropy_coef)
                changes, opt_state = optimizer.update(grads, opt_state, params)
                params = optax.apply_updates(params, changes)
                return (params, opt_state), metrics

            (params, opt_state), metrics = jax.lax.scan(minibatch, (params, opt_state), idx)
            return (params, opt_state, key), metrics.mean(axis=0)

        (params, opt_state, _), metrics = jax.lax.scan(epoch, (params, opt_state, key), None, length=epochs)
        return params, opt_state, metrics.mean(axis=0)

    return optimizer, update


def save_checkpoint(path, params, config):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    leaves, _ = jax.tree.flatten(params)
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **{f"arr_{i}": np.asarray(x) for i, x in enumerate(leaves)})
    blob = buffer.getvalue()
    path.with_suffix(".tmp.npz").write_bytes(blob)
    path.with_suffix(".tmp.npz").replace(path)
    metadata = dict(config, checkpoint_format=2, arrays_sha256=hashlib.sha256(blob).hexdigest())
    path.with_suffix(".tmp.json").write_text(json.dumps(metadata, indent=2) + "\n")
    path.with_suffix(".tmp.json").replace(path.with_suffix(".json"))


def load_checkpoint(path):
    config = json.loads(Path(path).with_suffix(".json").read_text())
    if config.get("observation_version") != 2 or config.get("obs_dim") != 70:
        raise ValueError("Checkpoint must use observation v2 (70 features); old environment is incompatible")
    if config.get("arrays_sha256") and hashlib.sha256(Path(path).read_bytes()).hexdigest() != config["arrays_sha256"]:
        raise ValueError("Checkpoint checksum mismatch")
    template = init_network(config["seed"], obs_dim=config["obs_dim"], hidden=config["hidden"])
    leaves, tree = jax.tree.flatten(template)
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != {f"arr_{i}" for i in range(len(leaves))}:
            raise ValueError("Unexpected checkpoint parameter set")
        arrays = [data[f"arr_{i}"] for i in range(len(leaves))]
        if any(a.shape != leaf.shape or a.dtype != np.float32 or not np.isfinite(a).all()
               for a, leaf in zip(arrays, leaves)):
            raise ValueError("Checkpoint parameter shape/dtype/values do not match its architecture")
        params = jax.tree.unflatten(tree, [jnp.asarray(a) for a in arrays])
    return params, config
