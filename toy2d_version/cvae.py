"""
A small Conditional VAE implemented in plain numpy (no torch available in
this environment). Architecture mirrors a standard CVAE:

    Encoder: q(z | state, action)  -> mu, logvar
    Decoder: p(action | state, z)  -> reconstructed action

Trained with the usual VAE loss = reconstruction MSE + beta * KL(q || N(0,I)).

At inference time we no longer have the true action to encode, so we
sample z ~ N(0, I) and let the decoder produce ONE coherent action
consistent with that sampled mode, instead of averaging over all modes
the way a plain MSE regressor does.
"""

import numpy as np


def tanh(x):
    return np.tanh(x)


def dtanh(y):
    # derivative of tanh given the tanh OUTPUT y
    return 1 - y ** 2


class CVAE:
    def __init__(self, state_dim=2, action_dim=2, latent_dim=2, hidden=32, seed=0):
        rng = np.random.default_rng(seed)
        self.latent_dim = latent_dim

        def init(shape):
            return rng.normal(0, 0.3, size=shape)

        enc_in = state_dim + action_dim
        self.W1 = init((enc_in, hidden)); self.b1 = np.zeros(hidden)
        self.Wmu = init((hidden, latent_dim)); self.bmu = np.zeros(latent_dim)
        self.Wlv = init((hidden, latent_dim)); self.blv = np.zeros(latent_dim)

        dec_in = state_dim + latent_dim
        self.W2 = init((dec_in, hidden)); self.b2 = np.zeros(hidden)
        self.W3 = init((hidden, action_dim)); self.b3 = np.zeros(action_dim)

        self.params = ["W1", "b1", "Wmu", "bmu", "Wlv", "blv", "W2", "b2", "W3", "b3"]
        # Adam optimizer state
        self.m = {p: np.zeros_like(getattr(self, p)) for p in self.params}
        self.v = {p: np.zeros_like(getattr(self, p)) for p in self.params}
        self.t = 0

    def encode(self, s, a):
        x = np.concatenate([s, a], axis=1)
        h1 = tanh(x @ self.W1 + self.b1)
        mu = h1 @ self.Wmu + self.bmu
        logvar = h1 @ self.Wlv + self.blv
        return x, h1, mu, logvar

    def decode(self, s, z):
        x2 = np.concatenate([s, z], axis=1)
        h2 = tanh(x2 @ self.W2 + self.b2)
        a_hat = h2 @ self.W3 + self.b3
        return x2, h2, a_hat

    def forward(self, s, a, rng):
        x, h1, mu, logvar = self.encode(s, a)
        std = np.exp(0.5 * logvar)
        eps = rng.normal(size=mu.shape)
        z = mu + eps * std
        x2, h2, a_hat = self.decode(s, z)
        return dict(x=x, h1=h1, mu=mu, logvar=logvar, std=std, eps=eps,
                    z=z, x2=x2, h2=h2, a_hat=a_hat)

    def loss(self, a, cache, beta=0.02):
        recon = np.mean(np.sum((cache["a_hat"] - a) ** 2, axis=1))
        kl = -0.5 * np.mean(np.sum(
            1 + cache["logvar"] - cache["mu"] ** 2 - np.exp(cache["logvar"]), axis=1))
        return recon + beta * kl, recon, kl

    def backward(self, s, a, cache, beta=0.02):
        n = a.shape[0]
        grads = {}

        # --- reconstruction loss grad wrt a_hat ---
        d_ahat = 2 * (cache["a_hat"] - a) / n            # (n, action_dim)
        grads["W3"] = cache["h2"].T @ d_ahat
        grads["b3"] = d_ahat.sum(axis=0)
        d_h2 = d_ahat @ self.W3.T
        d_h2_pre = d_h2 * dtanh(cache["h2"])
        grads["W2"] = cache["x2"].T @ d_h2_pre
        grads["b2"] = d_h2_pre.sum(axis=0)
        d_x2 = d_h2_pre @ self.W2.T
        d_z_recon = d_x2[:, -self.latent_dim:]

        # --- reparameterization: z = mu + eps*std, std = exp(0.5*logvar) ---
        d_mu_recon = d_z_recon.copy()
        d_logvar_recon = d_z_recon * cache["eps"] * 0.5 * cache["std"]

        # --- KL loss grad wrt mu, logvar ---
        d_mu_kl = beta * cache["mu"] / n
        d_logvar_kl = beta * 0.5 * (np.exp(cache["logvar"]) - 1) / n

        d_mu = d_mu_recon + d_mu_kl
        d_logvar = d_logvar_recon + d_logvar_kl

        grads["Wmu"] = cache["h1"].T @ d_mu
        grads["bmu"] = d_mu.sum(axis=0)
        grads["Wlv"] = cache["h1"].T @ d_logvar
        grads["blv"] = d_logvar.sum(axis=0)

        d_h1 = d_mu @ self.Wmu.T + d_logvar @ self.Wlv.T
        d_h1_pre = d_h1 * dtanh(cache["h1"])
        grads["W1"] = cache["x"].T @ d_h1_pre
        grads["b1"] = d_h1_pre.sum(axis=0)

        return grads

    def adam_step(self, grads, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.t += 1
        for p in self.params:
            g = grads[p]
            self.m[p] = beta1 * self.m[p] + (1 - beta1) * g
            self.v[p] = beta2 * self.v[p] + (1 - beta2) * (g ** 2)
            m_hat = self.m[p] / (1 - beta1 ** self.t)
            v_hat = self.v[p] / (1 - beta2 ** self.t)
            update = lr * m_hat / (np.sqrt(v_hat) + eps)
            setattr(self, p, getattr(self, p) - update)

    def fit(self, states, actions, epochs=300, batch_size=128, lr=1e-3, seed=1,
            beta=0.02, verbose=True):
        rng = np.random.default_rng(seed)
        n = states.shape[0]
        for epoch in range(epochs):
            perm = rng.permutation(n)
            epoch_loss = 0.0
            for i in range(0, n, batch_size):
                idx = perm[i:i + batch_size]
                s_b, a_b = states[idx], actions[idx]
                cache = self.forward(s_b, a_b, rng)
                loss, recon, kl = self.loss(a_b, cache, beta=beta)
                grads = self.backward(s_b, a_b, cache, beta=beta)
                self.adam_step(grads, lr=lr)
                epoch_loss += loss * len(idx)
            if verbose and (epoch % 50 == 0 or epoch == epochs - 1):
                print(f"epoch {epoch:4d}  loss={epoch_loss/n:.4f}  "
                      f"recon={recon:.4f}  kl={kl:.4f}")

    def sample_action(self, s, z=None, rng=None):
        """Decode a single action for state s, given a latent z (or a
        freshly sampled one)."""
        s = np.asarray(s).reshape(1, -1)
        if z is None:
            if rng is None:
                rng = np.random.default_rng()
            z = rng.normal(size=(1, self.latent_dim))
        else:
            z = np.asarray(z).reshape(1, -1)
        _, _, a_hat = self.decode(s, z)
        return a_hat[0]
