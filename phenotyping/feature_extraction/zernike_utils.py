import numpy as np


def zernfun(n, m, r, theta, normalized=False):
    n = np.atleast_1d(n).astype(int)
    m = np.atleast_1d(m).astype(int)
    r = np.asarray(r, dtype=float).ravel()
    theta = np.asarray(theta, dtype=float).ravel()

    if len(n) != len(m):
        raise ValueError("n and m must have the same length")
    if np.any((n - m) % 2 != 0):
        raise ValueError("All N and M must differ by multiples of 2")
    if np.any(m > n):
        raise ValueError("Each M must be less than or equal to its corresponding N")
    if np.any((r < 0) | (r > 1)):
        raise ValueError("All R values must be between 0 and 1")

    m_abs = np.abs(m)
    rpowers = []
    for j in range(len(n)):
        rpowers.extend(range(m_abs[j], n[j] + 1, 2))
    rpowers = np.unique(rpowers)

    if rpowers[0] == 0:
        rpowern = [r ** p for p in rpowers[1:]]
        rpowern = np.column_stack(rpowern) if len(rpowers) > 1 else np.empty((len(r), 0))
        rpowern = np.column_stack([np.ones(len(r)), rpowern])
    else:
        rpowern = [r ** p for p in rpowers]
        rpowern = np.column_stack(rpowern)

    z = np.zeros((len(r), len(n)))
    for j in range(len(n)):
        s = np.arange(0, (n[j] - m_abs[j]) // 2 + 1)
        pows = np.arange(n[j], m_abs[j] - 1, -2)
        for k in range(len(s) - 1, -1, -1):
            p = (1 - 2 * (s[k] % 2)) * np.prod(np.arange(2, n[j] - s[k] + 1)) / (
                np.prod(np.arange(2, s[k] + 1))
                * np.prod(np.arange(2, ((n[j] - m_abs[j]) // 2 - s[k]) + 1))
                * np.prod(np.arange(2, ((n[j] + m_abs[j]) // 2 - s[k]) + 1))
            )
            idx = np.where(pows[k] == rpowers)[0][0]
            z[:, j] += p * rpowern[:, idx]

        if normalized:
            z[:, j] *= np.sqrt((1 + (m[j] != 0)) * (n[j] + 1) / np.pi)

    pos = m > 0
    neg = m < 0
    if np.any(pos):
        z[:, pos] *= np.cos(theta[:, None] * m_abs[pos][None, :])
    if np.any(neg):
        z[:, neg] *= np.sin(theta[:, None] * m_abs[neg][None, :])

    return z
