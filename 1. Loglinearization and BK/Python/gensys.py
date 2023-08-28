# Imports
import numpy as np
import scipy.stats
import scipy.linalg

def gensys(g0, g1, Const, psi, pi, div=None, realsmall=0.000001):
    unique = False
    eu = [0, 0]
    nunstab = 0
    zxz = False

    n = g1.shape[0]

    if div is None:
        div = 1.01
        fixdiv = False
    else:
        fixdiv = True
    a, b, q, z = scipy.linalg.qz(g0, g1, 'complex')

    q = q.T.conjugate()
    pi = pi.astype(np.complex128)
    Const = Const.astype(np.complex128)

    for i in range(n):
        if not fixdiv:
            if np.abs(a[i, i]) > 0:
                divhat = np.abs(b[i, i] / a[i, i])
                if (1 + realsmall < divhat) and divhat <= div:
                    div = 0.5 * (1 + divhat)
        soma = 1 if ((np.abs(b[i, i])) > (div * np.abs(a[i, i]))) else 0
        nunstab = nunstab + soma
        if (np.abs(a[i, i]) < realsmall) and (np.abs(b[i, i]) < realsmall):
            zxz = True

    if not zxz:
        a, b, q, z, _ = qzdiv(div, a, b, q, z)
    else:
        eu = [-2, -2]
        return None, None, None, eu

    q1 = q[:n - nunstab, :]
    q2 = q[n - nunstab:, :]
    etawt = q2 @ pi
    neta = pi.shape[1]

    # Case for no stable roots
    if nunstab == 0:
        bigev = np.zeros((0), dtype=np.int64)
        ueta = np.zeros((0, 0), dtype=np.complex128)
        deta = np.zeros((0, 0))
        veta = np.zeros((neta, 0), dtype=np.complex128)
    else:
        ueta, deta, veta = scipy.linalg.svd(etawt)
        deta = np.diag(deta)
        veta = veta.T.conjugate()
        md = min(deta.shape)
        bigev = np.where(np.diag(deta[:md, :md]) > realsmall)[0]
        ueta = ueta[:, bigev]
        veta = veta[:, bigev]
        deta = np.diag(deta)[bigev]
        deta = np.diag(deta)
    if bigev.size >= nunstab:
        eu[0] = 1
    if nunstab == n:
        ueta1 = np.zeros((0, 0), dtype=np.complex128)
        deta1 = np.zeros((0, 0))
        veta1 = np.zeros((neta, 0), dtype=np.complex128)
    else:
        etawt1 = q1 @ pi
        ueta1, deta1, veta1 = scipy.linalg.svd(etawt1)
        deta1 = np.diag(deta1)
        veta1 = veta1.T.conjugate()
        md = min(deta1.shape)
        bigev = np.where(np.real(np.diag(deta1[:md, :md])) > realsmall)[0]
        ueta1 = ueta1[:, bigev]
        veta1 = veta1[:, bigev]
        deta1 = np.diag(deta1)[bigev]
        deta1 = np.diag(deta1)

    if 0 in veta1.shape:
        unique = True
    else:
        loose = veta1 - veta @ veta.T @ veta1
        ul, dl, vl = scipy.linalg.svd(loose)
        dl = np.diag(dl)
        nloose = np.sum(np.abs(np.diag(dl))) > realsmall * n

        if not nloose:
            unique = True

    if unique:
        eu[1] = 1

    deta = deta.astype(np.complex128)
    deta1 = deta1.astype(np.complex128)
    psi = psi.astype(np.complex128)

    tmat = np.hstack((np.eye(n - nunstab), -(
            ueta @ (scipy.linalg.inv(deta) @ veta.conjugate().T) @ veta1 @ deta1 @ ueta1.conjugate().T).conjugate().T))
    G0 = np.vstack((tmat @ a, np.hstack((np.zeros((nunstab, n - nunstab)), np.eye(nunstab)))))
    G1 = np.vstack((tmat @ b, np.zeros((nunstab, n))))

    G0I = scipy.linalg.inv(G0)
    G1 = G0I @ G1
    impact = G0I @ np.vstack((tmat @ q @ psi, np.zeros((nunstab, psi.shape[1]))))

    usix = list(range(n - nunstab, n))

    C = Const
    C = G0I @ np.concatenate((tmat.dot(q).dot(C), np.linalg.inv(a[usix][:, usix] - b[usix][:, usix]) @ q2 @ C))

    G1 = np.array((z @ G1 @ z.conjugate().T).real)
    C = (z @ C).real
    impact = (z @ impact).real

    return G1, C, impact, eu

def qzswitch(i, A, B, Q, Z):
    eps = 2.2204e-16
    realsmall = np.sqrt(eps) * 10

    a, b, c = A[i, i], A[i, i + 1], A[i + 1, i + 1]
    d, e, f = B[i, i], B[i, i + 1], B[i + 1, i + 1]

    if (abs(c) < realsmall) and (abs(f) < realsmall):
        if abs(a) < realsmall:
            return A, B, Q, Z
        else:
            wz = np.array([[b], [-a]])
            wz = wz / ((wz.conjugate().T @ wz)[0, 0] ** 0.5)
            wz = np.array([[wz[0][0], wz[1][0].conjugate()], [wz[1][0], -wz[0][0].conjugate()]])
            xy = np.eye(2, dtype=np.complex128)
    elif (abs(a) < realsmall) and (abs(d) < realsmall):
        if abs(c) < realsmall:
            return A, B, Q, Z
        else:
            wz = np.eye(2, dtype=np.complex128)
            xy = np.array([c, -b])
            xy = xy / np.array([[xy @ xy.conjugate().T]])[0, 0] ** 0.5
            xy = np.array([[xy[1].conjugate(), -xy[0].conjugate()], [xy[0], xy[1]]])
    else:

        wz = np.array([c * e - f * b, np.conjugate((c * d - f * a))])
        xy = np.array([np.conjugate(b * d - e * a), np.conjugate(c * d - f * a)])

        wz2 = np.array([[c * e - f * b, np.conjugate((c * d - f * a))]])
        xy2 = np.array([[np.conjugate(b * d - e * a), np.conjugate(c * d - f * a)]])

        n = (wz2 @ wz2.T.conjugate())[0, 0] ** 0.5
        m = (xy2 @ xy2.T.conjugate())[0, 0] ** 0.5

        if np.real(m) < eps * 100:
            return A, B, Q, Z
        else:
            wz = wz / n
            xy = xy / m
            wz = np.array([[wz[0], wz[1]], [np.conjugate(-wz[1]), np.conjugate(wz[0])]])
            xy = np.array([[xy[0], xy[1]], [np.conjugate(-xy[1]), np.conjugate(xy[0])]])

        if xy is None:
            return A, B, Q, Z

    A[i:i + 2, :] = xy @ A[i:i + 2, :]
    B[i:i + 2, :] = xy @ B[i:i + 2, :]
    A[:, i:i + 2] = A[:, i:i + 2] @ wz
    B[:, i:i + 2] = B[:, i:i + 2] @ wz
    Z[:, i:i + 2] = Z[:, i:i + 2] @ wz
    Q[i:i + 2, :] = xy @ Q[i:i + 2, :]

    return A, B, Q, Z

def qzdiv(stake, A, B, Q, Z, v=None):
    n = A.shape[0]

    root = np.zeros((np.diag(A).shape[0], 2), np.complex128)

    root[:, 0] = np.abs(np.diag(A))
    root[:, 1] = np.abs(np.diag(B))

    root[:, 0] = root[:, 0] - (np.real(root[:, 0]) < 1.e-13) * (root[:, 0] + root[:, 1])
    root[:, 1] = root[:, 1] / root[:, 0]

    for i in range(n - 1, -1, -1):
        m = None
        for j in range(i, -1, -1):
            if (np.real(root[j, 1]) > stake) or (np.real(root[j, 1]) < -0.1):
                m = j
                break

        if m is None:
            return A, B, Q, Z, v

        for k in range(m, i):
            A, B, Q, Z = qzswitch(k, A, B, Q, Z)
            temp = root[k, 1]
            root[k, 1] = root[k + 1, 1]
            root[k + 1, 1] = temp

            if not (v is None):
                temp = v[:, k]
                v[:, k] = v[:, k + 1]
                v[:, k + 1] = temp

    return A, B, Q, Z, v
