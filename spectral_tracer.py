#!/usr/bin/env python3
"""
Spectral BVH path tracer (NumPy + Numba + Tk).

    pip install numpy numba
    python spectral_tracer.py                      interactive window
    python spectral_tracer.py --headless --spp 128 --out render.png
    python spectral_tracer.py --bench              BVH vs brute force
    python spectral_tracer.py --selftest           BVH vs brute-force check
"""
import argparse
import math
import struct
import sys
import time
import zlib

import numpy as np

try:
    from numba import njit, prange
    HAVE_NUMBA = True
except ImportError:
    HAVE_NUMBA = False
    prange = range

    def njit(*a, **k):
        if len(a) == 1 and callable(a[0]) and not k:
            return a[0]
        return lambda f: f

NS = 81
LAMBDA = np.linspace(380.0, 780.0, NS)
EPS = 1e-4

XYZ2RGB = np.array([[3.2404542, -1.5371385, -0.4985314],
                    [-0.9692660, 1.8760108, 0.0415560],
                    [0.0556434, -0.2040259, 1.0572252]])


def _g(x, mu, s1, s2):
    s = np.where(x < mu, s1, s2)
    return np.exp(-0.5 * ((x - mu) / s) ** 2)


def cie_xyz_np(l):
    x = (1.056 * _g(l, 599.8, 37.9, 31.0) + 0.362 * _g(l, 442.0, 16.0, 26.7)
         - 0.065 * _g(l, 501.1, 20.4, 26.2))
    y = 0.821 * _g(l, 568.8, 46.9, 40.5) + 0.286 * _g(l, 530.9, 16.3, 31.1)
    z = 1.217 * _g(l, 437.0, 11.8, 36.0) + 0.681 * _g(l, 459.0, 26.0, 13.8)
    return x, y, z


def blackbody(l_nm, T):
    l = np.asarray(l_nm, float) * 1e-9
    return 1.0 / (l ** 5 * np.expm1(1.438777e-2 / (l * T)))


def emission(T):
    lf = np.arange(380.0, 781.0)
    s = blackbody(lf, T)
    xf, yf, zf = cie_xyz_np(lf)
    Y = (s * yf).sum()
    xyz = np.array([(s * xf).sum(), Y, (s * zf).sum()]) / Y
    return blackbody(LAMBDA, T) / Y, xyz


def const(v):
    return np.full(NS, float(v))


def sig(edge, width, lo, hi):
    return lo + (hi - lo) / (1.0 + np.exp(-(LAMBDA - edge) / width))


def bump(center, width, lo, hi):
    return lo + (hi - lo) * np.exp(-0.5 * ((LAMBDA - center) / width) ** 2)


U = np.uint64
K1, K2, K3 = U(0x9E3779B97F4A7C15), U(0xBF58476D1CE4E5B9), U(0x94D049BB133111EB)
RM = U(2685821657736338717)
S7, S11, S13, S17, S27, S30, S31 = U(7), U(11), U(13), U(17), U(27), U(30), U(31)
INV53 = 1.0 / 9007199254740992.0
FAST = dict(cache=True, error_model='numpy', fastmath=True)


@njit(cache=True)
def mix(a, b, c):
    z = U(a) * K1 + U(b) * K2 + U(c) * K3
    z = (z ^ (z >> S30)) * K2
    z = (z ^ (z >> S27)) * K3
    z = z ^ (z >> S31)
    if z == U(0):
        z = U(88172645463325252)
    return z


@njit(cache=True)
def rnd(s):
    x = s[0]
    x ^= x << S13
    x ^= x >> S7
    x ^= x << S17
    s[0] = x
    return ((x * RM) >> S11) * INV53


@njit(**FAST)
def cie_xyz(l):
    s = 37.9 if l < 599.8 else 31.0
    a = math.exp(-0.5 * ((l - 599.8) / s) ** 2)
    s = 16.0 if l < 442.0 else 26.7
    b = math.exp(-0.5 * ((l - 442.0) / s) ** 2)
    s = 20.4 if l < 501.1 else 26.2
    c = math.exp(-0.5 * ((l - 501.1) / s) ** 2)
    x = 1.056 * a + 0.362 * b - 0.065 * c
    s = 46.9 if l < 568.8 else 40.5
    a = math.exp(-0.5 * ((l - 568.8) / s) ** 2)
    s = 16.3 if l < 530.9 else 31.1
    b = math.exp(-0.5 * ((l - 530.9) / s) ** 2)
    y = 0.821 * a + 0.286 * b
    s = 11.8 if l < 437.0 else 36.0
    a = math.exp(-0.5 * ((l - 437.0) / s) ** 2)
    s = 26.0 if l < 459.0 else 13.8
    b = math.exp(-0.5 * ((l - 459.0) / s) ** 2)
    z = 1.217 * a + 0.681 * b
    return x, y, z


@njit(**FAST)
def spec_lookup(tab, lam):
    f = (lam - 380.0) / 5.0
    if f <= 0.0:
        return tab[0]
    i = int(f)
    if i >= tab.shape[0] - 1:
        return tab[tab.shape[0] - 1]
    t = f - i
    return tab[i] * (1.0 - t) + tab[i + 1] * t


@njit(cache=True, error_model='numpy')
def box_area(mn, mx):
    dx = mx[0] - mn[0]
    dy = mx[1] - mn[1]
    dz = mx[2] - mn[2]
    return 2.0 * (dx * dy + dy * dz + dz * dx)


@njit(cache=True, error_model='numpy')
def build_bvh(tmin, tmaxa, cent):
    """Binned-SAH build. Interior nodes have count 0 and children left[i], left[i]+1."""
    n = tmin.shape[0]
    NB = 16
    idx = np.arange(n)
    cap = 2 * n + 2
    nmin = np.zeros((cap, 3))
    nmax = np.zeros((cap, 3))
    left = np.zeros(cap, np.int32)
    first = np.zeros(cap, np.int32)
    count = np.zeros(cap, np.int32)
    stack = np.zeros(256, np.int32)
    count[0] = n
    used = 1
    sp = 1
    while sp > 0:
        sp -= 1
        node = stack[sp]
        f = first[node]
        c = count[node]
        mn = np.full(3, 1e30)
        mx = np.full(3, -1e30)
        for k in range(f, f + c):
            i = idx[k]
            for a in range(3):
                if tmin[i, a] < mn[a]:
                    mn[a] = tmin[i, a]
                if tmaxa[i, a] > mx[a]:
                    mx[a] = tmaxa[i, a]
        for a in range(3):
            nmin[node, a] = mn[a]
            nmax[node, a] = mx[a]
        if c <= 1:
            continue
        parent_area = box_area(mn, mx)
        best_cost = 1e30
        best_axis = -1
        best_split = 0.0
        for axis in range(3):
            cmn = 1e30
            cmx = -1e30
            for k in range(f, f + c):
                v = cent[idx[k], axis]
                if v < cmn:
                    cmn = v
                if v > cmx:
                    cmx = v
            if cmx - cmn < 1e-9:
                continue
            scale = NB / (cmx - cmn)
            bcnt = np.zeros(NB, np.int64)
            bmn = np.full((NB, 3), 1e30)
            bmx = np.full((NB, 3), -1e30)
            for k in range(f, f + c):
                i = idx[k]
                b = int((cent[i, axis] - cmn) * scale)
                if b >= NB:
                    b = NB - 1
                bcnt[b] += 1
                for a in range(3):
                    if tmin[i, a] < bmn[b, a]:
                        bmn[b, a] = tmin[i, a]
                    if tmaxa[i, a] > bmx[b, a]:
                        bmx[b, a] = tmaxa[i, a]
            lcnt = np.zeros(NB - 1, np.int64)
            larea = np.zeros(NB - 1)
            amn = np.full(3, 1e30)
            amx = np.full(3, -1e30)
            run = 0
            for b in range(NB - 1):
                run += bcnt[b]
                if bcnt[b] > 0:
                    for a in range(3):
                        amn[a] = min(amn[a], bmn[b, a])
                        amx[a] = max(amx[a], bmx[b, a])
                lcnt[b] = run
                if run > 0:
                    larea[b] = box_area(amn, amx)
            rmn = np.full(3, 1e30)
            rmx = np.full(3, -1e30)
            rc = 0
            for b in range(NB - 1, 0, -1):
                rc += bcnt[b]
                if bcnt[b] > 0:
                    for a in range(3):
                        rmn[a] = min(rmn[a], bmn[b, a])
                        rmx[a] = max(rmx[a], bmx[b, a])
                if rc > 0 and lcnt[b - 1] > 0:
                    cost = lcnt[b - 1] * larea[b - 1] + rc * box_area(rmn, rmx)
                    if cost < best_cost:
                        best_cost = cost
                        best_axis = axis
                        best_split = cmn + b / scale
        if best_axis < 0:
            continue
        if 0.5 + best_cost / parent_area >= c and c <= 8:
            continue
        i = f
        j = f + c - 1
        while i <= j:
            if cent[idx[i], best_axis] < best_split:
                i += 1
            else:
                tmp = idx[i]
                idx[i] = idx[j]
                idx[j] = tmp
                j -= 1
        lc = i - f
        if lc == 0 or lc == c:
            seg = idx[f:f + c].copy()
            order = np.argsort(cent[seg, best_axis])
            idx[f:f + c] = seg[order]
            lc = c // 2
        l = used
        used += 2
        left[node] = l
        count[node] = 0
        first[l] = f
        count[l] = lc
        first[l + 1] = f + lc
        count[l + 1] = c - lc
        stack[sp] = l
        sp += 1
        stack[sp] = l + 1
        sp += 1
    return nmin[:used].copy(), nmax[:used].copy(), left[:used].copy(), \
        first[:used].copy(), count[:used].copy(), idx, used


@njit(**FAST)
def tri_isect(ox, oy, oz, dx, dy, dz, tri, i, tmax):
    e1x = tri[i, 3]
    e1y = tri[i, 4]
    e1z = tri[i, 5]
    e2x = tri[i, 6]
    e2y = tri[i, 7]
    e2z = tri[i, 8]
    px = dy * e2z - dz * e2y
    py = dz * e2x - dx * e2z
    pz = dx * e2y - dy * e2x
    det = e1x * px + e1y * py + e1z * pz
    if abs(det) < 1e-14:
        return -1.0, 0.0, 0.0
    inv = 1.0 / det
    tx = ox - tri[i, 0]
    ty = oy - tri[i, 1]
    tz = oz - tri[i, 2]
    u = (tx * px + ty * py + tz * pz) * inv
    if u < 0.0 or u > 1.0:
        return -1.0, 0.0, 0.0
    qx = ty * e1z - tz * e1y
    qy = tz * e1x - tx * e1z
    qz = tx * e1y - ty * e1x
    v = (dx * qx + dy * qy + dz * qz) * inv
    if v < 0.0 or u + v > 1.0:
        return -1.0, 0.0, 0.0
    t = (e2x * qx + e2y * qy + e2z * qz) * inv
    if t < 1e-7 or t >= tmax:
        return -1.0, 0.0, 0.0
    return t, u, v


@njit(**FAST)
def box_hit(ox, oy, oz, ix, iy, iz, bb, node, tmax):
    t0 = (bb[node, 0] - ox) * ix
    t1 = (bb[node, 3] - ox) * ix
    tn = min(t0, t1)
    tf = max(t0, t1)
    t0 = (bb[node, 1] - oy) * iy
    t1 = (bb[node, 4] - oy) * iy
    tn = max(tn, min(t0, t1))
    tf = min(tf, max(t0, t1))
    t0 = (bb[node, 2] - oz) * iz
    t1 = (bb[node, 5] - oz) * iz
    tn = max(tn, min(t0, t1))
    tf = min(tf, max(t0, t1))
    tn = max(tn, 0.0)
    tf = min(tf, tmax)
    if tn <= tf:
        return tn
    return 1e30


@njit(**FAST)
def intersect(ox, oy, oz, dx, dy, dz, tmax, geo, bvh, stack, tstack, brute, anyhit):
    tri, nrm, tmat = geo
    bb, bn = bvh
    best_t = tmax
    best_i = -1
    bu = 0.0
    bv = 0.0
    if brute:
        for i in range(tri.shape[0]):
            t, u, v = tri_isect(ox, oy, oz, dx, dy, dz, tri, i, best_t)
            if t > 0.0:
                best_t = t
                best_i = i
                bu = u
                bv = v
                if anyhit:
                    return best_t, best_i, bu, bv
        return best_t, best_i, bu, bv
    if abs(dx) < 1e-12:
        dx = 1e-12
    if abs(dy) < 1e-12:
        dy = 1e-12
    if abs(dz) < 1e-12:
        dz = 1e-12
    ix = 1.0 / dx
    iy = 1.0 / dy
    iz = 1.0 / dz
    tn = box_hit(ox, oy, oz, ix, iy, iz, bb, 0, best_t)
    if tn > 1e29:
        return best_t, best_i, bu, bv
    stack[0] = 0
    tstack[0] = tn
    sp = 1
    while sp > 0:
        sp -= 1
        node = stack[sp]
        if tstack[sp] >= best_t:
            continue
        a = bn[node, 0]
        c = bn[node, 1]
        if c > 0:
            for i in range(a, a + c):
                t, u, v = tri_isect(ox, oy, oz, dx, dy, dz, tri, i, best_t)
                if t > 0.0:
                    best_t = t
                    best_i = i
                    bu = u
                    bv = v
                    if anyhit:
                        return best_t, best_i, bu, bv
        else:
            dl = box_hit(ox, oy, oz, ix, iy, iz, bb, a, best_t)
            dr = box_hit(ox, oy, oz, ix, iy, iz, bb, a + 1, best_t)
            if dl < dr:
                if dr < 1e29:
                    stack[sp] = a + 1
                    tstack[sp] = dr
                    sp += 1
                if dl < 1e29:
                    stack[sp] = a
                    tstack[sp] = dl
                    sp += 1
            else:
                if dl < 1e29:
                    stack[sp] = a
                    tstack[sp] = dl
                    sp += 1
                if dr < 1e29:
                    stack[sp] = a + 1
                    tstack[sp] = dr
                    sp += 1
    return best_t, best_i, bu, bv


@njit(**FAST)
def trace(ox, oy, oz, dx, dy, dz, lam, maxdepth, geo, bvh, mats, light, env,
          stack, tstack, brute, rs):
    # material types: 0 diffuse, 1 mirror, 2 dispersive dielectric, 3 emitter, 4 thin film
    tri, nrm, tmat = geo
    mtype, mrefl, memit, mA, mB = mats
    has_light = light[14] > 0.5
    L = 0.0
    T = 1.0
    spec = True
    for depth in range(maxdepth):
        t, ti, bu, bv = intersect(ox, oy, oz, dx, dy, dz, 1e30, geo, bvh,
                                  stack, tstack, brute, False)
        if ti < 0:
            L += T * spec_lookup(env, lam)
            break
        m = tmat[ti]
        px = ox + dx * t
        py = oy + dy * t
        pz = oz + dz * t
        w = 1.0 - bu - bv
        nx = w * nrm[ti, 0] + bu * nrm[ti, 3] + bv * nrm[ti, 6]
        ny = w * nrm[ti, 1] + bu * nrm[ti, 4] + bv * nrm[ti, 7]
        nz = w * nrm[ti, 2] + bu * nrm[ti, 5] + bv * nrm[ti, 8]
        il = 1.0 / math.sqrt(nx * nx + ny * ny + nz * nz)
        nx *= il
        ny *= il
        nz *= il
        mt = mtype[m]
        cosd = dx * nx + dy * ny + dz * nz
        if mt == 3:
            if spec and cosd < 0.0:
                L += T * spec_lookup(memit[m], lam)
            break
        inside = cosd > 0.0
        if inside:
            nx = -nx
            ny = -ny
            nz = -nz
        cosi = -(dx * nx + dy * ny + dz * nz)
        ndx = 0.0
        ndy = 0.0
        ndz = 0.0
        if mt == 0:
            R = spec_lookup(mrefl[m], lam)
            if has_light:
                r1 = rnd(rs)
                r2 = rnd(rs)
                wx = light[0] + r1 * light[3] + r2 * light[6] - px
                wy = light[1] + r1 * light[4] + r2 * light[7] - py
                wz = light[2] + r1 * light[5] + r2 * light[8] - pz
                d2 = wx * wx + wy * wy + wz * wz
                dist = math.sqrt(d2)
                wx /= dist
                wy /= dist
                wz /= dist
                cx = wx * nx + wy * ny + wz * nz
                cl = -(wx * light[9] + wy * light[10] + wz * light[11])
                if cx > 0.0 and cl > 0.0:
                    st, si, su, sv = intersect(px + nx * EPS, py + ny * EPS, pz + nz * EPS,
                                               wx, wy, wz, dist - 2.0 * EPS,
                                               geo, bvh, stack, tstack, brute, True)
                    if si < 0:
                        Le = spec_lookup(memit[int(light[13])], lam)
                        L += T * R * Le * cx * cl * light[12] / (d2 * math.pi)
            T *= R
            r1 = rnd(rs)
            r2 = rnd(rs)
            rr = math.sqrt(r1)
            phi = 2.0 * math.pi * r2
            if abs(nx) > 0.9:
                ax = 0.0
                ay = 1.0
            else:
                ax = 1.0
                ay = 0.0
            tx = ay * nz
            ty = -ax * nz
            tz = ax * ny - ay * nx
            il = 1.0 / math.sqrt(tx * tx + ty * ty + tz * tz)
            tx *= il
            ty *= il
            tz *= il
            bx = ny * tz - nz * ty
            by = nz * tx - nx * tz
            bz = nx * ty - ny * tx
            lx = rr * math.cos(phi)
            ly = rr * math.sin(phi)
            lz = math.sqrt(max(0.0, 1.0 - r1))
            ndx = lx * tx + ly * bx + lz * nx
            ndy = lx * ty + ly * by + lz * ny
            ndz = lx * tz + ly * bz + lz * nz
            spec = False
        elif mt == 1:
            T *= spec_lookup(mrefl[m], lam)
            ndx = dx + 2.0 * cosi * nx
            ndy = dy + 2.0 * cosi * ny
            ndz = dz + 2.0 * cosi * nz
        elif mt == 4:
            nf = mB[m]
            ct = math.sqrt(max(0.0, 1.0 - (1.0 - cosi * cosi) / (nf * nf)))
            phase = 4.0 * math.pi * nf * mA[m] * ct / lam
            T *= 0.06 + 0.9 * (0.5 - 0.5 * math.cos(phase))
            ndx = dx + 2.0 * cosi * nx
            ndy = dy + 2.0 * cosi * ny
            ndz = dz + 2.0 * cosi * nz
        else:
            um = lam * 1e-3
            ior = mA[m] + mB[m] / (um * um)
            if inside:
                n1v = ior
                n2v = 1.0
            else:
                n1v = 1.0
                n2v = ior
            eta = n1v / n2v
            sin2t = eta * eta * (1.0 - cosi * cosi)
            do_reflect = sin2t >= 1.0
            cost = 0.0
            if not do_reflect:
                cost = math.sqrt(1.0 - sin2t)
                rs_ = (n1v * cosi - n2v * cost) / (n1v * cosi + n2v * cost)
                rp_ = (n2v * cosi - n1v * cost) / (n2v * cosi + n1v * cost)
                if rnd(rs) < 0.5 * (rs_ * rs_ + rp_ * rp_):
                    do_reflect = True
            if do_reflect:
                ndx = dx + 2.0 * cosi * nx
                ndy = dy + 2.0 * cosi * ny
                ndz = dz + 2.0 * cosi * nz
            else:
                k = eta * cosi - cost
                ndx = eta * dx + k * nx
                ndy = eta * dy + k * ny
                ndz = eta * dz + k * nz
        if mt != 0:
            spec = True
        side = EPS if ndx * nx + ndy * ny + ndz * nz >= 0.0 else -EPS
        ox = px + nx * side
        oy = py + ny * side
        oz = pz + nz * side
        dx = ndx
        dy = ndy
        dz = ndz
        if depth >= 3:
            q = min(0.95, max(0.05, T))
            if rnd(rs) > q:
                break
            T /= q
    return L


@njit(parallel=True, **FAST)
def render_frame(acc, rows, frame, seed, cam, maxdepth, clampv, brute,
                 geo, bvh, mats, light, env):
    H = acc.shape[0]
    W = acc.shape[1]
    ulo = (0.85691062 - math.tanh((538.0 - 380.0) / 138.888889)) / 1.82750197
    uhi = (0.85691062 - math.tanh((538.0 - 780.0) / 138.888889)) / 1.82750197
    aspect = W / H
    for i in prange(H):
        y = rows[i]
        stack = np.empty(128, np.int32)
        tstack = np.empty(128, np.float64)
        rs = np.empty(1, np.uint64)
        for x in range(W):
            rs[0] = mix(x, y, seed * 65537 + frame + 1)
            off = (mix(x, y, seed) >> S11) * INV53
            u = (off + frame * 0.6180339887498949) % 1.0
            uu = ulo + u * (uhi - ulo)
            lam = 538.0 - 138.888889 * math.atanh(0.85691062 - 1.82750197 * uu)
            ch = math.cosh(0.0072 * (lam - 538.0))
            pdf = 0.0039398042 / (ch * ch) / (uhi - ulo)
            px = (2.0 * (x + rnd(rs)) / W - 1.0) * cam[12] * aspect
            py = (1.0 - 2.0 * (y + rnd(rs)) / H) * cam[12]
            dx = cam[3] + px * cam[6] + py * cam[9]
            dy = cam[4] + px * cam[7] + py * cam[10]
            dz = cam[5] + px * cam[8] + py * cam[11]
            il = 1.0 / math.sqrt(dx * dx + dy * dy + dz * dz)
            dx *= il
            dy *= il
            dz *= il
            ox = cam[0]
            oy = cam[1]
            oz = cam[2]
            if cam[13] > 0.0:
                ft = cam[14] / (dx * cam[3] + dy * cam[4] + dz * cam[5])
                fx = ox + dx * ft
                fy = oy + dy * ft
                fz = oz + dz * ft
                r = cam[13] * math.sqrt(rnd(rs))
                a = 2.0 * math.pi * rnd(rs)
                lx = r * math.cos(a)
                ly = r * math.sin(a)
                ox += cam[6] * lx + cam[9] * ly
                oy += cam[7] * lx + cam[10] * ly
                oz += cam[8] * lx + cam[11] * ly
                dx = fx - ox
                dy = fy - oy
                dz = fz - oz
                il = 1.0 / math.sqrt(dx * dx + dy * dy + dz * dz)
                dx *= il
                dy *= il
                dz *= il
            Lr = trace(ox, oy, oz, dx, dy, dz, lam, maxdepth, geo, bvh, mats,
                       light, env, stack, tstack, brute, rs)
            if Lr > clampv:
                Lr = clampv
            xb, yb, zb = cie_xyz(lam)
            s = Lr / pdf
            acc[y, x, 0] += xb * s
            acc[y, x, 1] += yb * s
            acc[y, x, 2] += zb * s


@njit(cache=True, fastmath=True)
def tone(v):
    v = max(v, 0.0)
    v = (v * (2.51 * v + 0.03)) / (v * (2.43 * v + 0.59) + 0.14)
    v = min(max(v, 0.0), 1.0)
    if v <= 0.0031308:
        v = 12.92 * v
    else:
        v = 1.055 * v ** (1.0 / 2.4) - 0.055
    return np.uint8(v * 255.0 + 0.5)


@njit(parallel=True, cache=True, fastmath=True)
def tonemap(acc, m, out):
    for y in prange(acc.shape[0]):
        for x in range(acc.shape[1]):
            a = acc[y, x, 0]
            b = acc[y, x, 1]
            c = acc[y, x, 2]
            out[y, x, 0] = tone(m[0, 0] * a + m[0, 1] * b + m[0, 2] * c)
            out[y, x, 1] = tone(m[1, 0] * a + m[1, 1] * b + m[1, 2] * c)
            out[y, x, 2] = tone(m[2, 0] * a + m[2, 1] * b + m[2, 2] * c)


@njit(cache=True, error_model='numpy')
def selftest_kernel(org, dirs, geo, bvh, out):
    stack = np.empty(128, np.int32)
    tstack = np.empty(128, np.float64)
    for k in range(org.shape[0]):
        for brute in range(2):
            t, i, a, b = intersect(org[k, 0], org[k, 1], org[k, 2], dirs[k, 0], dirs[k, 1],
                                   dirs[k, 2], 1e30, geo, bvh, stack, tstack, brute == 1, False)
            out[k, brute] = t


KINDS = {'diffuse': 0, 'mirror': 1, 'glass': 2, 'light': 3, 'film': 4}
_ICO = {}


def icosphere(level):
    if level in _ICO:
        return _ICO[level]
    t = (1 + 5 ** 0.5) / 2
    verts = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t),
             (0, -1, -t), (0, 1, -t), (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
    verts = [np.array(v, float) / np.linalg.norm(v) for v in verts]
    faces = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9),
             (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8), (3, 9, 4), (3, 4, 2),
             (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10),
             (8, 6, 7), (9, 8, 1)]
    cache = {}

    def mid(i, j):
        key = (min(i, j), max(i, j))
        if key not in cache:
            p = verts[i] + verts[j]
            verts.append(p / np.linalg.norm(p))
            cache[key] = len(verts) - 1
        return cache[key]

    for _ in range(level):
        new = []
        for a, b, c in faces:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            new += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        faces = new
    out = (np.array(verts), np.array(faces))
    _ICO[level] = out
    return out


def bvh_depth(left, count):
    depth, fr = 0, np.array([0])
    while True:
        inner = fr[count[fr] == 0]
        if inner.size == 0:
            return depth
        fr = np.concatenate([left[inner], left[inner] + 1])
        depth += 1


class SceneData:
    def __init__(self, name):
        self.name = name
        self.parts = []
        self.mt, self.mr, self.mA0, self.mB0 = [], [], [], []
        self.light_mat, self.glass, self.film = -1, [], []
        self.light_q = None
        self.light_intensity = 20.0
        self.env_intensity = 0.0
        self.cam = dict(target=(0, 1, 0), yaw=0.0, pitch=0.0, dist=8.0, fov=40.0)

    def material(self, kind, refl=None, A=0.0, B=0.0):
        self.mt.append(KINDS[kind])
        self.mr.append(np.zeros(NS) if refl is None else np.clip(refl, 0, 1))
        self.mA0.append(A)
        self.mB0.append(B)
        mid = len(self.mt) - 1
        if kind == 'light':
            self.light_mat = mid
        elif kind == 'glass':
            self.glass.append(mid)
        elif kind == 'film':
            self.film.append(mid)
        return mid

    def _add(self, a, b, c, na, nb, nc, m):
        a, b, c = (np.asarray(x, float).reshape(-1, 3) for x in (a, b, c))
        na, nb, nc = (np.asarray(x, float).reshape(-1, 3) for x in (na, nb, nc))
        mm = np.broadcast_to(np.asarray(m, np.int32), (len(a),)).copy()
        self.parts.append((a, b, c, na, nb, nc, mm))

    def quad(self, p0, p1, p2, p3, n, m):
        n = np.array(n, float)
        n /= np.linalg.norm(n)
        P = [np.array(p, float) for p in (p0, p1, p2, p3)]
        nn = np.tile(n, (2, 1))
        self._add([P[0], P[0]], [P[1], P[2]], [P[2], P[3]], nn, nn, nn, m)

    def sphere(self, c, r, m, level):
        verts, faces = icosphere(level)
        F = verts[faces]
        c = np.array(c, float)
        self._add(c + r * F[:, 0], c + r * F[:, 1], c + r * F[:, 2],
                  F[:, 0], F[:, 1], F[:, 2], m)

    def solid(self, tris, m):
        T = np.array(tris, float)
        centre = T.reshape(-1, 3).mean(0)
        n = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
        n /= np.linalg.norm(n, axis=1, keepdims=True)
        flip = ((T.mean(1) - centre) * n).sum(1) < 0
        n[flip] *= -1
        self._add(T[:, 0], T[:, 1], T[:, 2], n, n, n, m)

    def prism(self, base, rot_y, m, side=1.8, depth=1.4):
        h = side * math.sqrt(3) / 2
        c, s = math.cos(rot_y), math.sin(rot_y)

        def tf(x, y, z):
            return np.array([c * x + s * z, y, -s * x + c * z]) + np.array(base, float)

        f = [tf(x, y, depth / 2) for x, y in [(-side / 2, 0), (side / 2, 0), (0, h)]]
        b = [tf(x, y, -depth / 2) for x, y in [(-side / 2, 0), (side / 2, 0), (0, h)]]
        tris = [(f[0], f[1], f[2]), (b[0], b[1], b[2])]
        for i in range(3):
            j = (i + 1) % 3
            tris += [(f[i], f[j], b[j]), (f[i], b[j], b[i])]
        self.solid(tris, m)

    def set_light(self, corner, u, v, m):
        corner, u, v = (np.array(x, float) for x in (corner, u, v))
        nrm = np.array([0.0, -1.0, 0.0])
        self.quad(corner, corner + u, corner + u + v, corner + v, nrm, m)
        self.light_q = (corner, u, v, nrm, np.linalg.norm(np.cross(u, v)), m)

    def finalize(self):
        cat = [np.concatenate([p[i] for p in self.parts]) for i in range(7)]
        v0, v1, v2, n0, n1, n2, tm = cat
        mn = np.minimum(np.minimum(v0, v1), v2)
        mx = np.maximum(np.maximum(v0, v1), v2)
        nmin, nmax, left, first, count, idx, used = build_bvh(mn, mx, (v0 + v1 + v2) / 3.0)
        f64 = lambda a: np.ascontiguousarray(a, np.float64)
        i32 = lambda a: np.ascontiguousarray(a, np.int32)
        tri = np.hstack([v0, v1 - v0, v2 - v0])[idx]
        nrm = np.hstack([n0, n1, n2])[idx]
        self.geo = (f64(tri), f64(nrm), i32(tm[idx]))
        self.bvh = (f64(np.hstack([nmin, nmax])),
                    i32(np.stack([np.where(count > 0, first, left), count], 1)))
        self.n_tris = len(tri)
        self.n_nodes = used
        self.n_leaves = int((count > 0).sum())
        self.depth = bvh_depth(left, count)
        nm = len(self.mt)
        self.mtype = i32(self.mt)
        self.mrefl = f64(np.array(self.mr))
        self.memit = np.zeros((nm, NS))
        self.mA = f64(self.mA0)
        self.mB = f64(self.mB0)
        self.env = np.zeros(NS)
        self.mats = (self.mtype, self.mrefl, self.memit, self.mA, self.mB)
        L = np.zeros(15)
        if self.light_q is not None:
            c, u, v, n, area, m = self.light_q
            L[0:3], L[3:6], L[6:9], L[9:12] = c, u, v, n
            L[12], L[13], L[14] = area, m, 1.0
        self.light = L
        return self

    def apply(self, temp, dispersion, film_nm):
        for m in self.glass:
            self.mB[m] = self.mB0[m] * dispersion
        for m in self.film:
            self.mA[m] = film_nm
        if self.light_mat >= 0:
            self.memit[self.light_mat] = emission(temp)[0] * self.light_intensity
        self.env[:] = emission(9500.0)[0] * self.env_intensity


def scene_cornell():
    S = SceneData('cornell')
    white = S.material('diffuse', const(0.74))
    red = S.material('diffuse', sig(600, 12, 0.04, 0.68))
    green = S.material('diffuse', bump(535, 38, 0.05, 0.55))
    gold = S.material('mirror', sig(510, 25, 0.35, 0.93))
    glass = S.material('glass', A=1.5046, B=0.0042)
    film = S.material('film', A=420.0, B=1.33)
    lamp = S.material('light')
    x0, x1, y0, y1, z0, z1 = -2, 2, 0, 4, -2, 2
    S.quad((x0, y0, z0), (x0, y0, z1), (x1, y0, z1), (x1, y0, z0), (0, 1, 0), white)
    S.quad((x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1), (0, -1, 0), white)
    S.quad((x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0), (0, 0, 1), white)
    S.quad((x0, y0, z0), (x0, y1, z0), (x0, y1, z1), (x0, y0, z1), (1, 0, 0), red)
    S.quad((x1, y0, z0), (x1, y0, z1), (x1, y1, z1), (x1, y1, z0), (-1, 0, 0), green)
    S.set_light((-0.7, 3.99, -0.7), (1.4, 0, 0), (0, 0, 1.4), lamp)
    S.sphere((-0.85, 0.85, 0.2), 0.85, glass, 4)
    S.sphere((0.95, 0.7, -0.7), 0.7, gold, 3)
    S.sphere((0.7, 0.42, 1.0), 0.42, film, 3)
    S.light_intensity = 20.0
    S.cam = dict(target=(0, 2, 0), yaw=0.0, pitch=0.04, dist=8.3, fov=38.0)
    return S.finalize()


def scene_studio():
    S = SceneData('studio')
    cream = S.material('diffuse', const(0.78))
    slate = S.material('diffuse', bump(470, 60, 0.04, 0.22))
    gold = S.material('mirror', sig(510, 25, 0.35, 0.93))
    glass = S.material('glass', A=1.5046, B=0.0042)
    film = S.material('film', A=420.0, B=1.33)
    lamp = S.material('light')
    n = 7
    for i in range(-n, n):
        for j in range(-n, n):
            S.quad((i, 0, j), (i, 0, j + 1), (i + 1, 0, j + 1), (i + 1, 0, j), (0, 1, 0),
                   cream if (i + j) % 2 == 0 else slate)
    S.set_light((0.5, 6.0, -2.0), (2.2, 0, 0), (0, 0, 2.2), lamp)
    S.sphere((1.7, 1.0, 0.3), 1.0, glass, 4)
    S.prism((-1.9, 0.0, 0.4), math.radians(25), glass)
    S.sphere((-0.3, 0.6, -1.9), 0.6, gold, 3)
    S.sphere((-0.2, 0.55, 2.0), 0.55, film, 3)
    S.light_intensity = 45.0
    S.env_intensity = 0.65
    S.cam = dict(target=(0, 0.8, 0), yaw=0.0, pitch=0.27, dist=9.0, fov=38.0)
    return S.finalize()


SCENES = {'cornell': scene_cornell, 'studio': scene_studio}


class Camera:
    def __init__(self, target, yaw, pitch, dist, fov):
        self.target = np.array(target, float)
        self.yaw, self.pitch, self.dist, self.fov = yaw, pitch, dist, fov
        self._home = (self.target.copy(), yaw, pitch, dist)

    def home(self):
        self.target, self.yaw, self.pitch, self.dist = (
            self._home[0].copy(), *self._home[1:])

    def basis(self):
        cp = math.cos(self.pitch)
        eye = self.target + self.dist * np.array(
            [cp * math.sin(self.yaw), math.sin(self.pitch), cp * math.cos(self.yaw)])
        f = self.target - eye
        f /= np.linalg.norm(f)
        right = np.cross(f, [0, 1, 0])
        right /= np.linalg.norm(right)
        return eye, f, right, np.cross(right, f)

    def array(self, aperture):
        eye, f, r, u = self.basis()
        return np.array([*eye, *f, *r, *u, math.tan(math.radians(self.fov) / 2),
                         aperture, self.dist])

    def pan(self, dx, dy, view_h):
        _, _, r, u = self.basis()
        k = 2 * self.dist * math.tan(math.radians(self.fov) / 2) / view_h
        self.target += (-dx * r + dy * u) * k


class Renderer:
    def __init__(self, scene, w, h):
        self.params = dict(temp=6500.0, dispersion=6.0, film=420.0,
                           aperture=0.0, depth=8, brute=False)
        self.load_scene(scene)
        self.resize(w, h)

    def load_scene(self, name):
        self.scene = SCENES[name]()
        c = self.scene.cam
        self.camera = Camera(c['target'], c['yaw'], c['pitch'], c['dist'], c['fov'])
        self.name = name
        self.update_params()

    def update_params(self, **kw):
        self.params.update(kw)
        p = self.params
        self.scene.apply(p['temp'], p['dispersion'], p['film'])
        rgb = XYZ2RGB @ emission(p['temp'])[1]
        self.wb = rgb[1] / rgb
        self.reset()

    def resize(self, w, h):
        self.w, self.h = w, h
        self.acc = np.zeros((h, w, 3))
        self.out = np.empty((h, w, 3), np.uint8)
        self.rows = np.random.permutation(h).astype(np.int64)
        self.reset()

    def reset(self):
        if hasattr(self, 'acc'):
            self.acc[:] = 0.0
        self.frames = 0
        self.seed = int(np.random.randint(1, 1 << 30))

    def step(self):
        s, p = self.scene, self.params
        render_frame(self.acc, self.rows, self.frames, self.seed,
                     self.camera.array(p['aperture']), int(p['depth']), 100.0,
                     bool(p['brute']), s.geo, s.bvh, s.mats, s.light, s.env)
        self.frames += 1

    def image(self, exposure=0.0, wb=False):
        m = XYZ2RGB * (self.wb[:, None] if wb else 1.0) * (2.0 ** exposure / max(1, self.frames))
        tonemap(self.acc, np.ascontiguousarray(m), self.out)
        return self.out

    def info(self):
        s = self.scene
        return (f"{s.n_tris:,} triangles | BVH: {s.n_nodes:,} nodes, {s.n_leaves:,} leaves, "
                f"depth {s.depth}")


def warmup(renderer):
    w, h = renderer.w, renderer.h
    renderer.resize(16, 12)
    renderer.step()
    renderer.image()
    renderer.resize(w, h)


def save_png(path, img):
    h, w, _ = img.shape
    raw = b''.join(b'\x00' + img[y].tobytes() for y in range(h))

    def chunk(t, d):
        return struct.pack('>I', len(d)) + t + d + struct.pack('>I', zlib.crc32(t + d) & 0xffffffff)

    with open(path, 'wb') as f:
        f.write(b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0))
                + chunk(b'IDAT', zlib.compress(raw, 6)) + chunk(b'IEND', b''))


def run_selftest(scene):
    r = Renderer(scene, 8, 8)
    s = r.scene
    rng = np.random.default_rng(1)
    n = 3000
    c = np.array(s.cam['target'])
    org = c + rng.normal(size=(n, 3)) * 5.0
    dirs = (c + rng.normal(size=(n, 3)) * 1.5) - org
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    out = np.zeros((n, 2))
    selftest_kernel(org, dirs, s.geo, s.bvh, out)
    hit = out[:, 0] < 1e29
    bad = np.abs(out[:, 0] - out[:, 1]) > 1e-9
    print(f"[{scene}] {n} random rays, {hit.sum()} hits, {bad.sum()} mismatches "
          f"-> {'PASS' if bad.sum() == 0 else 'FAIL'}")
    return bad.sum() == 0


def run_bench(scene):
    r = Renderer(scene, 160, 120)
    print(r.info())
    warmup(r)
    for brute in (False, True):
        r.update_params(brute=brute)
        t0 = time.perf_counter()
        for _ in range(2):
            r.step()
        dt = (time.perf_counter() - t0) / 2
        print(f"  {'brute force' if brute else 'BVH':12s} {dt * 1e3:9.1f} ms / frame "
              f"({160 * 120 / dt / 1e6:.2f} Mpaths/s)")
        if not brute:
            t_bvh = dt
    print(f"  BVH speed-up: {dt / t_bvh:.1f}x with {r.scene.n_tris:,} triangles")


def run_headless(a):
    r = Renderer(a.scene, a.width, a.height)
    r.update_params(temp=a.temp, dispersion=a.dispersion, depth=a.depth, brute=False)
    print(r.info())
    print("compiling kernels (first run only)...")
    warmup(r)
    t0 = time.perf_counter()
    for i in range(a.spp):
        r.step()
        if (i + 1) % max(1, a.spp // 10) == 0:
            print(f"  {i + 1}/{a.spp} spp  ({time.perf_counter() - t0:.1f}s)", flush=True)
    save_png(a.out, r.image(a.exposure, a.wb))
    print("saved", a.out)


def run_gui(a):
    import tkinter as tk
    from tkinter import ttk

    DW, DH = 640, 480
    BG, PANEL, FG, DIM, TROUGH, BTN = '#16181c', '#1e2127', '#d7dae0', '#7f8590', '#343a44', '#2b3038'
    FONT, SMALL = ('Helvetica', 10), ('Helvetica', 9)
    RES = {"Full": 1, "1/2": 2, "1/4": 4, "1/8": 8}

    class Slider:
        def __init__(self, parent, label, lo, hi, init, step, cmd, fmt='{:.0f}'):
            self.step, self.fmt, self.cmd, self.last = step, fmt, cmd, None
            f = tk.Frame(parent, bg=PANEL)
            f.pack(fill='x', pady=(0, 8))
            top = tk.Frame(f, bg=PANEL)
            top.pack(fill='x')
            tk.Label(top, text=label, bg=PANEL, fg=FG, font=FONT).pack(side='left')
            self.val = tk.Label(top, text=fmt.format(init), bg=PANEL, fg=DIM, font=FONT)
            self.val.pack(side='right')
            self.var = tk.DoubleVar(value=init)
            ttk.Scale(f, from_=lo, to=hi, variable=self.var, command=self._chg,
                      style='Dark.Horizontal.TScale').pack(fill='x')

        def get(self):
            return round(self.var.get() / self.step) * self.step

        def _chg(self, _):
            v = self.get()
            if v == self.last:
                return
            self.last = v
            self.val.config(text=self.fmt.format(v))
            self.cmd()

    class App:
        def __init__(self, root):
            self.root = root
            root.title("Spectral Path Tracer")
            root.configure(bg=BG)
            root.resizable(False, False)
            self.scale = self.cur = 2
            self.pv = 4
            print("Building scene + compiling kernels (first run takes a few seconds)...")
            self.r = Renderer(a.scene, DW // 2, DH // 2)
            warmup(self.r)
            self.paused, self.drag, self.photo = False, None, None
            self.last_move, self.ms, self.max_spp = 0.0, 0.0, 4096
            self.flash_until, self.flash_text = 0.0, ''

            st = ttk.Style()
            st.theme_use('clam')
            st.configure('Dark.Horizontal.TScale', background='#6ea8fe', troughcolor=TROUGH,
                         lightcolor='#6ea8fe', darkcolor='#6ea8fe', bordercolor=PANEL,
                         sliderlength=14, troughrelief='flat')
            st.map('Dark.Horizontal.TScale', background=[('active', '#8fbcff')])
            st.configure('TCombobox', fieldbackground=BTN, background=BTN, foreground=FG,
                         arrowcolor=FG, bordercolor=BTN, lightcolor=BTN, darkcolor=BTN,
                         selectbackground=BTN, selectforeground=FG)
            st.map('TCombobox', fieldbackground=[('readonly', BTN)], foreground=[('readonly', FG)])
            st.configure('Dark.TCheckbutton', background=PANEL, foreground=FG, font=FONT)
            st.map('Dark.TCheckbutton', background=[('active', PANEL)])
            root.option_add('*TCombobox*Listbox.background', BTN)
            root.option_add('*TCombobox*Listbox.foreground', FG)

            self.canvas = tk.Canvas(root, width=DW, height=DH, bg='black', highlightthickness=0)
            self.canvas.grid(row=0, column=0)
            self.item = self.canvas.create_image(0, 0, anchor='nw')
            self.hud = self._text(12, DH - 10, 'sw', FG)
            self.hint = self._text(DW // 2, 16, 'n', '#b8bcc4')
            self._set(self.hint, 'drag to orbit  ·  right-drag to pan  ·  scroll to zoom')

            panel = tk.Frame(root, bg=PANEL, width=240)
            panel.grid(row=0, column=1, sticky='ns')
            panel.pack_propagate(False)
            body = tk.Frame(panel, bg=PANEL)
            body.pack(fill='both', expand=True, padx=16, pady=14)

            row = tk.Frame(body, bg=PANEL)
            row.pack(fill='x', pady=(0, 12))
            self.scene_var = tk.StringVar(value=a.scene)
            self.res_var = tk.StringVar(value="1/2")
            for var, vals, w, cb in ((self.scene_var, list(SCENES), 9, self.on_scene),
                                     (self.res_var, list(RES), 6, self.on_scale)):
                c = ttk.Combobox(row, textvariable=var, values=vals, width=w, state='readonly')
                c.pack(side='left', padx=(0, 6))
                c.bind('<<ComboboxSelected>>', cb)

            self.s_temp = Slider(body, "Lamp temperature", 2000, 12000, 6500, 100, self.on_param, '{:.0f} K')
            self.s_disp = Slider(body, "Dispersion", 0, 25, 6, 0.5, self.on_param, '{:.1f}×')
            self.s_film = Slider(body, "Film thickness", 100, 900, 420, 10, self.on_param, '{:.0f} nm')
            self.s_exp = Slider(body, "Exposure", -4, 4, 0, 0.1, self.push, '{:+.1f} EV')

            btns = tk.Frame(body, bg=PANEL)
            btns.pack(fill='x', pady=(4, 10))
            for text, cmd in (("Pause", self.toggle), ("Save", self.save), ("Reset view", self.home)):
                b = tk.Button(btns, text=text, command=cmd, bg=BTN, fg=FG, bd=0, font=SMALL,
                              activebackground=TROUGH, activeforeground=FG, padx=9, pady=4,
                              highlightthickness=0, relief='flat')
                b.pack(side='left', padx=(0, 6))
                if text == "Pause":
                    self.pause_btn = b

            self.adv_open = False
            self.adv_btn = tk.Button(body, text="▸  Advanced", command=self.toggle_adv, bg=PANEL,
                                     fg=DIM, bd=0, font=SMALL, activebackground=PANEL,
                                     activeforeground=FG, anchor='w', highlightthickness=0)
            self.adv_btn.pack(fill='x')
            self.adv = tk.Frame(body, bg=PANEL)
            self.s_ap = Slider(self.adv, "Aperture (DoF)", 0, 0.5, 0, 0.01, self.on_param, '{:.2f}')
            self.s_depth = Slider(self.adv, "Max bounces", 1, 16, 8, 1, self.on_param)
            self.wb = tk.BooleanVar(value=False)
            self.brute = tk.BooleanVar(value=False)
            ttk.Checkbutton(self.adv, text="White-balance the lamp", variable=self.wb,
                            style='Dark.TCheckbutton', command=self.push).pack(anchor='w', pady=2)
            ttk.Checkbutton(self.adv, text="Brute force (no BVH)", variable=self.brute,
                            style='Dark.TCheckbutton', command=self.on_param).pack(anchor='w', pady=2)

            self.info = tk.Label(panel, bg=PANEL, fg=DIM, font=('Helvetica', 8), justify='left',
                                 wraplength=210, anchor='w', text=self.r.info())
            self.info.pack(side='bottom', fill='x', padx=16, pady=(0, 12))

            c = self.canvas
            c.bind('<ButtonPress-1>', lambda e: self.press(e, 'orbit'))
            c.bind('<ButtonPress-2>', lambda e: self.press(e, 'pan'))
            c.bind('<ButtonPress-3>', lambda e: self.press(e, 'pan'))
            for b in ('<B1-Motion>', '<B2-Motion>', '<B3-Motion>'):
                c.bind(b, self.motion)
            c.bind('<MouseWheel>', lambda e: self.zoom(0.9 if e.delta > 0 else 1.1))
            c.bind('<Button-4>', lambda e: self.zoom(0.9))
            c.bind('<Button-5>', lambda e: self.zoom(1.1))
            root.bind('<space>', lambda e: self.toggle())
            root.bind('s', lambda e: self.save())
            root.bind('r', lambda e: self.home())
            root.bind('<ButtonRelease>', lambda e: setattr(self, 'drag', None))
            self.tick()

        def _text(self, x, y, anchor, color):
            sh = self.canvas.create_text(x + 1, y + 1, anchor=anchor, fill='black', font=SMALL)
            fg = self.canvas.create_text(x, y, anchor=anchor, fill=color, font=SMALL)
            return (sh, fg)

        def _set(self, item, text):
            for i in item:
                self.canvas.itemconfig(i, text=text)

        def toggle_adv(self):
            self.adv_open = not self.adv_open
            self.adv_btn.config(text=("▾  Advanced" if self.adv_open else "▸  Advanced"))
            if self.adv_open:
                self.adv.pack(fill='x', pady=(8, 0), after=self.adv_btn)
            else:
                self.adv.pack_forget()

        def moved(self):
            self.last_move = time.time()
            self.r.reset()
            if self.hint:
                for i in self.hint:
                    self.canvas.delete(i)
                self.hint = None

        def press(self, e, mode):
            self.drag = (mode, e.x, e.y)
            self.canvas.focus_set()

        def motion(self, e):
            if not self.drag:
                return
            mode, x, y = self.drag
            dx, dy = e.x - x, e.y - y
            cam = self.r.camera
            if mode == 'orbit':
                cam.yaw -= dx * 0.008
                cam.pitch = max(-1.5, min(1.5, cam.pitch + dy * 0.008))
            else:
                cam.pan(dx, dy, DH)
            self.drag = (mode, e.x, e.y)
            self.moved()

        def zoom(self, f):
            cam = self.r.camera
            cam.dist = max(1.0, min(40.0, cam.dist * f))
            self.moved()

        def home(self):
            self.r.camera.home()
            self.moved()

        def toggle(self):
            self.paused = not self.paused
            self.pause_btn.config(text="Resume" if self.paused else "Pause")

        def on_param(self, *_):
            self.r.update_params(depth=self.s_depth.get(), temp=self.s_temp.get(),
                                 dispersion=self.s_disp.get(), film=self.s_film.get(),
                                 aperture=self.s_ap.get(), brute=self.brute.get())

        def on_scale(self, *_):
            self.scale = RES[self.res_var.get()]
            self.pv = min(8, self.scale * 2)
            self.canvas.focus_set()

        def on_scene(self, *_):
            self.r.load_scene(self.scene_var.get())
            self.on_param()
            self.info.config(text=self.r.info())
            self.moved()
            self.canvas.focus_set()

        def save(self):
            name = f"render_{self.r.name}_{self.r.frames}spp.png"
            save_png(name, self.r.image(self.s_exp.get(), self.wb.get()))
            self.flash_text, self.flash_until = f"saved {name}", time.time() + 2.5

        def push(self, *_):
            if self.r.frames == 0:
                return
            img = self.r.image(self.s_exp.get(), self.wb.get())
            h, w, _ = img.shape
            ph = tk.PhotoImage(data=b'P6 %d %d 255\n' % (w, h) + img.tobytes(), format='PPM')
            k = DW // w
            self.photo = ph.zoom(k) if k > 1 else ph
            self.canvas.itemconfig(self.item, image=self.photo)

        def tick(self):
            moving = (time.time() - self.last_move) < 0.3 or self.drag is not None
            want = self.pv if moving else self.scale
            if want != self.cur:
                self.cur = want
                self.r.resize(DW // want, DH // want)
            if not self.paused and (moving or self.r.frames < self.max_spp):
                t0 = time.perf_counter()
                n = 0
                while True:
                    self.r.step()
                    n += 1
                    dt = time.perf_counter() - t0
                    if moving or dt > 0.04 or self.r.frames >= self.max_spp:
                        break
                self.ms = dt * 1e3 / n
                self.push()
                if moving:
                    if self.ms > 80 and self.pv < 8:
                        self.pv *= 2
                    elif self.ms < 15 and self.pv > self.scale:
                        self.pv //= 2
            if time.time() < self.flash_until:
                txt = self.flash_text
            else:
                txt = (f"{self.r.frames} spp  ·  {self.ms:.0f} ms  ·  {self.r.w}×{self.r.h}"
                       + ("  ·  brute force" if self.brute.get() else "")
                       + ("  ·  paused" if self.paused else ""))
            self._set(self.hud, txt)
            self.root.after(1, self.tick)

    root = tk.Tk()
    App(root)
    root.mainloop()


def main():
    ap = argparse.ArgumentParser(description="Spectral BVH path tracer")
    ap.add_argument('--scene', default='cornell', choices=list(SCENES))
    ap.add_argument('--headless', action='store_true', help="render to PNG, no window")
    ap.add_argument('--bench', action='store_true', help="time BVH vs brute force")
    ap.add_argument('--selftest', action='store_true', help="check BVH against brute force")
    ap.add_argument('--width', type=int, default=640)
    ap.add_argument('--height', type=int, default=480)
    ap.add_argument('--spp', type=int, default=128)
    ap.add_argument('--depth', type=int, default=8)
    ap.add_argument('--temp', type=float, default=6500.0, help="lamp colour temperature (K)")
    ap.add_argument('--dispersion', type=float, default=6.0, help="Cauchy B multiplier")
    ap.add_argument('--exposure', type=float, default=0.0)
    ap.add_argument('--wb', action='store_true', help="white-balance the lamp")
    ap.add_argument('--out', default='render.png')
    a = ap.parse_args()
    if not HAVE_NUMBA:
        print("WARNING: numba not found - running in pure Python (extremely slow). "
              "pip install numba", file=sys.stderr)
    if a.selftest:
        ok = all([run_selftest(s) for s in SCENES])
        sys.exit(0 if ok else 1)
    elif a.bench:
        run_bench(a.scene)
    elif a.headless:
        run_headless(a)
    else:
        run_gui(a)


if __name__ == '__main__':
    main()