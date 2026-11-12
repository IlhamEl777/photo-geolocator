#!/usr/bin/env python3
"""photo-geolocator-01 entry: model-free photo geolocation (pure Python stdlib).

Approach
--------
No model calls. Each photo is decoded with a small baseline-JPEG entropy
decoder that recovers only the DC coefficients, i.e. the 8x8-block average
colours (a coarse YCbCr thumbnail) -- cheap colour/layout cues such as mean
colour, sky/ground band colours, and green/sky/sand block shares.

Those cues are matched against a reference set fitted on the competition's
own 75 public practice photos (kit practice pool, answers published): the 8
nearest reference photos in standardised feature space, blended with the
fitted global prior anchor (the haversine-optimal point of the practice
distribution, repeated 8 times as shrinkage). The guess is the spatial
median of that blend.

Calibration note: leave-one-instance-out validation on the practice pool
(mean error, km) -- this model 7,610; global prior alone 7,610 (the cues do
not beat the prior on 75 photos, so the blend is shrunk hard onto it);
region oracle 2,420 (headroom if cues were stronger). Deterministic: no
clock, no randomness, fixed neighbour count.
"""
import json, math, os, struct, sys, urllib.request

import struct

def _huff_tables(data, tables):
    i = 0
    while i < len(data):
        cls, tid = data[i] >> 4, data[i] & 15
        counts = list(data[i+1:i+17])
        total = sum(counts)
        symbols = list(data[i+17:i+17+total])
        # canonical codes -> dict {(len,code): symbol} and max code per len
        table = {}
        code = 0
        k = 0
        for ln in range(1, 17):
            for _ in range(counts[ln-1]):
                table[(ln, code)] = symbols[k]
                k += 1
                code += 1
            code <<= 1
        tables[(cls, tid)] = table
        i += 17 + total

class _Bits:
    __slots__ = ('data', 'pos', 'acc', 'cnt')
    def __init__(self, data):
        self.data = data; self.pos = 0; self.acc = 0; self.cnt = 0
    def get(self, n):
        while self.cnt < n:
            b = self.data[self.pos]; self.pos += 1
            self.acc = (self.acc << 8) | b; self.cnt += 8
        self.cnt -= n
        v = (self.acc >> self.cnt) & ((1 << n) - 1)
        self.acc &= (1 << self.cnt) - 1
        return v
    def symbol(self, table):
        code = 0
        for ln in range(1, 17):
            code = (code << 1) | self.get(1)
            v = table.get((ln, code))
            if v is not None:
                return v
        raise ValueError('bad huffman code')

def dc_grids(jpeg):
    """Return (Y,Cb,Cr) block grids (lists of rows of floats, block averages 0..255ish) or None."""
    if jpeg[:2] != b'\xff\xd8':
        return None
    qts = {}; dht = {}; comps = []
    w = h = 0; restart = 0; sof = None
    i = 2
    scan = None
    while i < len(jpeg) - 1:
        if jpeg[i] != 0xFF:
            i += 1; continue
        m = jpeg[i+1]
        if m == 0xD9:
            break
        if m == 0xDA:  # SOS
            ln = struct.unpack('>H', jpeg[i+2:i+4])[0]
            scan = jpeg[i+2+ln:]
            break
        if m == 0x01 or 0xD0 <= m <= 0xD8:
            i += 2; continue
        ln = struct.unpack('>H', jpeg[i+2:i+4])[0]
        seg = jpeg[i+4:i+2+ln]
        if m == 0xDB:
            j = 0
            while j < len(seg):
                pq, tq = seg[j] >> 4, seg[j] & 15
                if pq == 0:
                    qts[tq] = list(seg[j+1:j+65]); j += 65
                else:
                    qts[tq] = list(struct.unpack('>64H', seg[j+1:j+129])); j += 129
        elif m == 0xC4:
            _huff_tables(seg, dht)
        elif m == 0xC0 or m == 0xC1:
            sof = m
            h, w = struct.unpack('>HH', seg[1:5])
            n = seg[5]
            for c in range(n):
                cid, samp, tq = seg[6+3*c:9+3*c]
                comps.append({'id': cid, 'hs': samp >> 4, 'vs': samp & 15, 'tq': tq})
        elif m == 0xC2:
            return None  # progressive: give up
        elif m == 0xDD:
            restart = struct.unpack('>H', seg[:2])[0]
        i += 2 + ln
    if scan is None or not comps:
        return None
    # scan header -> component table selectors
    # re-parse SOS header: need it; find it again
    # (we stored only entropy data; selectors are in the header we skipped)
    # -> handled by caller passing full jpeg: re-locate SOS
    return _decode(jpeg, scan, comps, qts, dht, w, h, restart)

def _decode(jpeg, scan, comps, qts, dht, w, h, restart):
    # locate SOS header for table selectors
    i = jpeg.find(b'\xff\xda')
    ln = struct.unpack('>H', jpeg[i+2:i+4])[0]
    hdr = jpeg[i+4:i+2+ln]
    ns = hdr[0]
    sel = {}
    for c in range(ns):
        cid, tt = hdr[1+2*c], hdr[2+2*c]
        sel[cid] = (tt >> 4, tt & 15)
    # clean entropy data: un-stuff FF00, split at restart markers
    # Build segments: list of byte chunks separated by restart markers
    chunks = []
    cur = bytearray()
    j = 0
    while j < len(scan):
        b = scan[j]
        if b == 0xFF:
            if j + 1 >= len(scan):
                break
            nb = scan[j+1]
            if nb == 0x00:
                cur.append(0xFF); j += 2; continue
            if 0xD0 <= nb <= 0xD7:
                chunks.append(bytes(cur)); cur = bytearray(); j += 2; continue
            break  # EOI or other marker
        cur.append(b); j += 1
    chunks.append(bytes(cur))
    hmax = max(c['hs'] for c in comps); vmax = max(c['vs'] for c in comps)
    mcux = (w + 8*hmax - 1) // (8*hmax); mcuy = (h + 8*vmax - 1) // (8*vmax)
    grids = {}
    for c in comps:
        bw = mcux * c['hs']; bh = mcuy * c['vs']
        grids[c['id']] = [[0.0]*bw for _ in range(bh)]
    preds = {c['id']: 0 for c in comps}
    bits = _Bits(chunks[0]); chunk_i = 0
    mcu_total = mcux * mcuy
    since_restart = 0
    try:
        for my in range(mcuy):
            for mx in range(mcux):
                if restart and since_restart == restart:
                    chunk_i += 1
                    bits = _Bits(chunks[chunk_i])
                    preds = {c['id']: 0 for c in comps}
                    since_restart = 0
                since_restart += 1
                for c in comps:
                    cid = c['id']
                    dct, act = sel[cid]
                    dtab = dht[(0, dct)]; atab = dht[(1, act)]
                    q0 = qts[c['tq']][0]
                    g = grids[cid]
                    for by in range(c['vs']):
                        for bx in range(c['hs']):
                            # DC
                            s = bits.symbol(dtab)
                            if s:
                                diff = bits.get(s)
                                if diff < (1 << (s-1)):
                                    diff -= (1 << s) - 1
                            else:
                                diff = 0
                            preds[cid] += diff
                            g[my*c['vs']+by][mx*c['hs']+bx] = preds[cid] * q0 / 8.0 + 128.0
                            # AC: skip 63 coefficients
                            k = 1
                            while k < 64:
                                sym = bits.symbol(atab)
                                run, size = sym >> 4, sym & 15
                                if size == 0:
                                    if run == 15:
                                        k += 16; continue
                                    break  # EOB
                                k += run
                                bits.get(size)
                                k += 1
    except Exception:
        return None
    if len(comps) == 1:
        g = grids[comps[0]['id']]
        return g, None, None
    # map component ids: 1=Y, 2=Cb, 3=Cr (JFIF)
    ids = [c['id'] for c in comps]
    Y = grids[ids[0]]
    Cb = grids[ids[1]] if len(ids) > 1 else None
    Cr = grids[ids[2]] if len(ids) > 2 else None
    return Y, Cb, Cr


# ---------------- feature extraction (matches the calibration code) ----------------
def _band(grid, r0, r1):
    a = int(len(grid)*r0); b = max(a+1, int(len(grid)*r1))
    vals = [v for row in grid[a:b] for v in row]
    return sum(vals)/len(vals)

def photo_features(data):
    out = dc_grids(data)
    if out is None:
        return None
    Y, Cb, Cr = out
    def mean(g):
        return sum(sum(r) for r in g)/(len(g)*len(g[0]))
    f = {}
    f['y_mean'] = mean(Y); f['cb_mean'] = mean(Cb) if Cb else 128.0; f['cr_mean'] = mean(Cr) if Cr else 128.0
    f['y_top'] = _band(Y, 0.0, 0.25); f['y_mid'] = _band(Y, 0.375, 0.625); f['y_bot'] = _band(Y, 0.75, 1.0)
    if Cb:
        f['cb_top'] = _band(Cb, 0.0, 0.25); f['cr_top'] = _band(Cr, 0.0, 0.25)
        f['cb_bot'] = _band(Cb, 0.75, 1.0); f['cr_bot'] = _band(Cr, 0.75, 1.0)
        tot = 0; green = 0; sky = 0; sand = 0
        for r in range(len(Cb)):
            for c in range(len(Cb[0])):
                cb, cr = Cb[r][c], Cr[r][c]
                tot += 1
                if 92 <= cb <= 126 and 96 <= cr <= 126: green += 1
                if cb >= 138 and cr <= 124: sky += 1
                if cr >= 140 and cb <= 124: sand += 1
        f['green'] = green/tot; f['sky'] = sky/tot; f['sand'] = sand/tot
    else:
        f.update({'cb_top': 128.0, 'cr_top': 128.0, 'cb_bot': 128.0, 'cr_bot': 128.0, 'green': 0.0, 'sky': 0.0, 'sand': 0.0})
    f['sat'] = abs(f['cb_mean']-128) + abs(f['cr_mean']-128)
    return [f[k] for k in FEAT_ORDER]

FEAT_ORDER = ['y_mean','cb_mean','cr_mean','y_top','y_mid','y_bot','cb_top','cr_top','cb_bot','cr_bot','green','sky','sand','sat']
ANCHOR = (56.780, 27.026)
MU = [float(x) for x in "119.823,133.427,119.615,146.835,115.048,97.360,144.169,112.093,126.049,126.103,0.110,0.281,0.025,16.884".split(',')]
SD = [float(x) for x in "24.074,9.218,8.866,44.760,34.877,48.200,15.879,13.679,8.600,8.072,0.132,0.241,0.072,14.540".split(',')]
REF = []
for _row in "115.83,128.12,121.47,152.72,119.79,83.68,132.42,119.59,121.80,125.35,0.36,0.10,0.00,6.65,27.0170,83.2277;106.85,129.20,120.93,50.66,65.19,198.70,128.30,122.83,131.28,118.31,0.07,0.00,0.00,8.27,55.8545,38.5799;131.36,131.43,120.35,189.37,117.51,89.22,138.90,115.37,125.07,126.74,0.16,0.29,0.00,11.08,40.6937,-106.3503;125.13,112.45,134.39,162.14,95.24,124.89,125.43,127.81,106.62,139.41,0.00,0.02,0.20,21.94,-16.4951,-41.0308;105.43,134.58,119.32,59.89,88.94,157.61,134.40,118.65,136.93,119.67,0.05,0.23,0.00,15.27,33.4580,135.7893;109.15,132.23,120.35,99.82,91.43,137.06,133.81,120.51,135.08,114.48,0.02,0.09,0.00,11.87,6.8041,79.9436;90.61,127.88,127.18,151.52,89.52,28.95,130.65,125.59,127.03,128.79,0.00,0.02,0.00,0.94,-14.9070,-46.9267;80.56,130.68,118.03,102.78,70.74,51.19,150.53,114.04,121.68,122.58,0.42,0.36,0.00,12.65,-8.8388,148.0307;152.07,140.94,117.40,192.27,184.75,76.11,148.26,110.93,134.92,124.78,0.03,0.67,0.00,23.54,25.6219,-108.5469;122.92,136.52,113.24,153.85,85.65,137.77,149.05,100.83,129.62,117.67,0.24,0.31,0.00,23.28,-26.4035,-69.3713;89.45,125.71,129.42,113.29,81.88,67.22,128.48,126.06,126.18,131.02,0.10,0.00,0.01,3.71,2.7157,98.9402;91.51,128.07,124.10,67.69,134.16,48.90,128.22,120.88,129.11,127.62,0.19,0.10,0.01,3.97,21.0289,-89.6044;93.41,132.77,118.41,155.32,41.62,89.61,149.90,107.41,122.55,125.12,0.22,0.23,0.00,14.35,-42.2453,171.3258;117.89,145.71,99.88,98.49,142.32,116.62,160.48,84.09,122.65,126.85,0.07,0.61,0.01,45.82,-38.1724,-63.2355;167.28,129.29,129.21,252.26,148.38,91.13,128.87,128.09,129.93,129.56,0.00,0.00,0.00,2.50,53.7771,44.8009;119.59,128.87,127.16,167.68,108.59,111.49,137.20,119.27,128.48,129.95,0.00,0.12,0.00,1.71,-20.6435,-40.5585;155.24,131.81,118.38,179.76,122.71,180.37,142.99,111.52,126.34,121.26,0.16,0.29,0.03,13.44,-8.6712,34.9416;58.00,127.53,124.52,78.26,64.96,34.49,128.10,120.08,132.82,124.77,0.15,0.04,0.00,3.94,41.6873,-123.9248;90.81,131.42,117.88,190.23,50.21,27.41,137.04,108.80,129.62,122.94,0.06,0.11,0.00,13.55,5.4942,-6.5628;98.99,130.66,123.53,164.86,84.47,70.64,131.63,121.73,129.76,125.52,0.10,0.03,0.00,7.13,8.5132,77.6526;151.12,123.30,133.30,131.02,151.38,145.20,148.68,116.48,116.66,136.66,0.00,0.15,0.36,10.00,-23.6994,-65.4520;113.59,129.50,124.56,142.80,144.16,55.63,131.97,123.76,124.77,127.65,0.08,0.00,0.00,4.94,21.0725,-86.7792;89.94,135.63,121.08,130.01,78.21,60.14,146.29,112.93,133.36,124.17,0.02,0.28,0.00,14.55,-46.0799,168.3938;116.64,131.13,119.45,173.55,123.45,62.37,133.78,118.66,129.90,119.35,0.10,0.03,0.00,11.68,-22.2500,-49.1214;128.88,131.59,124.68,172.98,137.04,79.38,139.58,119.55,124.80,129.07,0.00,0.20,0.00,6.91,21.7696,-102.9712;129.44,129.05,125.14,159.43,130.77,100.73,138.15,118.50,122.04,132.95,0.08,0.25,0.00,3.91,27.3908,69.0983;96.70,137.64,117.64,146.86,121.88,25.27,141.54,114.44,128.19,127.46,0.00,0.39,0.00,20.00,7.4027,-7.5569;120.24,145.82,114.64,97.36,131.02,131.07,172.29,97.48,123.46,133.19,0.08,0.54,0.00,31.19,50.5206,-101.0010;125.13,126.96,127.93,203.09,115.39,52.28,144.40,114.17,122.79,133.55,0.01,0.36,0.08,1.10,-29.1856,134.9440;127.12,126.34,132.42,143.79,102.12,144.82,144.35,118.66,116.53,139.08,0.00,0.25,0.35,6.07,21.8449,59.5751;112.59,129.24,119.71,184.89,56.81,100.66,139.29,116.61,127.30,116.96,0.35,0.32,0.05,9.53,64.6323,21.0320;120.95,153.05,104.90,169.12,159.44,36.23,175.94,88.74,137.13,120.45,0.00,0.72,0.00,48.16,60.6748,-135.2228;176.29,126.99,116.82,203.05,152.99,156.82,138.81,106.40,122.61,122.08,0.39,0.16,0.00,12.19,53.5012,-113.7627;87.94,137.61,116.88,90.14,78.86,113.73,164.18,104.49,115.24,127.15,0.27,0.36,0.00,20.72,63.5535,28.3816;106.01,125.08,128.30,175.74,97.76,54.75,130.11,126.13,121.77,130.76,0.17,0.00,0.00,3.21,43.4786,144.4280;115.67,144.80,112.64,82.99,118.31,164.52,167.70,98.87,116.98,133.94,0.04,0.59,0.01,32.17,30.2535,-90.4025;98.06,136.70,115.67,139.13,130.66,29.39,148.32,107.21,130.20,122.09,0.13,0.34,0.00,21.03,54.5797,24.1895;140.59,139.93,118.01,139.09,164.41,97.44,147.92,113.44,123.10,130.44,0.00,0.55,0.01,21.92,35.3578,51.3376;128.55,149.93,106.19,138.42,163.03,70.65,159.58,101.31,148.35,106.41,0.00,0.92,0.00,43.75,78.0807,14.1537;142.11,143.26,96.97,143.34,187.45,88.63,173.80,68.34,122.01,123.03,0.05,0.52,0.00,46.29,-15.7624,46.4746;135.11,136.10,113.40,79.00,176.94,161.87,125.04,123.94,145.04,106.73,0.20,0.51,0.00,22.70,38.0135,-87.5284;97.45,140.94,106.15,21.75,110.76,170.37,131.72,121.61,153.92,85.13,0.02,0.48,0.00,34.79,43.2792,3.3148;125.38,132.48,114.75,152.66,152.67,71.08,142.28,106.95,124.38,124.55,0.24,0.26,0.00,17.73,42.5436,12.2590;139.59,122.79,126.86,180.65,148.14,93.88,128.41,124.43,114.22,131.47,0.05,0.01,0.00,6.36,52.6437,77.0860;121.91,131.45,113.65,113.96,99.74,151.50,162.18,97.39,115.33,127.58,0.36,0.34,0.00,17.80,-8.7476,160.6944;150.02,125.97,122.97,203.03,88.62,143.82,132.92,116.73,120.48,127.24,0.20,0.01,0.00,7.06,34.5642,69.1210;131.80,136.80,116.39,167.41,136.28,75.19,151.80,104.79,129.61,123.47,0.05,0.41,0.00,20.41,32.1486,44.3227;182.81,116.28,131.75,218.81,121.77,224.37,128.79,126.44,105.13,139.22,0.03,0.00,0.10,15.47,-37.1845,-72.3925;107.99,128.26,128.26,198.94,70.45,54.07,131.93,127.02,128.15,128.95,0.00,0.00,0.00,0.52,-29.9640,-51.0977;127.30,115.39,127.69,91.04,127.82,170.60,116.59,122.35,116.34,132.90,0.49,0.00,0.00,12.92,25.4063,105.5087;115.18,143.06,122.75,138.11,166.40,51.86,156.12,116.84,129.51,128.91,0.00,0.53,0.00,20.32,58.6588,-124.3461;155.83,126.56,125.29,216.57,171.25,75.91,128.40,123.79,130.86,122.54,0.07,0.04,0.00,4.15,34.0073,35.6687;119.74,132.68,123.40,150.78,122.17,85.82,141.41,117.86,123.21,129.84,0.01,0.33,0.00,9.28,39.7563,19.7436;168.70,131.55,114.57,196.92,144.44,158.33,145.93,103.82,124.48,122.16,0.42,0.31,0.00,16.98,36.6129,-79.5741;145.22,131.53,123.84,181.89,122.01,122.03,146.89,112.04,124.54,131.50,0.01,0.25,0.00,7.69,48.1754,40.6106;128.66,124.05,128.61,213.83,94.69,86.88,128.42,126.99,120.34,131.15,0.10,0.00,0.01,4.56,-9.3786,-38.2300;120.83,125.22,126.42,174.16,84.39,106.93,135.96,115.18,122.95,131.15,0.02,0.18,0.00,4.36,56.2257,59.2353;116.23,155.49,86.57,94.76,107.67,151.26,188.40,60.75,108.22,131.21,0.19,0.61,0.14,68.92,-15.9961,39.6116;110.72,127.18,116.85,173.89,71.58,82.86,137.24,106.62,121.52,123.28,0.54,0.12,0.00,11.98,-13.5135,-39.9635;85.46,127.39,127.84,151.48,53.82,53.06,128.21,126.75,127.91,128.19,0.01,0.00,0.00,0.77,42.8983,74.5743;122.58,139.90,112.15,120.23,121.80,142.60,160.13,101.57,123.75,123.40,0.21,0.53,0.02,27.75,10.7905,-5.6863;80.50,138.29,119.24,128.64,68.50,23.22,144.49,112.02,135.69,123.26,0.00,0.49,0.00,19.04,57.0940,63.9780;91.01,141.44,116.20,99.19,99.03,67.90,163.32,106.40,126.05,126.67,0.25,0.44,0.00,25.23,62.3453,25.0256;109.44,147.45,108.97,143.07,118.90,42.16,173.04,85.36,133.47,123.05,0.00,0.54,0.00,38.48,51.5135,128.1961;134.66,127.92,127.44,219.60,136.13,49.53,127.46,128.75,126.67,127.56,0.00,0.01,0.00,0.64,66.3566,13.0034;150.86,151.24,109.32,119.82,160.89,181.26,182.11,90.86,105.94,142.77,0.02,0.62,0.23,41.92,63.7811,-137.8011;114.40,136.43,115.60,154.88,93.27,78.92,166.86,86.69,124.91,132.12,0.06,0.31,0.00,20.83,53.2648,14.3485;123.19,141.86,115.98,154.15,109.16,91.41,158.96,106.49,132.62,124.53,0.14,0.50,0.00,25.88,8.0281,99.1017;153.01,129.39,129.80,227.47,164.66,60.18,135.48,123.82,127.98,132.65,0.00,0.14,0.03,3.19,51.4754,45.7935;73.41,142.57,116.44,96.48,71.26,39.55,146.68,113.27,135.74,121.92,0.00,0.64,0.00,26.13,67.0855,33.1626;106.23,125.89,129.09,135.42,146.06,40.90,127.56,127.90,126.37,129.44,0.00,0.00,0.00,3.19,33.4351,126.9096;138.42,147.12,111.53,162.71,130.08,99.80,158.31,102.95,143.50,117.73,0.01,0.85,0.00,35.59,0.3810,24.1545;112.28,128.56,126.70,132.72,68.34,134.87,143.54,115.46,119.48,135.69,0.04,0.27,0.00,1.86,48.3873,-81.5586;123.93,116.77,132.30,122.79,87.88,192.70,123.95,123.12,120.48,131.71,0.17,0.00,0.23,15.54,2.4425,20.1327;117.21,161.61,100.27,126.15,147.82,72.53,180.80,89.73,130.23,119.25,0.10,0.77,0.00,61.34,-27.1298,-56.3207".split(';'):
    _v = [float(x) for x in _row.split(',')]
    REF.append((_v[:14], _v[14], _v[15]))

def _spatial_median(pts):
    x = sum(p[0] for p in pts)/len(pts); y = sum(p[1] for p in pts)/len(pts)
    for _ in range(40):
        nx = ny = den = 0.0
        for a, b in pts:
            d = max(1e-6, math.hypot(a-x, b-y))
            nx += a/d; ny += b/d; den += 1/d
        x, y = nx/den, ny/den
    return (x, y)

def predict(feat):
    if feat is None:
        return ANCHOR
    def dist(rf):
        return sum(((feat[i]-rf[i])/SD[i])**2 for i in range(14))
    near = sorted(REF, key=lambda r: dist(r[0]))[:8]
    pts = [(la, lo) for _, la, lo in near] + [ANCHOR]*8
    return _spatial_median(pts)

# ---------------- env loop ----------------
def _get(url, binary=False, timeout=60):
    req = urllib.request.Request(url, headers={'Accept': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = r.read()
    return data if binary else json.loads(data.decode('utf-8'))

def _post(url, payload, timeout=60):
    body = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(url, data=body, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode('utf-8'))

def main():
    env = os.environ.get('MAKEFUN_ENV_URL', '').rstrip('/')
    if not env:
        return
    listing = _get(env + '/photos')
    photos = listing.get('photos', [])
    answers = []
    for p in photos:
        pid = p['id']
        guess = ANCHOR
        try:
            url = p.get('url') or ('/photos/%s.jpg' % pid)
            if url.startswith('/'):
                url = env + url
            data = _get(url, binary=True)
            guess = predict(photo_features(data))
        except Exception:
            guess = ANCHOR
        answers.append({'id': pid, 'lat': round(guess[0], 6), 'lon': round(guess[1], 6)})
    if answers:
        try:
            _post(env + '/answers', {'answers': answers})
        except Exception:
            # retry once per photo so a single bad entry cannot sink the batch
            for a in answers:
                try:
                    _post(env + '/answers', {'answers': [a]})
                except Exception:
                    pass

if __name__ == '__main__':
    main()
