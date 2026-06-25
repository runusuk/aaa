#!/usr/bin/env python3
# 曲を解析して BPM と最初の拍のオフセットを求め、beatmaps.json を作る。
# 一定テンポを仮定（子ども向け楽曲は十分これで合う）。ゲームは offset + n*(60/bpm) で拍を生成する。
import json, sys, os, glob
import numpy as np
import miniaudio

SR = 22050  # 解析用サンプルレート

def load_mono(path):
    dec = miniaudio.decode_file(path, output_format=miniaudio.SampleFormat.FLOAT32,
                                nchannels=1, sample_rate=SR)
    y = np.array(dec.samples, dtype=np.float32)
    return y, SR

def onset_envelope(y, sr, hop=512, win=1024):
    # スペクトルフラックス（立ち上がり量）を1フレームごとに計算
    n = 1 + (len(y) - win) // hop if len(y) >= win else 0
    if n <= 1:
        return np.zeros(1), hop / sr
    window = np.hanning(win).astype(np.float32)
    prev = None
    flux = np.empty(n, dtype=np.float32)
    for i in range(n):
        seg = y[i*hop:i*hop+win] * window
        mag = np.abs(np.fft.rfft(seg))
        if prev is None:
            flux[i] = 0.0
        else:
            d = mag - prev
            d[d < 0] = 0.0           # 増えた分だけ（立ち上がり）
            flux[i] = float(d.sum())
        prev = mag
    # 平滑化＆正規化
    flux -= flux.mean()
    flux[flux < 0] = 0.0
    if flux.max() > 0:
        flux /= flux.max()
    return flux, hop / sr

def estimate_bpm(env, frame_dt, bpm_min=60, bpm_max=185):
    # オンセット包絡の自己相関からテンポを推定
    env = env - env.mean()
    ac = np.correlate(env, env, mode='full')[len(env)-1:]
    lag_min = int(round((60.0 / bpm_max) / frame_dt))
    lag_max = int(round((60.0 / bpm_min) / frame_dt))
    lag_max = min(lag_max, len(ac) - 1)
    if lag_max <= lag_min:
        return 100.0
    seg = ac[lag_min:lag_max+1].copy()
    # テンポは速め(短いラグ)が選ばれやすいよう軽く重み付け
    lags = np.arange(lag_min, lag_max+1)
    best_lag = lags[int(np.argmax(seg))]
    bpm = 60.0 / (best_lag * frame_dt)
    # 子ども向けは速すぎ/遅すぎを補正（オクターブ調整）
    while bpm > 150: bpm /= 2
    while bpm < 70:  bpm *= 2
    return float(bpm)

def estimate_offset(env, frame_dt, bpm):
    # 拍周期に対して、どの位相だと立ち上がりが最大になるかを探す → 最初の拍位置
    period = 60.0 / bpm
    period_frames = period / frame_dt
    best_phase, best_score = 0.0, -1.0
    steps = 48
    for s in range(steps):
        phase = s / steps * period_frames  # フレーム単位
        idx = phase
        score = 0.0
        cnt = 0
        while idx < len(env):
            score += env[int(idx)]
            idx += period_frames
            cnt += 1
        if cnt:
            score /= cnt
        if score > best_score:
            best_score, best_phase = score, phase
    return float(best_phase * frame_dt % period)

def analyze(path):
    y, sr = load_mono(path)
    dur = len(y) / sr
    env, frame_dt = onset_envelope(y, sr)
    bpm = estimate_bpm(env, frame_dt)
    offset = estimate_offset(env, frame_dt, bpm)
    return {"bpm": round(bpm, 2), "offset": round(offset, 3), "duration": round(dur, 2)}

def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    songs = json.load(open(os.path.join(root, "songs.json"), encoding="utf-8"))
    out = {}
    for s in songs:
        p = os.path.join(root, s)
        if not os.path.exists(p):
            print("skip (missing):", s); continue
        try:
            res = analyze(p)
            out[s] = res
            print(f"{s}: BPM={res['bpm']} offset={res['offset']}s dur={res['duration']}s")
        except Exception as e:
            print(f"ERROR {s}: {e}")
    json.dump(out, open(os.path.join(root, "beatmaps.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("wrote beatmaps.json")

if __name__ == "__main__":
    main()
