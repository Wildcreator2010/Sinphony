#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Sinphony — 傅里叶级数歌曲文件 <-> WAV 双向转换工具
#
# Copyright 2026 wildcreator2010
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Sinphony — 傅里叶级数歌曲文件 <-> WAV 双向转换工具

把一首歌曲的波形拆解成的千万元正弦波分量, 反向重建为 WAV,
或将 WAV 分解为同样的正弦波级数文本。

txt 格式 (song_fourier.txt):
  unit/s: <Δf>            -- 频率分辨率 (Hz)
  accuracy: <N>           -- 精度等级
  R_sin_count: <n>        -- 右声道正弦项数
  L_sin_count: <n>        -- 左声道正弦项数
  R:
  <dc>                    -- 右声道直流偏置
  <A> *sin <f> *x + <φ>   -- 正弦项: 振幅 * sin(频率Hz * 时间 + 相位)
  L:
  ...                     -- 同上, 左声道

重建原理:
  每个正弦项的 <f> 即为频率 (Hz), unit/s 是频率分辨率 Δf。
  bin 序号 k = round(f / Δf), 把振幅/相位累加进对应频谱 bin,
  做一次 IFFT 得到时域波形, 再重采样到目标采样率。

用法:
  python sinphony.py to-wav   <input.txt> <output.wav> [--sr 44100]
  python sinphony.py to-txt   <input.wav> <output.txt> [--accuracy 1]
  python sinphony.py info     <input.txt>
"""

import sys
import os
import re
import math
import struct
import wave
import argparse
import numpy as np

# ---------------------------------------------------------------------------
# 解析 txt 文件
# ---------------------------------------------------------------------------

RE_HEADER = re.compile(r"^(\w[\w/]*)\s*:\s*(.*)$")
RE_SINE_TERM = re.compile(
    r"^\s*(-?[\d.eE+\-]+)\s*\*sin\s+(-?[\d.eE+\-]+)\s*\*x\s+\+\s+(-?[\d.eE+\-]+)\s*$"
)
RE_DC_TERM = re.compile(r"^\s*(-?[\d.eE+\-]+)\s*$")


def parse_fourier_txt(filepath, channel="both", verbose=True):
    """
    解析傅里叶级数 txt 文件, 返回 dict:
      {
        "unit_per_s": float,
        "accuracy": int,
        "sample_rate": float,
        "duration": float,
        "R": {"dc": float, "terms": [(amp, omega_rad_s, phase_rad), ...], "sin_count": int},
        "L": {...},
      }
    """
    result = {
        "unit_per_s": None,
        "accuracy": 1,
        "R": {"dc": 0.0, "terms": [], "sin_count": 0},
        "L": {"dc": 0.0, "terms": [], "sin_count": 0},
    }

    section = None          # "R" or "L"
    parsed_dc = {"R": False, "L": False}

    total = os.path.getsize(filepath)
    processed = 0
    terms_r = 0
    terms_l = 0

    if verbose:
        mb = total / (1024 * 1024)
        print(f"[parse] 文件大小: {mb:.0f} MB, 开始流式解析...")

    header_done = False

    with open(filepath, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n\r")
            processed += len(line.encode("utf-8")) + 1

            # --- header: all key:value lines up to first blank line ---
            if not header_done:
                m = RE_HEADER.match(line)
                if m:
                    key = m.group(1)
                    val = m.group(2).strip()
                    if key == "unit/s":
                        result["unit_per_s"] = float(val)
                    elif key == "accuracy":
                        result["accuracy"] = int(val)
                    elif key == "R_sin_count":
                        result["R"]["sin_count"] = int(val)
                    elif key == "L_sin_count":
                        result["L"]["sin_count"] = int(val)
                    continue
                elif not line.strip():
                    # blank line → header section ended
                    header_done = True
                    continue
                else:
                    header_done = True  # first non-header line

            # --- section markers ---
            if line in ("R:", "L:"):
                section = line[0]
                continue

            if section is None:
                continue

            if not line.strip():
                continue

            ch = result[section]

            # DC term: first numeric line after section header
            if not parsed_dc[section]:
                m = RE_DC_TERM.match(line)
                if m:
                    ch["dc"] = float(m.group(1))
                    parsed_dc[section] = True
                    continue

            # Sine term
            m = RE_SINE_TERM.match(line)
            if m:
                amp = float(m.group(1))
                omega = float(m.group(2))
                phase = float(m.group(3))
                ch["terms"].append((amp, omega, phase))
                if section == "R":
                    terms_r += 1
                else:
                    terms_l += 1

            # Print progress every ~5% of file
            pct_now = int(processed / total * 100)
            pct_prev = int((processed - len(line.encode("utf-8")) - 1) / total * 100) if processed > 0 else -1
            if verbose and pct_now != pct_prev and pct_now % 5 == 0:
                print(f"  ... {pct_now}% ({terms_r+terms_l} terms)")

    if verbose:
        print(f"[parse] 完成: R {terms_r} 项, L {terms_l} 项")

    # Compute derived params
    u = result["unit_per_s"]  # Δf: frequency resolution (Hz)
    result["duration"] = 1.0 / u  # T = 1/Δf
    # Sample rate will be set during to-wav

    return result


# ---------------------------------------------------------------------------
# txt -> wav
# ---------------------------------------------------------------------------

def txt_to_wav(input_txt, output_wav, target_sr=44100, channel="both",
               verbose=True):
    """
    将傅里叶级数 txt 流式重建为 WAV 音频文件.
    单次遍历文件, 直接累加频谱, 不缓存千万级 term 列表.
    """
    # ---------- 第一遍: 只读头部, 获取元信息 ----------
    meta = {}
    with open(input_txt, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n\r")
            m = RE_HEADER.match(line)
            if m:
                key, val = m.group(1), m.group(2).strip()
                if key == "unit/s":
                    meta["unit_per_s"] = float(val)
                elif key == "accuracy":
                    meta["accuracy"] = int(val)
                elif key == "R_sin_count":
                    meta["R_sin_count"] = int(val)
                elif key == "L_sin_count":
                    meta["L_sin_count"] = int(val)
            elif not line.strip():
                break  # header end

    delta_f = meta["unit_per_s"]  # Δf (Hz 频率分辨率) == unit/s
    T = 1.0 / delta_f              # 信号时长 (秒)
    n_R = meta["R_sin_count"]
    n_L = meta["L_sin_count"]

    if verbose:
        print(f"[txt->wav] Δf = {delta_f:.9f} Hz")
        print(f"[txt->wav] duration T = {T:.3f} s ({T/60:.2f} min)")
        print(f"[txt->wav] R terms={n_R:,}  L terms={n_L:,}")

    # ---------- 分配频谱数组 ----------
    # bin index k = round(omega / Δf), omega 已是 Hz。
    # FFT 长度 N 需覆盖最大 k (N/2 >= k_max), 取 2 的幂以便 FFT 高效。
    # rfft 格式: 只需存 bins [0, N/2]
    do_R = channel in ("both", "R")
    do_L = channel in ("both", "L")

    def alloc_spectrum(n_terms):
        # 原始样本数 = 时长 * 采样率 (由 k_max 和时长共同决定)
        n = int(round(T * target_sr))
        # FFT 长度取 2 的幂, 至少覆盖 n
        n_fft = 1 << (n - 1).bit_length()
        n_nyquist = n_fft // 2 + 1
        return n_fft, n_nyquist, np.zeros(n_nyquist, dtype=np.complex128)

    spec_R = alloc_spectrum(n_R) if do_R else None
    spec_L = alloc_spectrum(n_L) if do_L else None

    # ---------- 第二遍: 流式累加频谱 ----------
    total = os.path.getsize(input_txt)
    processed = 0
    section = None
    dc_done = {"R": False, "L": False}
    dc_R = dc_L = 0.0
    count = {"R": 0, "L": 0}
    last_pct = -1

    if verbose:
        print(f"[txt->wav] streaming parse + spectrum accumulate...")

    with open(input_txt, "r", encoding="utf-8") as fh:
        for line in fh:
            processed += len(line.encode("ascii", "ignore")) + 1
            raw = line.rstrip("\n\r")

            if raw in ("R:", "L:"):
                section = raw[0]
                continue
            if section is None or not raw.strip():
                continue

            ch_key = section

            # DC term (first bare number in section)
            if not dc_done[ch_key]:
                m = RE_DC_TERM.match(raw)
                if m:
                    if ch_key == "R":
                        dc_R = float(m.group(1))
                    else:
                        dc_L = float(m.group(1))
                    dc_done[ch_key] = True
                    continue

            # Sine term
            m = RE_SINE_TERM.match(raw)
            if not m:
                continue

            amp = float(m.group(1))
            omega = float(m.group(2))
            phase = float(m.group(3))

            if ch_key == "R" and spec_R is not None:
                n_fft, n_nyquist, spectrum = spec_R
                k = int(round(omega / delta_f))
                if 1 <= k < n_nyquist:
                    c = -0.5j * n_fft * amp * np.exp(1j * phase)
                    spectrum[k] += c
            elif ch_key == "L" and spec_L is not None:
                n_fft, n_nyquist, spectrum = spec_L
                k = int(round(omega / delta_f))
                if 1 <= k < n_nyquist:
                    c = -0.5j * n_fft * amp * np.exp(1j * phase)
                    spectrum[k] += c

            count[ch_key] += 1

            # progress
            pct = int(processed * 100 / total)
            if pct != last_pct and pct % 5 == 0:
                if verbose:
                    print(f"  ... {pct}% (R {count['R']:,} / L {count['L']:,})")
                last_pct = pct

    if verbose:
        print(f"[txt->wav] parse done: R {count['R']:,} terms, L {count['L']:,} terms")

    # ---------- 各声道 IFFT → 时域 → 写 WAV ----------
    channels_out = []

    for ch_key, spec, dc in (("R", spec_R, dc_R), ("L", spec_L, dc_L)):
        if spec is None:
            continue
        n_fft, n_nyquist, spectrum = spec
        spectrum[0] = dc * n_fft  # DC bin, 用 n_fft 缩放

        if verbose:
            fs_fft = n_fft / T
            print(f"[txt->wav] {ch_key}: N_fft={n_fft:,}, "
                  f"fs_fft≈{fs_fft:.0f} Hz, duration={T:.2f}s")

        full = np.zeros(n_fft, dtype=np.complex128)
        full[:n_nyquist] = spectrum
        del spectrum  # free rfft array
        for k in range(1, n_fft // 2):
            full[n_fft - k] = np.conj(full[k])
        if n_fft % 2 == 0:
            full[n_fft // 2] = full[n_nyquist - 1].real

        if verbose:
            print(f"[txt->wav] IFFT {ch_key}...")
        time_signal = np.fft.ifft(full).real  # n_fft 个样本, 时长 T
        del full

        # IFFT 输出采样率 fs_fft = n_fft/T。重采样到 target_sr:
        # 目标样本数 = round(T * target_sr), 保持时长不变 (不截断!)
        fs_fft = n_fft / T
        out_len = int(round(T * target_sr))
        old_x = np.arange(len(time_signal))
        new_x = np.linspace(0, len(time_signal) - 1, out_len)
        time_signal = np.interp(new_x, old_x, time_signal)

        peak = np.max(np.abs(time_signal))
        if peak > 0:
            time_signal = time_signal / peak * 0.95
        wav_data = np.clip(time_signal * 32767, -32768, 32767).astype(np.int16)
        del time_signal
        channels_out.append(wav_data)

    if not channels_out:
        print("[txt->wav] error: no channel data!")
        sys.exit(1)

    if len(channels_out) == 1:
        wav_out = channels_out[0]
    else:
        min_len = min(len(c) for c in channels_out)
        wav_out = np.column_stack([c[:min_len] for c in channels_out])

    _write_wav(output_wav, target_sr, wav_out)
    dur = wav_out.shape[0] / target_sr
    if verbose:
        ch = wav_out.shape[1] if wav_out.ndim > 1 else 1
        print(f"\n[txt->wav] Done! {output_wav}")
        print(f"   sample rate: {target_sr} Hz, channels: {ch}, duration: {dur:.2f} s")


# ---------------------------------------------------------------------------
# WAV 读写 (纯标准库, 无 scipy 依赖)
# ---------------------------------------------------------------------------

def _write_wav(path, sample_rate, data):
    """用标准库 wave 模块写 WAV 文件. data: int16 numpy 数组."""
    data = np.asarray(data, dtype=np.int16)
    if data.ndim == 1:
        n_channels = 1
        n_frames = len(data)
    else:
        n_channels = data.shape[1]
        n_frames = data.shape[0]

    with wave.open(path, "wb") as wf:
        wf.setnchannels(n_channels)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(int(sample_rate))
        wf.writeframes(data.tobytes())


def _read_wav(path):
    """用标准库 wave 模块读 WAV 文件, 返回 (sample_rate, int16 numpy 数组)."""
    with wave.open(path, "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sampwidth == 2:
        dtype = np.int16
    elif sampwidth == 1:
        dtype = np.uint8
    elif sampwidth == 4:
        dtype = np.int32
    elif sampwidth == 3:
        # 24-bit: rare, pad to int32
        raw = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        raw = np.pad(raw, ((0, 0), (0, 1)), mode="constant")
        data = raw.astype(np.int32)
        # sign extension
        data = np.where((data & 0x800000) != 0, data | ~0xFFFFFF, data)
        data = data.reshape(-1, n_channels) if n_channels > 1 else data.ravel()
        return sample_rate, data.astype(np.float64)
    else:
        raise ValueError(f"不支持的采样位宽: {sampwidth}")

    data = np.frombuffer(raw, dtype=dtype)

    if sampwidth == 1:
        # 8-bit WAV is unsigned, center at 128
        data = (data.astype(np.float64) - 128.0) * 256.0
        if n_channels > 1:
            data = data.reshape(-1, n_channels)
        return sample_rate, data

    if n_channels > 1:
        data = data.reshape(-1, n_channels)

    return sample_rate, data.astype(np.float64)


# ---------------------------------------------------------------------------
# wav -> txt
# ---------------------------------------------------------------------------

def wav_to_txt(input_wav, output_txt, accuracy=1, verbose=True):
    """
    将 WAV 文件转换为傅里叶级数 txt 格式.
    做 FFT, 提取每个 bin 的振幅/频率/相位, 按振幅降序写入.
    accuracy 参数暂留 (未来可用于控制保留项数).
    """
    sample_rate, audio = _read_wav(input_wav)
    if verbose:
        print(f"[wav→txt] 读取: {sample_rate} Hz, shape={audio.shape}")

    if audio.ndim == 1:
        channels = [audio]
    else:
        channels = [audio[:, 0], audio[:, 1]]

    n_samples = len(channels[0])
    if verbose:
        dur = n_samples / sample_rate
        print(f"[wav→txt] 时长: {dur:.3f} 秒, {n_samples} 样本")

    # FFT each channel
    channel_data = []
    for idx, ch_audio in enumerate(channels):
        label = "R" if idx == 0 else "L"
        ch_audio = ch_audio.astype(np.float64)

        if verbose:
            print(f"[wav→txt] FFT {label} 声道...")

        spectrum = np.fft.rfft(ch_audio)
        n_nyquist = len(spectrum)

        # DC component
        dc = spectrum[0].real / n_samples

        # Frequency resolution
        delta_f = sample_rate / n_samples          # Hz resolution (= unit/s)

        # Extract terms for bins k=1 to n_nyquist-1
        # Convert to A * sin(ωx + φ) format, where ω is in Hz (matching the
        # original file format).
        terms = []
        for k in range(1, n_nyquist):
            z = spectrum[k]
            amp_raw = np.abs(z)
            if amp_raw < 1e-15:
                continue

            # For a real signal x(t) = Σ A_k sin(2π f_k t + φ_k):
            #   RFFT bin k: X[k] = N/(2j) * e^{jφ_k} * A_k  (positive freq coeff)
            #   |X[k]| = N * A_k / 2
            #   arg(X[k]) = φ_k - π/2
            # So: A_k = 2 * |X[k]| / N,  φ_k = arg(X[k]) + π/2
            amp = 2.0 * amp_raw / n_samples
            phase = np.angle(z) + math.pi / 2.0
            # Normalize phase to (-π, π]
            phase = ((phase + math.pi) % (2 * math.pi)) - math.pi

            # Frequency (in Hz) — the x variable is time in seconds
            freq_hz = k * delta_f
            omega = freq_hz  # NOTE: Hz, not rad/s, to match original format

            terms.append((amp, omega, phase))

        # Sort by amplitude descending
        terms.sort(key=lambda x: x[0], reverse=True)

        channel_data.append({
            "label": label,
            "dc": dc,
            "terms": terms,
            "sin_count": len(terms),
        })

    # Write output
    if verbose:
        print(f"[wav→txt] 写入 {output_txt} ...")

    with open(output_txt, "w", encoding="utf-8") as fh:
        fh.write(f"unit/s: {delta_f:.16f}\n")
        fh.write(f"accuracy: {accuracy}\n")
        for cd in channel_data:
            fh.write(f"{cd['label']}_sin_count: {cd['sin_count']}\n")
        fh.write("\n")

        for cd in channel_data:
            fh.write(f"{cd['label']}:\n\n")
            # DC term
            fh.write(f"{cd['dc']:.16f}\n")
            # Sine terms
            for amp, omega, phase in cd["terms"]:
                fh.write(
                    f"{amp:.16f} *sin {omega:.16f} *x + {phase:.16f}\n"
                )
            fh.write("\n")

    if verbose:
        print(f"\n[wav->txt] Done! {output_txt}")
        for cd in channel_data:
            print(f"   {cd['label']}: DC={cd['dc']:.6f}, 正弦项={cd['sin_count']}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Sinphony — 傅里叶级数歌曲文件 ↔ WAV 双向转换工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python sinphony.py to-wav song_fourier.txt output.wav
  python sinphony.py to-wav song_fourier.txt output.wav --sr 48000
  python sinphony.py to-txt input.wav output.txt
  python sinphony.py info  song_fourier.txt
        """,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_wav = sub.add_parser("to-wav", help="txt → WAV")
    p_wav.add_argument("input", help="输入 txt 文件")
    p_wav.add_argument("output", help="输出 WAV 文件")
    p_wav.add_argument("--sr", type=int, default=44100, help="目标采样率 (默认 44100)")
    p_wav.add_argument("--ch", choices=["both", "R", "L"], default="both",
                       help="输出声道 (默认 both)")

    p_txt = sub.add_parser("to-txt", help="WAV → txt")
    p_txt.add_argument("input", help="输入 WAV 文件")
    p_txt.add_argument("output", help="输出 txt 文件")
    p_txt.add_argument("--accuracy", type=int, default=1, help="精度等级 (默认 1)")

    p_txt = sub.add_parser("info", help="查看 txt 文件信息")
    p_txt.add_argument("input", help="输入 txt 文件")

    args = parser.parse_args()

    if args.cmd == "to-wav":
        txt_to_wav(args.input, args.output, target_sr=args.sr,
                   channel=args.ch)
    elif args.cmd == "to-txt":
        wav_to_txt(args.input, args.output, accuracy=args.accuracy)
    elif args.cmd == "info":
        data = parse_fourier_txt(args.input, verbose=False)
        df = data["unit_per_s"]  # Δf (Hz 频率分辨率) == unit/s
        T = 1.0 / df
        print(f"unit/s (Δf):  {df:.16f} Hz")
        print(f"accuracy:     {data['accuracy']}")
        print(f"时长 T=1/Δf:  {T:.3f} 秒 ({T/60:.1f} 分钟)")
        for ch_key, label in [("R", "Right"), ("L", "Left")]:
            ch = data[ch_key]
            n = ch["sin_count"]
            dc = ch["dc"]
            print(f"\n{label} channel:")
            print(f"  sin terms:  {n:,}")
            print(f"  DC offset:  {dc:.10f}")
            if ch["terms"]:
                max_amp = max(t[0] for t in ch["terms"])
                min_amp = min(t[0] for t in ch["terms"])
                max_hz = max(t[1] for t in ch["terms"])  # omega 即 Hz
                print(f"  amplitudes: {min_amp:.2e} ~ {max_amp:.4f}")
                print(f"  max freq:   {max_hz:.1f} Hz")


if __name__ == "__main__":
    main()