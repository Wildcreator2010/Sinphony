# Sinphony 🎵

> 傅里叶级数歌曲文件 ↔ WAV 双向转换工具
> A two-way converter between a song's Fourier-series text and WAV audio.

**Sinphony** 把一首歌曲的波形拆解成的**千万级正弦波分量**(`song_fourier.txt`)反向重建为可播放的 WAV 音频,或者把 WAV 分解回同样格式的正弦波级数文本。

**Sinphony** reconstructs a playable WAV from a song that has been decomposed into millions of sine components (a `song_fourier.txt` file), and can also decompose a WAV back into that same Fourier-series text format.

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

---

## 原理 / How it works

文件里每个正弦项形如:

```
<A> *sin <f> *x + <φ>
```

它表示 `A · sin(2π·f·t + φ)`，其中:

- `A` = 振幅 (amplitude)
- `f` = 频率，单位 **Hz** (frequency, in Hz)
- `φ` = 相位 (phase, radians)
- `x` = 时间，单位秒 (time, in seconds)

文件头的 `unit/s` 是频率分辨率 **Δf (Hz)**。转换时:

1. 对每个项计算 bin 序号 `k = round(f / Δf)`；
2. 把 `A` 和 `φ` 累加进对应频域的复数 bin；
3. 做一次 **IFFT** 得到时域波形；
4. 重采样到目标采样率，归一化后写成 16-bit WAV。

Each sine term means `A · sin(2π·f·t + φ)`, where `A` is amplitude, `f` is the frequency in **Hz**, `φ` is the phase in radians, and `x` is time in seconds. The header field `unit/s` is the frequency resolution **Δf (Hz)**. The converter maps each term to its frequency bin (`k = round(f / Δf)`), accumulates amplitude/phase into a complex spectrum, performs a single **IFFT** to get the time-domain waveform, resamples to the target rate, and writes a normalized 16-bit WAV.

### 文件格式 / Text format

```
unit/s: 0.02543417717671088523
accuracy: 1
R_sin_count: 5446476
L_sin_count: 5447168

R:

0.0003112172910129
0.0300030222733719 *sin 56.8457132219898469 *x + -3.0410987220695032
...

L:

...
```

- `unit/s` — 频率分辨率 Δf (Hz)
- `accuracy` — 精度等级
- `R_sin_count` / `L_sin_count` — 左右声道正弦项数
- 每个声道段先是一个直流偏置 (DC offset)，随后是 `A *sin f *x + φ` 项

---

## 安装 / Install

需要 Python 3.8+ 和 NumPy。

```bash
# 建议使用虚拟环境
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install numpy
```

Requires Python 3.8+ and NumPy.

---

## 使用 / Usage

### txt → WAV (重建音频 / build audio)

```bash
python sinphony.py to-wav song_fourier.txt output.wav
python sinphony.py to-wav song_fourier.txt output.wav --sr 44100   # 指定采样率
python sinphony.py to-wav song_fourier.txt output.wav --ch both    # both / R / L
```

### WAV → txt (分解 / decompose)

```bash
python sinphony.py to-txt input.wav output.txt
python sinphony.py to-txt input.wav output.txt --accuracy 1
```

### 查看信息 / Inspect

```bash
python sinphony.py info song_fourier.txt
```

---

## 处理大文件 / Large files

原始文本可达数百 MB、上千万行。Sinphony 采用**流式解析**：

- 单次遍历文件，边读边累加频谱，不缓存千万级项列表；
- 使用 NumPy 向量化与 FFT，内存占用稳定；
- 左右声道分别完成 IFFT 后合并输出。

The source text can be hundreds of MB with tens of millions of lines. Sinphony uses **streaming parsing**: it reads the file once and accumulates the spectrum on the fly (never caching the whole term list), uses vectorized NumPy / FFT, and keeps memory usage predictable.

---

## 示例项目 / Example

生成的 `song_final.wav` (44.1 kHz, 39.32 s, stereo) 正是由一个约 758 MB、2178 万行、含 1089 万个正弦项的 `song_fourier.txt` 重建而来。

A sample `song_final.wav` (44.1 kHz, 39.32 s, stereo) was reconstructed from a ~758 MB `song_fourier.txt` containing 21.78 million lines and ~10.89 million sine terms.

---

## 许可 / License

本项目采用 **Apache License 2.0** 发布。[查看全文](LICENSE)

Copyright © 2026 [wildcreator2010](https://github.com/wildcreator2010)

Released under the **Apache License 2.0**. See [LICENSE](LICENSE) for details.
