# Spectral BVH Path Tracer

A spectral path tracer in a single Python file, with a BVH and an interactive Tk window.

## Install

```bash
pip install numpy numba
```

Tkinter is needed for the window and ships with most Python installs.

## Run

```bash
python spectral_tracer.py                          # interactive window
python spectral_tracer.py --scene studio           # cornell (default) or studio
python spectral_tracer.py --headless --spp 256 --width 800 --height 600 --out render.png
python spectral_tracer.py --bench                  # BVH vs brute force timing
python spectral_tracer.py --selftest               # BVH vs brute force correctness
```

The first launch takes a few seconds while Numba compiles. After that it is cached.

## Controls

| Input | Action |
|---|---|
| Left-drag | Orbit |
| Right-drag | Pan |
| Scroll | Zoom |
| Space | Pause |
| `s` | Save PNG |
| `r` | Reset view |

The side panel has sliders for lamp temperature, dispersion, film thickness and exposure. Aperture, bounces, white balance and a brute-force toggle are under Advanced.

## Headless options

`--scene`, `--width`, `--height`, `--spp`, `--depth`, `--temp`, `--dispersion`, `--exposure`, `--wb`, `--out`
