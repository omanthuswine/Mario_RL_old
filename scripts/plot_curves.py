"""Vẽ đường cong học từ progress.csv và evals.csv của một hay nhiều lần chạy (mục 3, 4 của báo cáo).

    python scripts/plot_curves.py runs/1-1/tv1-e0-s1
    python scripts/plot_curves.py runs/1-1/tv1-e0-s1 runs/1-1/tv2-e1-s1 --labels "E0 γ=0,9" "E1 γ=0,99" \\
        --out bao-cao/hinh/e0-e1.pdf
    # các lần chạy cùng nhãn (ví dụ 2 seed) được gộp: đường trung bình, dải min-max
    python scripts/plot_curves.py runs/1-1/tv1-e0-s1 runs/1-1/tv2-e0-s2 runs/1-1/tv2-e1-s1 runs/1-1/tv1-e1-s2 \\
        --labels E0 E0 E1 E1 --smooth 5
    # chỉ vẽ một số ô
    python scripts/plot_curves.py runs/1-1/tv1-e0-s1 --panels flag eval_flag x_pos --cols 3

Mỗi ô là một đại lượng theo số bước, có trục y riêng. Đuôi tệp --out (.png, .pdf, .svg) quyết định định dạng;
báo cáo LaTeX nên dùng .pdf (ảnh véc-tơ).
"""
import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

# Bảng màu phân loại theo thứ tự cố định (đã kiểm tra mù màu trên nền sáng), kèm kiểu nét để in đen trắng.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
DASHES = ["-", "--", ":", "-."]
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

PANELS = {  # khóa: (tệp, cột, tiêu đề)
    "return": ("progress", "ep_return", "Lợi tức (TB 100 lượt gần nhất)"),
    "x_pos": ("progress", "x_pos", "Vị trí x cuối lượt (TB 100 lượt)"),
    "flag": ("progress", "flag_rate", "Tỉ lệ chạm cờ khi huấn luyện"),
    "eval_flag": ("evals", "flag_rate", "Tỉ lệ chạm cờ ở checkpoint"),
    "entropy": ("progress", "entropy", "Entropy"),
    "kl": ("progress", "approx_kl", "approxKL"),
    "clipfrac": ("progress", "clipfrac", "Tỉ lệ cắt"),
    "value_loss": ("progress", "value_loss", "Mất mát giá trị"),
}
FLAG_THRESHOLD = 0.8  # ngưỡng "vượt màn" của đề


def read_csv(path):
    """Đọc csv thành dict cột -> mảng số; bỏ qua cột không phải số (như x_pos của evals.csv)."""
    path = Path(path)
    if not path.exists():
        return {}
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    cols = {}
    for k in rows[0] if rows else []:
        try:
            cols[k] = np.array([float(r[k]) if r.get(k) not in ("", None) else np.nan for r in rows])
        except ValueError:
            pass
    return cols


def smooth(y, n):
    """Trung bình trượt n điểm về phía trước, giữ nguyên độ dài."""
    if n <= 1:
        return y
    return np.array([np.nanmean(y[max(0, i - n + 1):i + 1]) if np.isfinite(y[max(0, i - n + 1):i + 1]).any()
                     else np.nan for i in range(len(y))])


def combine(curves):
    """Gộp các đường (x, y) cùng nhãn: nội suy về lưới của đường đầu, trả (x, trung bình, min, max)."""
    if len(curves) == 1:
        x, y = curves[0]
        return x, y, None, None
    end = min(cx[-1] for cx, _ in curves)
    x = curves[0][0][curves[0][0] <= end]
    ys = np.stack([np.interp(x, cx, cy) for cx, cy in curves])
    return x, ys.mean(0), ys.min(0), ys.max(0)


def _style(ax, title, bottom):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=9, loc="left")
    ax.grid(True, color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelcolor=INK2, labelsize=7.5, length=0, pad=3)
    if bottom:
        ax.set_xlabel("Bước tác tử (triệu)", color=INK2, fontsize=8)


def plot(run_dirs, labels=None, panels=None, smooth_n=1, cols=4, out=None, title=None):
    labels = labels or [Path(r).name for r in run_dirs]
    if len(labels) != len(run_dirs):
        raise SystemExit(f"cần {len(run_dirs)} nhãn, nhận {len(labels)}")
    groups = {}  # nhãn -> [(progress, evals)], giữ thứ tự xuất hiện để màu đi theo nhãn
    for r, lab in zip(run_dirs, labels):
        if not (Path(r) / "progress.csv").exists():
            print(f"cảnh báo: không thấy {Path(r) / 'progress.csv'}")
        groups.setdefault(lab, []).append((read_csv(Path(r) / "progress.csv"), read_csv(Path(r) / "evals.csv")))
    if len(groups) > len(SERIES):
        raise SystemExit(f"tối đa {len(SERIES)} nhãn trên một hình; tách thành nhiều hình")
    keys = panels or list(PANELS)
    rows = -(-len(keys) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(2.7 * cols, 2.3 * rows + 0.5), squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    used = []

    for i, key in enumerate(keys):
        ax = axes[i // cols][i % cols]
        src_name, col, panel_title = PANELS[key]
        _style(ax, panel_title, bottom=i + cols >= len(keys))
        drawn = False
        for g, (lab, runs) in enumerate(groups.items()):
            curves = []
            for prog, ev in runs:
                src = prog if src_name == "progress" else ev
                if "global_step" in src and col in src and len(src[col]):
                    y = smooth(src[col], smooth_n) if src_name == "progress" else src[col]
                    curves.append((src["global_step"], y))
            if not curves:
                continue
            x, y, lo, hi = combine(curves)
            c, ls = SERIES[g], DASHES[g % len(DASHES)]
            if lo is not None:
                ax.fill_between(x / 1e6, lo, hi, color=c, alpha=0.15, lw=0)
            ax.plot(x / 1e6, y, color=c, ls=ls, lw=1.5, solid_capstyle="round", dash_capstyle="round",
                    marker="o" if src_name == "evals" else None, ms=3.5, mec=SURFACE, mew=0.8)
            drawn = True
        if not drawn:
            ax.text(0.5, 0.5, "không có dữ liệu", transform=ax.transAxes, ha="center", va="center",
                    color=MUTED, fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
        elif "flag" in key:
            ax.set_ylim(-0.03, 1.03)
            ax.axhline(FLAG_THRESHOLD, color=MUTED, lw=0.8)
            ax.annotate(f"ngưỡng {FLAG_THRESHOLD:g}", (0, FLAG_THRESHOLD),  # bên trái: đầu huấn luyện cờ còn ~0
                        xycoords=("axes fraction", "data"), xytext=(3, 3), textcoords="offset points",
                        ha="left", va="bottom", color=INK2, fontsize=7)
        if drawn:
            used.append(ax)
    xmax = max((ax.get_xlim()[1] for ax in used), default=1)
    for ax in used:  # mọi ô cùng khoảng trục x để so sánh theo cột dọc
        ax.set_xlim(0, xmax)
    for j in range(len(keys), rows * cols):
        axes[j // cols][j % cols].axis("off")

    top = 1.0
    if title:
        fig.suptitle(title, color=INK, fontsize=10, x=0.01, ha="left")
        top -= 0.06
    if len(groups) > 1:  # một lần chạy thì không cần chú giải
        handles = [Line2D([], [], color=SERIES[g], ls=DASHES[g % len(DASHES)], lw=1.5,
                          label=lab + (f" ({len(runs)} lần chạy: TB, dải min-max)" if len(runs) > 1 else ""))
                   for g, (lab, runs) in enumerate(groups.items())]
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, top), ncol=min(len(handles), 4),
                   frameon=False, fontsize=8, labelcolor=INK2, handlelength=2.5)
        top -= 0.07
    fig.tight_layout(rect=(0, 0, 1, top))
    out = Path(out or Path(run_dirs[0]) / "curves.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+", help="các thư mục run_dir (chứa progress.csv, evals.csv)")
    p.add_argument("--labels", nargs="+", help="nhãn mỗi lần chạy; cùng nhãn thì gộp (mặc định: tên thư mục)")
    p.add_argument("--panels", nargs="+", choices=list(PANELS), help="các ô cần vẽ (mặc định: cả 8)")
    p.add_argument("--smooth", type=int, default=1, help="trung bình trượt n lần cập nhật cho progress.csv")
    p.add_argument("--cols", type=int, default=4)
    p.add_argument("--title", default=None)
    p.add_argument("--out", default=None, help="mặc định <run đầu tiên>/curves.png")
    a = p.parse_args(argv)
    print(plot(a.runs, a.labels, a.panels, a.smooth, a.cols, a.out, a.title))


if __name__ == "__main__":
    main()
