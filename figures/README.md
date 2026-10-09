# Render scripts

Part of [vla-geometry](../README.md).

Each script reads the panel files `$VLAGEOM_DATA/panels/*.npz` that the last stage of a pipeline
writes and writes the rendered files to `$VLAGEOM_FIGURES` (default `$VLAGEOM_DATA/figures`). The
renders run on the CPU in about 10 s per figure.

Figure 1 is rendered as one file. Figures 2-7 are composites assembled by hand: robot sketches,
titles, panel letters and the placement of the panels are drawn in a layout program. For these
figures the repository renders the data panels as separate files (`.pdf` and `.png`, except
`fig6_a`, which is a PDF only unless `fig6.py --png-a` is given; one file per panel or block of
panels), and the README of each figure's pipeline says which files make up the figure:

| figure | script | rendered files (`$VLAGEOM_FIGURES/`) | panels described in |
|---|---|---|---|
| 1 | `fig1.py` | `fig1.png`, `fig1.pdf` | [Figure 1](#figure-1) below; the tiles in [`pipelines/`](../pipelines/README.md#figure-1) |
| 2 | `fig2.py` | `fig2_top`, `fig2_ee_grid`, `fig2_target_grid` | [`pipelines/position/`](../pipelines/position/README.md) |
| 3 | `fig3_distance.py` | `fig3a` .. `fig3d` | [`pipelines/distance/`](../pipelines/distance/README.md#figure-3-distance-to-the-target-object) |
| 4 | `fig4_contact.py` | `fig4a`, `fig4b` | [`pipelines/contact/`](../pipelines/contact/README.md) |
| 5 | `fig5.py` | `fig5_strip`, `fig5a` .. `fig5d` | [`pipelines/phase/`](../pipelines/phase/README.md) |
| 6 | `fig6.py` | `fig6_a` (PDF), `fig6_bcd` | [`pipelines/wrist_steer/`](../pipelines/wrist_steer/README.md) |
| 7 | `fig7_gripper.py` | `fig7a` .. `fig7d` | [`pipelines/gripper_fig7/`](../pipelines/gripper_fig7/README.md) |

The options of `fig2.py` .. `fig7_gripper.py` (`--panel` and the input and output locations) are
listed by `--help` and in the stage table of their pipeline's README; the options of `fig1.py` are
in [Figure 1](#figure-1) below.

## Figure 1

```bash
python figures/fig1.py        # $VLAGEOM_DATA/panels/*.npz -> $VLAGEOM_FIGURES/fig1.png, fig1.pdf
```

The script reads the four panel files `$VLAGEOM_DATA/panels/{gripper,wrist,target,distance}.npz`
(the docstring of `figures/fig1.py` lists the arrays it uses from each). It writes `fig1.png` and
`fig1.pdf` (25.2 x 9.15 in, 350 dpi) to `$VLAGEOM_FIGURES`, default `$VLAGEOM_DATA/figures`.
`--panels DIR` and `--out-dir DIR` override the two locations. It runs on the CPU in about 10 s.

- **Legend notes** are formatted from values in the panel files. The target R² comes from
  `target.npz:note_r2`. The two distance R² values come from `distance.npz:r2_m` and `r2_cm`. The
  wrist note comes from `wrist.npz:roll` and `Z3` (see [the wrist
  tile](../pipelines/wrist/README.md)).
- **Layout.** All four 3-D boxes have the same height, so the four floors render at the same size.
  That height and the gripper tile's floor are fixed layout constants (`BOX_HEIGHT` and
  `GRIPPER_FLOOR` in `figures/fig1.py`). The other tiles' floors are fitted to their data.
- **Fonts.** With Product Sans (see [Fonts](#fonts) below) and the pinned library versions, the
  render from the paper's panel arrays has the same bytes as the paper's PNG. The PDF has the same
  bytes apart from its creation date. With the DejaVu Sans fallback only the text differs, about 4%
  of the PNG pixels.

## Fonts

The figures use Product Sans, which is not distributed here. If you have the font files, set
`VLAGEOM_FONT_DIR` to their directory (or install the font system-wide). Without it the figures use
DejaVu Sans, which ships with matplotlib; text widths then differ slightly from the paper, and
everything else is unchanged (in Figure 6 b-d the axes margins are measured from the rendered
text, so they move with the font as well).

matplotlib writes the fonts a PDF embeds in the order of their file paths. In the paper's PDFs
matplotlib's own fonts (the Computer Modern math fonts in `matplotlib/mpl-data/fonts`) come before
Product Sans, as they do when the path of `VLAGEOM_FONT_DIR` sorts after the path of the
matplotlib installation. With a font directory that sorts before it, the PDFs hold the same
objects in another order: the PNGs are unchanged, but the PDF byte comparisons of `tests/` report
a difference.
