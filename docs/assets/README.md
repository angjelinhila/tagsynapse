# Figures

Hand-written SVGs, no external fonts (GitHub blocks them in `<img>`), system
font stack throughout. One palette so a reader learns the colours once:

| Role | Colour | Used for |
|---|---|---|
| Folksonomic / empirical | amber `#f59e0b` | user tags, `coUsedWith`, usage evidence (dashed) |
| Controlled / ontological | sky `#38bdf8` | FAST concepts, `broader` / `narrower` / `equivalent` / `associated` (solid) |
| Learned / inferred | violet `#a78bfa` | relation inference, GNN, semantic neighbours (dotted) |
| Human | green `#34d399` | tagging, validation |
| Surface | slate `#0f172a` / `#1e293b` | card backgrounds, so figures read the same in light and dark mode |

Dashed = observed, dotted = candidate, solid = asserted. Keep that when adding a figure.

`banner-dark.svg` / `banner-light.svg` share geometry; README swaps them with `<picture>`.
