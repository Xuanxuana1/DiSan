# DiSan README Paper Presentation Design

## Goal

Update the repository homepage to present DiSan as an EMNLP 2026 Main
Conference paper and make the paper's motivation and architecture immediately
understandable.

## Design

- Replace the anonymous-release heading with the paper name and full title.
- Add linked arXiv and EMNLP 2026 Main Conference badges.
- Add a short project summary and a navigation row for the paper, overview,
  setup, and experiments.
- Export the paper figures as web-friendly PNG files.
- Feature Figure 2 first as the architecture overview because its landscape
  layout remains readable on GitHub.
- Place Figure 1 in a later Motivation section because its portrait layout is
  more useful after the reader understands the project.
- Preserve the existing reproduction instructions and repository warnings.

## Verification

- Confirm both local image paths resolve.
- Confirm the arXiv URL returns successfully.
- Inspect the Markdown diff and rendered image dimensions.
- Push only after the working tree contains the intended README and image
  changes.
