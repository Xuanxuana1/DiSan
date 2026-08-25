# DiSan Project Page Migration Design

## Goal

Move the public-facing identity of the existing DiSan project page from the
legacy `RezinChow` repository to `Xuanxuana1/DiSan`, prominently show the EMNLP
2026 Main Conference acceptance, and make the site deployable from its current
directory.

## Page Changes

- Preserve the existing dark technical visual direction and page structure.
- Add an `EMNLP 2026 Main Conference Accepted` badge in the first viewport,
  next to the existing Intern-Shannon project badge.
- Update every repository link to `https://github.com/Xuanxuana1/DiSan`.
- Update every project-page link to `https://xuanxuana1.github.io/DiSan/`.
- Update the press-kit metadata with the acceptance status.

## Deployment

- Keep the static site in `disan-gh-pages/`.
- Add a GitHub Actions workflow using the official Pages actions.
- Upload `disan-gh-pages/` as the Pages artifact and deploy it on changes to
  the site or through manual dispatch.
- Require the repository owner to select GitHub Actions as the Pages source in
  repository settings.

## Verification

- Search for stale legacy URLs.
- Validate the workflow YAML and local asset paths.
- Serve the static site locally and inspect desktop and mobile screenshots.
- Check navigation, GitHub, arXiv, and citation-copy behavior.
