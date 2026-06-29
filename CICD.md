# CI/CD Guide

This document explains how CI/CD is organized for this repository, what GitHub
Actions is doing, how GitHub Pages deployment works in this project, and how to
operate the current documentation pipeline end to end.

## 1. What CI/CD means in this repository

In this repository, CI/CD currently focuses on documentation delivery rather
than model training automation.

- **CI** means automatically checking and building the documentation source.
- **CD** means automatically deploying the generated static site to GitHub
  Pages.

The current pipeline is intentionally simple:

1. push code to `main`
2. GitHub Actions starts the documentation workflow
3. Sphinx builds HTML from `doc/source`
4. the generated HTML is uploaded as a Pages artifact
5. GitHub Pages publishes the site

## 2. What GitHub Actions is

GitHub Actions is GitHub's hosted CI/CD system.

In practical terms, it does the same kind of job that systems like Jenkins do:

- detect repository events such as pushes or manual workflow dispatch
- check out repository code
- run build scripts
- run tests or validations
- deploy artifacts

The difference is mainly operational style:

- GitHub Actions is integrated directly into GitHub
- Jenkins is usually self-hosted and more manually managed

## 3. How this project uses GitHub Actions

This repository currently uses:

- `.github/workflows/docs.yml`

That workflow is responsible for documentation deployment.

### Trigger conditions

The workflow runs when:

- code is pushed to `main`
- the workflow is started manually from the GitHub Actions UI

### Permissions

The workflow requests these permissions:

- `contents: read`
- `pages: write`
- `id-token: write`

These are required for GitHub Pages deployment through Actions.

### Concurrency behavior

The workflow uses:

- one deployment group for Pages
- `cancel-in-progress: true`

That means if multiple documentation deployments are triggered in close
succession, older in-progress deployments can be canceled in favor of the newer
one. This is a common CI/CD optimization pattern.

## 4. Repository-specific documentation deployment flow

The current documentation path is:

- source files: `doc/source/`
- local build output: `doc/build/html/`
- deployment target: GitHub Pages

### Local build

The local build command is:

```bash
make -C doc html
```

This runs Sphinx and generates static HTML under:

```text
doc/build/html/
```

### Why `doc/build/` is not tracked

`doc/build/` is not committed because it is generated output.

This project keeps:

- documentation source under version control
- generated Pages output as a deployment artifact

This avoids:

- committing many generated HTML files
- source/build mismatch
- unnecessary repository noise

## 5. How GitHub Pages works in this repository

This project uses **GitHub Pages with GitHub Actions** as the source.

That means Pages does **not** publish directly from a tracked `docs/` directory
or a committed `gh-pages` branch.

Instead:

1. Actions builds the HTML
2. Actions uploads the build result
3. Pages deploys that uploaded artifact

## 6. How Pages was enabled for this project

The actual enablement flow for this repository was:

1. open the repository on GitHub
2. go to `Settings`
3. open `Pages`
4. under `Build and deployment`
5. set `Source` to `GitHub Actions`

After that, the deployment workflow can publish successfully.

## 7. What happened during the first failed deployment

The first deployment failed even though the workflow file was correct.

Why:

- the build job succeeded
- the artifact upload succeeded
- but GitHub Pages itself had not been fully enabled yet

The failure mode was a Pages deployment creation error, essentially because
GitHub could not create a Pages deployment before Pages was properly enabled for
the repository.

Once `Settings -> Pages -> Source -> GitHub Actions` was configured, the fix
was simply:

1. reopen the failed workflow run
2. click `Re-run all jobs`

That rerun used the already-correct workflow and then deployed successfully.

## 8. Current documentation workflow structure

The workflow has two jobs:

### `build`

This job:

- checks out the repository
- sets up Python
- installs documentation dependencies
- runs `make -C doc html`
- uploads `doc/build/html` as the Pages artifact

### `deploy`

This job:

- waits for `build`
- takes the uploaded artifact
- deploys it to GitHub Pages

## 9. How to operate this pipeline in the future

### Normal update flow

When documentation changes:

1. update files under `doc/source/`
2. optionally run locally:
   ```bash
   make -C doc html
   ```
3. commit and push to `main`
4. GitHub Actions runs automatically
5. GitHub Pages updates the public site

### If deployment fails

Check in this order:

1. `Actions` page
2. whether `build` failed or `deploy` failed
3. `Settings -> Pages`
4. confirm `Source` is `GitHub Actions`
5. rerun the workflow if configuration was corrected after the failed run

## 10. GitHub Actions limits for free users

The practical limits depend on repository visibility.

### Public repositories

For public repositories, Actions is usually much more permissive for normal
build usage.

### Private repositories

For private repositories, free usage is limited more tightly, especially in:

- total runner minutes
- artifact/log retention
- concurrency

Also note:

- Linux runners are usually the cheapest path
- macOS runners are much more expensive

This repository currently uses a Linux-based workflow, which is the right
default choice for cost control.

## 11. What happens in high-frequency push environments

In larger engineering organizations, CI/CD is not handled by letting every
single push run a full heavyweight pipeline to completion.

Common strategies include:

- canceling older in-progress runs
- running only the latest relevant commit
- path-based triggers
- branch-based workflow separation
- caching dependencies and build layers
- splitting lightweight checks from heavyweight deployment jobs
- using self-hosted runners when throughput becomes large

This repository already adopts one small but important version of that idea:

- deployment concurrency control for Pages

## 12. Why this design is appropriate for this project

This repository is a research/code/documentation project rather than a
high-frequency web product backend.

So the current design is appropriate because it is:

- simple
- maintainable
- low-overhead
- well integrated with GitHub
- sufficient for documentation publishing

If later needed, the pipeline can be extended to include:

- test workflows
- lint workflows
- packaging workflows
- benchmark or reproducibility checks

## 13. Current public documentation URL

The expected public Pages URL for this repository is:

```text
https://billzi2016.github.io/DeepChrInteract-v2/
```

## 14. Key files

- workflow: `.github/workflows/docs.yml`
- Sphinx source root: `doc/source/`
- local build root: `doc/build/html/`
- English repository overview: `README.md`
- Chinese repository overview: `README_CN.md`

