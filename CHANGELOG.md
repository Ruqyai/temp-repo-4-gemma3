# CHANGELOG


## v0.1.3 (2026-03-12)

### Bug Fixes

- **classifier**: Make log_prob an explicit constructor option
  ([`fb3aa86`](https://github.com/EleutherAI/delphi/commit/fb3aa863487683285a04f39f3c3134aa6cddb956))

### Continuous Integration

- Add uv quality gates and test discovery config
  ([`f851189`](https://github.com/EleutherAI/delphi/commit/f851189d1f61b2886176e37babc84c3198eb5a32))


## v0.1.2 (2026-03-12)

### Bug Fixes

- Add max_memory CLI support and fix vLLM prompt syntax
  ([#156](https://github.com/EleutherAI/delphi/pull/156),
  [`8a79bb7`](https://github.com/EleutherAI/delphi/commit/8a79bb77f5dbe815e087bb017bddfb8adfc23d4f))

* Add max_memory parameter to run config

Co-authored-by: Simon Schrader <simonschrader96@gmail.com>

* Use configurable max_memory for offline explainer

* Fix breaking change in prompt input formatting from vLLM

* [pre-commit.ci] auto fixes from pre-commit.com hooks

for more information, see https://pre-commit.ci

* Update vllm dependency version to after API breaking change

PR #18800: https://github.com/vllm-project/vllm/releases

---------

Co-authored-by: pre-commit-ci[bot] <66853113+pre-commit-ci[bot]@users.noreply.github.com>

- Bugs in _prepare_text and IntruderScorer prompt formatting
  ([#153](https://github.com/EleutherAI/delphi/pull/153),
  [`87f934d`](https://github.com/EleutherAI/delphi/commit/87f934d1b3728b1b6c038398a35287e0e24e4ad7))

* clean up some code and fix bug with preparing text with multiple false positives

* clean up intruderscorer a bit more

* another cleanup in _prepare_and_batch

* cleanup in Pipeline

* fix inconsistency with intruder example formatting

* simplify _generate error reporting

* ignore uv.lock

* undo code readability/style changes

* [pre-commit.ci] auto fixes from pre-commit.com hooks

for more information, see https://pre-commit.ci

---------

Co-authored-by: pre-commit-ci[bot] <66853113+pre-commit-ci[bot]@users.noreply.github.com>


## v0.1.1 (2025-08-27)

### Bug Fixes

- Make simulator workable
  ([`0ceaa10`](https://github.com/EleutherAI/delphi/commit/0ceaa10bd7b1b7beb411c488ff348c06fb868a67))

* bump vllm dependency to latest

* change simulation scoring default to all at once for local models

* use Role class instead of hardcoding

* delete old oai simulator scorer

* refactor

* [pre-commit.ci] auto fixes from pre-commit.com hooks

for more information, see https://pre-commit.ci

* linter fix

* replace lambda with function

---------

Co-authored-by: pre-commit-ci[bot] <66853113+pre-commit-ci[bot]@users.noreply.github.com>


## v0.1.0 (2025-06-17)

### Features

- Replaces print with logging ([#136](https://github.com/EleutherAI/delphi/pull/136),
  [`431d654`](https://github.com/EleutherAI/delphi/commit/431d6546c3a3a4ede107d5d4ebe0ffd75e5c923b))

* feat: replaces print with logging

* Change some infos to warnings in constructor

* [pre-commit.ci] auto fixes from pre-commit.com hooks

for more information, see https://pre-commit.ci

---------

Co-authored-by: Goncalo Paulo <30472805+SrGonao@users.noreply.github.com>

Co-authored-by: pre-commit-ci[bot] <66853113+pre-commit-ci[bot]@users.noreply.github.com>


## v0.0.2 (2025-06-02)

### Bug Fixes

- Initial release
  ([`6efa228`](https://github.com/EleutherAI/delphi/commit/6efa228c8d9c0fddc6cb21760c1adb1d51ffdc77))


## v0.0.1 (2025-04-21)

### Bug Fixes

- Add missing libraries
  ([`d6f7c72`](https://github.com/EleutherAI/delphi/commit/d6f7c72c0b9dd4fd12dc78aa75fd77d146b0199b))

- Add missing libraries
  ([`853d8ca`](https://github.com/EleutherAI/delphi/commit/853d8ca93256b8bda1395bab08199de45eb63926))

- Scope issue with sae causing it to not be loaded properly
  ([`94f40cc`](https://github.com/EleutherAI/delphi/commit/94f40cc8def426baa9be682e22c96d4c31a8b5ed))

- Scope issue with sae causing it to not be loaded properly
  ([`23caed7`](https://github.com/EleutherAI/delphi/commit/23caed746e0536da5a0739b6f0cdf12c678be467))

### Documentation

- Update README.md
  ([`4385b0b`](https://github.com/EleutherAI/delphi/commit/4385b0b3a9ee99fdbf5713a3f990ab5721b12d1e))

acessed -> accessed

- Update README.md
  ([`31b6896`](https://github.com/EleutherAI/delphi/commit/31b6896ae412903ab797da429f8c834a26a8dfed))

acessed -> accessed
