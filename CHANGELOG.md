# Changelog

## [1.2.0](https://github.com/Alberto-Codes/turboquant-vllm/compare/v1.1.1...v1.2.0) (2026-03-29)


### Features

* **benchmark:** add multi-model support for text-only models ([#32](https://github.com/Alberto-Codes/turboquant-vllm/issues/32)) ([f383895](https://github.com/Alberto-Codes/turboquant-vllm/commit/f383895f0e5970302f958bdfedeb94c1c7af572f))
* **kv-cache:** add context manager protocol and double-compression detection ([#24](https://github.com/Alberto-Codes/turboquant-vllm/issues/24)) ([d5c58f5](https://github.com/Alberto-Codes/turboquant-vllm/commit/d5c58f56875fb10997b5b563567bb30b6a7b0829))
* **triton:** add fused paged TQ4 decode attention kernel ([#37](https://github.com/Alberto-Codes/turboquant-vllm/issues/37)) ([ae7941e](https://github.com/Alberto-Codes/turboquant-vllm/commit/ae7941eb98f90925bd15ff7ccfbcb56637a8f3f5))
* **triton:** add fused paged TQ4 INT8 prefill kernel ([#41](https://github.com/Alberto-Codes/turboquant-vllm/issues/41)) ([bff651b](https://github.com/Alberto-Codes/turboquant-vllm/commit/bff651bcc23fe3b7f04432f53aa937cd9d879802))
* **triton:** add out parameter to tq4 compress/decompress wrappers ([#34](https://github.com/Alberto-Codes/turboquant-vllm/issues/34)) ([6fc60d8](https://github.com/Alberto-Codes/turboquant-vllm/commit/6fc60d8634600b1d5b520944d75a42b200c7c02d))
* **verify:** add verify CLI for compression quality checks ([#27](https://github.com/Alberto-Codes/turboquant-vllm/issues/27)) ([91cbf0e](https://github.com/Alberto-Codes/turboquant-vllm/commit/91cbf0e0da9c3355dc4a2f80150c0942d9ae7513))
* **vllm:** add CUDA graph buffer pre-allocation to TQ4 backend ([#35](https://github.com/Alberto-Codes/turboquant-vllm/issues/35)) ([2106d09](https://github.com/Alberto-Codes/turboquant-vllm/commit/2106d09aa34799cb57a857371a8d531c6c1c4298))
* **vllm:** add fused paged TQ4 decode backend integration and feature gating ([#39](https://github.com/Alberto-Codes/turboquant-vllm/issues/39)) ([fa6b220](https://github.com/Alberto-Codes/turboquant-vllm/commit/fa6b2208a98fc201447248a5c42133b8b2a31c3f))


### Bug Fixes

* **docs:** address Copilot review findings from Epic 6 PRs ([#40](https://github.com/Alberto-Codes/turboquant-vllm/issues/40)) ([f36ba66](https://github.com/Alberto-Codes/turboquant-vllm/commit/f36ba66f8db67d608cff52e2b3c24efb9714bac3)), closes [#38](https://github.com/Alberto-Codes/turboquant-vllm/issues/38)
* **experiments:** address code review findings for Story 6.5 ([073166c](https://github.com/Alberto-Codes/turboquant-vllm/commit/073166cb3127211840bda7dfee36d2335e337280))
* **git:** restore directory-only matching and IDE ignores in gitignore ([4ecc62e](https://github.com/Alberto-Codes/turboquant-vllm/commit/4ecc62e6b89819df58000a0e9cfba4f4cc00dee8))
* **test:** add gc.collect() before cuda empty_cache in verify GPU test ([e003f24](https://github.com/Alberto-Codes/turboquant-vllm/commit/e003f24fbf8bfdf0e192aefeca78f78c5c19426a))
* **triton:** resolve code review findings for fused paged TQ4 kernel ([ae7941e](https://github.com/Alberto-Codes/turboquant-vllm/commit/ae7941eb98f90925bd15ff7ccfbcb56637a8f3f5))
* **verify:** handle explicit None head_dim in _detect_model_config ([418661d](https://github.com/Alberto-Codes/turboquant-vllm/commit/418661d25a4f181ac174e4787605119e4c6ec4b6))
* **verify:** restrict --bits to valid choices [3, 4] ([91cbf0e](https://github.com/Alberto-Codes/turboquant-vllm/commit/91cbf0e0da9c3355dc4a2f80150c0942d9ae7513))
* **vllm:** guard INT8 prefill dispatch for single-sequence only ([bff651b](https://github.com/Alberto-Codes/turboquant-vllm/commit/bff651bcc23fe3b7f04432f53aa937cd9d879802))


### Performance Improvements

* **benchmark:** add experiment 018 CUDA graph decode latency ([#36](https://github.com/Alberto-Codes/turboquant-vllm/issues/36)) ([4ad2210](https://github.com/Alberto-Codes/turboquant-vllm/commit/4ad22107df0e1ba26a374e7b9992e59843e4fe51))
* **benchmark:** add experiment 018 fused decode smoke test log ([fa6b220](https://github.com/Alberto-Codes/turboquant-vllm/commit/fa6b2208a98fc201447248a5c42133b8b2a31c3f))
* **triton:** add kernel benchmarks and optimize autotune configs ([#42](https://github.com/Alberto-Codes/turboquant-vllm/issues/42)) ([073166c](https://github.com/Alberto-Codes/turboquant-vllm/commit/073166cb3127211840bda7dfee36d2335e337280))

## [1.1.1](https://github.com/Alberto-Codes/turboquant-vllm/compare/v1.1.0...v1.1.1) (2026-03-28)


### Performance Improvements

* **tests:** add TQ4 fixture, cache simulations, and mark slow tests ([#13](https://github.com/Alberto-Codes/turboquant-vllm/issues/13)) ([96fc960](https://github.com/Alberto-Codes/turboquant-vllm/commit/96fc960eeb033f2948980d13d1fcbc20c4af4313)), closes [#8](https://github.com/Alberto-Codes/turboquant-vllm/issues/8)

## [1.1.0](https://github.com/Alberto-Codes/turboquant-vllm/compare/v1.0.0...v1.1.0) (2026-03-27)


### Features

* **container:** vLLM Containerfile + per-layer cosine tests ([#3](https://github.com/Alberto-Codes/turboquant-vllm/issues/3)) ([a71abc7](https://github.com/Alberto-Codes/turboquant-vllm/commit/a71abc7616b765bd29410a8b3e88f92061d75c0d))


### Documentation

* add MkDocs site with API reference and container guide ([a71abc7](https://github.com/Alberto-Codes/turboquant-vllm/commit/a71abc7616b765bd29410a8b3e88f92061d75c0d))

## [1.0.0](https://github.com/Alberto-Codes/turboquant-vllm/compare/v0.1.0...v1.0.0) (2026-03-27)


### Features

* **packaging:** production README, Apache 2.0, release-please, GitHub templates ([#1](https://github.com/Alberto-Codes/turboquant-vllm/issues/1)) ([5ed0c12](https://github.com/Alberto-Codes/turboquant-vllm/commit/5ed0c12045049a1d50f58da0484b0e2d9e2297bb))


### Documentation

* **architecture:** fix stale turboquant_consumer package name ([5ed0c12](https://github.com/Alberto-Codes/turboquant-vllm/commit/5ed0c12045049a1d50f58da0484b0e2d9e2297bb))
* final roadmap update — success criteria met, packaging section added ([4239de0](https://github.com/Alberto-Codes/turboquant-vllm/commit/4239de0cb2db4e88bd9b83a46f9ed98cecb6e66b))
* import technical research from molmo-video-analyzer ([9750fe9](https://github.com/Alberto-Codes/turboquant-vllm/commit/9750fe9b0c8ccbdbcfc2bcb3fe17fbe6b8d662a4))

## Changelog
