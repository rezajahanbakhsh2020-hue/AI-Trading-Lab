# Production Deployment Operational Guide

This document details the operational deployment boundary for Project 1's continuous live execution worker and its outbound integration with Project 2.

## 1. System Architecture

The continuous production pipeline connects:
1. Authoritative Promoted-Candidate Resolution (`results/research_experiments/by_candidate/`)
2. `ProductionRuntimeAuthorization` & Authorization Receipt
3. Live BiQuote Market Data Snapshot Acquisition (`XAUUSD`)
4. Closed-Candle Gate & Deduplication Boundary
5. Canonical Downstream Execution (`evaluate_authorized_live_runtime`)
6. Canonical Decision, Signal & Risk Evaluation (`CanonicalLiveDecision`)
7. Project 2 Intelligence Publication (`ProductionIntelligencePublication`)
8. Project 2 Publisher (`Project2Publisher`) -> `POST /api/v1/integration/project1/ingest`

## 2. Deployment Prerequisites & Artifact Provisioning

### Authoritative Research Store Artifacts
The live worker strictly enforces fail-closed candidate resolution via `resolve_authoritative_promoted_candidate()`.
It does **NOT** manufacture candidates, candidate IDs, or synthetic fallbacks.

The repository includes canonical persisted candidate bindings under git tracking:
- `results/research_experiments/by_candidate/cand_moving_average_5m/candidate.json`
- `results/research_experiments/by_candidate/cand_momentum_5m/candidate.json`
- `results/research_experiments/by_candidate/cand_moving_average_1d/candidate.json`
- `results/research_experiments/by_candidate/cand_momentum_1d/candidate.json`

For the 5m continuous live production signal path, `--candidate-id cand_moving_average_5m` (or `cand_momentum_5m`) must be passed when multiple candidates exist for a timeframe to resolve ambiguity.

### Environment Variables
Configure the following environment variables on Render (or target deployment host):

```env
PYTHONPATH=.
PROJECT2_PUBLISH_ENABLED=true
PROJECT2_PUBLISH_URL=https://ai-trading-lab-platform-1.onrender.com/api/v1/integration/project1/ingest
PROJECT2_API_KEY=<SECRET_API_KEY>
```

*Note: `PROJECT2_API_KEY` is a sensitive credential and must be managed securely via Render Blueprint `sync: false` or environment settings. Never commit API keys to git.*

## 3. Render Background Worker Deployment

The deployment definition is provided in `render.yaml`:

```yaml
services:
  - type: worker
    name: p1-live-execution-worker
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: python run_live_execution.py --continuous --symbol XAUUSD --interval 5m --candidate-id cand_moving_average_5m --publish
    envVars:
      - key: PYTHONPATH
        value: "."
      - key: PROJECT2_PUBLISH_ENABLED
        value: "true"
      - key: PROJECT2_PUBLISH_URL
        value: "https://ai-trading-lab-platform-1.onrender.com/api/v1/integration/project1/ingest"
      - key: PROJECT2_API_KEY
        sync: false
```

## 4. Operational Startup & Fail-Closed Preflight

When the worker starts (`run_live_execution.py --continuous`), it executes a mandatory startup readiness preflight check (`verify_startup_readiness()`):

1. Resolves the authoritative promoted candidate artifact for the configured symbol (`XAUUSD`) and timeframe (`5m`).
2. If the candidate artifact is missing, ambiguous, or invalid, the worker logs `Startup readiness preflight failed` and exits immediately (`sys.exit(1)`).
3. If preflight succeeds, the continuous market polling loop starts, detecting closed candles, deduplicating cycles, and publishing contract v1.0 payloads to Project 2.

## 5. Deployment Proof & Verification Boundaries

1. **Repo-Side Production Readiness:** Verified via automated pytest suite (`tests/test_production_activation_deployment.py`).
2. **Render Service Creation:** Deploy the background worker using `render.yaml` on Render.
3. **Secret Configuration:** Supply `PROJECT2_API_KEY` in Render Dashboard.
4. **Authoritative Artifact Provisioning:** Ensure tracked candidate artifacts in `results/research_experiments/` are present.
5. **Live P1->P2 Delivery Proof:** Confirmed by inspecting Render worker logs for `Publish Result: {'status': 'PUBLISHED', ...}` or reviewing Project 2 dashboard/ingest logs.
