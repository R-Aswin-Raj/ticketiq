# TicketIQ — Self-Optimizing Support Triage Agent

TicketIQ is a FastAPI service that takes a support ticket, works out what it's
about, scores how the customer feels about each part of the product, pulls the
relevant policy, decides what to do, writes a reply, and then learns which
pipeline configuration works best from the feedback it gets back. The whole
thing runs as a resumable, dependency-aware DAG instead of one big function.

By default nothing leaves your machine (`LLM_MODE=mock`). You don't need an API
key, a model download, or a network connection to run the service, the tests, or
the RL experiment.

---

## Quick start

### Without Docker

```bash
git clone <repo> && cd ticketiq
pip install poetry
poetry install
cp .env.example .env

export PYTHONPATH=.
python scripts/generate_dataset.py                    # data/tickets.jsonl (180 tickets)
python scripts/train_classifier.py --save             # trains + prints held-out metrics
uvicorn ticketiq.main:app --reload
```

The interactive API lives at http://localhost:8000/docs.

### With Docker

```bash
docker compose up --build          # http://localhost:8000
# or
docker build -t ticketiq . && docker run -p 8000:8000 ticketiq
```

### Switching the LLM backend

It's one variable. Nothing else in the code changes.

```bash
LLM_MODE=mock                      # offline deterministic stub (default)

LLM_MODE=local                     # Ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2

LLM_MODE=cloud                     # any OpenAI-compatible endpoint
CLOUD_BASE_URL=https://api.openai.com/v1
CLOUD_MODEL=gpt-4o-mini
CLOUD_API_KEY=sk-...
```

To use Ollama, run `ollama pull llama3.2 && ollama serve` first, then set
`LLM_MODE=local`.

---

## Exercising the API

```bash
# 1. Submit a ticket
curl -s -X POST http://localhost:8000/ticket \
  -H 'Content-Type: application/json' \
  -d '{
        "subject": "Charged twice this month",
        "body": "We were billed twice on invoice INV-1201. Please refund the duplicate.",
        "tier": "pro",
        "customer_id": "cust-42",
        "order_id": "ord-991"
      }' | jq

# 2. Inspect real per-stage state from the workflow engine
curl -s http://localhost:8000/ticket/txn_XXXX/status | jq '.stages[] | {name, status, duration_s}'

# 3. Send feedback. This is the call that updates the bandit.
curl -s -X POST http://localhost:8000/feedback \
  -H 'Content-Type: application/json' \
  -d '{"transaction_id": "txn_XXXX", "score": 1}' | jq

# 4. Watch the bandit learn
curl -s http://localhost:8000/rl/stats | jq '.action_distribution'

# 5. Re-run a single stage without repeating upstream work
curl -s -X POST http://localhost:8000/ticket/txn_XXXX/rerun/agent | jq

# 6. See the DAG, including which stages run in parallel
curl -s http://localhost:8000/pipeline/graph | jq
```

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/ticket` | Run the full pipeline; returns category, aspect sentiment, snippets, agent trace, reply, config used, latency, transaction id |
| `POST` | `/feedback` | Binary feedback (1/0) turns into a reward and updates the bandit |
| `GET` | `/ticket/{id}/status` | Real per-stage state and outputs from the DAG store |
| `POST` | `/ticket/{id}/rerun/{stage}` | Re-run one stage, replaying completed upstream stages |
| `GET` | `/rl/stats` | Bandit Q-table, pull counts, action distribution |
| `GET` | `/pipeline/graph` | DAG structure and parallel levels |
| `GET` | `/healthz` | Liveness and active LLM mode |

---

## Architecture

```
POST /ticket
     │
     ├── classify ──────┐          (level 0 — these two run concurrently)
     ├── aspects ───────┤
     │                  ▼
     │              urgency        category + aspect sentiment + tier → RL state
     │                  ▼
     │           select_config     ← contextual bandit picks the arm
     │                  ▼
     │             retrieve        ← top-K from the arm
     │                  ▼
     │               agent         ← ReAct: decide → tool → observe → draft
     │                  ▼
     │              respond
     ▼
TicketResponse + transaction id     ──► POST /feedback ──► reward ──► bandit
```

```
ticketiq/
├── config.py          settings from env/.env; LLM_MODE lives here
├── schemas.py         Pydantic request/response models
├── main.py            FastAPI app
├── ml/                text, vectorizer, logistic_regression, metrics,
│                      classifier, sentiment, aspects, urgency
├── rag/               kb (chunking), embeddings, store (cosine index)
├── llm/               base protocol, mock, remote (Ollama/cloud), configs, factory
├── agent/             tools, react
├── rl/                bandit, service
├── workflow/          engine (DAG), state (SQLite)
└── pipeline/          stages (the DAG), service (application layer)
```

---

## 1. Classical ML, written from scratch

The brief rules out `model.fit()` from scikit-learn for the classifier. I went a
step further: scikit-learn isn't a dependency at all. The vectoriser, the
classifier, and the metrics are all hand-written. The vectoriser and metrics are
unit-tested against closed-form values, and the classifier against its
convergence behaviour and held-out accuracy.

The **TF-IDF** vectoriser (`ml/vectorizer.py`) builds a sparse dict per document:

```
tf(t,d)  = count(t,d) / |d|
idf(t)   = ln((1 + N) / (1 + df(t))) + 1        smoothed, always > 0
x(t,d)   = tf · idf, L2-normalised
```

It uses unigrams and bigrams. The subject gets repeated once before
vectorising, since word-for-word it carries more signal than the body.

The classifier is a **multinomial logistic regression**
(`ml/logistic_regression.py`), trained from scratch by gradient descent on the
softmax cross-entropy loss with L2 regularisation. It optimises `P(class | features)`
directly, so unlike Naive Bayes it makes no feature-independence assumption,
which matters on TF-IDF where a bigram and its own unigrams are strongly
correlated. The softmax output is a genuine posterior, which the agent and the
RL layer read as a confidence score:

```
z_c(x)   = w_c · x + b_c
P(c|x)   = softmax(z)_c = exp(z_c) / Σ_k exp(z_k)
loss     = −(1/N) Σ_x log P(y|x) + (λ/2) Σ_c ‖w_c‖²
∂/∂w_c   = (1/N) Σ_x (P(c|x) − 1[y=c]) x + λ w_c
```

The loss is convex, so gradient descent from a zero start reaches the global
optimum and training is deterministic without a seed.

**Why logistic regression.** The brief allows any classical model, and I picked
logistic regression because it is the principled fit for TF-IDF features: being
discriminative, it doesn't double-count the correlated terms (a bigram and its
unigrams) that a Naive Bayes independence assumption would, and it degrades
gracefully as the corpus grows and vocabulary gets noisier. It also emits real
probabilities, which the agent and bandit consume as a confidence score. What I
did not reach for: a linear SVM (classifies well on text but gives no
probabilities out of the box, which this pipeline needs) or a fine-tuned
transformer (against the "classical ML" requirement and pointless on a corpus
this small).

Running `python scripts/train_classifier.py` on the 180-ticket set with a
stratified 75/25 split gives:

```
samples=44  accuracy=0.955  macro_f1=0.954
class               prec     rec      f1     n
account            0.909   0.909   0.909    11
billing            1.000   0.909   0.952    11
feature_request    0.917   1.000   0.957    11
technical          1.000   1.000   1.000    11

accuracy across 10 splits: mean=0.964 min=0.932 max=1.000 stdev=0.021
```

A note on the dataset. The first version I generated from templates scored a
perfect 1.000. That doesn't mean the model was good; it means the benchmark was
useless. So I added 16 deliberately ambiguous tickets: billing problems written
in access vocabulary, a feature request phrased around refunds, one-word
messages like "Slow". The score fell to a believable ~0.96. I also report a
ten-seed mean next to the single-split number, because a 44-row test set has
real variance and one number would be cherry-picking. The errors that remain are
exactly the hard cases I planted, e.g. an account ticket about seat charges that
lands in billing.

**Aspect-level sentiment** lives in `ml/aspects.py` and `ml/sentiment.py`.
Aspects are spotted with a curated keyword lexicon covering eight product areas.
Each aspect is then scored only over the clauses that actually mention it, and
those clauses are split on contrastive conjunctions first. That clause splitting
is what makes the score aspect-level rather than one blurred average:

```
"Support was great but the dashboard crashes and exports time out."
  support response time  +0.53  positive
  performance            −0.81  negative
  data & reporting       −0.56  negative
```

Scoring uses `vaderSentiment` when it's installed and falls back to an
equivalent built-in lexicon scorer with negation and intensifier handling
otherwise. Both return a compound score in [−1, 1], so nothing downstream has to
know which one is running.

**Urgency** (`ml/urgency.py`) is a transparent weighted blend of tier, category,
worst-aspect negativity, and escalation-keyword pressure. It's clipped to [0, 1]
and bucketed low/medium/high for the RL state. I kept it rule-based on purpose. A
learned urgency model would drift underneath the bandit and muddy reward
attribution, and that trade isn't worth it here.

---

## 2. RAG

There are five markdown KB documents in `data/kb/` covering refund and billing
policy, account and access troubleshooting, technical and performance
troubleshooting, feature-request handling, and escalation rules (with an SLA
table).

Chunking is markdown-aware. Sections are cut on headings first and only split
further when they're too big, with overlap. Each chunk's heading is prepended
and weighted in the indexed text, because short queries match on headings far
more reliably than on prose.

The index itself is brute-force cosine over roughly 24 chunks. I picked an
in-memory index over FAISS or Chroma on purpose. At this corpus size a linear
scan is exact, instant, and one fewer heavy dependency to carry. `VectorStore.add`
and `.search` keep the same signature, so dropping FAISS in later touches a
single file.

I tried two embedding backends and kept the one that retrieved better:

| Query | Hashing embeddings | TF-IDF cosine (default) |
|---|---|---|
| "double charged and want a refund" | Refund eligibility ranked **5th** | ranked **2nd** |
| "SAML SSO login fails for everyone" | correct top-1 | correct top-1, higher margin |
| "password reset email never arrives" | correct top-1 | correct top-1 |

Rare, high-signal terms like SAML, VAT, or chargeback collide into shared buckets
under feature hashing, but each keeps its own dimension under TF-IDF. Set
`RAG_BACKEND=dense` to switch to hashing or `sentence-transformers`.

One weakness I want to be upfront about: pure lexical retrieval falls over on
vocabulary mismatch. "I want my money back" doesn't literally contain the word
"refund". The pipeline patches most of this by appending the predicted category
and detected aspect names to the retrieval query, which bridges the gap most of
the time. A production system should use dense embeddings; as above, that's one
env var away.

### The two LLM configurations (requirement 3)

Requirement 3 asks for two distinct LLM configurations for the RL layer to
choose between. Worth being precise about where that lives, because it's easy to
confuse with the *backend* setting:

* **Ticket classification is not an LLM.** That's the hand-rolled logistic
  regression from section 1. The LLM is only involved in the agent's reasoning
  and the final reply.
* **The two configurations are two system prompts** (`llm/configs.py`):
  `concise` (policy-first, at most four sentences, temperature 0.1) and
  `empathetic` (leads with impact, gives numbered next steps, temperature 0.35).
  Same model, two genuinely different instruction sets and latency profiles.
* Cross those two prompts with top-K ∈ {2, 5} and the bandit gets **four arms**
  to learn between.
* `LLM_MODE` (mock / local / cloud) is orthogonal. It swaps the whole backend
  under all four arms at once; it is not one of the choices the bandit makes.

---

## 3. Agentic layer

`agent/react.py` runs a bounded reason–act–observe loop:

1. **Guardrail before the model.** A deterministic regex policy check runs first.
   Data loss, legal or regulatory exposure, security incidents, and full outages
   escalate without ever asking an LLM. Those calls shouldn't ride on a small
   model's judgement.
2. **Thought and action.** One completion returns JSON that picks
   `answer_directly`, `tool_call`, or `escalate`. The parser tolerates code
   fences and surrounding prose, because local models emit both. If the output
   won't parse, it falls back to answering from context instead of erroring out.
3. **Observation.** `check_account_status(customer_id)` or
   `check_refund_eligibility(order_id)` runs. Both are deterministic mocks, which
   keeps tests and simulations reproducible. Real identifiers are injected
   server-side; the model never gets to supply them.
4. **Answer.** A second completion drafts the reply from the context plus the
   observation.

I cap the loop at one tool call by design. Unbounded ReAct is a latency and cost
risk, and latency is penalised directly by the reward function, so letting the
loop run free would fight the objective. The full reasoning trace, the tool
arguments, and the tool results all come back in the API response and go to the
logs.

If the LLM backend is down, the agent escalates to a human and the customer still
gets a holding reply. That's graceful degradation, not a 500.

---

## 4. Reinforcement learning

### The formulation

- **State** `s = (category, urgency_bucket, tier)` → 4 × 3 × 3 = **36 states**
- **Action** `a` ∈ {`concise-k2`, `concise-k5`, `empathetic-k2`, `empathetic-k5`}
- **Reward** `r = 10 × feedback − latency_seconds` (as specified in the brief)
- **Update** is an incremental sample mean:

```
N(s,a) ← N(s,a) + 1
Q(s,a) ← Q(s,a) + (r − Q(s,a)) / N(s,a)
```

That's exactly equivalent to recomputing the arithmetic mean, without keeping any
history around. There's a unit test that asserts the equivalence directly.

### Why a contextual bandit rather than Q-learning or a neural policy

Triage is a one-step decision: which configuration to use for this ticket. The
choice doesn't affect the next ticket's state, so there's no temporal credit to
assign. A discount factor would be modelling a dependency that isn't there, and
full Q-learning would burn samples learning that the transition matrix is
degenerate. A contextual bandit is the right formalism for a one-step decision
under uncertainty, full stop.

Tabular beats neural at 36 × 4 = 144 cells. It converges from realistic feedback
volumes where a deep policy would need orders of magnitude more, it's fully
inspectable through `/rl/stats`, it persists as a small JSON file, and there's no
training loop to babysit. The brief also rules out deep networks for this part
anyway.

For exploration, epsilon-greedy (ε = 0.15) is the default; UCB1 is implemented
too and selectable via `BANDIT_STRATEGY=ucb1`. Both pull every arm once per state
before they start exploiting, so no arm gets written off on zero evidence. I made
ε-greedy the default because its exploration cost is bounded and predictable: 15%
of tickets get a random configuration, which is a number you can defend to a
support director. UCB1 explores harder up front and is the better pick for an
offline sweep.

The Q-table is written atomically (temp file, then rename) after every feedback
event, so a restart doesn't throw away what's been learned.

### Evidence that it actually learns

`scripts/bandit_simulation.py` runs the loop against a hidden reward model with a
context-dependent optimum: concise replies win on low-urgency tickets, empathetic
replies win when urgency is high, and top-K=5 only helps on policy-heavy
categories. A non-contextual bandit couldn't capture that structure, so it's a
real test of the formulation rather than a softball.

```bash
python scripts/bandit_simulation.py --rounds 20000            # table below
python scripts/bandit_simulation.py --rounds 20000 --plot     # data/bandit_learning.png
python scripts/bandit_simulation.py --strategy ucb1
```

```
quarter       concise-k2     concise-k5  empathetic-k2  empathetic-k5   mean reward
Q1                 0.195          0.325          0.192          0.288         6.344
Q2                 0.204          0.335          0.163          0.299         6.535
Q3                 0.207          0.380          0.121          0.292         6.685
Q4                 0.226          0.398          0.082          0.294         6.618

cumulative regret      : 3951.5
mean regret per ticket : 0.198          (0.369 at 4k rounds — halves with data)
per-state argmax match : 66.7% of states
per-state variant match: 80.6% of states

mean reward  random=6.222   bandit(Q4)=6.618   oracle=6.756
```

What this actually shows, without spin:

- The action distribution shifts. `empathetic-k2`, which is dominated in every
  state, drops from 19% of pulls to 8%, while the two better arms grow.
- Mean reward climbs from 6.34 to 6.62 against a random-routing baseline of 6.22
  and an oracle ceiling of 6.76. So the bandit closes about 74% of the gap that's
  available to close.
- Regret per ticket roughly halves as data comes in (0.369 → 0.198), which is the
  sublinear-regret signature you'd hope for.
- Where it doesn't converge, and why. Full per-state argmax accuracy is 67%, not
  95%. With 36 states × 4 arms and Bernoulli feedback, the reward standard
  deviation is around 5 while some arm gaps are around 0.2, so telling `k2` from
  `k5` inside one variant needs thousands of samples per state. The *variant*
  choice, which is the decision that actually moves the reward, is learned in 81%
  of states. I'm reporting both numbers instead of just the flattering one. The
  production fix is state aggregation (pool tiers, or move to a LinUCB-style
  linear model that shares strength across states), noted in the next-steps list.

CI enforces this. The `rl-evidence` job fails the build if the bandit stops
beating a uniform-random router.

---

## 5. Workflow engine

`workflow/engine.py` is a hand-rolled async DAG runner, around 200 lines, with no
Airflow, Prefect, or Celery, which is what the brief prefers.

**Dependencies are explicit.** Each stage declares `depends_on`, and the
execution order is derived from that, never implied by the order I happened to
write the calls in. Stages get grouped into topological levels with Kahn's
algorithm, so everything inside one level is independent by construction and runs
under a single `asyncio.gather`.

```
level 0: classify, aspects   ← parallel
level 1: urgency
level 2: select_config
level 3: retrieve
level 4: agent
level 5: respond
```

The parallelism is checked by the wall clock, not by asserting something about
the graph shape: two 250 ms stages in one level finish in under 400 ms
(`test_independent_stages_actually_run_concurrently`). CPU-bound stages
(classification, aspect scoring, retrieval) run via `asyncio.to_thread` so they
don't block the event loop.

**Resumability.** Every stage writes its status and output to SQLite as it
finishes. Re-running a transaction replays completed stages from the store rather
than recomputing them; I measured a warm replay at 0.354 s → 0.005 s. A stage
that fails marks itself `failed`, downstream stages never start, and
`POST /ticket/{id}/rerun/{stage}` picks up from that point with the upstream
outputs replayed. `test_a_failed_stage_reruns_without_repeating_upstream_work`
asserts the upstream function is called exactly once across both runs.

The `agent` stage rehydrates its retrieved snippets from persisted state when
it's re-run on its own, so a single-stage rerun doesn't quietly depend on its
upstream neighbour still being in memory.

**State visibility.** `GET /ticket/{id}/status` reads straight from that store:
per-stage status, start and finish timestamps, duration, error text, and the
stage's own output. None of it is hardcoded.

Why SQLite over flag-files or a dict? Atomic writes, crash safety, it survives a
process restart, it's in the standard library, and WAL mode means a status read
never blocks a pipeline write.

---

## 6. Testing, quality, CI

```bash
pytest                                        # 140 tests
pytest --cov --cov-report=term-missing        # coverage report
pytest --cov --cov-report=html                # htmlcov/index.html
pytest tests/test_bandit.py -v                # one module

ruff check ticketiq tests scripts
black --check ticketiq tests scripts
mypy ticketiq
pre-commit install && pre-commit run --all-files
```

140 tests across five modules:

| Module | Covers |
|---|---|
| `test_classifier.py` | IDF against its closed form, L2 normalisation, min_df, logistic-regression class separation / probability distribution sums to 1 / gradient-descent loss reduction / unseen-feature safety / persistence, metrics against hand-computed precision/recall/F1, stratified split, persistence roundtrips, accuracy threshold |
| `test_bandit.py` | Reward formula and its validation, incremental mean == arithmetic mean, per-state isolation, cold-start coverage, greedy at ε=0, empirical exploration rate ≈ ε, convergence on the better arm, different arms learned per state, UCB1 optimism, persistence |
| `test_workflow.py` | Level grouping, diamond graphs, cycle/self-dep/unknown-dep/duplicate rejection, declaration-order independence, wall-clock concurrency, failure recording, downstream not started, resume without recomputing upstream, `only=`, state surviving a new store instance |
| `test_agent_rag.py` | Negation handling, clause-level aspect attribution, urgency ordering and bounds, frontmatter/heading chunking, retrieval correctness and top-K, deterministic tools, JSON extraction from fenced/prose/garbage output, every escalation guardrail, LLM-failure fallback, unparseable-decision fallback, two variants producing different replies |
| `test_api.py` | Every endpoint, all response fields, 422 validation cases, 404s, status reflecting real stage state, single-stage rerun preserving upstream timestamps, feedback → reward arithmetic, feedback visible in `/rl/stats`, end-to-end with an injected fake LLM, stage failure surfaced through the API |

The tests lean heavily on failure paths on purpose (broken backends, malformed
model output, cycles, unknown transactions, invalid payloads), since that's where
a system like this actually breaks in practice.

CI (`.github/workflows/ci.yml`) runs four jobs on every push: lint (ruff, black,
and advisory mypy), tests on Python 3.11 and 3.12 with coverage gated at 80%, the
RL evidence check, and a Docker build that gets smoke-tested against a live
container.

---

## Time-boxing: what I cut, and what I'd do next

Writing this down rather than leaving the gaps for you to find.

**Deliberate scope calls**
- In-memory index instead of FAISS/Chroma. Exact and instant at 24 chunks, and
  the interface is FAISS-shaped so the swap is one file.
- Hand-rolled DAG instead of Airflow. Single process, no scheduler, no retry
  backoff. The brief wanted to see how I reason about dependencies and state, not
  whether I know a specific orchestrator.
- One tool call per ticket. Latency is in the reward function, so unbounded loops
  work against the objective.
- Rule-based aspect extraction. A trained ABSA model is a project on its own, and
  the brief explicitly allows keyword spotting.
- `LLM_MODE=mock` as the default. This makes the whole system reproducible for a
  reviewer with no keys and no GPU. The mock respects the prompt variants and
  gives `concise` lower latency than `empathetic`, so the bandit still faces a
  real trade-off offline.

**Known limitations**
- The dataset is synthetic and template-derived. Real accuracy on production
  tickets would be lower; the reported number measures the pipeline, not the
  world.
- Sparse retrieval struggles with vocabulary mismatch (mitigated, not solved, see
  section 2).
- The bandit learns the variant choice well and the top-K choice slowly (see
  section 4).
- The mock's decision logic is keyword-based. With `LLM_MODE=local` the real
  model does that reasoning, and the agent's parsing fallbacks exist precisely
  because small models are unreliable at emitting clean JSON.
- No auth, rate limiting, or multi-worker coordination. The bandit and the SQLite
  state are process-local, so behind multiple uvicorn workers the Q-table would
  need to move to Redis or Postgres.

**Next steps, in the order I'd tackle them**
1. Move the bandit table to Redis so it's correct under multiple workers.
2. LinUCB or state aggregation to share strength across the 36 states and fix the
   slow top-K convergence.
3. Dense embeddings (`RAG_BACKEND=dense`) plus a retrieval eval set with recall@k,
   so retrieval changes get measured instead of argued.
4. Off-policy evaluation on logged feedback, so a config change can be assessed
   before it ships to live traffic.
5. Structured JSON logging with the transaction id on every line, and
   OpenTelemetry spans per stage. The DAG already produces the right boundaries
   for it.
