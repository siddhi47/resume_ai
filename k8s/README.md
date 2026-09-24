# Deploying resume-ai to Kubernetes

Tested on k3s. Nothing here is k3s-specific except the assumption that the node's
container runtime is containerd.

## 1. Build the image onto the node

The Deployment sets `imagePullPolicy: Never`, so the image is never fetched from a
registry — it must already exist in the node's container runtime.

```sh
docker build -t resume-ai:local .

# k3s uses containerd, not the Docker daemon, so the image has to be imported:
docker save resume-ai:local | sudo k3s ctr images import -
```

Confirm it landed:

```sh
sudo k3s ctr images ls | grep resume-ai
```

## 2. Create the Secret

The Deployment pulls its credentials with `envFrom.secretRef: resume-ai-secrets`.
Create it from your existing `.env` (which is gitignored and must stay that way):

```sh
kubectl create secret generic resume-ai-secrets --from-env-file=.env
```

Keys read by the app:

| Key | Required | Purpose |
|---|---|---|
| `SECRET_KEY` | yes | Flask session signing |
| `OPENAI_API_KEY` | only if `LLM_PROVIDER=openai` | OpenAI backend |
| `GITHUB_PERSONAL_ACCESS_TOKEN` | for GitHub features | project lookup via MCP |
| `GITHUB_USERNAME` | for GitHub features | whose repos to search |
| `GITHUB_MCP_MODE` | no | `hosted` (default) or `self_hosted` |
| `LANGSMITH_API_KEY` | no | tracing |
| `LANGSMITH_ENDPOINT` | no | tracing |
| `LANGSMITH_PROJECT` | no | tracing |
| `LANGSMITH_TRACING` | no | tracing |

Never commit the Secret manifest itself — `kubectl create secret` from the local
`.env` keeps the values off disk in this repo.

## 3. Apply

```sh
kubectl apply -f k8s/resume-ai.yaml
kubectl rollout status deploy/resume-ai
```

Reachable on `http://<node-ip>:30505`.

## Choosing the chat backend

`LLM_PROVIDER` selects at runtime; see `src/resume_bot/llm.py`.

- `openai` — uses the OpenAI API and `OPENAI_API_KEY`.
- `ollama` — uses `OLLAMA_MODEL` at `OLLAMA_BASE_URL`. No API cost.

Under Ollama the app also switches on extra safety scaffolding
(`needs_explicit_plan()`): tailoring runs a planning pass first, and the edit
vocabulary is restricted so the model cannot rewrite the substance of a bullet.
See `.clinerules` for why.

### Sizing the timeout

`--timeout 1800` in the Deployment is derived, not arbitrary. To re-derive it for a
different model, measure that model's two rates:

```sh
curl -s http://<ollama-host>:11434/api/generate \
  -d '{"model":"<model>","prompt":"Write 400 words about testing.",
       "stream":false,"options":{"num_predict":400}}' \
  | python3 -c 'import sys,json; d=json.load(sys.stdin); \
      print("gen", d["eval_count"]/(d["eval_duration"]/1e9), "tok/s"); \
      print("prompt", d["prompt_eval_count"]/(d["prompt_eval_duration"]/1e9), "tok/s")'
```

Then budget roughly: `(prompt_tokens / prompt_rate + output_tokens / gen_rate) × 3`
passes — planning, rewrite, and one validation retry. Set the gunicorn timeout above
that. A timeout that is too short does not surface as a timeout; the worker is killed
mid-generation and it reads as a crash.

Two model families behave very differently here. A Mixture-of-Experts model such as
`qwen3-coder:30b` activates a fraction of its parameters per token and measured
841 tok/s prompt / 75 tok/s generation on an M4 Max. The dense `qwen3:32b` on the
same hardware measured 163 / 22 — roughly five times slower to read and three times
slower to write, despite the similar parameter count.

### Model residency

If the Ollama host runs `OLLAMA_MAX_LOADED_MODELS=1`, only one model stays in memory.
Any other client asking that host for a different model evicts this one, and the next
request here pays a full reload (~72s for a 30GB model). Either point every client at
the same model, or give this app its own Ollama host.

## Persistence

`/var/lib/resume-ai-data` on the node, mounted as two subPaths:

| Host subPath | Container path | Contents |
|---|---|---|
| `data` | `/app/data` | SQLite checkpoints, generated artifacts |
| `resumes` | `/app/static/resumes` | uploaded `.tex` resume templates |

Both survive pod restarts and image rebuilds. Back up `data/` before any migration —
a corrupted LangGraph checkpoint makes the affected conversation unreadable, not just
unwritable.
