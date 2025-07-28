# 🐍 Python Code Helper – Infrastructure Layer

> **Status : Prototype fonctionnel + Infrastructure prête (Terraform/Kubernetes/Prometheus).**  
> Les démos locales (Ollama/FastAPI) fonctionnent ; le reste du dépôt contient **uniquement** le code nécessaire au déploiement production.

## 🔧 Contenu du dépôt

| Dossier | Rôle |
|---------|------|
| `terraform/` | Infrastructure-as-Code pour AWS : VPC, EKS, RDS, ElastiCache, Load Balancer |
| `k8s/` | Manifests Kubernetes (deployment, service, ingress, configmap, secrets, pvc) |
| `docker/monitoring/` | Stack Prometheus + Grafana prête pour K8s ou Docker Compose |
| `src/` | Application FastAPI + moteur RAG (Ollama, Sentence Transformers, vector store numpy) |
| `scripts/deploy.py` | Orchestration de déploiement (Terraform → K8s → monitoring) |
| `.github/workflows/ci-cd.yml` | Pipeline GitHub Actions : `terraform fmt/validate` + `kubectl --dry-run` |

## 🚀 Déploiement Production (Terraform + Kubernetes)

```bash
# 1) Provisionnement AWS
cd terraform
terraform init
terraform plan -out tf.plan
terraform apply tf.plan

# 2) Context Kubeconfig (EKS)
aws eks --region <region> update-kubeconfig --name <cluster-name>

# 3) Déploiement applicatif
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secrets.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/ingress.yaml

# 4) Monitoring
kubectl apply -f docker/monitoring/prometheus.yml
kubectl apply -f docker/monitoring/grafana/
```

## 🏗️ Architecture Technique Retenue

```
┌──────────────┐      ┌─────────────┐      ┌──────────────────┐
│  Terraform   │──►  │   AWS EKS   │──►  │  Kubernetes Apps  │
└──────────────┘      └─────────────┘      └──────────────────┘
        │                      │                     │
        ▼                      ▼                     ▼
  Prometheus/Grafana     FastAPI + RAG        Nginx Ingress + TLS
```

## ⚙️ Stack Applicative (prototype validé)

- **Backend :** FastAPI, Pydantic, Uvicorn
- **IA / ML :** Ollama (CodeLlama-7B), Sentence Transformers (embeddings), recherche vectorielle NumPy
- **Vector DB :** Alternative Chroma (API compatible, zéro dépendance externe)
- **Monitoring :** Prometheus (metrics) + Grafana (dashboards)

## 📜 Licence
MIT 

## 🧠 Architecture RAG détaillée & Implémentation

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                               RAG WORKFLOW                                    │
├─────────────────┬──────────────────────────┬──────────────────────────────────┤
│    Ingestion    │     Traitement           │        Stockage / Recherche      │
│                 │                          │                                  │
│ • GitHub Crawler│ • Analyse AST Python     │ • Embeddings Sentence Tfrm (384) │
│ • StackOverflow │ • Chunking sémantique    │ • Vector Store NumPy (Cosine)    │
│ • Web Scraping  │ • Enrichissement meta    │ • Alternative Chroma API         │
├─────────────────┴──────────────────────────┴──────────────────────────────────┤
│                           Génération & API (FastAPI)                          │
│ • Récupération top-K                         • Ollama CodeLlama-7B            │
│ • Filtrage/Ranking hybrid                    • Attribution des sources        │
│ • Streaming SSE / REST                       • Scoring confiance              │
├───────────────────────────────────────────────────────────────────────────────┤
│                   Observabilité & Ops (Prometheus / Grafana)                  │
│ • Metrics API  /metrics   • Dashboards  • Alerting • Traces                   │
└───────────────────────────────────────────────────────────────────────────────┘
```

### 🔑 Points d’implémentation

1. **Ingestion**
   - `src/ingestion/github_crawler.py` : collecte repos publics via PyGithub.
   - `src/ingestion/stackoverflow_collector.py` : questions/réponses via Stack Exchange API.

2. **Traitement & Chunking**
   - Analyse AST pour identifier fonctions/classes.
   - Chunking semantique avec chevauchement pour meilleur rappel.

3. **Stockage Vectoriel**
   - Embeddings générés localement → matrice NumPy.
   - Similarité cosinus + indexation par ID & métadonnées.
   - Interface compatible Chroma (module `src/vector/chroma_alternative.py`).

4. **API & Génération**
   - FastAPI (`web_app/main.py`) expose `/chat`, `/search`, `/generate`.
   - Ollama LLM (CodeLlama-7B) génère réponses contextualisées.
   - SSE pour streaming en temps réel.

5. **Infrastructure**
   - Terraform crée VPC + EKS + RDS/Redis + IAM + ALB.
   - Manifests K8s déploient FastAPI + Nginx Ingress + monitoring.
   - Prometheus scrape `/metrics`, Grafana dashboards importés via ConfigMap.

6. **CI/CD minimal (gratuit)**
   - GitHub Actions : `terraform fmt/validate` + `kubectl --dry-run` sur chaque PR.

> **Pourquoi cette approche ?** Garder un **prototype local fonctionnel** tout en fournissant une **piste de déploiement production** reproductible et infra-as-code. 