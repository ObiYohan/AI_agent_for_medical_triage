#!/usr/bin/env bash
# Démonstration de l'endpoint de triage /triage (API FastAPI + vLLM), étape par étape, pour la présentation des livrables.
#
#   ./scripts/demo.sh              toutes les étapes, avec une pause entre chacune (Entrée pour continuer)
#   ./scripts/demo.sh 5            une seule étape (1 à 8)
#   ./scripts/demo.sh --no-pause   sans pause (répétition, vérification avant la présentation)
#   API_URL=https://0biyohan-triage-api.hf.space ./scripts/demo.sh    démo cloud (après scripts/cloud.py start)
#
# Prérequis : pile démarrée (docker compose ou CD), clé d'API dans API_KEY ou dans .env (jamais affichée).

set -uo pipefail
cd "$(dirname "$0")/.."

API_URL="${API_URL:-http://localhost:8080}"
DEMO=scripts/demo
PAUSE=1
ONLY=""
for arg in "$@"; do
  case "$arg" in
    --no-pause) PAUSE=0 ;;
    [1-8]) ONLY="$arg" ;;
    -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
    *) echo "Argument inconnu : $arg (voir --help)"; exit 1 ;;
  esac
done

# Clé d'API : variable d'environnement, sinon fichier .env
if [ -z "${API_KEY:-}" ] && [ -f .env ]; then
  API_KEY=$(grep -E '^API_KEY=' .env | cut -d= -f2-)
fi
export API_KEY="${API_KEY:-}"

# Démo cloud (Space Hugging Face privé) : jeton Hugging Face en plus de la clé de l'API (jamais affiché)
CLOUD=0
PLATFORM=()
if [[ "$API_URL" == *.hf.space* ]]; then
  CLOUD=1
  HF_TOKEN="${HF_TOKEN:-$(cat ~/.cache/huggingface/token 2>/dev/null)}"
  if [ -n "$HF_TOKEN" ]; then PLATFORM=(-H "Authorization: Bearer $HF_TOKEN"); fi
fi

# Journal d'audit de la pile en cours : dossier de la pile déployée (~/p14-deploy) ou du projet
STACK_DIR=$(docker inspect -f '{{ index .Config.Labels "com.docker.compose.project.working_dir" }}' p14-api-1 2>/dev/null)
AUDIT_LOG="${AUDIT_LOG:-${STACK_DIR:-.}/logs/audit.jsonl}"

B=$'\e[1m'; C=$'\e[36m'; G=$'\e[32m'; Y=$'\e[33m'; R=$'\e[31m'; D=$'\e[2m'; N=$'\e[0m'

title() { printf '\n%s━━━ %s ━━━%s\n\n' "$B$C" "$1" "$N"; }
say()   { printf '%s%s%s\n' "$D" "$1" "$N"; }
cmd()   { printf '%s$ %s%s\n' "$G" "$1" "$N"; }
pause() { if [ "$PAUSE" = 1 ]; then read -rsp $'\n[Entrée pour continuer]\n'; fi; }

# Affiche un patient de démonstration (motif, symptômes, constantes)
show_request() {
  python3 - "$1" <<'EOF'
import json, sys, textwrap
p = json.load(open(sys.argv[1], encoding="utf-8"))
print(f"\033[1m{p['_titre']}\033[0m\n")
who = {"F": "Femme", "M": "Homme"}.get(p.get("sexe"), "Patient")
print(f"  {who}, {p['age']} ans — motif : {p['motif']}")
print(textwrap.fill(f"Symptômes : {p['symptomes']}", 100, initial_indent="  ", subsequent_indent="  ").replace("\n<|", "\\n<|"))
if p.get("antecedents"):
    print(f"  Antécédents : {p['antecedents']}")
if p.get("constantes"):
    print("  Constantes : " + ", ".join(f"{k} {v}" for k, v in p["constantes"].items()))
print(f"\n\033[2mNiveau attendu : {p['_attendu']}\033[0m")
EOF
}

# Envoie un patient à /triage et affiche la réponse (niveau, source, signes de gravité, explication, questions)
ask() {
  local file="$1" key="${2-$API_KEY}" quiet="${3:-}"
  if [ -z "$quiet" ]; then show_request "$file"; echo; fi
  if [ -n "$key" ]; then
    cmd "curl -X POST $API_URL/triage -H \"X-API-Key: \$API_KEY\" -d @$file"
  else
    cmd "curl -X POST $API_URL/triage -d @$file        # sans clé"
  fi
  local body status
  body=$(mktemp)
  status=$(curl -s -o "$body" -w '%{http_code} %{time_total}' -X POST "$API_URL/triage" "${PLATFORM[@]}" \
           ${key:+-H "X-API-Key: $key"} -H 'Content-Type: application/json' -d @"$file")
  python3 - "$file" "$status" "$body" <<'EOF'
import json, sys
p = json.load(open(sys.argv[1], encoding="utf-8"))
code, seconds = sys.argv[2].split()
body = json.load(open(sys.argv[3], encoding="utf-8"))
color = "\033[32m" if code == "200" else "\033[31m"
print(f"\n{color}HTTP {code}\033[0m  ({float(seconds) * 1000:.0f} ms côté client)")
if code == "200":
    ok = body["niveau"] == p["_attendu"]
    mark = "\033[32m✓ conforme à l'attendu\033[0m" if ok else "\033[33m✗ différent de l'attendu\033[0m"
    print(f"  Niveau          : \033[1m{body['niveau'].upper()}\033[0m   {mark}")
    print(f"  Source          : {body['source']} (avis du modèle : {body['niveau_modele']})")
    if body["signes_de_gravite"]:
        print(f"  Signes de gravité (constantes hors seuils) : {' ; '.join(body['signes_de_gravite'])}")
    print(f"  Justification   : {body['justification']}")
    print(f"  Conduite        : {body['conduite']}")
    print("  Questions complémentaires :")
    for q in body["questions_complementaires"]:
        print(f"    - {q}")
    print(f"  Latence serveur : {body['latence_ms']} ms   ·   {body['avertissement']}")
else:
    detail = body.get("detail")
    reason = detail[0]["msg"] if isinstance(detail, list) else detail  # 422 : motif de la validation
    print(f"  Refus : {reason}")
EOF
  rm -f "$body"
}

step1() {
  title "1. Le service est en ligne"
  say "L'API répond, et vérifie elle-même que le serveur d'inférence vLLM (GPU) est prêt."
  cmd "curl $API_URL/health"
  curl -s "${PLATFORM[@]}" "$API_URL/health" | python3 -m json.tool
}

step2() {
  title "2. Un patient décrit en langage naturel"
  say "Le modèle évalue le niveau de priorité ; l'API explique la décision et propose les questions à poser."
  ask "$DEMO/01_douleur_thoracique.json"
}

step3() {
  title "3. Des constantes graves : les garde-fous"
  say "Des seuils sur les constantes imposent l'urgence maximale ; ils ne peuvent que relever le niveau du modèle."
  ask "$DEMO/02_detresse_respiratoire.json"
}

step4() {
  title "4. Une situation bénigne : la limite du modèle"
  say "Le modèle n'a jamais appris le triage : il sur-trie (erreur prudente, mais qui engorgerait l'accueil)."
  ask "$DEMO/03_entorse.json"
}

step5() {
  title "5. Sécurité d'accès et des entrées"
  say "Sans clé d'API : la requête est refusée, et la tentative est tracée."
  ask "$DEMO/01_douleur_thoracique.json" "" quiet
  pause
  say "Fausses balises de gabarit : elles deviendraient de vrais tokens spéciaux (injection d'un faux message système)."
  ask "$DEMO/04_injection.json"
}

step6() {
  title "6. Traçabilité : le journal d'audit"
  say "Une ligne par requête (heures UTC), sans aucun texte patient : seule l'empreinte SHA-256 de l'entrée est conservée."
  local lines
  lines=$(mktemp)
  if [ "$CLOUD" = 1 ]; then
    # Démo cloud : le journal est lu dans les logs du conteneur du Space (lignes « AUDIT {…} »)
    cmd "logs du Space 0biyohan/triage-api | grep AUDIT | tail -n 6"
    uv run -q --no-project --with huggingface_hub python -c '
from huggingface_hub import HfApi
for line in HfApi().fetch_space_logs("0biyohan/triage-api"):
    if "AUDIT {" in str(line):
        print(str(line).split("AUDIT ", 1)[1])' > "$lines"
  else
    cmd "tail -n 6 $AUDIT_LOG"
    cp "$AUDIT_LOG" "$lines"
  fi
  tail -n 6 "$lines" | python3 <(cat <<'EOF'
import json, sys
for line in sys.stdin:
    r = json.loads(line)
    level = r.get("niveau", "-")
    latency = f"{r['latency_ms']} ms" if "latency_ms" in r else "-"
    digest = r["input_sha256"][:12] + "…" if "input_sha256" in r else "-"
    print(f"  {r['timestamp'][11:19]}  {r['status']:<12} niveau={level:<17} source={r.get('source', '-'):<18}"
          f" latence={latency:<7} entrée={digest}")
EOF
)
  echo
  cmd "grep -c \"constrictive\" <journal>     # le texte du patient apparaît-il dans le journal ?"
  printf '  %s occurrence(s)\n' "$(grep -c "constrictive" "$lines")"
  rm -f "$lines"
}

step7() {
  title "7. Tests d'intégration contre la pile réelle"
  say "Robustesse et audit : les mêmes tests sont relancés automatiquement à chaque déploiement."
  cmd "pytest tests/integration"
  # Démo cloud : pas de journal local à relire (les tests d'audit sont alors ignorés)
  local audit="$AUDIT_LOG"
  if [ "$CLOUD" = 1 ]; then audit=/nonexistent/audit.jsonl; fi
  INTEGRATION_API_URL="$API_URL" AUDIT_LOG="$audit" INTEGRATION_HF_TOKEN="${HF_TOKEN:-}" \
    uv run -q --with-requirements api/requirements.txt --with pytest pytest tests/integration -q -p no:warnings
}

step8() {
  title "8. Les autres livrables"
  say "À ouvrir dans le navigateur :"
  python3 - <<'EOF'
links = {
    "Suivi des entraînements": "https://dagshub.com/obiyohan/my-ml-mlflow.mlflow",
    "Modèle (poids finaux)": "https://huggingface.co/0biyohan/qwen3-1.7b-medical-triage-fr",
    "Pipeline CI/CD": "https://github.com/ObiYohan/AI_agent_for_medical_triage/actions",
    "Image de l'API (GHCR)": "https://github.com/ObiYohan?tab=packages",
    "Rapport technique": "reports/rapport.pdf",
}
for label, url in links.items():
    print(f"  {label:<25} {url}")  # alignement en caractères, pas en octets (accents)
EOF
}

if [ -z "$API_KEY" ]; then
  echo "${R}API_KEY absente : définissez-la ou renseignez-la dans .env${N}"; exit 1
fi
if [ -n "$ONLY" ]; then
  "step$ONLY"
else
  for i in 1 2 3 4 5 6 7 8; do "step$i"; [ "$i" -lt 8 ] && pause; done
fi
echo
