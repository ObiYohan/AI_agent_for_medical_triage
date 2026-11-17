#!/usr/bin/env bash
# Démonstration de l'endpoint de triage (API FastAPI + vLLM), étape par étape, pour la présentation des livrables.
#
#   ./scripts/demo.sh              toutes les étapes, avec une pause entre chacune (Entrée pour continuer)
#   ./scripts/demo.sh 5            une seule étape (1 à 8)
#   ./scripts/demo.sh --no-pause   sans pause (répétition, vérification avant la présentation)
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
    -h|--help) sed -n '2,9p' "$0"; exit 0 ;;
    *) echo "Argument inconnu : $arg (voir --help)"; exit 1 ;;
  esac
done

# Clé d'API : variable d'environnement, sinon fichier .env
if [ -z "${API_KEY:-}" ] && [ -f .env ]; then
  API_KEY=$(grep -E '^API_KEY=' .env | cut -d= -f2-)
fi
export API_KEY="${API_KEY:-}"

# Journal d'audit de la pile en cours : dossier de la pile déployée (~/p14-deploy) ou du projet
STACK_DIR=$(docker inspect -f '{{ index .Config.Labels "com.docker.compose.project.working_dir" }}' p14-api-1 2>/dev/null)
AUDIT_LOG="${AUDIT_LOG:-${STACK_DIR:-.}/logs/audit.jsonl}"

B=$'\e[1m'; C=$'\e[36m'; G=$'\e[32m'; Y=$'\e[33m'; R=$'\e[31m'; D=$'\e[2m'; N=$'\e[0m'

title() { printf '\n%s━━━ %s ━━━%s\n\n' "$B$C" "$1" "$N"; }
say()   { printf '%s%s%s\n' "$D" "$1" "$N"; }
cmd()   { printf '%s$ %s%s\n' "$G" "$1" "$N"; }
pause() { if [ "$PAUSE" = 1 ]; then read -rsp $'\n[Entrée pour continuer]\n'; fi; }

# Affiche un QCM de démonstration (cas, question, propositions)
show_request() {
  python3 - "$1" <<'EOF'
import json, sys, textwrap
q = json.load(open(sys.argv[1], encoding="utf-8"))
print(f"\033[1m{q['_titre']}\033[0m\n")
if q.get("cas_clinique"):
    print(textwrap.fill(q["cas_clinique"], 100), "\n")
print(q["question"].replace("\n", "\\n"))
for letter, text in q["propositions"].items():
    print(f"  {letter}. {text}")
print(f"\n\033[2mRéponse attendue : {q['_attendu']}\033[0m")
EOF
}

# Envoie un QCM à l'API et affiche la réponse (lettres, texte, validité, latence)
ask() {
  local file="$1" key="${2-$API_KEY}" quiet="${3:-}"
  if [ -z "$quiet" ]; then show_request "$file"; echo; fi
  if [ -n "$key" ]; then
    cmd "curl -X POST $API_URL/qcm -H \"X-API-Key: \$API_KEY\" -d @$file"
  else
    cmd "curl -X POST $API_URL/qcm -d @$file        # sans clé"
  fi
  local body status
  body=$(mktemp)
  status=$(curl -s -o "$body" -w '%{http_code} %{time_total}' -X POST "$API_URL/qcm" \
           ${key:+-H "X-API-Key: $key"} -H 'Content-Type: application/json' -d @"$file")
  python3 - "$file" "$status" "$body" <<'EOF'
import json, sys
q = json.load(open(sys.argv[1], encoding="utf-8"))
code, seconds = sys.argv[2].split()
body = json.load(open(sys.argv[3], encoding="utf-8"))
color = "\033[32m" if code == "200" else "\033[31m"
print(f"\n{color}HTTP {code}\033[0m  ({float(seconds) * 1000:.0f} ms côté client)")
if code == "200":
    ok = body["lettres"] == list(q["_attendu"])
    mark = "\033[32m✓ conforme au corrigé\033[0m" if ok else "\033[33m✗ différent du corrigé\033[0m"
    print(f"  Réponse        : {body['reponse']}")
    print(f"  Lettres        : {', '.join(body['lettres'])}   {mark}")
    print(f"  Format valide  : {body['format_valide']}")
    print(f"  Latence serveur: {body['latence_ms']} ms")
    print(f"  Identifiant    : {body['request_id']}")
    print(f"  Avertissement  : {body['avertissement']}")
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
  curl -s "$API_URL/health" | python3 -m json.tool
}

step2() {
  title "2. Un QCM médical du jeu de test"
  say "Le prompt est construit comme à l'entraînement ; vLLM génère la réponse sur le GPU."
  ask "$DEMO/01_qcm_embolie.json"
}

step3() {
  title "3. Un cas de triage : urgence vitale"
  say "Le modèle n'a jamais appris le triage : ce cas mesure ce qu'il en a déduit."
  ask "$DEMO/02_triage_choc.json"
}

step4() {
  title "4. Un cas de triage : situation bénigne"
  say "Limite connue : le modèle sur-trie les cas bénins (erreur prudente, mais coûteuse en charge)."
  ask "$DEMO/03_triage_benin.json"
}

step5() {
  title "5. Sécurité d'accès et des entrées"
  say "Sans clé d'API : la requête est refusée, et la tentative est tracée."
  ask "$DEMO/01_qcm_embolie.json" "" quiet
  pause
  say "Fausses balises de gabarit : elles deviendraient de vrais tokens spéciaux (injection d'un faux message système)."
  ask "$DEMO/04_injection.json"
}

step6() {
  title "6. Traçabilité : le journal d'audit"
  say "Une ligne par requête (heures UTC), sans aucun texte patient : seule l'empreinte SHA-256 de l'entrée est conservée."
  cmd "tail -n 6 $AUDIT_LOG"
  tail -n 6 "$AUDIT_LOG" | python3 <(cat <<'EOF'
import json, sys
for line in sys.stdin:
    r = json.loads(line)
    letters = ",".join(r.get("letters", [])) or "-"
    latency = f"{r['latency_ms']} ms" if "latency_ms" in r else "-"
    digest = r["input_sha256"][:16] + "…" if "input_sha256" in r else "-"
    print(f"  {r['timestamp'][11:19]}  {r['status']:<12} lettres={letters:<5} latence={latency:<8} entrée={digest}")
EOF
)
  echo
  cmd "grep -c \"embolie\" $AUDIT_LOG     # le texte du cas apparaît-il dans le journal ?"
  printf '  %s occurrence(s)\n' "$(grep -c "embolie" "$AUDIT_LOG")"
}

step7() {
  title "7. Tests d'intégration contre la pile réelle"
  say "Robustesse et audit : les mêmes tests sont relancés automatiquement à chaque déploiement."
  cmd "pytest tests/integration"
  INTEGRATION_API_URL="$API_URL" AUDIT_LOG="$AUDIT_LOG" \
    uv run -q --with-requirements api/requirements.txt --with pytest pytest tests/integration -q -p no:warnings
}

step8() {
  title "8. Les autres livrables"
  say "À ouvrir dans le navigateur :"
  python3 - <<'EOF'
links = {
    "Suivi des entraînements": "https://dagshub.com/obiyohan/my-ml-mlflow.mlflow",
    "Modèle (poids finaux)": "https://huggingface.co/0biyohan/qwen3-1.7b-medical-qcm-fr",
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
