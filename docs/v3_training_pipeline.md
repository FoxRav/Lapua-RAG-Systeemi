# `lapua-llm-v3` — koulutusputki

Tavoite: korvata ylikouluttava `lapua-llm-v2` (abstain sataa usein läpi
jopa kun konteksti vastaa) tasapainoisella `lapua-llm-v3`:lla, jossa
extract/abstain-esimerkit ovat 50/50 ja koulutusdata on suoraan
nykyisen korpuksen chunkeista (`decisions`-taulun päätökset +
ei-päätös-kategoriat).

Putki on suunniteltu niin, että **kaikki koodi on committoituna repoon**;
varsinainen GPU-koulutus on manuaalinen vaihe WSL2:ssa (~6–10 h
RTX 4050:llä).

| Vaihe | Skripti / komento | Kesto | Ympäristö |
|-------|-------------------|-------|-----------|
| 1. Datasetin rakennus | `scripts/build_v3_dataset.py` | 2–5 min | Windows PowerShell |
| 2. Datasetin auditointi | `scripts/audit_training_data.py` | < 1 min | Windows PowerShell |
| 3. Kopiointi WSL2:een | `Copy-Item ... \\wsl$\lapua-vllm\root\training` | < 1 min | Windows PowerShell |
| 4. Riippuvuudet | `pip install unsloth trl peft accelerate bitsandbytes datasets` | 3–5 min | WSL2 (kertaluonteinen) |
| 5. Koulutus | `scripts/train_v3_lora.py` | **6–10 h** | WSL2 + GPU, screen-session |
| 6. Validointi | inline Python-skripti (5 testitapausta) | 1–2 min | WSL2 |
| 7. Julkaisu | `huggingface-cli upload CCG-FAKTUM/lapua-llm-v3` | 2–5 min | WSL2 |
| 8. Käyttöönotto | `.env` + `start_vllm.sh` päivitys | < 1 min | Windows + WSL2 |

Tarkat komennot ovat `INSTRUCTIONS FOLDER/CURSOR_v0.9_OHJE.md`:ssä,
kohta **§A.1–A.5**. Seuraavat ovat niistä tiivistetty muistilista.

## 1. Datasetti (Windows)

```powershell
cd F:\-DEV-\76.PaddleOCR
.\.venv\Scripts\Activate.ps1
docker compose -f deploy\docker-compose.yml up -d qdrant

python scripts/build_v3_dataset.py `
    --output data/training/v3_dataset.jsonl `
    --max-extract 400 `
    --max-abstain 400 `
    --seed 42

python scripts/audit_training_data.py --path data/training/v3_dataset.jsonl
# Hyväksytään kun: extract ~50 %, abstain ~50 %, kumpikin > 45 %
```

Datasetin rakenne: `{"messages": [{"role": "user", ...},
{"role": "assistant", "content": "<JSON>"}]}`. JSON-vastaus on joko
`{"quote": "...", "chunk_index": 3, "no_match": false}` (extract) tai
`{"quote": "", "chunk_index": -1, "no_match": true}` (abstain).

## 2. Koulutus (WSL2)

Pysäytä vLLM (VRAM-konflikti), käynnistä koulutus screenissä:

```bash
wsl -d lapua-vllm
pkill -f "vllm serve" 2>/dev/null; sleep 2

screen -S v3train
/root/vllm-venv/bin/python /mnt/f/-DEV-/76.PaddleOCR/scripts/train_v3_lora.py \
    --data /root/training/v3_dataset.jsonl \
    --output /root/lapua-llm-v3 \
    --epochs 4 --batch-size 2 --grad-accum 4 --lr 2e-4
# Irrottaudu: Ctrl+A, D   |   takaisin: screen -r v3train
```

Onnistumisen merkki: `eval_loss` laskee monotonisesti 2:n ensimmäisen
epochin aikana ja tasantuu. Jos loss nousee, keskeytä ja tarkista
datasetti (ajettu ehkä extract-vino → abstain-esimerkkejä liian vähän).

## 3. Validointi ennen julkaisua

Aja inline 5 testitapauksen mini-eval (2 extract, 2 abstain,
1 sekaerhe). Hyväksymisraja: **4/5 = 80 %**. Jos epäonnistuu, korjaa
datasetti ja aja lisäepoch — älä julkaise.

## 4. Julkaisu ja käyttöönotto

```bash
huggingface-cli login
huggingface-cli upload CCG-FAKTUM/lapua-llm-v3 /root/lapua-llm-v3 \
    --repo-type model
```

```powershell
# Windows
(Get-Content .env) `
    -replace 'LAPUA_LLM_LORA=.*', 'LAPUA_LLM_LORA=CCG-FAKTUM/lapua-llm-v3' `
    -replace 'LAPUA_ANSWER_MODE=.*', 'LAPUA_ANSWER_MODE=synth' |
    Set-Content .env
```

WSL2:n `start_vllm.sh`:n `--lora-modules`-polku pitää päivittää uuteen
`/root/.cache/huggingface/hub/models--CCG-FAKTUM--lapua-llm-v3/
snapshots/<hash>`-polkuun, minkä jälkeen `bash /root/start_vllm.sh`.

## 5. Rollback

Jos `v3` huonompi kuin `v2` (esim. extract-accuracy laskee
eval-joukossa), palautus:

```powershell
(Get-Content .env) -replace 'LAPUA_LLM_LORA=.*', 'LAPUA_LLM_LORA=CCG-FAKTUM/lapua-llm-v2' |
    Set-Content .env
```

`extract`-mode toimii backupina koko ajan — `synth` on vain oletuksen
muutos, ei invasiivinen putki.

## Skriptit pähkinänkuoressa

- `scripts/build_v3_dataset.py` — käyttää `decisions`- ja
  `chunks`-tauluja; ottaa satunnaisotoksen `--max-extract` +
  `--max-abstain` (default 400+400) ja kirjoittaa
  `data/training/*.jsonl`.
- `scripts/audit_training_data.py` — raportoi extract/abstain-suhteen,
  duplikaatit, pisimmät/lyhimmät kontekstit.
- `scripts/train_v3_lora.py` — Unsloth-pohjainen LoRA-koulutusskripti
  Qwen2.5-1.5B-Instructin päälle; parametrit `--epochs`, `--batch-size`,
  `--grad-accum`, `--lr`, `--max-seq-length`. Kirjoittaa
  `/root/lapua-llm-v3/`-kansioon adapter-painot + tokenizerin.

Tehtävä on **valmiina koodin puolesta v0.9.0:ssa**. Varsinainen ajo
merkitään tehdyksi, kun `huggingface-cli upload` on onnistunut ja
`.env` + `start_vllm.sh` osoittavat uuteen malliin.
