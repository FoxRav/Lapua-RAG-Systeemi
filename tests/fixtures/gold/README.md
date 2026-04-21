# Kultajoukko (`lapua_gold_v1.jsonl`)

Tämä JSONL-tiedosto on `scripts/eval_rag.py`:n syöte laadun mittaamiseen.
Yksi rivi per kysymys, kentät:

| Kenttä              | Tyyppi           | Selitys                                                                        |
|---------------------|------------------|--------------------------------------------------------------------------------|
| `question`          | string           | Kysymys luonnollisella suomella.                                               |
| `expected_contains` | string[]         | Merkkijonot joiden TÄYTYY löytyä vastauksesta kun `should_abstain=false`.      |
| `doc_type`          | string           | `poytakirja` / `osavuosi` / `tilinpaatos` / `any` (kaikki).                    |
| `should_abstain`    | boolean          | `true` → off-topic-kysymys, vastauspalvelun täytyy pidättäytyä vastaamasta.    |

## Miten lisätä kysymyksiä

1. Aja live-API:lla varmistaaksesi että vastaus tulee oikein:

    ```powershell
    lapua-rag query "Kuka valittiin ..."
    ```

2. Kopioi oleellinen osa vastauksesta `expected_contains`-listaan (case-insensitive
   substring-matchi, joten älä käytä kokonaisia virkkeitä – pelkät nimet/numerot).
3. Off-topic-kysymyksiin laita `should_abstain: true` ja `expected_contains: []`.

## Tavoite

- 20–50 kysymystä, tasapaino ~50/50 substanssi- ja off-topic-kysymysten välillä.
- Vähintään yksi kysymys per `doc_type`.
- Kovat reranker-testit: hallituksen vs. valtuuston päätökset, pykälä-kolakolme
  -tyyppiset spesifikaatiot, summaus-kysymykset (odotetaan abstain kunnes
  `/v1/aggregate` on live).
