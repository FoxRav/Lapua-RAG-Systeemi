# Lapua-RAG – myyntinäkökulma

Lapua-RAG on paikallinen, yksityisyyden säilyttävä RAG-alusta, joka muuttaa
organisaation asiakirjamassat (PDF, skannatut, taulukkomuotoiset) **haettavaksi
rakenteelliseksi dataksi** ja vastaa luonnollisen kielen kysymyksiin omalla,
asiakaskohtaisesti hienosäädetyllä kielimallilla.

## Arvolupaus

* **Data pysyy talossa.** Ei OpenAI-kutsuja, ei pilveä. Kaikki ajautuu
  omilla koneilla tai omalla VPC:llä – GDPR- ja julkishallintokelpoinen.
* **Oma LoRA per asiakas.** Pohjalla Qwen2.5-1.5B-Instruct, päällä asiakkaan
  omaan asiakirjakieleen sovitettu LoRA-adapteri (esim.
  [`CCG-FAKTUM/lapua-llm-v2`](https://huggingface.co/CCG-FAKTUM/lapua-llm-v2)).
  Uusi asiakas = uusi adapteri, ei uusi järjestelmä.
* **SOTA OCR + layout.** PaddleOCR PP-StructureV3 purkaa vinot skannaukset,
  pykälät, taulukot, kaavat ja leimat. Tarvittaessa automaattinen VL-fallback
  vaikeimmille sivuille.
* **Rakenteellinen tulos**, ei pelkästään teksti. Jokaisesta asiakirjasta
  irrotetaan validoitu Pydantic-JSON (päätökset, pykälät, euromäärät,
  päivämäärät, taulukot Parquet-muodossa).
* **Jäljitettävät vastaukset.** Jokainen RAG-vastaus palautuu muodossa
  *Johtopäätös → Perustelut → Lähteet (§, kokous, pvm)* – sama formaatti
  jolla LoRA on opetettu.

## Tyypilliset käyttötapaukset

| Asiakassegmentti      | Esimerkkidata                                      | Kyselyt                                                |
|-----------------------|-----------------------------------------------------|--------------------------------------------------------|
| Kunnat                | Pöytäkirjat, tilinpäätökset, osavuosikatsaukset    | "Mitä § 78 päätettiin?", "Investointien toteuma Q1"     |
| Tilitoimistot         | Kirjanpito, kuitit, tilinpäätökset, veroilmoitukset| "Näytä kaikki > 50 k€ hankinnat 2024"                   |
| Juridiset tiimit       | Sopimukset, päätökset                              | "Mitkä NDA:t vanhenevat tämän vuoden aikana?"           |
| Teollisuus / ERP      | Projektiraportit, poikkeamaraportit                | "Kaikki laatupoikkeamat linjalla A"                     |

## Ero tyypilliseen "chatti PDF:n kanssa" -ratkaisuun

| Ominaisuus                    | Lapua-RAG                                   | Tyypilliset PDF-chatit |
|-------------------------------|---------------------------------------------|------------------------|
| Data pilvessä                 | Ei (lokaali / VPC)                          | Yleensä kyllä          |
| Oma hienosäädetty malli       | Kyllä, LoRA per asiakas                     | Ei                     |
| Tukee skannattuja PDF:iä      | Kyllä (PP-StructureV3 + VL fallback)        | Rajallinen             |
| Tukee taulukoita              | Kyllä (HTML → Parquet)                      | Teksti                 |
| Rakenteellinen JSON-ulostulo  | Kyllä (Pydantic-skeema, validoitu)          | Ei                     |
| Sitaatit pykälätarkasti       | § + sivu + bbox                             | Yleensä sivunumero     |
| Inkrementaalinen ingest       | Sisään jono, yksi kerrallaan, idempotentti  | Vaihtelee              |
| Kymmenien tuhansien sivujen   | Suunniteltu tähän                            | Ei tyypillisesti       |
  kestävyys                    |                                              |                        |

## Pipeline yhdellä silmäyksellä

1. PDF pudotetaan inbox-kansioon (tai ladataan `POST /v1/ingest`).
2. PP-StructureV3 GPU:lla → per-sivu markdown + JSON + taulukot.
3. Jälkikäsittely: merkistökorjaus, dokumenttityypin tunnistus, chunkkaus
   pykälärajoilla.
4. Embeddings (multilingual E5 / BGE-M3) → Qdrant.
5. BM25 (Finnish-stemmed SQLite FTS5) – hybriidiä varten.
6. Rakenne-ekstraktio Qwen + LoRA → validoitu JSON.
7. Valmis. Kysymykset `POST /v1/query` → *Johtopäätös → Perustelut → Lähteet*.

## Demon reitti

1. `docker compose up -d qdrant`
2. `pip install -e .[dev]`
3. `lapua-rag init`
4. `lapua-rag ingest "F:/...Talouden toteutumaraportti.pdf"`
5. `lapua-rag query "Mitä kaupunginhallitus päätti ympäristönsuojelulupien
   käsittelystä Q1 aikana?"`

## Hinnoittelumalli (aihio)

* **Setup**: kertaluontoinen käyttöönotto + asiakaskohtaisen LoRA-adapterin
  koulutus (3 000 esimerkkiä riittää v2-tasolla, ~3 h A10G-ajoaika).
* **Lisenssi**: kuukausimaksu / asiakirja-alaraja (esim. 0–50 k sivua /kk).
* **Ylittävä volyymi**: sivuhinta indeksoinnista.
* **Tuki & päivitykset**: SLA-tasot (business hours / 24×7).
