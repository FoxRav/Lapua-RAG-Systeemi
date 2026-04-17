from __future__ import annotations

from lapua_rag.models.document import DocumentType
from lapua_rag.postprocess.chunking import chunk_document, split_by_pykala

_MINUTES = """\
<!-- page: 0 -->

§ 12 Talousarvion tarkistus
Kaupunginhallitus päätti hyväksyä talousarvion muutokset.
Perustelut: kustannusten nousu.

<!-- page: 1 -->

§ 13 Henkilöstöasiat
Päätettiin palkata uusi kehityspäällikkö 1.4. alkaen.
"""


def test_split_by_pykala_finds_two_sections() -> None:
    chunks = split_by_pykala(_MINUTES)
    assert len(chunks) == 2
    assert chunks[0].section_id == "§ 12"
    assert chunks[1].section_id == "§ 13"


def test_split_by_pykala_attaches_pages() -> None:
    chunks = split_by_pykala(_MINUTES)
    assert chunks[0].page_start == 0
    assert chunks[1].page_start == 1


def test_chunk_document_falls_back_to_whole_text_for_unknown_type() -> None:
    chunks = chunk_document(text="Ei pykäliä", doc_type=DocumentType.TILINPAATOS)
    assert len(chunks) == 1
