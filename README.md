# Homepage Redesign, August 9 2026

## Deploy
1. Back up current index.html.
2. Copy index_new.html to the site root; rename to index.html when approved.
3. Unzip position_statements content into positions/ at the site root
   (already included in this package under positions/).
4. Images are referenced at existing img/ paths; no image changes needed.
5. Delete the REDESIGN NOTES comment at the top of the file.

## Design
- Palette from the trilogy covers: deep-space ink, cover gold, ACM spine red as rare accent.
- Signature element: the constellation spine under the hero; the three volumes are the
  three bright labeled nodes on the Path (clickable).
- Venue badges set in monospace, BibTeX vernacular.
- Single self-contained file: no external fonts, CSS, or JS. Responsive; reduced-motion safe.

## Content changes vs. previous page
- Added: Position Statements strip (six statements + index).
- Added to News: the 2026 nine-lecture AGI Trilogy tour.
- Research section reorganized into six programs following the trilogy's arc.
- Fixed: Google Scholar link placeholder replaced with real ID (pZmOvyYAAAAJ).
- Fixed: Volume 1 links now point to DOI 10.1145/3749421 (previously pointed at Volume 2's DOI in two entries).
- Fixed: Volume 2 subtitle rendered as "Causal Intelligence" (book's actual subtitle).
- Fixed: "What Is Lfe" typo, now "What Is Life".
- Fixed: Architectural Wisdom's "Volume III" link now targets the Books section
  instead of Volume 1's DOI.
- Publication list flattened (previous page had accidental nested list markers).
- All previous publications, books, teaching entries, and background content preserved.

## socrasynth_new.html (added Aug 9)
Redesign of SocraSynth.html in the same design system.
- Lineage narrative preserved (nine pillars, 2020-2025 eras) in the author's voice.
- New section: "The pillars became a trilogy" (2025-2026 second generation: ERM/RLER,
  Trivium, TRACE, Mnemosyne, TRW, CausalT5K/RAudit; Vol.2 published; Vol.3 forthcoming).
- Publications brought current: full 2026 block added; Vol.1 -> ACM DOI 3749421;
  SagaLLM -> VLDB DOI; REALM -> KDD DOI (KDD 2026); ICML 2025 title per proceedings;
  stale "under review" flags removed; "United Cognitive..." and "ResearcgGate" typos fixed.
- 2023-24 abstracts preserved inside collapsible <details> blocks.
- Temple figure referenced at cs372/MACI System1-to-System2.png (existing server path).
- Deploy: rename to SocraSynth.html when approved (or keep both and update homepage link).

## Link audit (Aug 9, second pass)
Every publication entry now carries at least one verified physical-copy URL.
Added/verified this pass:
- CRIT (CCWC 2023): IEEE Xplore doc 10099179 (DOI 10.1109/CCWC57344.2023.10099179), pp. 351-360. Added on both pages.
- Examining GPT-4 (CSCI 2023): IEEE Xplore doc 10590532. Added on both pages.
- ICML 2025: PMLR v267 chang25a verified (pp. 7332-7350); companion arXiv 2502.00136 added.
- NeurIPS SafeAI 2024 three-branch: ResearchGate 380515639 full text; link added on homepage (was missing).
- IEEE MIPR 2024: Xplore doc 10708009 added on lineage page.
- Unlocking the Wisdom: arXiv 2409.01007 full-text added beside Amazon on both pages.
- TRACE (2607.12480) and TRW (2607.21910) arXiv records spot-verified live.
- SocraHealth / SocraPlan (CSCI 2023): no individual Xplore records surfaced in search;
  ResearchGate full-texts retained as the physical copies.

## Spine redesign (Aug 9, per author)
Spine (final): four equally spaced nodes: Pre-LLM Era (Data-centric AI, ImageNet sponsor; Healthcare AI XPRIZE 2017) -> Vol.1 -> Vol.2 -> Vol.3; venues in parentheses
(ImageNet and 2007-12 data-centric removed from spine; remain in Pioneer section).
Key papers listed under each volume:
- V1: SocraSynth, CRIT, EVINCE, Checks-&-Balances ICML'25, SagaLLM/ALAS VLDB'25
- V2: REALM KDD'26, Causal Reasoning ACL'26/KDD'26, Semantic Anchoring ICML'26, ERM
- V3: TRACE, Mnemosyne, TRW (physical-digital synchronization), Architectural Wisdom
Author-suggested list plus four audit additions (EVINCE, ALAS, ERM, Architectural Wisdom);
each is one text line, deletable independently.
