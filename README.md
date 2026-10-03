# Business Entity Resolution — Amazon ML Challenge 2026

Given a Source-1 business, find its matching records in Source 2 and Source 3.

Pipeline: normalize → block (TF-IDF top-k over name + address) → pair features → LightGBM → per-entity decision rule. CPU only, no external data or pretrained models.

- Code, setup and run instructions: [code/business_entity_resolution/README.md](code/business_entity_resolution/README.md)
- Write-up: [Documentation_template.md](Documentation_template.md)

The dataset, trained models and submission files are not included in this repository.
