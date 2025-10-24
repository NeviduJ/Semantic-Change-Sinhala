## SiDiaC-v.2.0

A curated collection of Sinhala literary texts and related resources with OCR outputs and basic metadata for each work. This repository organizes original PDFs alongside finalized OCR text and machine-readable `metadata.json` files per title.

### Repository structure
- `Books_PDF/`: Original source PDFs for each title.
- `OCR_Final/`: Finalized OCR results per title, each directory containing:
  - `metadata.json`: Minimal bibliographic and processing metadata
  - `<title>.txt`: Plain text content extracted via OCR

### Example metadata.json
```json
{
  "title": "පන්සිය පනස් ජාතක පොත",
  "title_en": "Pansiya Panas Jathaka Potha",
  "author": "Unknown",
  "author_en": "",
  "genre": "Fiction; Religious",
  "issued_date": "1881",
  "written_date": "1303 - 1333",
  "ocr_confidence": 0.9987
}
```

### Goals
- Preserve Sinhala texts in accessible, searchable formats.
- Provide lightweight, consistent metadata to enable downstream use.

### Contributing
1. Add new works under `Books_PDF/<Work Name>.pdf` and `OCR_Final/<Work Name>/`.
2. Include both `metadata.json` and a UTF-8 encoded `<Work Name>.txt`.
3. Use existing `metadata.json` examples as a schema reference.
4. Open a pull request describing the additions or edits.

### Notes
- Filenames and directory names may include Sinhala characters.
- `ocr_confidence` is a heuristic score from the OCR process and may vary by work.

### License
Unless otherwise noted, PDFs retain their original rights. OCR text and metadata contributions in this repository are provided under an open license consistent with source permissions. Verify rights before adding new materials.