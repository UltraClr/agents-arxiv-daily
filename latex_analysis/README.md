# LaTeX Analysis for arXiv Papers

This module provides tools to download, parse, and analyze LaTeX source code from arXiv papers, offering more accurate extraction of methods, equations, and experimental details compared to PDF parsing.

## Features

- **Direct Source Access**: Download LaTeX source files from arXiv
- **Accurate Parsing**: Extract structure, equations, algorithms without PDF parsing errors
- **Content Extraction**: Automatically identify methods, key equations, experimental setups
- **LLM Analysis**: Use Claude/GPT to analyze extracted content and generate summaries

## Advantages over PDF Analysis

| Feature | PDF Analysis | LaTeX Analysis |
|---------|-------------|----------------|
| Formula extraction | Difficult/inaccurate | Perfect LaTeX format |
| Table structure | May lose formatting | Preserves tabular structure |
| Algorithm code | Cannot extract | Extracts algorithm environments |
| Accuracy | Depends on OCR/parser | 100% accurate source |
| Speed | Slower (PDF rendering) | Faster (text processing) |
| Dependencies | GROBID, pdfminer | Only Python stdlib + requests |

## Installation

```bash
# Required packages (already in requirements.txt)
pip install requests tqdm pyyaml openai
```

## Usage

### 1. Download LaTeX Source Files

```bash
python download_latex.py \
  --json_path ../docs/agent-arxiv-daily.json \
  --saved_path ./results/raw_latex
```

### 2. Parse LaTeX Files and Extract Content

```bash
python parse_latex.py \
  --latex_dir ./results/raw_latex \
  --saved_path ./results/parsed_content
```

### 3. Analyze with LLM

```bash
python analysis_papers.py \
  --parsed_content_path ./results/parsed_content \
  --saved_path ./results/llm_analysis \
  --api openai \
  --apikey .apikey
```

### 4. Or Run Complete Pipeline

```bash
python pipeline.py \
  --json_path ../docs/agent-arxiv-daily.json \
  --output_dir ./results \
  --api openai \
  --apikey .apikey
```

## Output Structure

### Parsed Content (JSON)
```json
{
  "arxiv_id": "2401.12345",
  "title": "Paper Title",
  "abstract": "...",
  "method_section": "...",
  "key_algorithms": ["...", "..."],
  "main_equations": ["$$L = ...$$", "$$...$$"],
  "loss_function": "$$\\mathcal{L} = ...$$",
  "experimental_setup": {
    "datasets": ["ImageNet", "COCO"],
    "metrics": ["Accuracy", "F1-Score"],
    "hyperparameters": {"lr": 0.001, "...": "..."}
  },
  "result_tables": ["...", "..."]
}
```

### LLM Analysis Output
```json
{
  "arxiv_id": "2401.12345",
  "summary": {
    "innovation": "...",
    "method_explanation": "...",
    "key_findings": "...",
    "reproducibility_assessment": "..."
  }
}
```


## Notes

- Not all arXiv papers have LaTeX source available (some are PDF-only)
- Multi-file LaTeX projects are automatically handled
- Custom LaTeX macros are partially expanded
- Large papers may hit LLM context limits (automatic chunking planned)

## Troubleshooting

**Error: "LaTeX source not available"**
- Some papers only provide PDF. The script will skip these automatically.

**Error: "Cannot find main .tex file"**
- The archive may have an unusual structure. Check `raw_latex/` manually.

**Error: "API rate limit exceeded"**
- Add delays between requests or use a different API key.

## Future Enhancements

- [ ] Support for BibTeX citation extraction
- [ ] Figure caption analysis
- [ ] Cross-reference resolution
- [ ] Automatic macro expansion
- [ ] Multi-paper comparison analysis
