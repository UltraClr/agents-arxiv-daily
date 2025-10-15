import os
import json
import argparse
import logging
from tqdm import tqdm
from openai_api import OpenAIClient

logging.basicConfig(format='[%(asctime)s %(levelname)s] %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)


def create_analysis_prompt(parsed_data):
    """
    Create analysis prompt from parsed LaTeX content
    @param parsed_data: dict from parse_latex.py output (title, authors, affiliations, urls, abstract, sections)
    @return: formatted prompt string
    """
    # Convert parsed_data to formatted JSON string
    import json

    prompt = f"""Analyze this research paper extracted from LaTeX source:

{json.dumps(parsed_data, indent=2, ensure_ascii=False)}

Please provide a comprehensive analysis in the following JSON format:

{{
  "metadata": {{
    "authors": ["list of author names from the paper"],
    "affiliations": ["list of institutions/universities"],
    "resources": {{
      "github": "GitHub repository URL if available, otherwise null",
      "huggingface": "HuggingFace model/dataset URL if available, otherwise null",
      "project_page": "Project website URL if available, otherwise null",
      "other_links": ["any other relevant URLs"]
    }}
  }},
  "analysis": {{
    "summary": "Provide a concise overview of the paper in 2–3 sentences, highlighting its main objective, approach, and findings.",
    "research_question": "What is the primary research question or objective of the paper?",
    "hypothesis": "What is the hypothesis or theses put forward by the authors?",
    "methodology": "What methodology does the paper employ? Briefly describe the study design, data sources, and analysis techniques.",
    "key_findings": "What are the key findings or results of the research?",
    "interpretation": "How do the authors interpret these findings in the context of the existing literature on the topic?",
    "conclusions": "What conclusions are drawn from the research?",
    "limitations": "Can you identify any limitations of the study mentioned by the authors?",
    "future_research": "What future research directions do the authors suggest?"
  }}
}}

IMPORTANT:
- Extract metadata (authors, affiliations, URLs) from the provided data
- Return valid JSON only, no additional text
- Be concise but thorough in the analysis section
- Focus on technical accuracy and practical insights"""

    return prompt


def get_title_from_arxiv_json(arxiv_id, json_path='../docs/agent-arxiv-daily.json'):
    """
    Get paper title from arXiv JSON metadata
    @param arxiv_id: arXiv ID
    @param json_path: Path to arXiv JSON file
    @return: Title string or None
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Search through all categories for this arxiv_id
        for papers in data.values():
            if arxiv_id in papers:
                # Parse markdown table row: |date|title|authors|link|code|
                row = papers[arxiv_id]
                parts = row.split('|')
                if len(parts) >= 3:
                    # Extract title (strip ** markdown bold)
                    title = parts[2].replace('**', '').strip()
                    return title
        return None
    except Exception as e:
        logging.warning(f'Failed to get title from arXiv JSON: {e}')
        return None


def get_category_from_arxiv_json(arxiv_id, arxiv_json_path):
    """
    Get category for an arxiv_id from agent-arxiv-daily.json
    @param arxiv_id: arXiv ID
    @param arxiv_json_path: Path to agent-arxiv-daily.json
    @return: Category name or 'Uncategorized'
    """
    try:
        with open(arxiv_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Search through all categories for this arxiv_id
        for category, papers in data.items():
            if arxiv_id in papers:
                return category
        return 'Uncategorized'
    except Exception as e:
        logging.warning(f'Failed to get category for {arxiv_id}: {e}')
        return 'Uncategorized'


def analyze_all_papers(parsed_content_path, saved_path, api='openai', arxiv_json_path='../docs/agent-arxiv-daily.json'):
    """
    Analyze multiple papers using LLM and save to a consolidated JSON file
    @param parsed_content_path: Directory containing parsed content JSON files
    @param saved_path: Path to save consolidated analysis JSON (e.g., ../docs/agent-arxiv-daily-analysis.json)
    @param api: LLM API to use ('openai' or 'claude')
    @param arxiv_json_path: Path to arXiv metadata JSON file
    @return: True if all successful, False otherwise
    """
    # Initialize API client
    if api == 'openai':
        # Use API key file for OpenAI
        try:
            api_key = os.environ.get('ANTHROPIC_AUTH_TOKEN')
            base_url = os.environ.get('CRS_BASE_URL')
            api_client = OpenAIClient(api_key, base_url)
        except Exception as e:
            logging.error(f'Failed to read API key: {e}')
            return False
    else:
        logging.error(f'Unsupported API: {api}')
        return False

    # Load existing analysis file if it exists
    analysis_data = {}
    if os.path.exists(saved_path):
        try:
            with open(saved_path, 'r', encoding='utf-8') as f:
                analysis_data = json.load(f)
            logging.info(f'Loaded existing analysis file with {sum(len(papers) for papers in analysis_data.values())} papers')
        except Exception as e:
            logging.warning(f'Failed to load existing analysis file: {e}')

    # Get all parsed JSON files
    json_files = [f for f in os.listdir(parsed_content_path)
                  if f.endswith('.json')]
    json_files.sort(reverse=True)  # Process newest first

    logging.info(f'Found {len(json_files)} papers to analyze')

    successful = 0
    failed = 0
    skipped = 0

    for json_file in tqdm(json_files, desc='Analyzing papers'):
        arxiv_id = json_file.replace('.json', '')

        # Check if already analyzed (in any category)
        already_analyzed = any(arxiv_id in papers for papers in analysis_data.values())
        if already_analyzed:
            logging.info(f'{arxiv_id}: Already analyzed, skipping')
            skipped += 1
            continue

        json_path = os.path.join(parsed_content_path, json_file)

        # Get category for this paper
        category = get_category_from_arxiv_json(arxiv_id, arxiv_json_path)

        # Load parsed content
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                parsed_data = json.load(f)
        except Exception as e:
            logging.error(f'{arxiv_id}: Failed to load parsed data - {e}')
            failed += 1
            continue

        # Get title and publish_date
        title = get_title_from_arxiv_json(arxiv_id, arxiv_json_path)
        if not title:
            title = parsed_data.get('title', f'Paper {arxiv_id}')

        publish_date = parsed_data.get('publish_date')

        # Create prompt and call LLM
        prompt = create_analysis_prompt(parsed_data)

        try:
            response = api_client.send_message(prompt)

            if not response:
                logging.error(f'{arxiv_id}: Empty response from API')
                failed += 1
                continue

            # Try to parse LLM response as JSON
            try:
                import re
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    analysis_json = json.loads(json_match.group(0))
                else:
                    analysis_json = json.loads(response)

                # Build final result
                result = {
                    'arxiv_id': arxiv_id,
                    'title': title,
                    'publish_date': publish_date,
                    'metadata': analysis_json.get('metadata', {}),
                    'analysis': analysis_json.get('analysis', {})
                }
            except json.JSONDecodeError:
                # Fallback: if LLM didn't return valid JSON, save as text
                logging.warning(f'{arxiv_id}: LLM response is not valid JSON, saving as text')
                result = {
                    'arxiv_id': arxiv_id,
                    'title': title,
                    'publish_date': publish_date,
                    'metadata': {
                        'authors': parsed_data.get('authors', []),
                        'affiliations': parsed_data.get('affiliations', []),
                        'resources': {'other_links': parsed_data.get('urls', [])}
                    },
                    'analysis': {'raw_text': response}
                }

            # Add to analysis_data under the appropriate category
            if category not in analysis_data:
                analysis_data[category] = {}

            analysis_data[category][arxiv_id] = result

            # Save after each paper (incremental save)
            os.makedirs(os.path.dirname(saved_path) or '.', exist_ok=True)
            with open(saved_path, 'w', encoding='utf-8') as f:
                json.dump(analysis_data, f, indent=2, ensure_ascii=False)

            logging.info(f'{arxiv_id}: Analysis complete and saved to {category}')
            successful += 1

            # Add delay to avoid rate limits
            import time
            logging.info('Waiting 10 seconds before next paper...')
            time.sleep(10)

        except Exception as e:
            logging.error(f'{arxiv_id}: API call failed - {e}')
            failed += 1
            continue

    logging.info(f'Analysis complete: {successful} new, {skipped} skipped, {failed} failed')
    logging.info(f'Total papers in analysis file: {sum(len(papers) for papers in analysis_data.values())}')
    return failed == 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze parsed LaTeX content using LLM.')
    parser.add_argument('--parsed_content_path', type=str,
                        default='./results/parsed_content',
                        help='Directory containing parsed content JSON files.')
    parser.add_argument('--saved_path', type=str,
                        default='../docs/agent-arxiv-daily-analysis.json',
                        help='Path to save consolidated analysis JSON file.')
    parser.add_argument('--api', type=str, default='openai',
                        choices=['openai', 'claude'],
                        help='LLM API to use (default: openai)')
    parser.add_argument('--arxiv_json_path', type=str,
                        default='../docs/agent-arxiv-daily.json',
                        help='Path to arXiv metadata JSON file')
    parser.add_argument('--default_url', type=str,
                        default=None,
                        help='API base URL (for OpenAI-compatible APIs)')

    args = parser.parse_args()

    import sys
    success = analyze_all_papers(
        args.parsed_content_path,
        args.saved_path,
        args.api,
        args.arxiv_json_path
    )
    sys.exit(0 if success else 1)
