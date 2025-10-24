import os
import json
import argparse
import logging
import yaml
import re
from tqdm import tqdm
from openai_api import OpenAIClient as OpenAIClientOrig
from claude_api import OpenAIClient as ClaudeClient

logging.basicConfig(format='[%(asctime)s %(levelname)s] %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)


def load_config(config_path='../config.yaml'):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def get_search_keywords(config):
    """Extract search keywords from config"""
    keywords_list = []
    for category, data in config.get('keywords', {}).items():
        keywords_list.extend(data.get('filters', []))
    return keywords_list


def add_to_blacklist(paper_title, blacklist_path='../blacklists.txt'):
    """Add paper title to blacklist file"""
    try:
        # Read existing blacklist
        if os.path.exists(blacklist_path):
            with open(blacklist_path, 'r', encoding='utf-8') as f:
                existing = set(line.strip() for line in f if line.strip())
        else:
            existing = set()

        # Add new title if not already present
        if paper_title not in existing:
            with open(blacklist_path, 'a', encoding='utf-8') as f:
                f.write(paper_title + '\n')
            logging.info(f'Added to blacklist: {paper_title}')
            return True
        else:
            logging.info(f'Already in blacklist: {paper_title}')
            return False
    except Exception as e:
        logging.error(f'Failed to add to blacklist: {e}')
        return False


def remove_paper_from_json_files(arxiv_id, title, config):
    """
    Remove paper from all JSON data files
    @param arxiv_id: arXiv ID to remove
    @param title: Paper title (for logging)
    @param config: Configuration dictionary containing JSON file paths
    @return: Number of files successfully updated
    """
    # Get JSON file paths from config
    json_files = [
        config.get('json_readme_path', '../docs/agent-arxiv-daily.json'),
        config.get('json_gitpage_path', '../docs/agent-arxiv-daily-web.json'),
        config.get('json_wechat_path', '../docs/agent-arxiv-daily-wechat.json')
    ]

    updated_count = 0

    for json_path in json_files:
        # Convert relative path to absolute if needed
        if not os.path.isabs(json_path):
            # Get path relative to analysis directory
            current_dir = os.path.dirname(os.path.abspath(__file__))
            json_path = os.path.join(current_dir, json_path)

        if not os.path.exists(json_path):
            logging.warning(f'{arxiv_id}: JSON file not found: {json_path}')
            continue

        try:
            # Load JSON data
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Find and remove the paper from all categories
            removed = False
            for category, papers in data.items():
                if arxiv_id in papers:
                    del papers[arxiv_id]
                    removed = True
                    logging.info(f'{arxiv_id}: Removed from category "{category}" in {os.path.basename(json_path)}')

            if removed:
                # Save updated data
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                updated_count += 1
                logging.info(f'{arxiv_id}: Successfully updated {os.path.basename(json_path)}')
            else:
                logging.debug(f'{arxiv_id}: Not found in {os.path.basename(json_path)}')

        except Exception as e:
            logging.error(f'{arxiv_id}: Failed to update {json_path} - {e}')
            continue

    if updated_count > 0:
        logging.info(f'{arxiv_id}: Removed "{title}" from {updated_count} JSON file(s)')

    return updated_count


def extract_relevance_from_json(analysis_json):
    """Extract relevance score from analysis JSON"""
    try:
        # Check for keyword_relevance field
        if 'keyword_relevance' in analysis_json:
            kr = analysis_json['keyword_relevance']
            if isinstance(kr, dict):
                return kr.get('score'), kr.get('reasoning'), kr.get('matching_keywords', [])
            elif isinstance(kr, (int, float)):
                return float(kr), None, []

        # Check for direct relevance_score field
        if 'relevance_score' in analysis_json:
            return float(analysis_json['relevance_score']), None, []

        return None, None, []
    except Exception as e:
        logging.error(f'Error extracting relevance from JSON: {e}')
        return None, None, []


def create_analysis_prompt(parsed_data, search_keywords=None, enable_validation=False):
    """
    Create analysis prompt from parsed LaTeX content
    @param parsed_data: dict from parse_latex.py output (title, authors, affiliations, urls, abstract, sections)
    @param search_keywords: list of keywords to validate against (optional)
    @param enable_validation: whether to include keyword relevance validation
    @return: formatted prompt string
    """
    # Convert parsed_data to formatted JSON string
    import json

    # Build keyword relevance section if validation is enabled
    keyword_relevance_field = ""
    keyword_instruction = ""
    if enable_validation and search_keywords:
        keywords_str = ', '.join(search_keywords)
        keyword_relevance_field = f""",
  "keyword_relevance": {{
    "score": 0.0,  // 0-10 scale: 0-3=barely related, 4-6=partially related, 7-10=highly relevant
    "reasoning": "Brief explanation of why this paper does or doesn't relate to the keywords",
    "matching_keywords": ["list of which specific keywords from [{keywords_str}] this paper addresses"]
  }}"""
        keyword_instruction = f"""
- Evaluate how relevant this paper is to these search keywords: {keywords_str}
- Provide a relevance score (0-10) and explain your reasoning"""

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
  }}{keyword_relevance_field}
}}

IMPORTANT:
- Extract metadata (authors, affiliations, URLs) from the provided data
- Return valid JSON only, no additional text
- Be concise but thorough in the analysis section
- Focus on technical accuracy and practical insights{keyword_instruction}"""

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


def analyze_all_papers(parsed_content_path, saved_path, api='openai', arxiv_json_path='../docs/agent-arxiv-daily.json', config_path='../config.yaml'):
    """
    Analyze multiple papers using LLM and save to a consolidated JSON file
    @param parsed_content_path: Directory containing parsed content JSON files
    @param saved_path: Path to save consolidated analysis JSON (e.g., ../docs/agent-arxiv-daily-analysis.json)
    @param api: LLM API to use ('openai' or 'claude')
    @param arxiv_json_path: Path to arXiv metadata JSON file
    @param config_path: Path to configuration file
    @return: True if all successful, False otherwise
    """
    # Load configuration for keyword validation
    config = load_config(config_path)
    enable_validation = config.get('enable_keyword_validation', False)
    relevance_threshold = config.get('keyword_relevance_threshold', 5.0)
    auto_blacklist = config.get('auto_blacklist', True)
    blacklist_path = config.get('black_list_path', '../blacklists.txt')

    # Get search keywords
    search_keywords = get_search_keywords(config)
    keywords_str = ', '.join(search_keywords)

    logging.info(f'Keyword validation: {enable_validation}')
    logging.info(f'Relevance threshold: {relevance_threshold}')
    logging.info(f'Search keywords: {keywords_str}')

    # Initialize API client
    try:
        api_key = os.environ.get('ANTHROPIC_AUTH_TOKEN')
        base_url = os.environ.get('CRS_BASE_URL')

        if api == 'claude':
            api_client = ClaudeClient(api_key, base_url)
            logging.info('Claude API client initialized')
        elif api == 'openai':
            api_client = OpenAIClientOrig(api_key, base_url)
            logging.info('OpenAI API client initialized')
        else:
            logging.error(f'Unsupported API: {api}')
            return False
    except Exception as e:
        logging.error(f'Failed to initialize API client: {e}')
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

        # Create prompt and call LLM (with keyword validation if enabled)
        prompt = create_analysis_prompt(parsed_data, search_keywords, enable_validation)

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

                # Process keyword relevance validation
                if enable_validation:
                    relevance_score, reasoning, matching_keywords = extract_relevance_from_json(analysis_json)

                    if relevance_score is not None:
                        logging.info(f'{arxiv_id}: Relevance score = {relevance_score}')
                        result['keyword_relevance'] = {
                            'score': relevance_score,
                            'reasoning': reasoning,
                            'matching_keywords': matching_keywords
                        }

                        # Add to blacklist if below threshold
                        if auto_blacklist and relevance_score < relevance_threshold:
                            add_to_blacklist(title, blacklist_path)
                            logging.warning(f'{arxiv_id}: Low relevance ({relevance_score}), added to blacklist: {title}')
                            result['blacklisted'] = True
                        else:
                            logging.info(f'{arxiv_id}: Relevance score acceptable ({relevance_score} >= {relevance_threshold})')
                            result['blacklisted'] = False
                    else:
                        logging.warning(f'{arxiv_id}: Could not extract relevance score from JSON')
                        result['keyword_relevance'] = None
                        result['blacklisted'] = False

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
                    'analysis': {'raw_text': response},
                    'blacklisted': False
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
    parser = argparse.ArgumentParser(description='Analyze parsed LaTeX content using LLM with optional keyword relevance validation.')
    parser.add_argument('--parsed_content_path', type=str,
                        default='./results/parsed_content',
                        help='Directory containing parsed content JSON files.')
    parser.add_argument('--saved_path', type=str,
                        default='../docs/agent-arxiv-daily-analysis.json',
                        help='Path to save consolidated analysis JSON file.')
    parser.add_argument('--api', type=str, default='claude',
                        choices=['openai', 'claude'],
                        help='LLM API to use (default: claude)')
    parser.add_argument('--arxiv_json_path', type=str,
                        default='../docs/agent-arxiv-daily.json',
                        help='Path to arXiv metadata JSON file')
    parser.add_argument('--config_path', type=str,
                        default='../config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--default_url', type=str,
                        default=None,
                        help='API base URL (for OpenAI-compatible APIs)')

    args = parser.parse_args()

    import sys
    success = analyze_all_papers(
        args.parsed_content_path,
        args.saved_path,
        args.api,
        args.arxiv_json_path,
        args.config_path
    )
    sys.exit(0 if success else 1)
