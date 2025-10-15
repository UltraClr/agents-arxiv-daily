import os
import json
import argparse
import logging
from tqdm import tqdm
from openai_api import OpenAIClient

logging.basicConfig(format='[%(asctime)s %(levelname)s] %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)


def load_old_trends(trends_path):
    """
    Load existing trends from file
    @param trends_path: Path to trends.txt file
    @return: Content of old trends, or None if file doesn't exist
    """
    if os.path.exists(trends_path):
        try:
            with open(trends_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            if content:
                logging.info(f'Loaded existing trends from {trends_path}')
                return content
        except Exception as e:
            logging.warning(f'Failed to load old trends: {e}')
    return None


def convert_analysis_to_text(analysis_data):
    """
    Convert analysis JSON to formatted text for LLM input
    @param analysis_data: Dict with arxiv_id, title, publish_date, metadata, analysis
    @return: Formatted text string
    """
    text = f"## Paper: {analysis_data.get('title', 'Unknown Title')}\n"
    text += f"ArXiv ID: {analysis_data.get('arxiv_id', 'Unknown')}\n"

    publish_date = analysis_data.get('publish_date')
    if publish_date:
        text += f"Published: {publish_date}\n"

    text += "\n### Authors and Affiliations\n"
    authors = analysis_data.get('metadata', {}).get('authors', [])
    if authors:
        text += f"Authors: {', '.join(authors[:5])}"  # First 5 authors
        if len(authors) > 5:
            text += " et al."
        text += "\n"

    affiliations = analysis_data.get('metadata', {}).get('affiliations', [])
    if affiliations:
        text += f"Affiliations: {', '.join(affiliations[:3])}\n"  # First 3 affiliations

    # Add analysis content
    analysis = analysis_data.get('analysis', {})

    if 'core_innovation' in analysis:
        text += f"\n### Core Innovation\n{analysis['core_innovation']}\n"

    if 'method_explanation' in analysis:
        text += f"\n### Method\n{analysis['method_explanation']}\n"

    if 'experimental_validation' in analysis:
        text += f"\n### Experimental Results\n{analysis['experimental_validation']}\n"

    if 'limitations' in analysis:
        text += f"\n### Limitations\n{analysis['limitations']}\n"

    if 'future_directions' in analysis:
        text += f"\n### Future Directions\n{analysis['future_directions']}\n"

    # Handle fallback case (raw_text)
    if 'raw_text' in analysis:
        text += f"\n### Analysis\n{analysis['raw_text']}\n"

    text += "\n" + "="*80 + "\n\n"
    return text


def analysis(paper_num, analysis_json_path, saved_path, api_key, base_url):
    """
    Analyze recent research trends from analyzed papers
    @param paper_num: Number of recent papers to analyze
    @param analysis_json_path: Path to consolidated analysis JSON file (agent-arxiv-daily-analysis.json)
    @param saved_path: Path to save trends.txt
    @param api_key: API key for OpenAI-compatible API
    @param base_url: Base URL for API
    """
    # Initialize API client
    try:
        api_client = OpenAIClient(api_key, base_url)
    except Exception as e:
        logging.error(f'Failed to initialize API client: {e}')
        return False

    # Load old trends if exists
    old_trends = load_old_trends(saved_path)

    # Load consolidated analysis JSON file
    if not os.path.exists(analysis_json_path):
        logging.error(f'Analysis file not found: {analysis_json_path}')
        return False

    try:
        with open(analysis_json_path, 'r', encoding='utf-8') as f:
            analysis_data = json.load(f)
    except Exception as e:
        logging.error(f'Failed to load analysis file: {e}')
        return False

    # Collect all papers from all categories
    all_papers = []
    for category, papers in analysis_data.items():
        for arxiv_id, paper_data in papers.items():
            all_papers.append(paper_data)

    if len(all_papers) == 0:
        logging.error('No papers found in analysis file')
        return False

    # Sort by publish_date (newest first), then by arxiv_id
    all_papers.sort(key=lambda p: (p.get('publish_date', ''), p.get('arxiv_id', '')), reverse=True)

    # Select top N papers
    selected_papers = all_papers[:paper_num]
    logging.info(f'Analyzing trends from {len(selected_papers)} papers (out of {len(all_papers)} total)')

    # Convert papers to text
    papers_text = []
    for paper_data in tqdm(selected_papers, desc='Processing papers'):
        try:
            paper_text = convert_analysis_to_text(paper_data)
            papers_text.append(paper_text)
        except Exception as e:
            arxiv_id = paper_data.get('arxiv_id', 'unknown')
            logging.warning(f'Failed to convert {arxiv_id}: {e}')
            continue

    if len(papers_text) == 0:
        logging.error('No papers processed successfully')
        return False

    # Combine all papers
    all_papers_text = "\n".join(papers_text)

    # Create prompt
    if old_trends:
        prompt = f"""You are analyzing research trends in AI/ML based on recent papers and previous trend analysis.

## Previous Trend Analysis:

{old_trends}

## New Papers to Analyze:

{all_papers_text}

## Task:

Based on the previous trend analysis and these new papers, please update the research trend summary. Your task is to:

1. Identify if the trends mentioned in the previous analysis are continuing, strengthening, or fading
2. Identify any NEW emerging trends from the recent papers
3. Synthesize the top 5 most important keywords/trends that define the current research direction
4. For each trend, provide detailed analysis covering:
   - Key themes and methodologies
   - Important findings and breakthroughs
   - Connections between different papers
   - Future trajectory of the research area

## Output Format:

For each of the top 5 trends, use this format:
<b>Trend Keyword</b>: Detailed analysis (2-3 paragraphs) covering the points above.

Focus on:
- Technical accuracy
- Identifying patterns across papers
- Understanding how the field is evolving
- Highlighting the most impactful research directions

Note: Papers are ordered by publication date (newest first). Consider the temporal aspect when analyzing trends."""

    else:
        prompt = f"""You are analyzing research trends in AI/ML based on recent academic papers.

## Papers to Analyze:

{all_papers_text}

## Task:

Based on these papers, identify the top 5 most prominent keywords/trends in recent research. For each trend, provide detailed analysis covering:

1. Key themes and methodologies
2. Important findings and breakthroughs
3. Connections and patterns across papers
4. Future trajectory of the research area

## Output Format:

For each of the top 5 trends, use this format:
<b>Trend Keyword</b>: Detailed analysis (2-3 paragraphs) covering the points above.

Focus on:
- Technical accuracy
- Identifying patterns across papers
- Understanding cutting-edge developments
- Highlighting the most impactful research directions

Note: Papers are ordered by publication date (newest first)."""

    # Send to LLM
    logging.info('Sending request to LLM for trend analysis...')
    try:
        response = api_client.send_message(prompt)

        if not response:
            logging.error('Empty response from API')
            return False

        # Save to file
        with open(saved_path, 'w', encoding='utf-8') as f:
            f.write(response)

        logging.info(f'Trends analysis saved to {saved_path}')
        return True

    except Exception as e:
        logging.error(f'API call failed: {e}')
        return False


if __name__ == '__main__':
    # Initialize argument parser
    parser = argparse.ArgumentParser(
        description='Analyze recent research trends from LLM-analyzed papers'
    )

    # Define arguments
    parser.add_argument('--paper_num', type=int, default=10,
                        help='Number of recent papers to analyze')
    parser.add_argument('--analysis_json_path', type=str,
                        default='../docs/agent-arxiv-daily-analysis.json',
                        help='Path to consolidated analysis JSON file')
    parser.add_argument('--saved_path', type=str,
                        default='../docs/trends.txt',
                        help='Path to save trends analysis (will overwrite)')
    parser.add_argument('--api_key', type=str,
                        default=None,
                        help='API key (defaults to ANTHROPIC_AUTH_TOKEN env var)')
    parser.add_argument('--base_url', type=str,
                        default=None,
                        help='API base URL (defaults to CRS_BASE_URL env var)')

    # Parse arguments
    args = parser.parse_args()

    # Get API credentials from env if not provided
    api_key = args.api_key or os.environ.get('ANTHROPIC_AUTH_TOKEN')
    base_url = args.base_url or os.environ.get('CRS_BASE_URL')

    if not api_key:
        logging.error('API key not provided. Set --api_key or ANTHROPIC_AUTH_TOKEN env variable')
        exit(1)

    # Create output directory if needed
    os.makedirs(os.path.dirname(args.saved_path) or '.', exist_ok=True)

    # Call analysis function
    success = analysis(
        args.paper_num,
        args.analysis_json_path,
        args.saved_path,
        api_key,
        base_url
    )

    exit(0 if success else 1)
