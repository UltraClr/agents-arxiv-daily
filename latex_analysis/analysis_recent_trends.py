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


def analysis(paper_num, analysis_prefix, saved_path, api_key, base_url):
    """
    Analyze recent research trends from analyzed papers
    @param paper_num: Number of recent papers to analyze
    @param analysis_prefix: Directory containing llm_analysis JSON files
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

    # Get list of analysis files and sort in reverse order (newest first)
    analysis_files = [f for f in os.listdir(analysis_prefix) if f.endswith('.json')]
    analysis_files.sort(reverse=True)

    if len(analysis_files) == 0:
        logging.error('No analysis files found')
        return False

    # Select top N papers
    selected_files = analysis_files[:paper_num]
    logging.info(f'Analyzing trends from {len(selected_files)} papers')

    # Load and convert papers to text
    papers_text = []
    for filename in tqdm(selected_files, desc='Loading papers'):
        filepath = os.path.join(analysis_prefix, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                analysis_data = json.load(f)

            papers_text.append(analysis_data)
        except Exception as e:
            logging.warning(f'Failed to load {filename}: {e}')
            continue

    if len(papers_text) == 0:
        logging.error('No papers loaded successfully')
        return False

    # Combine all papers
    all_papers_text = papers_text

    # Create prompt
    if old_trends:
        prompt = f"""You are analyzing research trends in AI/ML based on recent papers and previous trend analysis.

## Previous Trend Analysis:

{old_trends}

## New Papers to Analyze:

{all_papers_text}

## Task:

Based on the previous trend analysis and the collection of academic papers, please identify the top five most prominent keywords on recent trends.
Then summarize them in detail. Focus on synthesizing key themes, methodologies, findings, and any shifts in perspective or new areas of inquiry that these papers collectively highlight.
The summary should identify interconnectedness amongst the papers and indicate the direction in which the field of study is moving.
This overview should serve as an insightful guide for researchers seeking to understand the cutting-edge developments and the future trajectory of research within this discipline. 
The output format: '<b>keyword<b>': 'detailed content'."""

    else:
        prompt = f"""You are analyzing research trends in AI/ML based on recent academic papers.

## Papers to Analyze:

{all_papers_text}

## Task:

Based on the collection of academic papers, please identify the top five most prominent keywords on recent trends.
Then summarize them in detail. Focus on synthesizing key themes, methodologies, findings, and any shifts in perspective or new areas of inquiry that these papers collectively highlight.
The summary should identify interconnectedness amongst the papers and indicate the direction in which the field of study is moving.
This overview should serve as an insightful guide for researchers seeking to understand the cutting-edge developments and the future trajectory of research within this discipline. 
The output format: '<b>keyword<b>': 'detailed content'."""

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
    parser.add_argument('--analysis_prefix', type=str,
                        default='./results/llm_analysis',
                        help='Directory containing llm_analysis JSON files')
    parser.add_argument('--saved_path', type=str,
                        default='./results/trends.txt',
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
        args.analysis_prefix,
        args.saved_path,
        api_key,
        base_url
    )

    exit(0 if success else 1)
