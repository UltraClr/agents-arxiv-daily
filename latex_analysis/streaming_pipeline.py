"""
Streaming pipeline for LaTeX-based paper analysis
Process each paper as soon as it's downloaded: download -> parse -> analyze
"""
import os
import json
import argparse
import logging
import sys
import time
from tqdm import tqdm

# Import individual processing functions
from download_latex import download_latex_source, find_all_tex_files
from parse_latex import LaTeXParser, find_main_tex_file, load_arxiv_metadata
from analysis_papers import create_analysis_prompt, get_title_from_arxiv_json, get_category_from_arxiv_json
from openai_api import OpenAIClient

logging.basicConfig(format='[%(asctime)s %(levelname)s] %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)


def process_single_paper(arxiv_id, raw_latex_dir, parsed_content_dir,
                         api_client, analysis_data, arxiv_json_path, arxiv_metadata):
    """
    Process a single paper: download -> parse -> analyze
    @param arxiv_id: arXiv ID to process
    @param raw_latex_dir: Directory to save raw LaTeX
    @param parsed_content_dir: Directory to save parsed content
    @param api_client: LLM API client
    @param analysis_data: Dictionary to store analysis results (modified in place)
    @param arxiv_json_path: Path to arXiv metadata JSON
    @param arxiv_metadata: Pre-loaded arXiv metadata dictionary
    @return: True if successful, False otherwise
    """
    logging.info(f'\n{"="*60}')
    logging.info(f'Processing paper: {arxiv_id}')
    logging.info(f'{"="*60}')

    # Step 1: Download LaTeX source
    logging.info(f'{arxiv_id}: [1/3] Downloading LaTeX source...')
    extract_dir = download_latex_source(arxiv_id, raw_latex_dir)

    if not extract_dir:
        logging.error(f'{arxiv_id}: Download failed, skipping')
        return False

    # Verify LaTeX files exist
    tex_files = find_all_tex_files(extract_dir)
    if not tex_files:
        logging.warning(f'{arxiv_id}: No .tex files found, skipping')
        return False

    logging.info(f'{arxiv_id}: Found {len(tex_files)} .tex files')

    # Step 2: Parse LaTeX content
    logging.info(f'{arxiv_id}: [2/3] Parsing LaTeX content...')
    main_tex = find_main_tex_file(extract_dir)

    if not main_tex:
        logging.warning(f'{arxiv_id}: No main .tex file found, skipping')
        return False

    # Get title and publish_date from arXiv metadata
    title = arxiv_metadata.get(arxiv_id, {}).get('title')
    publish_date = arxiv_metadata.get(arxiv_id, {}).get('publish_date')

    # Parse the LaTeX file
    parser = LaTeXParser(main_tex)
    parsed_data = parser.parse(arxiv_id=arxiv_id, title=title, publish_date=publish_date)

    if not parsed_data:
        logging.error(f'{arxiv_id}: Parsing failed, skipping')
        return False

    # Save parsed content
    parsed_json_path = os.path.join(parsed_content_dir, f'{arxiv_id}.json')
    with open(parsed_json_path, 'w', encoding='utf-8') as f:
        json.dump(parsed_data, f, indent=2, ensure_ascii=False)
    logging.info(f'{arxiv_id}: Parsed content saved')

    # Step 3: LLM Analysis
    if not api_client:
        logging.info(f'{arxiv_id}: Skipping analysis (no API client)')
        return True

    logging.info(f'{arxiv_id}: [3/3] Running LLM analysis...')

    # Get category for this paper
    category = get_category_from_arxiv_json(arxiv_id, arxiv_json_path)

    # Get title
    if not title:
        title = get_title_from_arxiv_json(arxiv_id, arxiv_json_path)
    if not title:
        title = parsed_data.get('title', f'Paper {arxiv_id}')

    # Create prompt and call LLM
    prompt = create_analysis_prompt(parsed_data)

    try:
        response = api_client.send_message(prompt)

        if not response:
            logging.error(f'{arxiv_id}: Empty response from API')
            return False

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

        logging.info(f'{arxiv_id}: Analysis complete (category: {category})')
        return True

    except Exception as e:
        logging.error(f'{arxiv_id}: API call failed - {e}')
        return False


def run_streaming_pipeline(args):
    """
    Run the streaming LaTeX analysis pipeline
    Process papers one by one: download -> parse -> analyze
    """
    # Define paths
    output_dir = args.output_dir
    raw_latex_dir = os.path.join(output_dir, 'raw_latex')
    parsed_content_dir = os.path.join(output_dir, 'parsed_content')
    llm_analysis_path = '../docs/agent-arxiv-daily-analysis.json'

    # Create directories
    for directory in [raw_latex_dir, parsed_content_dir]:
        os.makedirs(directory, exist_ok=True)

    # Initialize API client if needed
    api_client = None
    if not args.skip_analysis:
        if not args.apikey:
            logging.warning('API key not found. Skipping analysis step.')
        else:
            try:
                api_client = OpenAIClient(args.apikey, os.environ.get('CRS_BASE_URL'))
                logging.info('API client initialized')
            except Exception as e:
                logging.error(f'Failed to initialize API client: {e}')

    # Load existing analysis file if it exists
    analysis_data = {}
    if os.path.exists(llm_analysis_path):
        try:
            with open(llm_analysis_path, 'r', encoding='utf-8') as f:
                analysis_data = json.load(f)
            logging.info(f'Loaded existing analysis file with {sum(len(papers) for papers in analysis_data.values())} papers')
        except Exception as e:
            logging.warning(f'Failed to load existing analysis file: {e}')

    # Load arXiv metadata (for titles and dates)
    arxiv_metadata = load_arxiv_metadata(args.json_path)

    # Load JSON with paper data to get all arxiv IDs
    with open(args.json_path, 'r') as file:
        data = json.load(file)

    # Collect all arxiv IDs
    all_ids = []
    for keyword in data.keys():
        for arxiv_id in data[keyword].keys():
            # Skip old format IDs (contain '/')
            if '/' in arxiv_id:
                continue
            all_ids.append(arxiv_id)

    # Limit if max_papers specified
    if args.max_papers:
        all_ids = all_ids[:args.max_papers]

    logging.info(f'\n{"="*60}')
    logging.info(f'Starting streaming pipeline for {len(all_ids)} papers')
    logging.info(f'{"="*60}\n')

    # Process statistics
    successful = 0
    failed = 0
    skipped = 0

    # Process each paper one by one
    for arxiv_id in tqdm(all_ids, desc='Processing papers'):
        # Check if already analyzed
        already_analyzed = any(arxiv_id in papers for papers in analysis_data.values())
        if already_analyzed and not args.force_reprocess:
            logging.info(f'{arxiv_id}: Already analyzed, skipping')
            skipped += 1
            continue

        # Process this paper
        success = process_single_paper(
            arxiv_id=arxiv_id,
            raw_latex_dir=raw_latex_dir,
            parsed_content_dir=parsed_content_dir,
            api_client=api_client,
            analysis_data=analysis_data,
            arxiv_json_path=args.json_path,
            arxiv_metadata=arxiv_metadata
        )

        if success:
            successful += 1

            # Save analysis file after each successful paper (incremental save)
            if api_client:
                os.makedirs(os.path.dirname(llm_analysis_path) or '.', exist_ok=True)
                with open(llm_analysis_path, 'w', encoding='utf-8') as f:
                    json.dump(analysis_data, f, indent=2, ensure_ascii=False)
                logging.info(f'Analysis file updated: {sum(len(papers) for papers in analysis_data.values())} total papers')

            # Add delay to avoid rate limits (only if API was used)
            if api_client and not args.skip_analysis:
                delay = args.delay_seconds
                logging.info(f'Waiting {delay} seconds before next paper...')
                time.sleep(delay)
        else:
            failed += 1

    # Summary
    logging.info('\n' + '='*60)
    logging.info('Streaming pipeline complete!')
    logging.info(f'Successful: {successful}')
    logging.info(f'Skipped: {skipped}')
    logging.info(f'Failed: {failed}')
    logging.info(f'Results saved to: {output_dir}')
    if api_client:
        logging.info(f'Analysis file: {llm_analysis_path}')
    logging.info('='*60 + '\n')

    return failed == 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Streaming pipeline for LaTeX-based paper analysis (process one paper at a time)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Input/Output
    parser.add_argument('--json_path', type=str,
                        default='../docs/agent-arxiv-daily.json',
                        help='Path to JSON file with arXiv IDs')
    parser.add_argument('--output_dir', type=str,
                        default='./results',
                        help='Base directory for all outputs')

    # API configuration
    parser.add_argument('--api', type=str, default='openai',
                        choices=['openai', 'claude'],
                        help='LLM API to use')
    parser.add_argument('--apikey', type=str, default=os.environ.get('ANTHROPIC_AUTH_TOKEN'),
                        help='API key or path to API key file')

    # Control flags
    parser.add_argument('--skip_analysis', action='store_true',
                        help='Skip LLM analysis step (only download and parse)')
    parser.add_argument('--force_reprocess', action='store_true',
                        help='Force reprocessing of already analyzed papers')

    # Limits
    parser.add_argument('--max_papers', type=int, default=None,
                        help='Maximum number of papers to process (for testing)')
    parser.add_argument('--delay_seconds', type=int, default=10,
                        help='Delay in seconds between papers (to avoid rate limits)')

    args = parser.parse_args()

    # Run streaming pipeline
    success = run_streaming_pipeline(args)
    sys.exit(0 if success else 1)
