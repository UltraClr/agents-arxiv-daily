"""
Streaming pipeline for LaTeX-based paper analysis with PDF fallback
Process each paper as soon as it's downloaded: download -> parse -> analyze
Automatically falls back to PDF analysis if LaTeX source is unavailable
"""
import os
import json
import argparse
import logging
import sys
import time
import re
from tqdm import tqdm
from pdfminer.high_level import extract_text

# Import LaTeX processing functions
from download_latex import download_latex_source, find_all_tex_files
from parse_latex import LaTeXParser, find_main_tex_file, load_arxiv_metadata
from analysis_papers import (
    create_analysis_prompt,
    get_title_from_arxiv_json,
    get_category_from_arxiv_json,
    load_config,
    get_search_keywords,
    extract_relevance_from_json,
    add_to_blacklist,
    remove_paper_from_json_files
)
from openai_api import OpenAIClient as OpenAIClientOrig
from claude_api import OpenAIClient as ClaudeClient
from generating_paper_analysis import json_to_md

# Import PDF processing functions (reuse existing code)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..'))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Simple PDF download function
import requests

def download_pdf(output_dir, arxiv_id, pdf_url):
    """Download PDF from arXiv"""
    os.makedirs(output_dir, exist_ok=True)
    pdf_path = os.path.join(output_dir, f'{arxiv_id}.pdf')

    try:
        response = requests.get(pdf_url, timeout=30)
        response.raise_for_status()

        with open(pdf_path, 'wb') as f:
            f.write(response.content)

        return pdf_path
    except Exception as e:
        logging.error(f'Failed to download PDF from {pdf_url}: {e}')
        return None

logging.basicConfig(format='[%(asctime)s %(levelname)s] %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)


def process_single_paper(arxiv_id, raw_latex_dir, raw_pdf_dir, parsed_content_dir,
                         parsed_pdf_dir, api_client, analysis_data, arxiv_json_path,
                         arxiv_metadata, config, enable_validation, search_keywords,
                         relevance_threshold, auto_blacklist, blacklist_path,
                         force_reprocess=False, enable_pdf_fallback=True):
    """
    Process a single paper: download -> parse -> analyze (with PDF fallback)
    @param arxiv_id: arXiv ID to process
    @param raw_latex_dir: Directory to save raw LaTeX
    @param raw_pdf_dir: Directory to save raw PDFs (for fallback)
    @param parsed_content_dir: Directory to save parsed LaTeX content
    @param parsed_pdf_dir: Directory to save parsed PDF content (for fallback)
    @param api_client: LLM API client
    @param analysis_data: Dictionary to store analysis results (modified in place)
    @param arxiv_json_path: Path to arXiv metadata JSON
    @param arxiv_metadata: Pre-loaded arXiv metadata dictionary
    @param force_reprocess: If True, reprocess even if already analyzed
    @return: True if successful, False otherwise, or 'skipped' if already analyzed
    """
    logging.info(f'\n{"="*60}')
    logging.info(f'Processing paper: {arxiv_id}')
    logging.info(f'{"="*60}')

    # Check if already analyzed (early exit before downloading)
    already_analyzed = any(arxiv_id in papers for papers in analysis_data.values())
    if already_analyzed and not force_reprocess:
        logging.info(f'{arxiv_id}: Already in analysis data, skipping all steps')
        return 'skipped'

    # Step 1: Download LaTeX source
    logging.info(f'{arxiv_id}: [1/3] Downloading LaTeX source...')
    extract_dir = download_latex_source(arxiv_id, raw_latex_dir)

    if not extract_dir:
        if enable_pdf_fallback:
            logging.warning(f'{arxiv_id}: LaTeX download failed, trying PDF fallback')
            return process_single_paper_pdf(
                arxiv_id, raw_pdf_dir, parsed_pdf_dir,
                api_client, analysis_data, arxiv_json_path,
                arxiv_metadata, config, enable_validation,
                search_keywords, relevance_threshold,
                auto_blacklist, blacklist_path
            )
        else:
            logging.error(f'{arxiv_id}: LaTeX download failed, skipping')
            return False

    # Verify LaTeX files exist
    tex_files = find_all_tex_files(extract_dir)
    if not tex_files:
        if enable_pdf_fallback:
            logging.warning(f'{arxiv_id}: No .tex files found, trying PDF fallback')
            return process_single_paper_pdf(
                arxiv_id, raw_pdf_dir, parsed_pdf_dir,
                api_client, analysis_data, arxiv_json_path,
                arxiv_metadata, config, enable_validation,
                search_keywords, relevance_threshold,
                auto_blacklist, blacklist_path
            )
        else:
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

    # Create prompt and call LLM (with keyword validation if enabled)
    prompt = create_analysis_prompt(parsed_data, search_keywords, enable_validation)

    try:
        response = api_client.send_message(prompt)

        if not response:
            logging.error(f'{arxiv_id}: Empty response from API')
            return False

        # Try to parse LLM response as JSON
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                analysis_json = json.loads(json_match.group(0))
            else:
                analysis_json = json.loads(response)

            # Extract metadata from LLM response
            llm_metadata = analysis_json.get('metadata', {})
            llm_resources = llm_metadata.get('resources', {})

            # Build final result with new fields
            result = {
                'arxiv_id': arxiv_id,
                'title': title,
                'publish_date': publish_date,
                'source_type': 'latex',

                # New metadata fields
                'first_author': llm_metadata.get('first_author'),
                'corresponding_author': llm_metadata.get('corresponding_author'),
                'author_affiliations': llm_metadata.get('author_affiliations', {}),

                # URLs (restructured)
                'urls': {
                    'github': llm_resources.get('github'),
                    'huggingface': llm_resources.get('huggingface'),
                    'project_page': llm_resources.get('project_page'),
                    'other': llm_resources.get('other_links', [])
                },

                # One sentence summary
                'one_sentence_summary': analysis_json.get('one_sentence_summary'),

                # Keep metadata for backward compatibility (optional)
                'metadata': llm_metadata
            }

            # Process keyword relevance validation (same logic as PDF)
            if enable_validation:
                relevance_score, reasoning, matching_keywords = extract_relevance_from_json(analysis_json)

                if relevance_score is not None:
                    logging.info(f'{arxiv_id}: Relevance score = {relevance_score}')
                    result['keyword_relevance_score'] = relevance_score
                    result['keyword_reasoning'] = reasoning
                    result['matching_keywords'] = matching_keywords

                    # Add to blacklist if below threshold
                    if auto_blacklist and relevance_score < relevance_threshold:
                        add_to_blacklist(title, blacklist_path)
                        # Remove from JSON data files
                        remove_paper_from_json_files(arxiv_id, title, config)
                        logging.warning(f'{arxiv_id}: Low relevance ({relevance_score}), added to blacklist and removed from JSON files')
                        result['blacklisted'] = True
                        result['blacklist_reason'] = f"Low relevance score: {relevance_score}"
                    else:
                        logging.info(f'{arxiv_id}: Relevance acceptable ({relevance_score} >= {relevance_threshold})')
                        result['blacklisted'] = False
                        result['blacklist_reason'] = ""
                else:
                    logging.warning(f'{arxiv_id}: Could not extract relevance score')
                    result['keyword_relevance_score'] = None
                    result['blacklisted'] = False
                    result['blacklist_reason'] = ""
            else:
                result['blacklisted'] = False
                result['blacklist_reason'] = ""
        except json.JSONDecodeError:
            # Fallback: if LLM didn't return valid JSON, save as text
            logging.warning(f'{arxiv_id}: LLM response is not valid JSON, saving as text')
            result = {
                'arxiv_id': arxiv_id,
                'title': title,
                'publish_date': publish_date,
                'source_type': 'latex',  # Mark source type
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

        logging.info(f'{arxiv_id}: Analysis complete (category: {category})')
        return True

    except Exception as e:
        logging.error(f'{arxiv_id}: API call failed - {e}')
        return False


def process_single_paper_pdf(arxiv_id, raw_pdf_dir, parsed_pdf_dir,
                             api_client, analysis_data, arxiv_json_path,
                             arxiv_metadata, config, enable_validation,
                             search_keywords, relevance_threshold,
                             auto_blacklist, blacklist_path):
    """
    PDF fallback processing pipeline
    Called when LaTeX source is unavailable
    Reuses pdf_analysis module components
    @return: True if successful, False otherwise
    """
    logging.info(f'{arxiv_id}: [PDF FALLBACK] LaTeX not available, using PDF analysis')

    # Step 1: Download PDF (reuse pdf_analysis/download_pdf.py)
    logging.info(f'{arxiv_id}: [1/3] Downloading PDF...')
    pdf_url = f'https://arxiv.org/pdf/{arxiv_id}.pdf'
    pdf_path = os.path.join(raw_pdf_dir, f'{arxiv_id}.pdf')

    try:
        os.makedirs(raw_pdf_dir, exist_ok=True)
        download_pdf(raw_pdf_dir, arxiv_id, pdf_url)

        if not os.path.exists(pdf_path):
            logging.error(f'{arxiv_id}: PDF download failed')
            return False

        logging.info(f'{arxiv_id}: PDF downloaded successfully')
    except Exception as e:
        logging.error(f'{arxiv_id}: PDF download error - {e}')
        return False

    # Step 2: Parse PDF
    logging.info(f'{arxiv_id}: [2/3] Parsing PDF...')
    parsed_md_path = os.path.join(parsed_pdf_dir, f'{arxiv_id}.md')

    if not os.path.exists(parsed_md_path):
        try:
            # Extract text using pdfminer
            text_content = extract_text(pdf_path)

            # Clean text
            text_content = re.sub(r"-\n", "", text_content)
            text_content = text_content.replace("\n\n\n", "\n\n")

            # Get title
            title = arxiv_metadata.get(arxiv_id, {}).get('title', f'Paper {arxiv_id}')
            if not title or title == f'Paper {arxiv_id}':
                title = get_title_from_arxiv_json(arxiv_id, arxiv_json_path)

            # Save as markdown
            os.makedirs(parsed_pdf_dir, exist_ok=True)
            with open(parsed_md_path, 'w', encoding='utf-8') as f:
                f.write(f"# {title}\n\n{text_content}")

            logging.info(f'{arxiv_id}: PDF parsed successfully')
        except Exception as e:
            logging.error(f'{arxiv_id}: PDF parsing failed - {e}')
            return False

    # Step 3: LLM Analysis
    if not api_client:
        logging.info(f'{arxiv_id}: Skipping analysis (no API client)')
        return True

    logging.info(f'{arxiv_id}: [3/3] Running LLM analysis...')

    # Read parsed content
    with open(parsed_md_path, 'r', encoding='utf-8') as f:
        parsed_text = f.read()

    # Build parsed_data structure (compatible with LaTeX format)
    title = arxiv_metadata.get(arxiv_id, {}).get('title')
    if not title:
        title = get_title_from_arxiv_json(arxiv_id, arxiv_json_path)

    publish_date = arxiv_metadata.get(arxiv_id, {}).get('publish_date')

    parsed_data = {
        'title': title or f'Paper {arxiv_id}',
        'abstract': '',  # PDF hard to extract accurately
        'authors': [],
        'affiliations': [],
        'urls': [],
        'sections': parsed_text[:100000],  # Limit length to avoid token limit
        'source_type': 'pdf'  # Mark source type
    }

    # Get paper category
    category = get_category_from_arxiv_json(arxiv_id, arxiv_json_path)

    # Create prompt (with keyword validation)
    prompt = create_analysis_prompt(parsed_data, search_keywords, enable_validation)

    try:
        response = api_client.send_message(prompt)

        if not response:
            logging.error(f'{arxiv_id}: Empty response from API')
            return False

        # Parse JSON response
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                analysis_json = json.loads(json_match.group(0))
            else:
                analysis_json = json.loads(response)

            # Extract metadata from LLM response (same as LaTeX)
            llm_metadata = analysis_json.get('metadata', {})
            llm_resources = llm_metadata.get('resources', {})

            # Build result with new fields (same structure as LaTeX)
            result = {
                'arxiv_id': arxiv_id,
                'title': title,
                'publish_date': publish_date,
                'source_type': 'pdf',

                # New metadata fields
                'first_author': llm_metadata.get('first_author'),
                'corresponding_author': llm_metadata.get('corresponding_author'),
                'author_affiliations': llm_metadata.get('author_affiliations', {}),

                # URLs (restructured)
                'urls': {
                    'github': llm_resources.get('github'),
                    'huggingface': llm_resources.get('huggingface'),
                    'project_page': llm_resources.get('project_page'),
                    'other': llm_resources.get('other_links', [])
                },

                # One sentence summary
                'one_sentence_summary': analysis_json.get('one_sentence_summary'),

                # Keep metadata for backward compatibility
                'metadata': llm_metadata
            }

            # Process keyword validation (same logic as LaTeX)
            if enable_validation:
                relevance_score, reasoning, matching_keywords = extract_relevance_from_json(analysis_json)

                if relevance_score is not None:
                    logging.info(f'{arxiv_id}: Relevance score = {relevance_score}')
                    result['keyword_relevance_score'] = relevance_score
                    result['keyword_reasoning'] = reasoning
                    result['matching_keywords'] = matching_keywords

                    # Add to blacklist if below threshold
                    if auto_blacklist and relevance_score < relevance_threshold:
                        add_to_blacklist(title, blacklist_path)
                        # Remove from JSON data files
                        remove_paper_from_json_files(arxiv_id, title, config)
                        logging.warning(f'{arxiv_id}: Low relevance ({relevance_score}), added to blacklist and removed from JSON files')
                        result['blacklisted'] = True
                        result['blacklist_reason'] = f"Low relevance score: {relevance_score}"
                    else:
                        logging.info(f'{arxiv_id}: Relevance acceptable ({relevance_score} >= {relevance_threshold})')
                        result['blacklisted'] = False
                        result['blacklist_reason'] = ""
                else:
                    logging.warning(f'{arxiv_id}: Could not extract relevance score')
                    result['keyword_relevance_score'] = None
                    result['blacklisted'] = False
                    result['blacklist_reason'] = ""
            else:
                result['blacklisted'] = False
                result['blacklist_reason'] = ""

        except json.JSONDecodeError:
            logging.warning(f'{arxiv_id}: Invalid JSON response, saving as text')
            result = {
                'arxiv_id': arxiv_id,
                'title': title,
                'publish_date': publish_date,
                'source_type': 'pdf',
                'metadata': {'authors': [], 'affiliations': [], 'resources': {}},
                'analysis': {'raw_text': response},
                'blacklisted': False
            }

        # Add to analysis data
        if category not in analysis_data:
            analysis_data[category] = {}

        analysis_data[category][arxiv_id] = result

        logging.info(f'{arxiv_id}: PDF analysis complete (category: {category})')
        return True

    except Exception as e:
        logging.error(f'{arxiv_id}: API call failed - {e}')
        return False


def run_streaming_pipeline(args):
    """
    Run the streaming LaTeX analysis pipeline with PDF fallback
    Process papers one by one: download -> parse -> analyze
    Automatically falls back to PDF if LaTeX is unavailable
    """
    # Load configuration for keyword validation
    config = load_config(args.config_path)
    enable_validation = config.get('enable_keyword_validation', False)
    relevance_threshold = config.get('keyword_relevance_threshold', 5.0)
    auto_blacklist = config.get('auto_blacklist', True)
    blacklist_path = "../blacklists.txt"

    # Get search keywords
    search_keywords = get_search_keywords(config)

    logging.info(f'Keyword validation: {enable_validation}')
    logging.info(f'Relevance threshold: {relevance_threshold}')
    logging.info(f'PDF fallback: {args.enable_pdf_fallback}')

    # Define paths
    output_dir = args.output_dir
    raw_latex_dir = os.path.join(output_dir, 'raw_latex')
    raw_pdf_dir = os.path.join(output_dir, 'raw_pdfs')          # Added for PDF fallback
    parsed_content_dir = os.path.join(output_dir, 'parsed_content')
    parsed_pdf_dir = os.path.join(output_dir, 'parsed_pdfs')    # Added for PDF fallback
    llm_analysis_path = '../docs/agent-arxiv-daily-analysis.json'

    # Create directories
    for directory in [raw_latex_dir, raw_pdf_dir, parsed_content_dir, parsed_pdf_dir]:
        os.makedirs(directory, exist_ok=True)

    # Initialize API client if needed
    api_client = None
    if not args.skip_analysis:
        if not args.apikey:
            logging.warning('API key not found. Skipping analysis step.')
        else:
            try:
                # Select API client based on --api argument
                if args.api == 'claude':
                    api_client = ClaudeClient(args.apikey, os.environ.get('CRS_BASE_URL'))
                    logging.info('Claude API client initialized')
                else:  # openai
                    api_client = OpenAIClientOrig(args.apikey, os.environ.get('CRS_BASE_URL'))
                    logging.info('OpenAI API client initialized')
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

    # Collect arxiv IDs by category, sorted by date (newest first)
    all_ids = []
    for keyword in data.keys():
        # Collect papers for this category with dates
        category_papers = []
        for arxiv_id in data[keyword].keys():
            # Skip old format IDs (contain '/')
            if '/' in arxiv_id:
                continue
            # Get publish date from metadata
            publish_date = arxiv_metadata.get(arxiv_id, {}).get('publish_date', '1970-01-01')
            category_papers.append((arxiv_id, publish_date))

        # Sort this category by date (newest first)
        category_papers.sort(key=lambda x: x[1], reverse=True)

        # Limit papers per category if max_papers specified
        if args.max_papers:
            category_papers = category_papers[:args.max_papers]
            logging.info(f'Category "{keyword}": selecting {len(category_papers)} newest papers (max_papers={args.max_papers})')
        else:
            logging.info(f'Category "{keyword}": processing all {len(category_papers)} papers')

        # Show date range for this category
        if category_papers:
            newest_date = category_papers[0][1]
            oldest_date = category_papers[-1][1]
            logging.info(f'  Date range: {newest_date} (newest) to {oldest_date} (oldest)')

        # Add to overall list
        all_ids.extend([arxiv_id for arxiv_id, _ in category_papers])

    logging.info(f'\n{"="*60}')
    logging.info(f'Starting streaming pipeline for {len(all_ids)} papers')
    logging.info(f'{"="*60}\n')

    # Process statistics
    successful = 0
    failed = 0
    skipped = 0
    newly_analyzed_papers = []  # Track newly analyzed paper IDs

    # Process each paper one by one
    for arxiv_id in tqdm(all_ids, desc='Processing papers'):
        # Process this paper (early skip check is inside process_single_paper)
        result = process_single_paper(
            arxiv_id=arxiv_id,
            raw_latex_dir=raw_latex_dir,
            raw_pdf_dir=raw_pdf_dir,                    # Added
            parsed_content_dir=parsed_content_dir,
            parsed_pdf_dir=parsed_pdf_dir,              # Added
            api_client=api_client,
            analysis_data=analysis_data,
            arxiv_json_path=args.json_path,
            arxiv_metadata=arxiv_metadata,
            config=config,                              # Added
            enable_validation=enable_validation,        # Added
            search_keywords=search_keywords,            # Added
            relevance_threshold=relevance_threshold,    # Added
            auto_blacklist=auto_blacklist,              # Added
            blacklist_path=blacklist_path,              # Added
            force_reprocess=args.force_reprocess,
            enable_pdf_fallback=args.enable_pdf_fallback  # Added
        )

        # Handle different results
        if result == 'skipped':
            skipped += 1
        elif result:  # True - success
            successful += 1
            newly_analyzed_papers.append(arxiv_id)  # Record newly analyzed paper

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
        else:  # False - failed
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

    # Save list of newly analyzed papers for webhook notification
    if api_client and newly_analyzed_papers:
        newly_analyzed_path = '../docs/newly_analyzed_papers.json'
        try:
            os.makedirs(os.path.dirname(newly_analyzed_path) or '.', exist_ok=True)
            with open(newly_analyzed_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()),
                    'count': len(newly_analyzed_papers),
                    'paper_ids': newly_analyzed_papers
                }, f, indent=2, ensure_ascii=False)
            logging.info(f'Newly analyzed papers list saved: {newly_analyzed_path}')
        except Exception as e:
            logging.warning(f'Failed to save newly analyzed papers list: {e}')

    # Generate final report if analysis was performed
    if api_client and not args.skip_report:
        logging.info('\n' + '='*60)
        logging.info('Generating final markdown report...')
        logging.info('='*60 + '\n')

        report_path = args.report_path or '../docs/paper_analysis.md'
        trends_file = args.trends_file if hasattr(args, 'trends_file') else None

        try:
            if json_to_md(
                analysis_json_path=llm_analysis_path,
                output_md_path=report_path,
                show_badges=args.show_badges if hasattr(args, 'show_badges') else False,
                repo_name=args.repo_name if hasattr(args, 'repo_name') else 'agent-arxiv-daily',
                trends_file=trends_file
            ):
                logging.info(f'Report successfully generated: {report_path}')
            else:
                logging.warning('Report generation completed with warnings')
        except Exception as e:
            logging.error(f'Failed to generate report: {e}')

    # Consider successful if:
    # 1. At least one paper was successfully processed, OR
    # 2. All papers were skipped (no new papers to analyze)
    # Only return failure if there were papers to process but ALL failed
    return successful > 0 or (failed == 0 and skipped > 0) or len(all_ids) == 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Streaming pipeline for LaTeX-based paper analysis with PDF fallback',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    default_api_key = os.environ.get('ANTHROPIC_AUTH_TOKEN')
    if not default_api_key:
        logging.warning('API key not found in env.')
    # Input/Output
    parser.add_argument('--json_path', type=str,
                        default='../docs/agent-arxiv-daily.json',
                        help='Path to JSON file with arXiv IDs')
    parser.add_argument('--output_dir', type=str,
                        default='./results',
                        help='Base directory for all outputs')

    # API configuration
    parser.add_argument('--api', type=str, default='claude',
                        choices=['openai', 'claude'],
                        help='LLM API to use (default: claude)')
    parser.add_argument('--apikey', type=str, default=default_api_key,
                        help='API key or path to API key file')

    # Control flags
    parser.add_argument('--skip_analysis', action='store_true',
                        help='Skip LLM analysis step (only download and parse)')
    parser.add_argument('--force_reprocess', action='store_true',
                        help='Force reprocessing of already analyzed papers')
    parser.add_argument('--enable_pdf_fallback', action='store_true',
                        default=True,
                        help='Enable PDF fallback when LaTeX source is unavailable (default: True)')
    parser.add_argument('--config_path', type=str,
                        default='../config.yaml',
                        help='Path to configuration file for keyword validation')

    # Limits
    parser.add_argument('--max_papers', type=int, default=None,
                        help='Maximum number of papers to process (for testing)')
    parser.add_argument('--delay_seconds', type=int, default=10,
                        help='Delay in seconds between papers (to avoid rate limits)')

    # Report generation
    parser.add_argument('--skip_report', action='store_true',
                        help='Skip final report generation')
    parser.add_argument('--report_path', type=str, default=None,
                        help='Path to output markdown report (default: ../docs/paper_analysis.md)')
    parser.add_argument('--show_badges', action='store_true',
                        help='Show GitHub badges in markdown report')
    parser.add_argument('--repo_name', type=str, default='agent-arxiv-daily',
                        help='Repository name for badges')
    parser.add_argument('--trends_file', type=str, default=None,
                        help='Path to trends file (e.g., ../docs/agent_trends.txt)')

    args = parser.parse_args()

    # Run streaming pipeline
    success = run_streaming_pipeline(args)
    sys.exit(0 if success else 1)
