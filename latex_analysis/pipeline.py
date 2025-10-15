"""
Complete pipeline for LaTeX-based paper analysis
Orchestrates: download -> parse -> analyze
"""
import os
import argparse
import logging
import sys

# Import our pipeline modules
from download_latex import download_all_papers
from parse_latex import parse_all_papers
from analysis_papers import analyze_all_papers

logging.basicConfig(format='[%(asctime)s %(levelname)s] %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)


def run_latex_analysis_pipeline(args):
    """
    Run the complete LaTeX analysis pipeline
    """
    # Define paths
    output_dir = args.output_dir
    raw_latex_dir = os.path.join(output_dir, 'raw_latex')
    parsed_content_dir = os.path.join(output_dir, 'parsed_content')
    llm_analysis_path = '../docs/agent-arxiv-daily-analysis.json'  # Changed from dir to file path

    # Create directories
    for directory in [raw_latex_dir, parsed_content_dir]:
        os.makedirs(directory, exist_ok=True)

    success_count = 0
    total_steps = 3

    # Step 1: Download LaTeX sources
    if not args.skip_download:
        logging.info('=' * 60)
        logging.info('Step 1/3: Downloading LaTeX sources')
        logging.info('=' * 60)

        try:
            if download_all_papers(
                json_path=args.json_path,
                saved_path=raw_latex_dir,
                max_papers=args.max_papers
            ):
                success_count += 1
                logging.info('Download step completed successfully')
            else:
                logging.warning('Download step completed with some failures')
        except Exception as e:
            logging.error(f'Download step failed: {e}')
    else:
        logging.info('Skipping download step')
        success_count += 1

    # Step 2: Parse LaTeX files
    if not args.skip_parse:
        logging.info('=' * 60)
        logging.info('Step 2/3: Parsing LaTeX files')
        logging.info('=' * 60)

        try:
            if parse_all_papers(
                latex_dir=raw_latex_dir,
                saved_path=parsed_content_dir,
                arxiv_json_path=args.json_path
            ):
                success_count += 1
                logging.info('Parse step completed successfully')
            else:
                logging.warning('Parse step completed with some failures')
        except Exception as e:
            logging.error(f'Parse step failed: {e}')
    else:
        logging.info('Skipping parse step')
        success_count += 1

    # Step 3: LLM Analysis
    if not args.skip_analysis:
        if not args.apikey:
            logging.warning('API key not found. Skipping analysis step.')
            success_count += 1
        else:
            logging.info('=' * 60)
            logging.info('Step 3/3: Running LLM Analysis')
            logging.info('=' * 60)

            try:
                if analyze_all_papers(
                    parsed_content_path=parsed_content_dir,
                    saved_path=llm_analysis_path,
                    api=args.api,
                    arxiv_json_path=args.json_path
                ):
                    success_count += 1
                    logging.info('Analysis step completed successfully')
                else:
                    logging.warning('Analysis step completed with some failures')
            except Exception as e:
                logging.error(f'Analysis step failed: {e}')
    else:
        logging.info('Skipping analysis step')
        success_count += 1

    # Summary
    logging.info('=' * 60)
    logging.info(f'Pipeline complete: {success_count}/{total_steps} steps successful')
    logging.info(f'Results saved to: {output_dir}')
    logging.info('=' * 60)

    if success_count == total_steps:
        logging.info('All steps completed successfully!')
        return True
    else:
        logging.warning('Some steps failed. Check logs above for details.')
        return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Complete pipeline for LaTeX-based paper analysis',
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
                        help='Path to API key file')
    parser.add_argument('--default_url', type=str,
                        default='https://api.openai.com',
                        help='API base URL')

    # Control flags
    parser.add_argument('--skip_download', action='store_true',
                        help='Skip download step (use existing files)')
    parser.add_argument('--skip_parse', action='store_true',
                        help='Skip parse step (use existing parsed files)')
    parser.add_argument('--skip_analysis', action='store_true',
                        help='Skip LLM analysis step')

    # Limits
    parser.add_argument('--max_papers', type=int, default=2,
                        help='Maximum number of papers to process (for testing)')

    args = parser.parse_args()

    # Run pipeline
    success = run_latex_analysis_pipeline(args)
    sys.exit(0 if success else 1)
