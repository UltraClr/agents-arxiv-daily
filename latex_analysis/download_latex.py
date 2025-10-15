import os
import json
import requests
import tarfile
import shutil
from tqdm import tqdm
import argparse
import logging

logging.basicConfig(format='[%(asctime)s %(levelname)s] %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

def download_latex_source(arxiv_id, save_dir):
    """
    Download LaTeX source from arXiv
    @param arxiv_id: arXiv ID (e.g., '2401.12345')
    @param save_dir: directory to save the downloaded files
    @return: path to extracted directory, or None if failed
    """
    # Clean arxiv_id (remove version if present)
    clean_id = arxiv_id.split('v')[0] if 'v' in arxiv_id else arxiv_id

    tar_path = os.path.join(save_dir, f'{clean_id}.tar.gz')
    extract_dir = os.path.join(save_dir, clean_id)

    # Skip if already downloaded and extracted
    if os.path.exists(extract_dir) and os.listdir(extract_dir):
        logging.info(f'{clean_id}: Already downloaded and extracted')
        return extract_dir

    # Skip if tar.gz exists (will extract later)
    if os.path.exists(tar_path):
        logging.info(f'{clean_id}: Archive exists, extracting...')
    else:
        # Download from arXiv e-print endpoint
        url = f'https://arxiv.org/e-print/{clean_id}'
        try:
            logging.info(f'{clean_id}: Downloading from {url}')
            response = requests.get(url, timeout=30)

            if response.status_code == 200:
                # Save the file
                with open(tar_path, 'wb') as f:
                    f.write(response.content)
                logging.info(f'{clean_id}: Downloaded successfully')
            else:
                logging.warning(f'{clean_id}: Failed to download (status {response.status_code})')
                return None

        except Exception as e:
            logging.error(f'{clean_id}: Download error - {e}')
            return None

    # Extract the archive
    try:
        os.makedirs(extract_dir, exist_ok=True)

        # Try to extract as tar.gz
        try:
            with tarfile.open(tar_path, 'r:gz') as tar:
                tar.extractall(extract_dir)
            logging.info(f'{clean_id}: Extracted as tar.gz')
        except:
            # Some papers are single .tex files, not archives
            # In this case, the downloaded file is already the .tex
            shutil.copy(tar_path, os.path.join(extract_dir, f'{clean_id}.tex'))
            logging.info(f'{clean_id}: Copied as single .tex file')

        return extract_dir

    except Exception as e:
        logging.error(f'{clean_id}: Extraction error - {e}')
        return None

# https://github.com/showlab/Paper2Video/blob/main/src/slide_code_gen_select_improvement.py#L596
def find_all_tex_files(root_dir):
    tex_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(".tex"):
                full_path = os.path.join(dirpath, filename)
                try:
                    with open(full_path, 'r', encoding='utf-8') as f:
                        tex_files.append(f.read())
                except Exception as e:
                    print(f"⚠️ Skip {full_path}: {e}")
                    continue
    return tex_files


def download_all_papers(json_path, saved_path, max_papers=None):
    """
    Download all LaTeX sources from arXiv based on JSON file
    @param json_path: Path to JSON file with arxiv IDs
    @param saved_path: Directory to save downloaded LaTeX sources
    @param max_papers: Maximum number of papers to download (None for all)
    @return: True if all successful, False otherwise
    """
    # Create output directory
    os.makedirs(saved_path, exist_ok=True)

    # Load JSON with paper data
    with open(json_path, 'r') as file:
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
    if max_papers:
        all_ids = all_ids[:max_papers]

    logging.info(f'Total papers to download: {len(all_ids)}')

    # Download and extract
    successful = 0
    failed = 0

    for arxiv_id in tqdm(all_ids, desc='Downloading LaTeX sources'):
        extract_dir = download_latex_source(arxiv_id, saved_path)

        if extract_dir:
            # Try to find main .tex file
            main_tex = find_all_tex_files(extract_dir)
            if main_tex:
                logging.info(f'{arxiv_id}: Found {len(main_tex)} .tex files')
                successful += 1
            else:
                logging.warning(f'{arxiv_id}: No .tex file found')
                failed += 1
        else:
            failed += 1

    logging.info(f'Download complete: {successful} successful, {failed} failed')
    return failed == 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download LaTeX source from arXiv.')
    parser.add_argument('--json_path', type=str,
                        default='../docs/agent-arxiv-daily.json',
                        help='Path to the JSON file with arxiv IDs.')
    parser.add_argument('--saved_path', type=str,
                        default='./results/raw_latex',
                        help='Path where the LaTeX sources will be saved.')
    parser.add_argument('--max_papers', type=int, default=None,
                        help='Maximum number of papers to download (for testing)')
    args = parser.parse_args()

    import sys
    success = download_all_papers(args.json_path, args.saved_path, args.max_papers)
    sys.exit(0 if success else 1)
