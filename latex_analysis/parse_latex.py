import os
import re
import json
import argparse
import logging
from tqdm import tqdm

logging.basicConfig(format='[%(asctime)s %(levelname)s] %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)


class LaTeXParser:
    """Parse LaTeX source files and extract structured content"""

    def __init__(self, tex_file_path):
        self.tex_file_path = tex_file_path
        self.base_dir = os.path.dirname(tex_file_path)
        self.content = ""
        self.sections = {}

    def load_content(self):
        """Load and preprocess LaTeX content"""
        try:
            with open(self.tex_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                self.content = f.read()
        except Exception as e:
            logging.error(f"Failed to load {self.tex_file_path}: {e}")
            return False

        # Remove comments
        self.content = self.remove_comments(self.content)

        # Handle \input and \include
        self.content = self.resolve_inputs(self.content)

        return True

    def remove_comments(self, text):
        """Remove LaTeX comments (lines starting with %)"""
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            # Remove comments but keep escaped %
            if '\\%' in line:
                line = line.replace('\\%', '__PERCENT__')
            # Remove comment part
            if '%' in line:
                line = line.split('%')[0]
            line = line.replace('__PERCENT__', '\\%')
            cleaned_lines.append(line)
        return '\n'.join(cleaned_lines)

    def resolve_inputs(self, text):
        r"""Recursively resolve \input{file} and \include{file}"""
        # Pattern for \input{filename} or \include{filename}
        pattern = r'\\(?:input|include)\{([^\}]+)\}'

        def replace_input(match):
            filename = match.group(1)
            # Add .tex if not present
            if not filename.endswith('.tex'):
                filename += '.tex'

            input_path = os.path.join(self.base_dir, filename)

            if os.path.exists(input_path):
                try:
                    with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
                        return f.read()
                except:
                    return match.group(0)  # Return original if failed
            return match.group(0)

        # Recursively replace inputs (max 3 levels to avoid infinite loop)
        for _ in range(3):
            new_text = re.sub(pattern, replace_input, text)
            if new_text == text:
                break
            text = new_text

        return text

    def extract_title_abstract(self):
        """Extract paper title and abstract"""
        title = ""
        abstract = ""

        # Extract title
        title_match = re.search(r'\\title\{([^\}]+)\}', self.content)
        if title_match:
            title = title_match.group(1).strip()
            # Remove LaTeX commands
            title = re.sub(r'\\[a-zA-Z]+\{([^\}]+)\}', r'\1', title)
            title = re.sub(r'\\[a-zA-Z]+', '', title)

        # Extract abstract
        abstract_match = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}',
                                   self.content, re.DOTALL)
        if abstract_match:
            abstract = abstract_match.group(1).strip()

        return title, abstract

    def extract_sections(self):
        """Extract nested section/subsection structure"""
        # Find all section-like commands with their positions
        pattern = r'\\(section|subsection|subsubsection)\*?\{([^\}]+)\}'
        matches = list(re.finditer(pattern, self.content))

        if not matches:
            return {}

        sections = {}
        current_section_name = None
        current_subsection_name = None

        for i, match in enumerate(matches):
            level = match.group(1)  # 'section', 'subsection', or 'subsubsection'
            title = match.group(2).strip()

            # Get content between this match and next match (or end of document)
            start_pos = match.end()
            end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(self.content)
            content = self.content[start_pos:end_pos].strip()  # No length limit - keep full content

            if level == 'section':
                # New top-level section
                current_section_name = title
                current_subsection_name = None
                sections[title] = {
                    'level': 'section',
                    'content': content
                }

            elif level == 'subsection' and current_section_name:
                # Subsection under current section
                current_subsection_name = title
                if current_section_name not in sections:
                    sections[current_section_name] = {'level': 'section', 'content': ''}
                sections[current_section_name][title] = {
                    'level': 'subsection',
                    'content': content
                }

            elif level == 'subsubsection' and current_section_name and current_subsection_name:
                # Subsubsection under current subsection
                if current_section_name not in sections:
                    sections[current_section_name] = {'level': 'section', 'content': ''}
                if current_subsection_name not in sections[current_section_name]:
                    sections[current_section_name][current_subsection_name] = {'level': 'subsection', 'content': ''}
                sections[current_section_name][current_subsection_name][title] = {
                    'level': 'subsubsection',
                    'content': content
                }

        self.sections = sections
        return sections

    def extract_urls(self):
        """Extract URLs from LaTeX (GitHub, HuggingFace, project pages, etc.)"""
        urls = []

        # Pattern for \url{...} and \href{...}{...}
        url_patterns = [
            r'\\url\{([^}]+)\}',
            r'\\href\{([^}]+)\}\{[^}]*\}',
        ]

        for pattern in url_patterns:
            matches = re.findall(pattern, self.content)
            urls.extend(matches)

        # Remove duplicates and filter common URLs
        urls = list(dict.fromkeys(urls))  # Remove duplicates while preserving order

        # Filter to keep only meaningful URLs (GitHub, HuggingFace, project pages, etc.)
        filtered_urls = []
        for url in urls:
            url_lower = url.lower()
            if any(domain in url_lower for domain in ['github.com', 'huggingface.co', 'arxiv.org',
                                                        'project', 'demo', 'paperswithcode.com']):
                filtered_urls.append(url)

        return filtered_urls[:10]  # Limit to first 10 URLs

    def extract_authors_and_affiliations(self):
        """Extract authors and affiliations from LaTeX"""
        authors = []
        affiliations = []

        # Method 1: Try to find individual \author{name} entries (ACM/IEEE style)
        author_pattern = r'\\author\{([^}]+)\}'
        author_matches = re.findall(author_pattern, self.content)

        if len(author_matches) > 1:
            # Multiple separate \author{} commands found (ACM/IEEE style)
            for author_text in author_matches:
                # Clean author name
                clean_name = re.sub(r'\\[a-zA-Z]+\{?[^}]*\}?', '', author_text)
                clean_name = re.sub(r'\s+', ' ', clean_name).strip()
                if clean_name and len(clean_name) > 2:
                    authors.append(clean_name)

            # Extract affiliations from \institution{} or \affiliation{} commands
            affil_patterns = [
                r'\\institution\{([^}]+)\}',
                r'\\affiliation\{[^}]*\\institution\{([^}]+)\}',
            ]
            for pattern in affil_patterns:
                matches = re.findall(pattern, self.content)
                for match in matches:
                    clean_affil = re.sub(r'\\[a-zA-Z]+', '', match)
                    clean_affil = re.sub(r'\s+', ' ', clean_affil).strip()
                    if clean_affil and len(clean_affil) > 3:
                        affiliations.append(clean_affil)

            # Remove duplicates while preserving order
            affiliations = list(dict.fromkeys(affiliations))
            return authors[:20], affiliations[:10]

        # Method 2: Single \author{...} block (ICLR/NeurIPS style)
        author_start = self.content.find(r'\author{')
        if author_start == -1:
            return [], []

        # Find matching closing brace
        start_pos = author_start + len(r'\author{')
        brace_count = 1
        pos = start_pos

        while pos < len(self.content) and brace_count > 0:
            if self.content[pos] == '{' and (pos == 0 or self.content[pos-1] != '\\'):
                brace_count += 1
            elif self.content[pos] == '}' and (pos == 0 or self.content[pos-1] != '\\'):
                brace_count -= 1
            pos += 1

        if brace_count == 0:
            author_block = self.content[start_pos:pos-1]

            # Split by double backslash to separate author lines from affiliation lines
            lines = author_block.split('\\\\')

            # Process each line
            for line in lines:
                # Check if line looks like affiliation (has $^{number}$ pattern at start)
                if re.match(r'^\s*\$\^{?\d+}?\$', line):
                    # This is an affiliation line
                    clean_line = line
                    clean_line = re.sub(r'\$\^{?\d+}?\$', '', clean_line)  # Remove markers
                    clean_line = re.sub(r'\\quad', ' ', clean_line)  # Replace \quad with space
                    clean_line = re.sub(r'\\[a-zA-Z]+', '', clean_line)  # Remove other commands
                    clean_line = re.sub(r'\s+', ' ', clean_line).strip()
                    if clean_line and len(clean_line) > 3:
                        affiliations.append(clean_line)
                else:
                    # This is an author line
                    clean_line = line
                    # Remove \thanks, \footnotemark, etc.
                    clean_line = re.sub(r'\\thanks\{[^}]*\}', '', clean_line)
                    clean_line = re.sub(r'\\footnotemark\[[^\]]*\]', '', clean_line)
                    clean_line = re.sub(r'\$\^{?\d+[,\d]*}?\$\*?', '', clean_line)  # Remove affiliation markers
                    clean_line = re.sub(r'\\quad', ' ', clean_line)
                    clean_line = re.sub(r'\\textbf\{([^}]*)\}', r'\1', clean_line)
                    clean_line = re.sub(r'\\[a-zA-Z]+', '', clean_line)
                    clean_line = re.sub(r'\s+', ' ', clean_line).strip()

                    # Split by common separators to get individual authors
                    if clean_line:
                        author_list = re.split(r'\s+and\s+|(?<=[a-z])\s+(?=[A-Z][a-z])', clean_line)
                        for author in author_list:
                            author = author.strip()
                            if author and len(author) > 2 and not author.isdigit():
                                authors.append(author)

        return authors[:20], affiliations[:10]  # Limit to first 20 authors and 10 affiliations

    def parse(self, arxiv_id=None, title=None):
        """
        Main parsing function - extract only basic, reliable information
        @param arxiv_id: arXiv ID (if not provided, use directory name)
        @param title: Paper title from arXiv metadata (if not provided, extract from LaTeX)
        """
        if not self.load_content():
            return None

        # Use provided arxiv_id or infer from directory
        if not arxiv_id:
            arxiv_id = os.path.basename(self.base_dir)

        # Use provided title or extract from LaTeX as fallback
        if not title:
            title, _ = self.extract_title_abstract()

        # Extract abstract (skip title since we have it)
        _, abstract = self.extract_title_abstract()
        authors, affiliations = self.extract_authors_and_affiliations()
        urls = self.extract_urls()
        sections = self.extract_sections()

        # Return only basic, reliably extractable information
        result = {
            'arxiv_id': arxiv_id,
            'title': title,  # From arXiv metadata (preferred) or LaTeX (fallback)
            'authors': authors,
            'affiliations': affiliations,
            'urls': urls,
            'abstract': abstract,
            'sections': sections
        }

        return result


def find_main_tex_file(latex_dir):
    """Find the main .tex file in a directory"""
    tex_files = []
    for root, dirs, files in os.walk(latex_dir):
        for file in files:
            if file.endswith('.tex'):
                tex_files.append(os.path.join(root, file))

    if not tex_files:
        return None

    # Look for file with \documentclass
    for tex_file in tex_files:
        try:
            with open(tex_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read(5000)
                if '\\documentclass' in content or '\\begin{document}' in content:
                    return tex_file
        except:
            continue

    # Fallback: look for main.tex or similar
    for tex_file in tex_files:
        basename = os.path.basename(tex_file).lower()
        if basename in ['main.tex', 'paper.tex', 'manuscript.tex']:
            return tex_file

    return tex_files[0] if tex_files else None


def load_arxiv_metadata(json_path):
    """
    Load arXiv metadata from JSON file
    @param json_path: Path to arXiv JSON file
    @return: Dict mapping arxiv_id -> {title, ...}
    """
    metadata = {}
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Parse all papers from all categories
        for papers in data.values():
            for arxiv_id, row in papers.items():
                # Parse markdown table row: |date|title|authors|link|code|
                parts = row.split('|')
                if len(parts) >= 3:
                    title = parts[2].replace('**', '').strip()
                    metadata[arxiv_id] = {'title': title}
    except Exception as e:
        logging.warning(f'Failed to load arXiv metadata: {e}')

    return metadata


def parse_all_papers(latex_dir, saved_path, arxiv_json_path='../docs/agent-arxiv-daily.json'):
    """
    Parse all LaTeX papers in a directory
    @param latex_dir: Directory containing extracted LaTeX sources
    @param saved_path: Directory to save parsed content JSON files
    @param arxiv_json_path: Path to arXiv metadata JSON file
    @return: True if all successful, False otherwise
    """
    os.makedirs(saved_path, exist_ok=True)

    # Load arXiv metadata (for titles)
    arxiv_metadata = load_arxiv_metadata(arxiv_json_path)

    # Get all subdirectories (each is an arxiv paper)
    paper_dirs = [d for d in os.listdir(latex_dir)
                  if os.path.isdir(os.path.join(latex_dir, d))]

    logging.info(f'Found {len(paper_dirs)} paper directories')

    successful = 0
    failed = 0

    for arxiv_id in tqdm(paper_dirs, desc='Parsing LaTeX files'):
        output_json = os.path.join(saved_path, f'{arxiv_id}.json')

        # Skip if already parsed
        if os.path.exists(output_json):
            logging.info(f'{arxiv_id}: Already parsed')
            successful += 1
            continue

        latex_dir_full = os.path.join(latex_dir, arxiv_id)
        main_tex = find_main_tex_file(latex_dir_full)

        if not main_tex:
            logging.warning(f'{arxiv_id}: No main .tex file found')
            failed += 1
            continue

        # Get title from arXiv metadata
        title = arxiv_metadata.get(arxiv_id, {}).get('title')

        # Parse the LaTeX file
        parser = LaTeXParser(main_tex)
        result = parser.parse(arxiv_id=arxiv_id, title=title)

        if result:
            # Save to JSON
            with open(output_json, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            logging.info(f'{arxiv_id}: Parsed successfully')
            successful += 1
        else:
            logging.error(f'{arxiv_id}: Parsing failed')
            failed += 1

    logging.info(f'Parsing complete: {successful} successful, {failed} failed')
    return failed == 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parse LaTeX files and extract content.')
    parser.add_argument('--latex_dir', type=str,
                        default='./results/raw_latex',
                        help='Directory containing extracted LaTeX sources.')
    parser.add_argument('--saved_path', type=str,
                        default='./results/parsed_content',
                        help='Directory to save parsed content JSON files.')
    parser.add_argument('--arxiv_json_path', type=str,
                        default='../docs/agent-arxiv-daily.json',
                        help='Path to arXiv metadata JSON file.')
    args = parser.parse_args()

    import sys
    success = parse_all_papers(args.latex_dir, args.saved_path, args.arxiv_json_path)
    sys.exit(0 if success else 1)
