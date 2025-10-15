import os
import json
import argparse
import logging
import datetime

logging.basicConfig(format='[%(asctime)s %(levelname)s] %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)


def sort_papers(papers):
    """
    Sort papers by date (newest first)
    @param papers: dict of {arxiv_id: paper_data}
    @return: sorted dict
    """
    # Create list of (date, arxiv_id, paper_data) tuples
    paper_list = []
    for arxiv_id, paper_data in papers.items():
        publish_date = paper_data.get('publish_date', '')
        paper_list.append((publish_date, arxiv_id, paper_data))

    # Sort by date (descending)
    paper_list.sort(reverse=True, key=lambda x: x[0])

    # Return as ordered dict
    sorted_papers = {}
    for _, arxiv_id, paper_data in paper_list:
        sorted_papers[arxiv_id] = paper_data

    return sorted_papers


def format_analysis_content(paper_data):
    """
    Format paper analysis into readable text for markdown
    @param paper_data: dict with analysis fields
    @return: formatted string
    """
    analysis = paper_data.get('analysis', {})

    content_parts = []

    # Authors and affiliations
    metadata = paper_data.get('metadata', {})
    authors = metadata.get('authors', [])
    if authors:
        author_str = ', '.join(authors[:5])
        if len(authors) > 5:
            author_str += ' et al.'
        content_parts.append(f"**Authors**: {author_str}")

    affiliations = metadata.get('affiliations', [])
    if affiliations:
        affil_str = ', '.join(affiliations[:3])
        content_parts.append(f"**Affiliations**: {affil_str}")

    # Resources (GitHub, HuggingFace, etc.)
    resources = metadata.get('resources', {})
    resource_links = []
    if resources.get('github'):
        resource_links.append(f"[GitHub]({resources['github']})")
    if resources.get('huggingface'):
        resource_links.append(f"[HuggingFace]({resources['huggingface']})")
    if resources.get('project_page'):
        resource_links.append(f"[Project Page]({resources['project_page']})")
    if resource_links:
        content_parts.append(f"**Resources**: {' | '.join(resource_links)}")

    content_parts.append("")  # Empty line

    # New structure fields (preferred)
    if 'summary' in analysis:
        content_parts.append(f"**Summary**: {analysis['summary']}")
        content_parts.append("")

    if 'research_question' in analysis:
        content_parts.append(f"**Research Question**: {analysis['research_question']}")
        content_parts.append("")

    if 'hypothesis' in analysis:
        content_parts.append(f"**Hypothesis**: {analysis['hypothesis']}")
        content_parts.append("")

    if 'methodology' in analysis:
        content_parts.append(f"**Methodology**: {analysis['methodology']}")
        content_parts.append("")

    if 'key_findings' in analysis:
        content_parts.append(f"**Key Findings**: {analysis['key_findings']}")
        content_parts.append("")

    if 'interpretation' in analysis:
        content_parts.append(f"**Interpretation**: {analysis['interpretation']}")
        content_parts.append("")

    if 'conclusions' in analysis:
        content_parts.append(f"**Conclusions**: {analysis['conclusions']}")
        content_parts.append("")

    if 'limitations' in analysis:
        content_parts.append(f"**Limitations**: {analysis['limitations']}")
        content_parts.append("")

    if 'future_research' in analysis:
        content_parts.append(f"**Future Research**: {analysis['future_research']}")

    # Backward compatibility: old structure fields
    if not any(key in analysis for key in ['summary', 'research_question', 'methodology']):
        # Old structure
        if 'core_innovation' in analysis:
            content_parts.append(f"**Core Innovation**: {analysis['core_innovation']}")
            content_parts.append("")

        if 'method_explanation' in analysis:
            content_parts.append(f"**Method**: {analysis['method_explanation']}")
            content_parts.append("")

        if 'experimental_validation' in analysis:
            content_parts.append(f"**Experimental Results**: {analysis['experimental_validation']}")
            content_parts.append("")

        if 'future_directions' in analysis:
            content_parts.append(f"**Future Directions**: {analysis['future_directions']}")

    # Fallback for raw_text
    if 'raw_text' in analysis and len(content_parts) <= 3:  # Only metadata
        content_parts.append(analysis['raw_text'])

    return '<br>'.join(content_parts)


def json_to_md(analysis_json_path, output_md_path, show_badges=False, repo_name='agent-arxiv-daily', trends_file=None):
    """
    Convert agent-arxiv-daily-analysis.json to markdown with collapsible sections
    @param analysis_json_path: Path to agent-arxiv-daily-analysis.json
    @param output_md_path: Path to output markdown file
    @param show_badges: Whether to show GitHub badges
    @param repo_name: Repository name for badges
    @param trends_file: Optional path to category-specific trends file (e.g., agent_trends.txt)
    """
    # Load analysis data
    if not os.path.exists(analysis_json_path):
        logging.error(f'Analysis file not found: {analysis_json_path}')
        return False

    try:
        with open(analysis_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        logging.error(f'Failed to load analysis file: {e}')
        return False

    # Get current date
    date_now = datetime.date.today().strftime('%Y.%m.%d')

    # Write markdown file
    with open(output_md_path, 'w', encoding='utf-8') as f:
        # Header with badges
        if show_badges:
            f.write(f"[![Contributors][contributors-shield]][contributors-url]\n")
            f.write(f"[![Forks][forks-shield]][forks-url]\n")
            f.write(f"[![Stargazers][stars-shield]][stars-url]\n")
            f.write(f"[![Issues][issues-shield]][issues-url]\n\n")

        # Title
        f.write("# Agent ArXiv Daily - Paper Analysis\n\n")
        f.write(f"## Updated on {date_now}\n\n")

        # Description
        f.write("This page contains AI-generated analysis of recent papers. ")
        f.write("The analysis is generated using Claude AI via OpenAI-compatible API.\n\n")
        f.write("**Note**: The generated contents are not guaranteed to be 100% accurate.\n\n")

        # Count total papers
        total_papers = sum(len(papers) for papers in data.values())
        f.write(f"**Total Papers Analyzed**: {total_papers}\n\n")

        # Table of contents
        f.write("<details>\n")
        f.write("  <summary>Table of Contents</summary>\n")
        f.write("  <ol>\n")
        for category in data.keys():
            if data[category]:
                category_anchor = category.replace(' ', '-').lower()
                paper_count = len(data[category])
                f.write(f"    <li><a href=\"#{category_anchor}\">{category}</a> ({paper_count} papers)</li>\n")
        f.write("  </ol>\n")
        f.write("</details>\n\n")

        # Process each category
        for category, papers in data.items():
            if not papers:
                continue

            # Category header
            f.write(f"## {category}\n\n")

            # Load category-specific trends if available
            trends_content = None
            if trends_file and os.path.exists(trends_file):
                try:
                    with open(trends_file, 'r', encoding='utf-8') as tf:
                        trends_content = tf.read().strip()
                    logging.info(f'Loaded trends from {trends_file}')
                except Exception as e:
                    logging.warning(f'Failed to load trends: {e}')

            # Insert trends if available (between category header and detailed analysis)
            if trends_content:
                f.write("<details>\n")
                f.write("<summary><b>📊 Research Trends</b></summary>\n")
                f.write("<p>\n\n")
                f.write(f"{trends_content}\n\n")
                f.write("</p>\n")
                f.write("</details>\n\n")
                f.write("---\n\n")

            # Detailed paper analysis section
            f.write("### 📄 Detailed Paper Analysis\n\n")

            # Sort papers by date
            sorted_papers = sort_papers(papers)

            # Write each paper as collapsible section
            for arxiv_id, paper_data in sorted_papers.items():
                title = paper_data.get('title', 'Unknown Title')
                publish_date = paper_data.get('publish_date', 'Unknown Date')

                # Format: <details><summary>date title (first author) [PDF]</summary> analysis </details>
                pdf_link = f"http://arxiv.org/pdf/{arxiv_id}.pdf"
                arxiv_abs_link = f"http://arxiv.org/abs/{arxiv_id}"

                # Get first author
                metadata = paper_data.get('metadata', {})
                authors = metadata.get('authors', [])
                first_author = authors[0] if authors else 'Unknown Author'

                # Format analysis content
                analysis_content = format_analysis_content(paper_data)

                # Write collapsible section
                f.write(f'<details>\n')
                f.write(f'<summary><b>{publish_date}</b> {title} ({first_author}) ')
                f.write(f'<a href="{arxiv_abs_link}">arXiv</a> | <a href="{pdf_link}">PDF</a></summary>\n')
                f.write(f'<p>\n\n')
                f.write(f'{analysis_content}\n\n')
                f.write(f'</p>\n')
                f.write(f'</details>\n\n')

            f.write("\n")

            # Back to top link
            top_anchor = f"#updated-on-{date_now.replace('.', '')}"
            f.write(f'<p align="right">(<a href="{top_anchor}">back to top</a>)</p>\n\n')

        # Footer with badges
        if show_badges:
            f.write(f"[contributors-shield]: https://img.shields.io/github/contributors/yourusername/{repo_name}.svg?style=for-the-badge\n")
            f.write(f"[contributors-url]: https://github.com/yourusername/{repo_name}/graphs/contributors\n")
            f.write(f"[forks-shield]: https://img.shields.io/github/forks/yourusername/{repo_name}.svg?style=for-the-badge\n")
            f.write(f"[forks-url]: https://github.com/yourusername/{repo_name}/network/members\n")
            f.write(f"[stars-shield]: https://img.shields.io/github/stars/yourusername/{repo_name}.svg?style=for-the-badge\n")
            f.write(f"[stars-url]: https://github.com/yourusername/{repo_name}/stargazers\n")
            f.write(f"[issues-shield]: https://img.shields.io/github/issues/yourusername/{repo_name}.svg?style=for-the-badge\n")
            f.write(f"[issues-url]: https://github.com/yourusername/{repo_name}/issues\n\n")

    logging.info(f'Markdown file generated: {output_md_path}')
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate markdown file with paper analysis from JSON'
    )

    parser.add_argument('--analysis_json_path', type=str,
                        default='../docs/agent-arxiv-daily-analysis.json',
                        help='Path to agent-arxiv-daily-analysis.json')
    parser.add_argument('--output_md_path', type=str,
                        default='../docs/paper_analysis.md',
                        help='Path to output markdown file')
    parser.add_argument('--show_badges', action='store_true',
                        help='Show GitHub badges in markdown')
    parser.add_argument('--repo_name', type=str,
                        default='agent-arxiv-daily',
                        help='Repository name for badges')
    parser.add_argument('--trends_file', type=str,
                        default='../docs/agent_trends.txt',
                        help='Path to trends file (e.g., agent_trends.txt)')

    args = parser.parse_args()

    success = json_to_md(
        args.analysis_json_path,
        args.output_md_path,
        args.show_badges,
        args.repo_name,
        args.trends_file
    )

    exit(0 if success else 1)
