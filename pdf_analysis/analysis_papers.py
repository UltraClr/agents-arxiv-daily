from claude_api import Client
from openai_api import OpenAIClient
from random import randint
from time import sleep
import os, json
from tqdm import tqdm
import argparse
import yaml
import re
import logging

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

def extract_relevance_score(response_text):
    """
    Extract relevance score from LLM response.
    Looks for patterns like:
    - "Keyword Relevance Score: 7.5"
    - "relevance_score": 7.5
    - Score: 8/10
    """
    try:
        # Try JSON format first
        if '{' in response_text:
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                try:
                    data = json.loads(json_match.group(0))
                    if 'relevance_score' in data:
                        return float(data['relevance_score'])
                    if 'keyword_relevance' in data:
                        kr = data['keyword_relevance']
                        if isinstance(kr, dict) and 'score' in kr:
                            return float(kr['score'])
                except:
                    pass

        # Try pattern matching
        patterns = [
            r'[Rr]elevance [Ss]core[:\s]+(\d+(?:\.\d+)?)',
            r'[Ss]core[:\s]+(\d+(?:\.\d+)?)\s*/?\s*10',
            r'"relevance_score"[:\s]+(\d+(?:\.\d+)?)',
        ]

        for pattern in patterns:
            match = re.search(pattern, response_text)
            if match:
                return float(match.group(1))

        logging.warning('Could not extract relevance score from response')
        return None
    except Exception as e:
        logging.error(f'Error extracting relevance score: {e}')
        return None

def get_paper_title_from_arxiv_json(arxiv_id, json_path='../docs/agent-arxiv-daily.json'):
    """Get paper title from arXiv JSON metadata"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Search through all categories for this arxiv_id
        for papers in data.values():
            if arxiv_id in papers:
                # Parse markdown table/list format
                row = papers[arxiv_id]
                # Extract title (strip ** markdown bold)
                if '**' in row:
                    title_match = re.search(r'\*\*(.+?)\*\*', row)
                    if title_match:
                        return title_match.group(1).strip()
        return None
    except Exception as e:
        logging.warning(f'Failed to get title from arXiv JSON: {e}')
        return None

def convet_to_file_upload_format(text_path):
    file_name = os.path.basename(text_path)
    file_size = os.path.getsize(text_path)

    return {
        "file_name": file_name,
        "file_type": "text/plain",
        "file_size": file_size,
        "extracted_content": open(text_path).read()
    }

def analysis_papers(args):
    # Extract arguments
    prompt_name = args.prompt_name
    prompt_content = args.prompt_content
    claude_results = args.claude_results
    text_parsed_saved_path = args.text_parsed_saved_path

    # Load configuration
    config = load_config(args.config_path)
    enable_validation = config.get('enable_keyword_validation', False)
    relevance_threshold = config.get('keyword_relevance_threshold', 5.0)
    auto_blacklist = config.get('auto_blacklist', True)

    # Get search keywords for validation
    search_keywords = get_search_keywords(config)
    keywords_str = ', '.join(search_keywords)

    logging.info(f'Keyword validation: {enable_validation}')
    logging.info(f'Relevance threshold: {relevance_threshold}')
    logging.info(f'Search keywords: {keywords_str}')

    # Modify prompt to include keyword relevance if validation is enabled
    if enable_validation:
        relevance_question = f"""
9. Keyword Relevance Score: On a scale of 0-10, how relevant is this paper to the following search keywords: {keywords_str}?
   - 0-3: Barely related or off-topic
   - 4-6: Partially related, mentions some concepts but not the main focus
   - 7-10: Highly relevant and directly addresses these topics

Please include your relevance score in your response. You can format it as "Keyword Relevance Score: X.X" or include it in a JSON response.
"""
        prompt_content = prompt_content.rstrip() + relevance_question
        logging.info('Added keyword relevance check to prompt')

    # Initialize API client
    if args.api == 'claudeai':
        api_client = Client(open(args.apikey).read().replace("\n", ""))
    else:
        api_client = OpenAIClient(open(args.apikey).read().replace("\n", ""), args.default_url)

    # Write prompt content to file
    os.makedirs(claude_results, exist_ok=True)
    open(os.path.join(claude_results, prompt_name+".txt"),'w').write(prompt_content)
    
    # Create directory for saving results
    saved_prefix = os.path.join(claude_results, prompt_name)
    os.makedirs(saved_prefix, exist_ok=True)

    # Get list of papers and sort in reverse order
    lists = [f for f in os.listdir(text_parsed_saved_path) if os.path.isfile(os.path.join(text_parsed_saved_path, f))]
    lists.sort(reverse=True)

    # Process each PDF
    for pdf_name in tqdm(lists):
        
        print(pdf_name)
        
        # Skip system files
        if pdf_name == '.DS_Store':
            continue
        pdf_name, _ = os.path.splitext(pdf_name)
        
        text_parsed_path = os.path.join(text_parsed_saved_path, pdf_name + ".md")
        saved_to_json_path = os.path.join(saved_prefix, pdf_name + ".json")
        if os.path.exists(saved_to_json_path):
            continue
        
        # Send message to API
        if args.api == 'claudeai':
            upload_file_format = convet_to_file_upload_format(text_parsed_path)
            conversation_id = api_client.create_new_chat()['uuid']
            response = api_client.send_message(upload_file_format, prompt_content, conversation_id)

            # Skip if no response received
            if response is None:
                print(f'Error, checking {pdf_name}')
                continue

            response_text = response.decode("utf-8")

            # Save response to JSON file
            json_result = {'conversation_id': conversation_id, 'response': response_text}
        else:
            conversation_id = 'openai'

            with open(text_parsed_path, 'r') as f:
                text_parsed_content = f.read()

            response_text = api_client.send_message(text_parsed_content + prompt_content)
            json_result = {'conversation_id': conversation_id, 'response': response_text}

        # Process keyword relevance validation
        if enable_validation and auto_blacklist:
            relevance_score = extract_relevance_score(response_text)

            if relevance_score is not None:
                logging.info(f'{pdf_name}: Relevance score = {relevance_score}')
                json_result['relevance_score'] = relevance_score

                # Add to blacklist if below threshold
                if relevance_score < relevance_threshold:
                    paper_title = get_paper_title_from_arxiv_json(pdf_name, args.arxiv_json_path)
                    if paper_title:
                        add_to_blacklist(paper_title, config.get('black_list_path', '../blacklists.txt'))
                        logging.warning(f'{pdf_name}: Low relevance ({relevance_score}), added to blacklist: {paper_title}')
                        json_result['blacklisted'] = True
                    else:
                        logging.warning(f'{pdf_name}: Low relevance ({relevance_score}), but could not find title')
                        json_result['blacklisted'] = False
                else:
                    logging.info(f'{pdf_name}: Relevance score acceptable ({relevance_score} >= {relevance_threshold})')
                    json_result['blacklisted'] = False
            else:
                logging.warning(f'{pdf_name}: Could not extract relevance score from response')
                json_result['relevance_score'] = None
                json_result['blacklisted'] = False

        # Save final result
        with open(saved_to_json_path, 'w') as f:
            json.dump(json_result, f, indent=2, ensure_ascii=False)
        # break

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze papers with optional keyword relevance validation.')
    parser.add_argument('--prompt_name', type=str, default='prompt1')
    parser.add_argument('--prompt_content', type=str, default="Please carefully review the following academic paper. After a thorough reading, summarize the essential elements by answering the following questions in a concise manner:\n \
                1.What is the primary research question or objective of the paper?\n\
                2.What is the hypothesis or theses put forward by the authors?\n\
                3.What methodology does the paper employ? Briefly describe the study design, data sources, and analysis techniques.\n\
                4.What are the key findings or results of the research?\n\
                5.How do the authors interpret these findings in the context of the existing literature on the topic?\n\
                6.What conclusions are drawn from the research?\n\
                7.Can you identify any limitations of the study mentioned by the authors?\n\
                8.What future research directions do the authors suggest?\n")

    parser.add_argument('--text_parsed_saved_path', type=str, default='./results/text_parsed/rich_markdown/')
    parser.add_argument('--claude_results', type=str, default='./results/claude_results/')
    parser.add_argument('--apikey', type=str, default='.apikey')
    parser.add_argument('--api', type=str, default='openai', choices=['openai', 'claudeai'])
    parser.add_argument('--default_url', type=str, default='https://api.xi-ai.cn') # or you can change the url to the default : https://api.openai.com
    parser.add_argument('--config_path', type=str, default='../config.yaml', help='Path to configuration file')
    parser.add_argument('--arxiv_json_path', type=str, default='../docs/agent-arxiv-daily.json', help='Path to arXiv metadata JSON')
    args = parser.parse_args()

    analysis_papers(args)



