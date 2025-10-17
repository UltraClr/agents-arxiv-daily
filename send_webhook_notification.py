#!/usr/bin/env python3
"""
DingTalk webhook notification script for arXiv daily updates
Uses LLM to generate a comprehensive summary of newly analyzed papers
"""

import os
import sys
import json
import requests
import time
import hmac
import hashlib
import base64
import urllib.parse
from typing import Dict, List, Optional
from latex_analysis.openai_api import OpenAIClient


def load_newly_analyzed_papers(newly_analyzed_path: str) -> Optional[Dict]:
    """Load the list of newly analyzed papers"""
    try:
        with open(newly_analyzed_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading newly analyzed papers: {e}")
        return None


def load_analysis_data(analysis_json_path: str) -> Optional[Dict]:
    """Load the complete analysis JSON data"""
    try:
        with open(analysis_json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading analysis data: {e}")
        return None


def collect_papers_info(paper_ids: List[str], analysis_data: Dict) -> List[Dict]:
    """Collect information for newly analyzed papers"""
    papers = []
    for category, paper_list in analysis_data.items():
        if isinstance(paper_list, dict):
            for paper_id, paper_info in paper_list.items():
                if paper_id in paper_ids and isinstance(paper_info, dict):
                    papers.append({
                        'id': paper_id,
                        'title': paper_info.get('title', 'Unknown'),
                        'category': category,
                        'analysis': paper_info.get('analysis', {})
                    })
    return papers


def generate_llm_summary_prompt(papers: List[Dict]) -> str:
    """Generate prompt for LLM to summarize papers"""

    prompt_lines = [
        "请对以下新分析的 arXiv 论文进行整体总结。要求：",
        "1. 用简洁的语言概括今日论文的主要研究方向和趋势",
        "2. 突出最有价值或最有创新性的研究",
        "3. 总结控制在300字以内",
        "4. 用中文输出，语言要专业但易懂",
        "",
        "今日论文列表：",
        ""
    ]

    for i, paper in enumerate(papers, 1):
        prompt_lines.append(f"{i}. 【{paper['category']}】{paper['title']}")

        analysis = paper['analysis']
        if isinstance(analysis, dict):
            if 'summary' in analysis:
                summary_text = analysis['summary'][:300]
                prompt_lines.append(f"   研究内容: {summary_text}")
            elif 'research_question' in analysis:
                prompt_lines.append(f"   研究问题: {analysis['research_question'][:200]}")

        prompt_lines.append("")

    prompt_lines.append("请生成一个简洁的总结：")

    return '\n'.join(prompt_lines)


def call_llm_for_summary(prompt: str, api_key: str, base_url: Optional[str] = None) -> Optional[str]:
    """Call OpenAI-compatible API to generate summary"""

    try:
        client = OpenAIClient(api_key, base_url)

        response = client.send_message(prompt)

        return response

    except Exception as e:
        print(f"Error calling LLM API: {e}")
        return None


def format_dingtalk_message(summary: str, papers: List[Dict], timestamp: str) -> str:
    """Format message for DingTalk markdown"""

    lines = [
        "# 📚 arXiv 每日论文分析更新",
        "",
        f"**更新时间：** {timestamp}",
        f"**新分析论文数：** {len(papers)} 篇",
        "",
        "---",
        "",
        "## 📝 今日综述",
        "",
        summary,
        "",
        "---",
        "",
        "## 📄 论文列表",
        ""
    ]

    for i, paper in enumerate(papers[:10], 1):  # Show max 10 papers
        lines.append(f"{i}. **[{paper['category']}]** {paper['title'][:60]}...")
        lines.append(f"   🔗 https://arxiv.org/abs/{paper['id']}")
        lines.append("")

    if len(papers) > 10:
        lines.append(f"... 以及其他 {len(papers) - 10} 篇论文")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("**查看完整报告：** https://github.com/ultraclr/agent-arxiv-daily/blob/main/docs/paper_analysis.md")

    return '\n'.join(lines)


def generate_dingtalk_sign(secret: str) -> tuple:
    """Generate DingTalk webhook signature"""
    timestamp = str(round(time.time() * 1000))
    secret_enc = secret.encode('utf-8')
    string_to_sign = '{}\n{}'.format(timestamp, secret)
    string_to_sign_enc = string_to_sign.encode('utf-8')
    hmac_code = hmac.new(secret_enc, string_to_sign_enc, digestmod=hashlib.sha256).digest()
    sign = urllib.parse.quote_plus(base64.b64encode(hmac_code))
    return timestamp, sign


def send_dingtalk_webhook(webhook_url: str, content: str, secret: Optional[str] = None) -> bool:
    """Send notification to DingTalk webhook with optional signature"""

    try:
        # Add signature parameters if secret is provided
        url = webhook_url
        if secret:
            timestamp, sign = generate_dingtalk_sign(secret)
            separator = '&' if '?' in url else '?'
            url = f"{webhook_url}{separator}timestamp={timestamp}&sign={sign}"
            print(f"   Using signed webhook (timestamp: {timestamp})")

        payload = {
            'msgtype': 'markdown',
            'markdown': {
                'title': 'arXiv 每日论文分析更新',
                'text': content
            }
        }

        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()

        result = response.json()
        if result.get('errcode') == 0:
            print(f"✅ DingTalk webhook notification sent successfully")
            return True
        else:
            print(f"❌ DingTalk API error: {result.get('errmsg', 'Unknown error')}")
            return False

    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to send DingTalk webhook: {e}")
        return False


def main():
    # Get configuration from environment variables
    webhook_url = os.environ.get('DINGTALK_WEBHOOK_URL')
    webhook_secret = os.environ.get('DINGTALK_WEBHOOK_SECRET')
    api_key = os.environ.get('ANTHROPIC_AUTH_TOKEN') or os.environ.get('OPENAI_API_KEY')
    base_url = os.environ.get('CRS_BASE_URL')

    if not webhook_url:
        print("⚠️  DINGTALK_WEBHOOK_URL not set, skipping notification")
        return 0

    if not api_key:
        print("⚠️  API key not set (ANTHROPIC_AUTH_TOKEN or OPENAI_API_KEY), skipping notification")
        return 0

    # Get file paths
    newly_analyzed_path = os.environ.get('NEWLY_ANALYZED_PATH', './docs/newly_analyzed_papers.json')
    analysis_json_path = os.environ.get('ANALYSIS_JSON_PATH', './docs/agent-arxiv-daily-analysis.json')

    # Load newly analyzed papers list
    print(f"📖 Loading newly analyzed papers from: {newly_analyzed_path}")
    newly_analyzed = load_newly_analyzed_papers(newly_analyzed_path)

    if not newly_analyzed:
        print("❌ Failed to load newly analyzed papers list")
        return 1

    paper_ids = newly_analyzed.get('paper_ids', [])
    timestamp = newly_analyzed.get('timestamp', 'Unknown time')
    paper_count = len(paper_ids)

    print(f"   Found {paper_count} newly analyzed papers")

    if paper_count == 0:
        print("ℹ️  No new papers analyzed, skipping notification")
        return 0

    # Load complete analysis data
    print(f"📖 Loading analysis data from: {analysis_json_path}")
    analysis_data = load_analysis_data(analysis_json_path)

    if not analysis_data:
        print("❌ Failed to load analysis data")
        return 1

    # Collect paper information
    print("📝 Collecting paper information...")
    papers = collect_papers_info(paper_ids, analysis_data)

    if not papers:
        print("❌ No paper details found")
        return 1

    # Generate LLM prompt
    print("🤖 Generating LLM summary prompt...")
    prompt = generate_llm_summary_prompt(papers)

    # Call LLM for summary
    print("🤖 Calling LLM to generate comprehensive summary...")
    llm_summary = call_llm_for_summary(prompt, api_key, base_url)

    if not llm_summary:
        print("❌ Failed to generate LLM summary")
        return 1

    print("\n" + "="*60)
    print("LLM Generated Summary:")
    print("="*60)
    print(llm_summary)
    print("="*60 + "\n")

    # Format message for DingTalk
    print("📝 Formatting message for DingTalk...")
    message = format_dingtalk_message(llm_summary, papers, timestamp)

    # Send webhook
    print(f"📤 Sending notification to DingTalk...")
    success = send_dingtalk_webhook(webhook_url, message, webhook_secret)

    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
