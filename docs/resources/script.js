let allPapers = [];
let filteredPapers = [];
let analysisData = {};
let activeCategory = 'all';
let currentSort = 'date';

// Load and parse the JSON data
async function loadPapers() {
    try {
        // Load both web data and analysis data
        const [webResponse, analysisResponse] = await Promise.all([
            fetch('agent-arxiv-daily-web.json'),
            fetch('agent-arxiv-daily-analysis.json')
        ]);

        const webData = await webResponse.json();
        analysisData = await analysisResponse.json();

        // Parse the data structure
        allPapers = [];
        for (const [category, papers] of Object.entries(webData)) {
            for (const [paperId, paperMarkdown] of Object.entries(papers)) {
                const paper = parseMarkdownPaper(paperMarkdown, category, paperId);
                if (paper) {
                    // Add analysis data if available
                    if (analysisData[category] && analysisData[category][paperId]) {
                        paper.analysis = analysisData[category][paperId];
                    }
                    allPapers.push(paper);
                }
            }
        }

        filteredPapers = [...allPapers];
        initializeFilters();
        sortPapers();
        renderPapers();
        updateStats();
    } catch (error) {
        console.error('Error loading papers:', error);
        document.getElementById('papersContainer').innerHTML = `
            <div class="no-results">
                <h2>Error Loading Papers</h2>
                <p>Please make sure the data file is available.</p>
            </div>
        `;
    }
}

// Parse markdown format: |**date**|**title**|authors|[id](url)|code|
function parseMarkdownPaper(markdown, category, paperId) {
    const parts = markdown.split('|').filter(p => p.trim());
    if (parts.length < 4) return null;

    const date = parts[0].replace(/\*\*/g, '').trim();
    const title = parts[1].replace(/\*\*/g, '').trim();
    const authors = parts[2].trim();

    // Extract arXiv link
    const arxivMatch = parts[3].match(/\[([^\]]+)\]\(([^)]+)\)/);
    const arxivId = arxivMatch ? arxivMatch[1] : paperId;
    const arxivUrl = arxivMatch ? arxivMatch[2] : `http://arxiv.org/abs/${paperId}`;

    // Extract code link
    const codeLink = parts[4] && parts[4].trim() !== 'null' ? parts[4].trim() : null;

    return {
        id: paperId,
        date: date,
        title: title,
        authors: authors,
        category: category,
        arxivId: arxivId,
        arxivUrl: arxivUrl,
        codeUrl: codeLink
    };
}

function initializeFilters() {
    const categories = ['all', ...new Set(allPapers.map(p => p.category))];
    const filterContainer = document.getElementById('categoryFilters');

    filterContainer.innerHTML = categories.map(cat =>
        `<div class="filter-tag ${cat === 'all' ? 'active' : ''}" data-category="${cat}">
            ${cat === 'all' ? 'All Papers' : cat}
        </div>`
    ).join('');

    // Add click listeners
    filterContainer.querySelectorAll('.filter-tag').forEach(tag => {
        tag.addEventListener('click', () => {
            filterContainer.querySelectorAll('.filter-tag').forEach(t => t.classList.remove('active'));
            tag.classList.add('active');
            activeCategory = tag.dataset.category;
            filterPapers();
        });
    });

    // Sort buttons
    document.querySelectorAll('.sort-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.sort-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            currentSort = btn.dataset.sort;
            sortPapers();
            renderPapers();
        });
    });

    document.getElementById('categoriesCount').textContent = categories.length - 1;
}

function filterPapers() {
    const searchTerm = document.getElementById('searchInput').value.toLowerCase();

    filteredPapers = allPapers.filter(paper => {
        const matchesCategory = activeCategory === 'all' || paper.category === activeCategory;
        const matchesSearch = searchTerm === '' ||
            paper.title.toLowerCase().includes(searchTerm) ||
            paper.authors.toLowerCase().includes(searchTerm) ||
            paper.category.toLowerCase().includes(searchTerm);

        return matchesCategory && matchesSearch;
    });

    sortPapers();
    renderPapers();
    updateStats();
}

function sortPapers() {
    filteredPapers.sort((a, b) => {
        switch(currentSort) {
            case 'date':
                return new Date(b.date) - new Date(a.date);
            case 'date-old':
                return new Date(a.date) - new Date(b.date);
            case 'title':
                return a.title.localeCompare(b.title);
            default:
                return 0;
        }
    });
}

function renderPapers() {
    const container = document.getElementById('papersContainer');

    if (filteredPapers.length === 0) {
        container.innerHTML = `
            <div class="no-results">
                <h2>No Papers Found</h2>
                <p>Try adjusting your search or filters</p>
            </div>
        `;
        return;
    }

    container.innerHTML = `
        <div class="papers-grid">
            ${filteredPapers.map(paper => {
                // Get first affiliation if available
                const affiliation = paper.analysis?.metadata?.affiliations?.[0] || '';

                return `
                <div class="paper-card" onclick='showPaperDetails(${JSON.stringify(paper.id)})'>
                    <span class="paper-date">${paper.date}</span>
                    <div class="category-badge">${paper.category}</div>
                    <h3 class="paper-title">${paper.title}</h3>
                    <p class="paper-authors">${paper.authors}</p>
                    ${affiliation ? `<p class="paper-affiliation">${affiliation}</p>` : ''}
                    <p class="paper-id">arXiv:${paper.arxivId}</p>
                    <div class="paper-links">
                        <a href="${paper.arxivUrl}" class="paper-link link-arxiv" target="_blank" onclick="event.stopPropagation()">
                            📄 arXiv
                        </a>
                        ${paper.codeUrl && paper.codeUrl !== 'null'
                            ? `<a href="${paper.codeUrl}" class="paper-link link-code" target="_blank" onclick="event.stopPropagation()">💻 Code</a>`
                            : '<span class="paper-link link-disabled">💻 No Code</span>'}
                    </div>
                </div>
            `}).join('')}
        </div>
    `;
}

function showPaperDetails(paperId) {
    const paper = allPapers.find(p => p.id === paperId);
    if (!paper) return;

    const modal = document.getElementById('paperModal');
    const modalBody = document.getElementById('modalBody');
    const modalTitle = document.getElementById('modalTitle');
    const modalMeta = document.getElementById('modalMeta');

    // Set title and basic info
    modalTitle.textContent = paper.title;
    modalMeta.innerHTML = `
        <span>📅 ${paper.date}</span>
        <span>📑 arXiv:${paper.arxivId}</span>
        <span>🏷️ ${paper.category}</span>
    `;

    // Build modal content
    let content = '';

    if (paper.analysis) {
        const analysis = paper.analysis;
        const metadata = analysis.metadata || {};
        const analysisContent = analysis.analysis || {};

        // Authors
        if (metadata.authors && metadata.authors.length > 0) {
            content += `
                <div class="modal-section">
                    <h3 class="modal-section-title">Authors</h3>
                    <div class="authors-list">
                        ${metadata.authors.map(author =>
                            `<span class="author-tag">${author}</span>`
                        ).join('')}
                    </div>
                </div>
            `;
        }

        // Affiliations
        if (metadata.affiliations && metadata.affiliations.length > 0) {
            content += `
                <div class="modal-section">
                    <h3 class="modal-section-title">Affiliations</h3>
                    <div class="affiliations-list">
                        ${metadata.affiliations.map(aff =>
                            `<div class="affiliation-item">${aff}</div>`
                        ).join('')}
                    </div>
                </div>
            `;
        }

        // Resources
        if (metadata.resources) {
            const links = [];
            if (metadata.resources.github) links.push({ url: metadata.resources.github, label: '💻 GitHub', icon: '💻' });
            if (metadata.resources.huggingface) links.push({ url: metadata.resources.huggingface, label: '🤗 HuggingFace', icon: '🤗' });
            if (metadata.resources.project_page) links.push({ url: metadata.resources.project_page, label: '🌐 Project Page', icon: '🌐' });

            if (links.length > 0) {
                content += `
                    <div class="modal-section">
                        <h3 class="modal-section-title">Resources</h3>
                        <div class="resources-links">
                            ${links.map(link =>
                                `<a href="${link.url}" class="resource-link" target="_blank">${link.icon} ${link.label}</a>`
                            ).join('')}
                        </div>
                    </div>
                `;
            }
        }

        // Summary
        if (analysisContent.summary) {
            content += `
                <div class="modal-section">
                    <h3 class="modal-section-title">Summary</h3>
                    <div class="modal-section-content">${analysisContent.summary}</div>
                </div>
            `;
        }

        // Research Question
        if (analysisContent.research_question) {
            content += `
                <div class="modal-section">
                    <h3 class="modal-section-title">Research Question</h3>
                    <div class="modal-section-content">${analysisContent.research_question}</div>
                </div>
            `;
        }

        // Methodology
        if (analysisContent.methodology) {
            content += `
                <div class="modal-section">
                    <h3 class="modal-section-title">Methodology</h3>
                    <div class="modal-section-content">${analysisContent.methodology}</div>
                </div>
            `;
        }

        // Key Findings
        if (analysisContent.key_findings) {
            content += `
                <div class="modal-section">
                    <h3 class="modal-section-title">Key Findings</h3>
                    <div class="modal-section-content">${analysisContent.key_findings}</div>
                </div>
            `;
        }

        // Conclusions
        if (analysisContent.conclusions) {
            content += `
                <div class="modal-section">
                    <h3 class="modal-section-title">Conclusions</h3>
                    <div class="modal-section-content">${analysisContent.conclusions}</div>
                </div>
            `;
        }

        // Limitations
        if (analysisContent.limitations) {
            content += `
                <div class="modal-section">
                    <h3 class="modal-section-title">Limitations</h3>
                    <div class="modal-section-content">${analysisContent.limitations}</div>
                </div>
            `;
        }

        // Future Research
        if (analysisContent.future_research) {
            content += `
                <div class="modal-section">
                    <h3 class="modal-section-title">Future Research</h3>
                    <div class="modal-section-content">${analysisContent.future_research}</div>
                </div>
            `;
        }
    } else {
        content = `
            <div class="modal-section">
                <p>Detailed analysis not available for this paper.</p>
                <p>You can view the paper on <a href="${paper.arxivUrl}" target="_blank">arXiv</a>.</p>
            </div>
        `;
    }

    modalBody.innerHTML = content;
    modal.classList.add('active');
    document.body.style.overflow = 'hidden';
}

function closeModal() {
    const modal = document.getElementById('paperModal');
    modal.classList.remove('active');
    document.body.style.overflow = 'auto';
}

function updateStats() {
    document.getElementById('totalPapers').textContent = allPapers.length;
    document.getElementById('visiblePapers').textContent = filteredPapers.length;
}

// Search functionality
document.getElementById('searchInput').addEventListener('input', filterPapers);

// Close modal on overlay click
document.addEventListener('DOMContentLoaded', () => {
    const modal = document.getElementById('paperModal');
    modal.addEventListener('click', (e) => {
        if (e.target === modal) {
            closeModal();
        }
    });
});

// Load papers on page load
loadPapers();
