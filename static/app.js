class SummaryApp {
    constructor() {
        this.initializeEventListeners();
    }

    initializeEventListeners() {
        const form = document.getElementById('summary-form');
        const textInput = document.getElementById('text-input');
        
        form.addEventListener('submit', (e) => this.handleSubmit(e));
        textInput.addEventListener('input', () => this.updateCharCount());
        
        // Enter key support (Ctrl+Enter pour soumettre)
        textInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
                this.handleSubmit(e);
            }
        });
    }

    updateCharCount() {
        const textInput = document.getElementById('text-input');
        const charCount = document.getElementById('char-count');
        charCount.textContent = textInput.value.length;
        
        // Warning visuel si approche limite
        if (textInput.value.length > 3500) {
            charCount.style.color = '#DC3545';
            charCount.style.fontWeight = 'bold';
        } else {
            charCount.style.color = '';
            charCount.style.fontWeight = '';
        }
    }

    async handleSubmit(event) {
        event.preventDefault();
        
        const textInput = document.getElementById('text-input');
        const modeSelect = document.getElementById('mode-select');
        const submitBtn = document.getElementById('submit-btn');
        const resultsDiv = document.getElementById('results');
        
        const text = textInput.value.trim();
        const optimized = modeSelect.value === 'true';
        
        // Validation
        if (!text) {
            this.showError('Veuillez saisir un texte à résumer.');
            return;
        }
        
        if (text.length > 4000) {
            this.showError('Le texte dépasse 4000 caractères.');
            return;
        }
        
        this.setLoadingState(true);
        resultsDiv.style.display = 'none';
        
        try {
            const response = await this.generateSummary(text, optimized);
            
            if (response.error) {
                throw new Error(response.error);
            }
            
            this.displayResults(response);
            resultsDiv.style.display = 'block';
            
        } catch (error) {
            this.showError('Erreur lors de la génération : ' + error.message);
        } finally {
            this.setLoadingState(false);
        }
    }

    async generateSummary(text, optimized) {
        const response = await fetch('/summarize', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                text: text,
                optimized: optimized
            })
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || `Erreur serveur (${response.status})`);
        }

        return await response.json();
    }

    displayResults(data) {
        const summaryContent = document.getElementById('summary-content');
        const wordCount = document.getElementById('word-count');
        const energyValue = document.getElementById('energy-value');
        const latencyValue = document.getElementById('latency-value');
        
        summaryContent.textContent = data.summary;
        wordCount.textContent = data.summary.split(/\s+/).length;
        energyValue.textContent = data.energy_wh.toFixed(6);
        latencyValue.textContent = data.latency_ms.toFixed(2);
        
        // Scroll doux vers les résultats
        document.getElementById('results').scrollIntoView({ 
            behavior: 'smooth',
            block: 'start'
        });
    }

    setLoadingState(loading) {
        const submitBtn = document.getElementById('submit-btn');
        const btnText = submitBtn.querySelector('.btn-text');
        const btnLoading = submitBtn.querySelector('.btn-loading');
        
        if (loading) {
            btnText.style.display = 'none';
            btnLoading.style.display = 'inline';
            submitBtn.disabled = true;
        } else {
            btnText.style.display = 'inline';
            btnLoading.style.display = 'none';
            submitBtn.disabled = false;
        }
    }

    showError(message) {
        // Simple alert pour rester sobre
        alert('❌ ' + message);
    }
}

// Initialisation de l'application
document.addEventListener('DOMContentLoaded', () => {
    new SummaryApp();
});
