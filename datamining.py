import re
import PyPDF2
from collections import Counter
import numpy as np
from datetime import datetime
import spacy
import matplotlib
matplotlib.use('Agg')  # Set the backend before importing pyplot
import matplotlib.pyplot as plt
from io import BytesIO
import seaborn as sns
from wordcloud import WordCloud
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict

class PDFDataMiner:
    def __init__(self):
        # Add thread-safe backend configuration
        plt.switch_backend('Agg')
        # Initialize spaCy for NLP tasks
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            import os
            os.system("python -m spacy download en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")

    def extract_metadata(self, pdf_path):
        """Extract PDF metadata"""
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            metadata = reader.metadata
            info = {
                'Title': metadata.get('/Title', 'N/A'),
                'Author': metadata.get('/Author', 'N/A'),
                'Creation Date': metadata.get('/CreationDate', 'N/A'),
                'Number of Pages': len(reader.pages)
            }
            return info

    def extract_statistics(self, text):
        """Extract numerical statistics and p-values"""
        # Find p-values
        p_values = re.findall(r'p\s*[<=>]\s*0\.\d+', text)
        # Find percentages
        percentages = re.findall(r'\d+\.?\d*\s*%', text)
        # Find numerical values with units
        measurements = re.findall(r'\d+\.?\d*\s*(?:kg|m|cm|mm|g|mg|ml|km|hz|Hz)', text)
        
        return {
            'p_values': p_values[:10],  # Limit to top 10
            'percentages': percentages[:10],
            'measurements': measurements[:10]
        }

    def extract_citations(self, text):
        """Extract citations in various formats"""
        # APA style citations
        apa_citations = re.findall(r'\([A-Za-z]+(?:\s+et\s+al\.)?(?:,\s+\d{4})\)', text)
        # Harvard style citations
        harvard_citations = re.findall(r'\([A-Za-z]+\s+and\s+[A-Za-z]+,\s+\d{4}\)', text)
        
        all_citations = apa_citations + harvard_citations
        return list(set(all_citations))[:15]  # Return unique citations, limited to 15

    def extract_key_terms(self, text):
        """Extract and count important terms using NLP"""
        doc = self.nlp(text)
        
        # Extract noun phrases and named entities
        terms = []
        terms.extend([chunk.text.lower() for chunk in doc.noun_chunks])
        terms.extend([ent.text.lower() for ent in doc.ents])
        
        # Count frequencies
        term_freq = Counter(terms)
        return dict(term_freq.most_common(20))

    def generate_word_frequency_plot(self, term_frequencies):
        """Generate a bar plot of word frequencies"""
        plt.figure(figsize=(10, 6))
        terms = list(term_frequencies.keys())[:10]  # Top 10 terms
        frequencies = list(term_frequencies.values())[:10]
        
        plt.bar(range(len(terms)), frequencies)
        plt.xticks(range(len(terms)), terms, rotation=45, ha='right')
        plt.xlabel('Terms')
        plt.ylabel('Frequency')
        plt.title('Top 10 Key Terms Frequency')
        
        # Save plot to bytes buffer
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        return buf

    def generate_sentiment_plot(self, text):
        """Generate sentiment analysis visualization"""
        # Split text into more meaningful chunks (sentences instead of paragraphs)
        sentences = [sent.strip() for sent in text.split('.') if len(sent.strip()) > 20]
        sentiments = []
        
        # Analyze sentiment for each meaningful sentence
        for sentence in sentences[:30]:  # Analyze first 30 sentences for better visualization
            try:
                blob = TextBlob(sentence)
                sentiment = blob.sentiment.polarity
                if sentiment != 0:  # Only include non-zero sentiments
                    sentiments.append(sentiment)
            except:
                continue

        if not sentiments:  # If no sentiments were found, add dummy data
            sentiments = [0]

        plt.figure(figsize=(10, 4))
        plt.plot(range(len(sentiments)), sentiments, marker='o')
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.3)
        plt.fill_between(range(len(sentiments)), sentiments, alpha=0.2)
        plt.xlabel('Sentence Number')
        plt.ylabel('Sentiment (Negative → Positive)')
        plt.title('Sentiment Flow Throughout Document')
        plt.grid(True, alpha=0.3)

        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        return buf

    def generate_wordcloud(self, text):
        """Generate word cloud visualization"""
        wordcloud = WordCloud(
            width=800, 
            height=400,
            background_color='white',
            max_words=100
        ).generate(text)

        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Word Cloud of Key Terms')

        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        return buf

    def generate_text_structure(self, text):
        """Generate comprehensive text structure visualizations"""
        # Split text into sections and paragraphs
        sections = []
        paragraphs = []
        sentences = []
        
        # Try to split by common section indicators
        potential_sections = re.split(r'\n\s*(?:[IVX]+\.|[0-9]+\.|Introduction|Methods|Results|Discussion|Conclusion)', text)
        
        for section in potential_sections:
            if len(section.strip()) > 100:
                sections.append(section.strip())
                # Split section into paragraphs
                section_paragraphs = [p.strip() for p in section.split('\n\n') if len(p.strip()) > 50]
                paragraphs.extend(section_paragraphs)
                # Split paragraphs into sentences
                for para in section_paragraphs:
                    sent_list = [s.strip() for s in re.split(r'[.!?]+', para) if len(s.strip()) > 20]
                    sentences.extend(sent_list)
        
        if not sections:
            sections = [s.strip() for s in text.split('\n\n') if len(s.strip()) > 100]
        
        # Calculate various metrics
        section_metrics = []
        for section in sections:
            words = section.split()
            if len(words) > 0:
                unique_words = len(set(words))
                section_metrics.append({
                    'length': len(words),
                    'avg_word_length': sum(len(word) for word in words) / len(words),
                    'lexical_density': unique_words / len(words),
                    'sentence_count': len([s for s in re.split(r'[.!?]+', section) if len(s.strip()) > 0])
                })
        
        if not section_metrics:
            section_metrics = [{'length': 0, 'avg_word_length': 0, 'lexical_density': 0, 'sentence_count': 0}]
        
        # Create multiple visualizations
        plt.figure(figsize=(15, 10))
        
        # 1. Section Length Distribution
        plt.subplot(2, 2, 1)
        lengths = [m['length'] for m in section_metrics]
        plt.plot(range(len(lengths)), lengths, marker='o', color='blue', alpha=0.6)
        plt.axhline(y=np.mean(lengths), color='r', linestyle='--', label='Average')
        plt.xlabel('Section Number')
        plt.ylabel('Words per Section')
        plt.title('Section Length Distribution')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # 2. Word Complexity
        plt.subplot(2, 2, 2)
        word_lengths = [m['avg_word_length'] for m in section_metrics]
        plt.plot(range(len(word_lengths)), word_lengths, marker='s', color='green', alpha=0.6)
        plt.axhline(y=np.mean(word_lengths), color='r', linestyle='--', label='Average')
        plt.xlabel('Section Number')
        plt.ylabel('Average Word Length')
        plt.title('Word Complexity by Section')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # 3. Lexical Density
        plt.subplot(2, 2, 3)
        densities = [m['lexical_density'] for m in section_metrics]
        plt.bar(range(len(densities)), densities, alpha=0.6, color='purple')
        plt.axhline(y=np.mean(densities), color='r', linestyle='--', label='Average')
        plt.xlabel('Section Number')
        plt.ylabel('Lexical Density')
        plt.title('Lexical Richness by Section')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # 4. Sentence Distribution
        plt.subplot(2, 2, 4)
        sentence_counts = [m['sentence_count'] for m in section_metrics]
        plt.bar(range(len(sentence_counts)), sentence_counts, alpha=0.6, color='orange')
        plt.axhline(y=np.mean(sentence_counts), color='r', linestyle='--', label='Average')
        plt.xlabel('Section Number')
        plt.ylabel('Number of Sentences')
        plt.title('Sentence Distribution by Section')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Adjust layout
        plt.tight_layout()
        
        # Save first set of plots
        buf1 = BytesIO()
        plt.savefig(buf1, format='png', bbox_inches='tight', dpi=100)
        plt.close()
        
        # Create second figure for additional metrics
        plt.figure(figsize=(15, 5))
        
        # 5. Paragraph Length Distribution
        plt.subplot(1, 2, 1)
        para_lengths = [len(p.split()) for p in paragraphs]
        plt.hist(para_lengths, bins=20, alpha=0.6, color='teal')
        plt.axvline(x=np.mean(para_lengths), color='r', linestyle='--', label='Average')
        plt.xlabel('Words per Paragraph')
        plt.ylabel('Frequency')
        plt.title('Paragraph Length Distribution')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # 6. Sentence Length Distribution
        plt.subplot(1, 2, 2)
        sent_lengths = [len(s.split()) for s in sentences]
        plt.hist(sent_lengths, bins=20, alpha=0.6, color='brown')
        plt.axvline(x=np.mean(sent_lengths), color='r', linestyle='--', label='Average')
        plt.xlabel('Words per Sentence')
        plt.ylabel('Frequency')
        plt.title('Sentence Length Distribution')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Adjust layout
        plt.tight_layout()
        
        # Save second set of plots
        buf2 = BytesIO()
        plt.savefig(buf2, format='png', bbox_inches='tight', dpi=100)
        plt.close()
        
        return buf1, buf2

    def generate_readability_metrics(self, text):
        """Generate improved readability analysis visualization"""
        # Split into smaller chunks for more granular analysis
        chunks = []
        words = text.split()
        chunk_size = 100  # Analyze every 100 words
        
        for i in range(0, len(words), chunk_size):
            chunk = ' '.join(words[i:i + chunk_size])
            if len(chunk.strip()) > 0:
                chunks.append(chunk)
        
        metrics = []
        for chunk in chunks:
            sentences = [s.strip() for s in re.split(r'[.!?]+', chunk) if len(s.strip()) > 0]
            if not sentences:
                continue
            
            words = chunk.split()
            syllables = sum(self._count_syllables(word) for word in words)
            
            if len(sentences) > 0 and len(words) > 0:
                # Calculate multiple readability metrics
                flesch_score = 206.835 - 1.015 * (len(words)/len(sentences)) - 84.6 * (syllables/len(words))
                gunning_fog = 0.4 * ((len(words)/len(sentences)) + 100 * (syllables/len(words)))
                metrics.append({
                    'flesch_score': flesch_score,
                    'words_per_sentence': len(words)/len(sentences),
                    'syllables_per_word': syllables/len(words),
                    'gunning_fog': gunning_fog
                })
        
        if not metrics:
            return None
        
        plt.figure(figsize=(15, 10))
        
        # 1. Flesch Reading Ease Score Flow
        plt.subplot(2, 2, 1)
        scores = [m['flesch_score'] for m in metrics]
        plt.plot(scores, marker='o', color='blue', alpha=0.6)
        plt.axhline(y=np.mean(scores), color='r', linestyle='--', label=f'Average: {np.mean(scores):.1f}')
        plt.xlabel('Document Progress (100-word chunks)')
        plt.ylabel('Flesch Reading Ease')
        plt.title('Reading Ease Flow\n(Higher = Easier to Read)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # 2. Sentence Complexity Trend
        plt.subplot(2, 2, 2)
        wps = [m['words_per_sentence'] for m in metrics]
        plt.plot(wps, marker='s', color='green', alpha=0.6)
        plt.axhline(y=np.mean(wps), color='r', linestyle='--', label=f'Average: {np.mean(wps):.1f}')
        plt.xlabel('Document Progress (100-word chunks)')
        plt.ylabel('Words per Sentence')
        plt.title('Sentence Complexity Trend')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # 3. Word Complexity Distribution
        plt.subplot(2, 2, 3)
        spw = [m['syllables_per_word'] for m in metrics]
        plt.hist(spw, bins=20, color='purple', alpha=0.6)
        plt.axvline(x=np.mean(spw), color='r', linestyle='--', label=f'Average: {np.mean(spw):.2f}')
        plt.xlabel('Syllables per Word')
        plt.ylabel('Frequency')
        plt.title('Word Complexity Distribution')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # 4. Gunning Fog Index
        plt.subplot(2, 2, 4)
        fog = [m['gunning_fog'] for m in metrics]
        plt.plot(fog, marker='D', color='orange', alpha=0.6)
        plt.axhline(y=np.mean(fog), color='r', linestyle='--', label=f'Average: {np.mean(fog):.1f}')
        plt.xlabel('Document Progress (100-word chunks)')
        plt.ylabel('Gunning Fog Index')
        plt.title('Text Complexity\n(Years of Education Needed)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        plt.close()
        return buf

    def generate_content_patterns(self, text):
        """Generate content pattern analysis visualization"""
        # Extract entities and their categories
        doc = self.nlp(text)
        entity_categories = defaultdict(list)
        for ent in doc.ents:
            entity_categories[ent.label_].append(ent.text)
        
        # Count entities by category
        category_counts = {k: len(set(v)) for k, v in entity_categories.items()}
        
        plt.figure(figsize=(15, 5))
        
        # Entity distribution
        plt.subplot(1, 2, 1)
        categories = list(category_counts.keys())
        counts = list(category_counts.values())
        plt.bar(range(len(categories)), counts, color='teal', alpha=0.6)
        plt.xticks(range(len(categories)), categories, rotation=45, ha='right')
        plt.xlabel('Entity Category')
        plt.ylabel('Unique Entities')
        plt.title('Distribution of Named Entities')
        plt.grid(True, alpha=0.3)
        
        # Content density heatmap
        plt.subplot(1, 2, 2)
        paragraphs = text.split('\n\n')
        densities = []
        for i, para in enumerate(paragraphs[:20]):  # First 20 paragraphs
            doc = self.nlp(para)
            density = {
                'Technical Terms': len([t for t in doc if t.pos_ in ['NOUN', 'PROPN']]),
                'Actions': len([t for t in doc if t.pos_ == 'VERB']),
                'Descriptions': len([t for t in doc if t.pos_ in ['ADJ', 'ADV']])
            }
            densities.append(density)
        
        density_matrix = np.array([[d[key] for key in ['Technical Terms', 'Actions', 'Descriptions']] 
                                 for d in densities])
        
        sns.heatmap(density_matrix.T, 
                    xticklabels=range(1, len(densities) + 1),
                    yticklabels=['Technical Terms', 'Actions', 'Descriptions'],
                    cmap='YlOrRd', annot=True, fmt='d')
        plt.xlabel('Paragraph Number')
        plt.title('Content Density Analysis')
        
        plt.tight_layout()
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        plt.close()
        return buf

    def _count_syllables(self, word):
        """Helper method to count syllables in a word"""
        word = word.lower()
        count = 0
        vowels = 'aeiouy'
        if word[0] in vowels:
            count += 1
        for index in range(1, len(word)):
            if word[index] in vowels and word[index-1] not in vowels:
                count += 1
        if word.endswith('e'):
            count -= 1
        if count == 0:
            count += 1
        return count

    def analyze_pdf(self, pdf_path):
        """Main method to analyze PDF and return all findings"""
        # Extract text
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text()

        # Perform all analyses
        metadata = self.extract_metadata(pdf_path)
        statistics = self.extract_statistics(text)
        citations = self.extract_citations(text)
        key_terms = self.extract_key_terms(text)
        
        # Generate visualization
        term_freq_plot = self.generate_word_frequency_plot(key_terms)
        sentiment_plot = self.generate_sentiment_plot(text)
        wordcloud_plot = self.generate_wordcloud(text)
        text_structure_plot1, text_structure_plot2 = self.generate_text_structure(text)
        readability_plot = self.generate_readability_metrics(text)
        content_plot = self.generate_content_patterns(text)

        return {
            'metadata': metadata,
            'statistics': statistics,
            'citations': citations,
            'key_terms': key_terms,
            'term_freq_plot': term_freq_plot,
            'sentiment_plot': sentiment_plot,
            'wordcloud_plot': wordcloud_plot,
            'text_structure_plot1': text_structure_plot1,
            'text_structure_plot2': text_structure_plot2,
            'readability_plot': readability_plot,
            'content_plot': content_plot
        }