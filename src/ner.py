import spacy

from openai import OpenAI

import json
import re

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from collections import defaultdict
from tqdm import tqdm


class SpacyNER:
    def __init__(self,spacy_model):
        self.spacy_model = spacy.load(spacy_model)

    def batch_ner(self, hash_id_to_passage, max_workers):
        passage_list = list(hash_id_to_passage.values())
        batch_size = len(passage_list) // max_workers
        docs_list = self.spacy_model.pipe(passage_list,batch_size=batch_size)
        passage_hash_id_to_entities = {}
        sentence_to_entities = defaultdict(list)
        for idx,doc in enumerate(docs_list):
            passage_hash_id = list(hash_id_to_passage.keys())[idx]
            single_passage_hash_id_to_entities,single_sentence_to_entities = self.extract_entities_sentences(doc,passage_hash_id)
            passage_hash_id_to_entities.update(single_passage_hash_id_to_entities)
            # Ensure all sentence-level entities are also in passage-level entities
            for sent, ents in single_sentence_to_entities.items():
                for e in ents:
                    if e not in passage_hash_id_to_entities[passage_hash_id]:
                        passage_hash_id_to_entities[passage_hash_id].append(e)
                    if e not in sentence_to_entities[sent]:
                        sentence_to_entities[sent].append(e)
        return passage_hash_id_to_entities,sentence_to_entities
            
    def extract_entities_sentences(self, doc,passage_hash_id):
        sentence_to_entities = defaultdict(list)
        unique_entities = set()
        passage_hash_id_to_entities = {}
        # pdb.set_trace()  # 注释掉调试断点
        for ent in doc.ents:
            if ent.label_ == "ORDINAL" or ent.label_ == "CARDINAL":
                continue
            sent_text = ent.sent.text
            ent_text = ent.text
            if ent_text not in sentence_to_entities[sent_text]:
                sentence_to_entities[sent_text].append(ent_text)
            unique_entities.add(ent_text)
        passage_hash_id_to_entities[passage_hash_id] = list(unique_entities)
        return passage_hash_id_to_entities,sentence_to_entities

    def question_ner(self, question: str):
        doc = self.spacy_model(question)
        question_entities = set()
        for ent in doc.ents:
            if ent.label_ == "ORDINAL" or ent.label_ == "CARDINAL":
                continue
            question_entities.add(ent.text.lower())
        return question_entities


from transformers import AutoTokenizer, AutoModelForTokenClassification
from collections import defaultdict
import torch
import re


class IndicNER:
    def __init__(self, model_name="ai4bharat/IndicNER"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        self.model.eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()

    def _normalize_text(self, text: str) -> str:
        """Collapse all whitespace variants to a single space."""
        return re.sub(r'\s+', ' ', text).strip()

    def _clean_entity(self, entity: str) -> str:
        """Strip punctuation and whitespace that bleeds in from sentence boundaries."""
        cleaned = re.sub(r'^[\s,।\.\-\"\']+', '', entity)
        cleaned = re.sub(r'[\s,।\.\-\"\']+$', '', cleaned)
        return cleaned.strip()

    def _get_word_labels(self, sentence: str):
        """
        Tokenize a sentence and return per-word NER labels using word_ids()
        to assign only the head subword token's label to each word.
        Returns list of (word, label) tuples.
        """
        sentence = self._normalize_text(sentence)
        tok_sentence = self.tokenizer(
            sentence, return_tensors='pt', truncation=True, max_length=512
        )
        # word_ids() must be called on the non-cuda tokenizer output
        word_ids = tok_sentence.word_ids()

        if torch.cuda.is_available():
            tok_sentence = {k: v.cuda() for k, v in tok_sentence.items()}

        with torch.no_grad():
            logits = self.model(**tok_sentence).logits.argmax(-1)
            predicted_tokens_classes = [
                self.model.config.id2label[t.item()] for t in logits[0]
            ]

        predicted_labels = []
        previous_token_id = 0
        for word_index in range(len(word_ids)):
            if word_ids[word_index] is None:
                previous_token_id = word_ids[word_index]
            elif word_ids[word_index] == previous_token_id:
                previous_token_id = word_ids[word_index]
            else:
                predicted_labels.append(predicted_tokens_classes[word_index])
                previous_token_id = word_ids[word_index]

        words = sentence.split(' ')
        # guard: truncation may produce fewer labels than words
        return list(zip(words[:len(predicted_labels)], predicted_labels))

    def _extract_entities_from_word_labels(self, word_labels):
        """
        Extract entity strings from (word, label) pairs using BIO scheme.
        Consecutive B-/I- tags of the same type are merged into one span.
        Cleans each extracted entity before adding to the set.
        """
        entities = set()
        current_entity_words = []
        current_entity_type = None

        for word, label in word_labels:
            if label.startswith('B-'):
                if current_entity_words:
                    cleaned = self._clean_entity(' '.join(current_entity_words))
                    if cleaned:
                        entities.add(cleaned)
                current_entity_words = [word]
                current_entity_type = label[2:]
            elif label.startswith('I-') and current_entity_type == label[2:]:
                current_entity_words.append(word)
            else:
                if current_entity_words:
                    cleaned = self._clean_entity(' '.join(current_entity_words))
                    if cleaned:
                        entities.add(cleaned)
                current_entity_words = []
                current_entity_type = None

        # flush any trailing entity
        if current_entity_words:
            cleaned = self._clean_entity(' '.join(current_entity_words))
            if cleaned:
                entities.add(cleaned)

        return entities

    def _split_telugu_sentences(self, text: str):
        """
        Split Telugu text into sentences using appropriate delimiters.
        Telugu uses purna virama (।), period (.), question mark (?), exclamation (!),
        and sometimes English punctuation mixed with Telugu text.
        """
        # Split by Telugu purna virama, period, question mark, exclamation, and newlines
        # Keep the delimiter to maintain context
        sentences = re.split(r'([।\.!?\n]+)', text)
        
        # Reconstruct sentences with their delimiters
        result = []
        for i in range(0, len(sentences) - 1, 2):
            sent = sentences[i].strip()
            if i + 1 < len(sentences):
                delimiter = sentences[i + 1].strip()
                if sent:  # Only add non-empty sentences
                    # Combine sentence with its delimiter for better context
                    result.append(sent)
            elif sent:  # Last sentence without delimiter
                result.append(sent)
        
        # Handle case where last item is a sentence without delimiter
        if len(sentences) % 2 == 1 and sentences[-1].strip():
            result.append(sentences[-1].strip())
        
        # Filter out very short fragments (less than 3 characters) that are likely noise
        result = [s for s in result if len(s) >= 3]
        
        return result if result else [text]  # Return original text if no sentences found

    def batch_ner(self, hash_id_to_passage, max_workers=None):
        """
        Drop-in replacement for SpacyNER.batch_ner.
        Returns (passage_hash_id_to_entities, sentence_to_entities).
        """
        passage_hash_id_to_entities = {}
        sentence_to_entities = defaultdict(list)

        for passage_hash_id, passage in hash_id_to_passage.items():
            passage = self._normalize_text(passage)

            # Passage-level entities
            word_labels = self._get_word_labels(passage)
            unique_entities = self._extract_entities_from_word_labels(word_labels)
            passage_hash_id_to_entities[passage_hash_id] = list(unique_entities)

            # Sentence-level entities using Telugu-aware sentence splitting
            sentences = self._split_telugu_sentences(passage)
            
            # Track which entities have been assigned to sentences
            entities_with_sentences = set()
            
            for sent in sentences:
                sent_word_labels = self._get_word_labels(sent)
                sent_entities = self._extract_entities_from_word_labels(sent_word_labels)
                for e in sent_entities:
                    entities_with_sentences.add(e)
                    # Ensure sentence-level entities are also in passage-level entities
                    if e not in unique_entities:
                        unique_entities.add(e)
                        passage_hash_id_to_entities[passage_hash_id].append(e)
                    if e not in sentence_to_entities[sent]:
                        sentence_to_entities[sent].append(e)
            
            # For entities that weren't assigned to any sentence, assign them to the full passage
            # This ensures all entities have at least one sentence mapping
            orphan_entities = unique_entities - entities_with_sentences
            if orphan_entities:
                # If we have sentences, assign to the first one; otherwise use the full passage
                fallback_sent = sentences[0] if sentences else passage
                for e in orphan_entities:
                    if e not in sentence_to_entities[fallback_sent]:
                        sentence_to_entities[fallback_sent].append(e)

        return passage_hash_id_to_entities, sentence_to_entities

    def question_ner(self, question: str):
        """
        Drop-in replacement for SpacyNER.question_ner.
        Returns a set of lowercased entity strings.
        """
        word_labels = self._get_word_labels(question)
        entities = self._extract_entities_from_word_labels(word_labels)
        return {e.lower() for e in entities}