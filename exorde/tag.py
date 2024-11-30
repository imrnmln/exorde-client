import json
import modin.pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from finvader import finvader
from huggingface_hub import hf_hub_download
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import tensorflow as tf
import swifter
import logging
from exorde.models import (
    Classification,
    LanguageScore,
    Sentiment,
    Embedding,
    TextType,   
    Emotion,
    Irony,
    Age,
    Gender,
    Analysis,
)

logging.basicConfig(level=logging.INFO)

def initialize_models(device):
    logging.info("[TAGGING] Initializing models to be pre-ready for batch processing:")
    models = {}
    
    logging.info("[TAGGING] Loading model: MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33")
    models['zs_pipe'] = pipeline(
        "zero-shot-classification",
        model="MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33",
        device=device
    )
    logging.info("[TAGGING] Loading model: sentence-transformers/all-MiniLM-L6-v2")
    models['sentence_transformer'] = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    
    text_classification_models = [
        ("Emotion", "SamLowe/roberta-base-go_emotions"),
        ("Irony", "cardiffnlp/twitter-roberta-base-irony"),
        ("TextType", "marieke93/MiniLM-evidence-types"),
    ]
    for col_name, model_name in text_classification_models:
        logging.info(f"[TAGGING] Loading model: {model_name}")
        models[col_name] = pipeline(
            "text-classification",
            model=model_name,
            top_k=None,
            device=device,
            max_length=512,
            padding=True,
        )
    
    logging.info("[TAGGING] Loading model: bert-large-uncased")
    models['bert_tokenizer'] = AutoTokenizer.from_pretrained("bert-large-uncased")
    logging.info("[TAGGING] Loading model: vaderSentiment")
    models['sentiment_analyzer'] = SentimentIntensityAnalyzer()
    try:
        emoji_lexicon = hf_hub_download(
            repo_id="ExordeLabs/SentimentDetection",
            filename="emoji_unic_lexicon.json",
        )
        loughran_dict = hf_hub_download(
            repo_id="ExordeLabs/SentimentDetection", filename="loughran_dict.json"
        )
        logging.info("[TAGGING] Loading Loughran_dict & unic_emoji_dict for sentiment_analyzer.")
        with open(emoji_lexicon) as f:
            unic_emoji_dict = json.load(f)
        with open(loughran_dict) as f:
            Loughran_dict = json.load(f)
        models['sentiment_analyzer'].lexicon.update(Loughran_dict)
        models['sentiment_analyzer'].lexicon.update(unic_emoji_dict)
    except Exception as e:
        logging.info("[TAGGING] Error loading Loughran_dict & unic_emoji_dict for sentiment_analyzer. Doing without.")
    
    logging.info("[TAGGING] Loading model: mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")
    models['fdb_tokenizer'] = AutoTokenizer.from_pretrained("mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")
    logging.info("[TAGGING] Loading model: mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")
    models['fdb_model'] = AutoModelForSequenceClassification.from_pretrained("mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")
    models['fdb_pipe'] = pipeline(
        "text-classification",
        model=models['fdb_model'],
        tokenizer=models['fdb_tokenizer'],
        top_k=None, 
        max_length=512,
        padding=True,
    )
    
    logging.info("[TAGGING] Loading model: lxyuan/distilbert-base-multilingual-cased-sentiments-student")
    models['gdb_tokenizer'] = AutoTokenizer.from_pretrained("lxyuan/distilbert-base-multilingual-cased-sentiments-student")
    logging.info("[TAGGING] Loading model: lxyuan/distilbert-base-multilingual-cased-sentiments-student")
    models['gdb_model'] = AutoModelForSequenceClassification.from_pretrained("lxyuan/distilbert-base-multilingual-cased-sentiments-student")
    models['gdb_pipe'] = pipeline(
        "text-classification",
        model=models['gdb_model'],
        tokenizer=models['gdb_tokenizer'],
        top_k=None, 
        max_length=512,
        padding=True,
    )
    logging.info("[TAGGING] Models loaded successfully.")
    
    return models

def tag(documents: list[str], lab_configuration):
    models = lab_configuration["models"]
    
    # Convert documents to DataFrame
    tmp = pd.DataFrame({"Translation": documents})
    
    # Assertions
    assert tmp["Translation"].notna().all()
    assert len(tmp["Translation"]) > 0    

    logging.info("Starting Tagging Batch pipeline...")

    # Load models once
    model = models['sentence_transformer']
    zs_pipe = models['zs_pipe']
    classification_labels = list(lab_configuration["labeldict"].keys())
    text_classification_models = ["Emotion", "Irony", "TextType"]
    tokenizer = models['bert_tokenizer']
    sentiment_analyzer = models['sentiment_analyzer']
    fdb_pipe = models['fdb_pipe']
    gdb_pipe = models['gdb_pipe']

    # Apply operations using Modin's parallelization
    tmp["Embedding"] = tmp["Translation"].apply(lambda x: list(model.encode(x).astype(float)))
    tmp["Classification"] = tmp["Translation"].apply(lambda x: zs_pipe(x, candidate_labels=classification_labels))
    tmp["Embedded"] = tmp["Translation"].apply(lambda x: np.array(
        tokenizer.encode_plus(
            x,
            add_special_tokens=True,
            max_length=512,
            truncation=True,
            padding="max_length",
            return_attention_mask=False,
            return_tensors="tf",
        )["input_ids"][0]
    ).reshape(1, -1))

    for col_name in text_classification_models:
        pipe = models[col_name]
        tmp[col_name] = tmp["Translation"].apply(lambda x: [(y["label"], float(y["score"])) for y in pipe(x)[0]])

    # Sentiment Analysis Functions
    def vader_sentiment(text):
        return round(sentiment_analyzer.polarity_scores(text)["compound"], 2)

    def fin_vader_sentiment(text):
        return round(finvader(text, use_sentibignomics=True, use_henry=True, indicator='compound'), 2)

    def fdb_sentiment(text):
        prediction = fdb_pipe(text)
        return round({e["label"]: round(e["score"], 3) for e in prediction[0]}.get('positive', 0) - 
                     {e["label"]: round(e["score"], 3) for e in prediction[0]}.get('negative', 0), 3)

    def gdb_sentiment(text):
        prediction = gdb_pipe(text)
        return round({e["label"]: round(e["score"], 3) for e in prediction[0]}.get('positive', 0) - 
                     {e["label"]: round(e["score"], 3) for e in prediction[0]}.get('negative', 0), 3)

    def compounded_financial_sentiment(text):
        fin_vader_sent = fin_vader_sentiment(text)
        fin_distil_score = fdb_sentiment(text)
        return round(0.70 * fin_distil_score + 0.30 * fin_vader_sent, 2)

    def compounded_sentiment(text):
        gen_distilbert_sentiment = gdb_sentiment(text)
        vader_sent = vader_sentiment(text)
        compounded_fin_sentiment = compounded_financial_sentiment(text)
        
        if abs(compounded_fin_sentiment) >= 0.6:
            return round((0.30 * gen_distilbert_sentiment + 0.10 * vader_sent + 0.60 * compounded_fin_sentiment), 2)
        elif abs(compounded_fin_sentiment) >= 0.4:
            return round((0.40 * gen_distilbert_sentiment + 0.20 * vader_sent + 0.40 * compounded_fin_sentiment), 2)
        elif abs(compounded_fin_sentiment) >= 0.1:
            return round((0.60 * gen_distilbert_sentiment + 0.25 * vader_sent + 0.15 * compounded_fin_sentiment), 2)
        else:
            return round((0.60 * gen_distilbert_sentiment + 0.40 * vader_sent), 2)

    # Apply sentiment analysis
    tmp["Sentiment"] = tmp["Translation"].apply(compounded_sentiment)
    tmp["FinancialSentiment"] = tmp["Translation"].apply(compounded_financial_sentiment)

    # Remove temporary columns after use
    del tmp["Embedded"]

    # Format output
    return [
        Analysis(
            classification=Classification(label=item["Classification"]["labels"][0], 
                                          score=round(item["Classification"]["scores"][0], 4)),
            language_score=LanguageScore(1.0),  # Default value
            sentiment=Sentiment(item["Sentiment"]),
            embedding=Embedding(item["Embedding"]),
            gender=Gender(male=0.5, female=0.5),  # Mock gender
            text_type=TextType(**{
                k: round(v, 4) for k, v in {t[0]: t[1] for t in item["TextType"]}.items()
            } | {k: 0.0 for k in ['assumption', 'anecdote', 'none', 'definition', 'testimony', 'other', 'study'] 
                 if k not in {t[0] for t in item["TextType"]}}),
            emotion=Emotion(**{
                k: round(v, 4) for k, v in {e[0]: e[1] for e in item["Emotion"]}.items()
            } | {k: 0.0 for k in ['love', 'admiration', 'joy', 'approval', 'caring', 'excitement', 'gratitude', 'desire', 
                                  'anger', 'optimism', 'disapproval', 'grief', 'annoyance', 'pride', 'curiosity', 'neutral', 
                                  'disgust', 'disappointment', 'realization', 'fear', 'relief', 'confusion', 'remorse', 
                                  'embarrassment', 'surprise', 'sadness', 'nervousness'] 
                 if k not in {e[0] for e in item["Emotion"]}}),
            irony=Irony(**{
                k: round(v, 4) for k, v in {i[0]: i[1] for i in item["Irony"]}.items()
            } | {k: 0.0 for k in ['irony', 'non_irony'] if k not in {i[0] for i in item["Irony"]}}),
            age=Age(below_twenty=0.0, twenty_thirty=0.0, thirty_forty=0.0, forty_more=0.0),  # Untrained model
            financial_sentiment=Sentiment(item["FinancialSentiment"])  # Added Financial Sentiment
        ) for _, item in tmp.iterrows()
    ]
