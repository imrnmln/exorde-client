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

def tag(documents: List[str], lab_configuration: Dict) -> List[Analysis]:
    models = lab_configuration["models"]
    
    for doc in documents:
        assert isinstance(doc, str)

    # Create a Frame from the documents list
    tmp = dt.Frame(Translation=documents)

    assert not tmp["Translation"].isna().any1()[0]  # Check if any value is NA
    assert tmp.nrows > 0

    logging.info("Starting Tagging Batch pipeline...")
    model = models['sentence_transformer']
    
    # Convert string to list for embedding
    tmp["Embedding"] = tmp["Translation"].to_list()[0].apply(
        lambda x: Embedding(list(model.encode(x).astype(float)))
    )

    zs_pipe = models['zs_pipe']
    classification_labels = list(lab_configuration["labeldict"].keys())
    tmp["Classification"] = tmp["Translation"].to_list()[0].apply(
        lambda x: Classification(
            label=zs_pipe(x, candidate_labels=classification_labels)['labels'][0],
            score=round(zs_pipe(x, candidate_labels=classification_labels)['scores'][0], 4)
        )
    )

    text_classification_models = ["Emotion", "Irony", "TextType"]
    for col_name in text_classification_models:
        pipe = models[col_name]
        tmp[col_name] = tmp["Translation"].to_list()[0].apply(
            lambda x: globals()[col_name](**{y["label"]: float(y["score"]) for y in pipe(x)[0]})
        )

    tokenizer = models['bert_tokenizer']
    tmp["Embedded"] = tmp["Translation"].to_list()[0].apply(
        lambda x: np.array(
            tokenizer.encode_plus(
                x,
                add_special_tokens=True,
                max_length=512,
                truncation=True,
                padding="max_length",
                return_attention_mask=False,
                return_tensors="tf",
            )["input_ids"][0]
        ).reshape(1, -1)
    )

    sentiment_analyzer = models['sentiment_analyzer']
    fdb_pipe = models['fdb_pipe']
    gdb_pipe = models['gdb_pipe']

    def vader_sentiment(text):
        return Sentiment(round(sentiment_analyzer.polarity_scores(text)["compound"], 2))
    
    def fin_vader_sentiment(text):
        return Sentiment(round(finvader(text, use_sentibignomics=True, use_henry=True, indicator='compound'), 2))

    def fdb_sentiment(text):
        prediction = fdb_pipe(text)
        fdb_sentiment_dict = {e["label"]: round(e["score"], 3) for e in prediction[0]}
        return Sentiment(round(fdb_sentiment_dict["positive"] - fdb_sentiment_dict["negative"], 3))

    def gdb_sentiment(text):
        prediction = gdb_pipe(text)
        gen_distilbert_sent = {e["label"]: round(e["score"], 3) for e in prediction[0]}
        return Sentiment(round(gen_distilbert_sent["positive"] - gen_distilbert_sent["negative"], 3))
    
    def compounded_financial_sentiment(text):
        fin_vader_sent = fin_vader_sentiment(text)
        fin_distil_score = fdb_sentiment(text)
        return Sentiment(round((0.70 * fin_distil_score.value + 0.30 * fin_vader_sent.value), 2))
        
    def compounded_sentiment(text):
        gen_distilbert_sentiment = gdb_sentiment(text)
        vader_sent = vader_sentiment(text)
        compounded_fin_sentiment = compounded_financial_sentiment(text)
        if abs(compounded_fin_sentiment.value) >= 0.6:
            return Sentiment(round((0.30 * gen_distilbert_sentiment.value + 0.10 * vader_sent.value + 0.60 * compounded_fin_sentiment.value), 2))
        elif abs(compounded_fin_sentiment.value) >= 0.4:
            return Sentiment(round((0.40 * gen_distilbert_sentiment.value + 0.20 * vader_sent.value + 0.40 * compounded_fin_sentiment.value), 2))
        elif abs(compounded_fin_sentiment.value) >= 0.1:
            return Sentiment(round((0.60 * gen_distilbert_sentiment.value + 0.25 * vader_sent.value + 0.15 * compounded_fin_sentiment.value), 2))
        else:
            return Sentiment(round((0.60 * gen_distilbert_sentiment.value + 0.40 * vader_sent.value), 2))

    # Applying sentiment functions
    tmp["Sentiment"] = tmp["Translation"].to_list()[0].apply(compounded_sentiment)
    tmp["FinancialSentiment"] = tmp["Translation"].to_list()[0].apply(compounded_financial_sentiment)

    # Remove the 'Embedded' column as it's no longer needed
    tmp = tmp[:, [c for c in tmp.names if c != "Embedded"]]

    # Convert to dictionary for further processing
    tmp_dict = tmp.to_dict()

    _out = []
    for i in range(len(tmp_dict["Translation"])):
        sentiment = tmp_dict["Sentiment"][i]
        embedding = tmp_dict["Embedding"][i]
        classification = tmp_dict["Classification"][i]
        gender = Gender(male=0.5, female=0.5)  # Assuming mock values as before
        text_type = tmp_dict["TextType"][i]
        emotion = tmp_dict["Emotion"][i]
        irony = tmp_dict["Irony"][i]
        age = Age(below_twenty=0.0, twenty_thirty=0.0, thirty_forty=0.0, forty_more=0.0)  # Mock data, adjust if needed
        language_score = LanguageScore(1.0)  # Mock data

        analysis = Analysis(
            language_score=language_score,
            sentiment=sentiment,
            classification=classification,
            embedding=embedding,
            gender=gender,
            text_type=text_type,
            emotion=emotion,
            irony=irony,
            age=age,
        )

        _out.append(analysis)
    return _out
