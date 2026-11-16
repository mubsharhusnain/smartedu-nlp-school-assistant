"""Assignment 2: BoW, TF-IDF, n-grams, Word2Vec, fastText, probes, and saved interface.

Run from repository root:
    python src/assignment2.py

Input is the exact cleaned corpus from Assignment 1:
    data/processed/educationq_cleaned.csv
Required columns: cleaned_text, category
"""
from pathlib import Path
import json, math, re, warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec, FastText

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "processed" / "educationq_cleaned.csv"
OUT = ROOT / "outputs" / "assignment2"
MODELS = ROOT / "models"
OUT.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)

                                                                               
def load_corpus():
    if not INPUT.exists():
        raise FileNotFoundError(
            f"Missing A1 cleaned corpus: {INPUT}\n"
            "First complete Assignment 1 and place educationq_cleaned.csv in data/processed/."
        )
    df = pd.read_csv(INPUT).dropna(subset=["cleaned_text"]).copy()
    df["cleaned_text"] = df["cleaned_text"].astype(str)
    df = df[df["cleaned_text"].str.strip().ne("")]
    if df.empty:
        raise ValueError("The Assignment 1 cleaned corpus is empty.")
    if "category" not in df.columns:
        df["category"] = ""
    return df

def tokens_for_training(text):
    return re.findall(r"\b[\w']+\b", str(text).lower(), flags=re.UNICODE)

def sparsity_percent(matrix):
    total = matrix.shape[0] * matrix.shape[1]
    return 100.0 * (1.0 - (matrix.nnz / total)) if total else 0.0

def safe_top_terms(row, feature_names, n=5, largest=True):
    vals = row.toarray().ravel()
    ids = np.argsort(vals)
    ids = ids[::-1] if largest else ids
    chosen = [(str(feature_names[i]), float(vals[i])) for i in ids if vals[i] > 0][:n]
    return chosen

def hand_worked_check():
    """Hand arithmetic and sklearn confirmation on 3 tiny documents."""
    toy = [
        "machine learning helps students",
        "machine learning supports teachers",
        "students study learning"
    ]
                                                                     
    vec = TfidfVectorizer(ngram_range=(1,1), use_idf=True, smooth_idf=True, norm=None)
    mat = vec.fit_transform(toy)
    terms = vec.get_feature_names_out().tolist()
    term = "machine"
    idx = terms.index(term)
    df_term = sum(term in set(tokens_for_training(d)) for d in toy)
    idf_hand = math.log((1 + len(toy)) / (1 + df_term)) + 1
    tf_hand = toy[0].split().count(term)
    hand_tfidf = tf_hand * idf_hand
    library_tfidf = float(mat[0, idx])
                                                                 
    bigram = "machine learning"
    bi = CountVectorizer(ngram_range=(2,2))
    bi_mat = bi.fit_transform(toy)
    bi_terms = bi.get_feature_names_out().tolist()
    bi_idx = bi_terms.index(bigram)
    bigram_hand = sum(len(re.findall(r"(?<!\w)machine\s+learning(?!\w)", d)) for d in toy)
    bigram_library = int(bi_mat[:, bi_idx].sum())
    return {
        "toy_documents": toy,
        "tfidf_formula": "idf(t)=ln((1+N)/(1+df(t)))+1; tfidf=raw_tf*idf (norm=None)",
        "tfidf_term": term, "N": len(toy), "document_frequency": df_term,
        "tf_in_document_1": tf_hand, "idf_hand": idf_hand,
        "tfidf_hand": hand_tfidf, "tfidf_sklearn": library_tfidf,
        "tfidf_matches_library": bool(np.isclose(hand_tfidf, library_tfidf)),
        "bigram": bigram, "bigram_hand_count": bigram_hand,
        "bigram_library_count": bigram_library,
        "bigram_matches_library": bool(bigram_hand == bigram_library)
    }

def train_and_save(df):
    docs = df["cleaned_text"].tolist()
                                                               
    bow = CountVectorizer(ngram_range=(1,1), token_pattern=r"(?u)\b\w+\b")
    X_bow = bow.fit_transform(docs)
    tfidf = TfidfVectorizer(ngram_range=(1,1), token_pattern=r"(?u)\b\w+\b")
    X_tfidf = tfidf.fit_transform(docs)
    joblib.dump(bow, MODELS/"bow_vectorizer.joblib")
    joblib.dump(tfidf, MODELS/"tfidf_vectorizer.joblib")
    joblib.dump(tfidf, MODELS/"tfidf_vectorizer.joblib")                                   

    ngram_rows=[]
    for n in (1,2,3):
        v=CountVectorizer(ngram_range=(n,n), token_pattern=r"(?u)\b\w+\b")
        m=v.fit_transform(docs)
        ngram_rows.append({"ngram":f"{n}-gram","features":int(m.shape[1]),
                           "matrix_rows":int(m.shape[0]),"matrix_columns":int(m.shape[1]),
                           "sparsity_percent":round(sparsity_percent(m),6)})
                                                      
    v_all=CountVectorizer(ngram_range=(1,3), token_pattern=r"(?u)\b\w+\b")
    m_all=v_all.fit_transform(docs)
    joblib.dump(v_all, MODELS/"ngram_1_3_vectorizer.joblib")

    feature_names=tfidf.get_feature_names_out()
    first_row=X_tfidf[0]
    high=safe_top_terms(first_row,feature_names,5,True)
    idf_order=np.argsort(tfidf.idf_)
    low=[(str(feature_names[i]),float(tfidf.idf_[i])) for i in idf_order[:5]]                                                             
    count_summary={
        "documents":int(X_bow.shape[0]),
        "bow_shape":[int(X_bow.shape[0]),int(X_bow.shape[1])],
        "bow_sparsity_percent":round(sparsity_percent(X_bow),6),
        "tfidf_shape":[int(X_tfidf.shape[0]),int(X_tfidf.shape[1])],
        "tfidf_sparsity_percent":round(sparsity_percent(X_tfidf),6),
        "first_document":docs[0],
        "high_tfidf_terms_first_document":high,
        "lowest_idf_common_terms":low,
        "ngram_growth":ngram_rows,
        "combined_1_to_3_gram_shape":[int(m_all.shape[0]),int(m_all.shape[1])],
        "combined_1_to_3_gram_sparsity_percent":round(sparsity_percent(m_all),6)
    }
    sentences=[tokens_for_training(x) for x in docs]
    sentences=[x for x in sentences if x]
                                                                        
    common=dict(vector_size=100,window=5,min_count=1,workers=1,seed=42,epochs=30)
    cbow=Word2Vec(sentences=sentences,sg=0,**common)
    skip=Word2Vec(sentences=sentences,sg=1,**common)
    fast=FastText(sentences=sentences,vector_size=100,window=5,min_count=1,
                  workers=1,seed=42,epochs=30,min_n=3,max_n=6)
    cbow.save(str(MODELS/"word2vec_cbow.model"))
    skip.save(str(MODELS/"word2vec_skipgram.model"))
    fast.save(str(MODELS/"fasttext.model"))

    vocab=set(cbow.wv.index_to_key)
    queries=["education","student","teacher","school","learning"]
    queries=[q for q in queries if q in vocab]
    if len(queries)<3:
        queries=[w for w in cbow.wv.index_to_key[:5]]
    probes={}
    for model_name,model in [("cbow",cbow),("skipgram",skip),("fasttext",fast)]:
        probes[model_name]={}
        for q in queries:
            try: probes[model_name][q]=[(w,float(s)) for w,s in model.wv.most_similar(q,topn=8)]
            except KeyError: probes[model_name][q]="OOV"
                                                                
    pair_candidates=[("student","teacher"),("school","education"),("student","banana"),("teacher","computer")]
    similarities={}
    for model_name,model in [("cbow",cbow),("skipgram",skip),("fasttext",fast)]:
        similarities[model_name]={}
        for a,b in pair_candidates:
            if a in model.wv and b in model.wv:
                similarities[model_name][f"{a} :: {b}"]=float(model.wv.similarity(a,b))
            else:
                similarities[model_name][f"{a} :: {b}"]="OOV / not in model vocabulary"
    rare_candidates=[w for w in cbow.wv.index_to_key if len(w)>=7]
    rare_word=min(rare_candidates,key=lambda w:cbow.wv.get_vecattr(w,"count")) if rare_candidates else cbow.wv.index_to_key[-1]
    typo=rare_word[:-1]+"x" if len(rare_word)>2 else rare_word+"x"
    oov_demo={
        "rare_word":rare_word,"misspelled_query":typo,
        "word2vec_has_rare":rare_word in cbow.wv,
        "word2vec_has_misspelling":typo in cbow.wv,
        "fasttext_has_rare":rare_word in fast.wv,
        "fasttext_can_vectorize_misspelling":typo in fast.wv,
        "explanation":"Word2Vec only has vectors for vocabulary entries; FastText can compose vectors from character n-grams for unseen strings, though quality is not guaranteed."
    }
    analogy={}
                                                                                  
    a,b,c="teacher","school","student"
    if all(x in cbow.wv for x in (a,b,c)):
        for model_name,model in [("cbow",cbow),("skipgram",skip),("fasttext",fast)]:
            try: analogy[model_name]=model.wv.most_similar(positive=[a,c],negative=[b],topn=5)
            except Exception as e: analogy[model_name]=f"Failed: {e}"
    else:
        analogy={"status":"Not run because one or more analogy terms are absent from the trained vocabulary.",
                 "terms":[a,b,c]}
    summary={
        "count_based":count_summary,
        "embedding_hyperparameters":{"vector_size":100,"window":5,"min_count":1,"epochs":30,
            "workers":1,"seed":42,"cbow_sg":0,"skipgram_sg":1,"fasttext_min_n":3,"fasttext_max_n":6},
        "embedding_vocabulary_sizes":{"cbow":len(cbow.wv),"skipgram":len(skip.wv),"fasttext":len(fast.wv)},
        "similar_words":probes,"cosine_similarity_pairs":similarities,
        "rare_oov_demo":oov_demo,"analogy_probe":analogy,
        "corpus_warning":"Embedding quality depends on corpus size and domain coverage; interpret neighbors cautiously."
    }
    summary["hand_worked_numerical"]=hand_worked_check()
    (OUT/"assignment2_summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False,default=str),encoding="utf-8")
    return summary
def vectorize_document(raw_document, representation="tfidf"):
    """Phase 3 interface: return a vector using a saved representation; no retraining."""
    if representation=="bow":
        model=joblib.load(MODELS/"bow_vectorizer.joblib")
        return model.transform([raw_document])
    if representation=="tfidf":
        model=joblib.load(MODELS/"tfidf_vectorizer.joblib")
        return model.transform([raw_document])
    if representation=="ngram":
        model=joblib.load(MODELS/"ngram_1_3_vectorizer.joblib")
        return model.transform([raw_document])
    if representation in ("cbow","skipgram","fasttext"):
        model_path=MODELS/({"cbow":"word2vec_cbow.model","skipgram":"word2vec_skipgram.model","fasttext":"fasttext.model"}[representation])
        model=FastText.load(str(model_path)) if representation=="fasttext" else Word2Vec.load(str(model_path))
        toks=tokens_for_training(raw_document)
        vecs=[model.wv[t] for t in toks if t in model.wv]
        return np.mean(vecs,axis=0) if vecs else np.zeros(model.vector_size,dtype=np.float32)
    raise ValueError("representation must be bow, tfidf, ngram, cbow, skipgram, or fasttext")

if __name__=="__main__":
    df=load_corpus()
    result=train_and_save(df)
    print(json.dumps(result,indent=2,ensure_ascii=False,default=str))
    print("\nSaved vectorizers and embeddings in models/; summary in outputs/assignment2/.")
    
