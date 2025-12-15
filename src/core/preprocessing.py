import re
import pandas as pd
from pymorphy3 import MorphAnalyzer
from stop_words import get_stop_words
from tqdm import tqdm
tqdm.pandas()


morph = MorphAnalyzer()
rus_stopwords = set(get_stop_words('ru'))


def preprocess_text(df: pd.DataFrame, col: str) -> None:
    df[col] = df[col].astype(str).progress_apply(lambda text: 
        ' '.join([
            morph.parse(word)[0].normal_form 
            for word in re.sub(r'[^a-zA-Zа-яА-ЯёЁ0-9\s.,-]', '', text.lower()).split() 
            if word and word not in rus_stopwords
        ])
    )