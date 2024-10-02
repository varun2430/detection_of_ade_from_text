import re
import torch
from nltk.tokenize import sent_tokenize


def predict(text, model, tokenizer):
    model.eval()
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,
        return_token_type_ids=False,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt',
        truncation=True
    )
    
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        _, preds = torch.max(logits, dim=1)

    return preds.item()

def classify_sentences(paragraph, model, tokenizer):
    sentences = sent_tokenize(paragraph)
    cleaned_sentences = [re.sub(r'\s+', ' ', sentence).strip() for sentence in sentences]

    sentences = [sentence for sentence in cleaned_sentences if len(sentence) > 5]

    classified_sentences = [(sentence, predict(sentence, model, tokenizer)) for sentence in sentences]
    return classified_sentences

