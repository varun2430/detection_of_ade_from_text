import httpx
import json
from groq import Groq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import BertTokenizer, BertForSequenceClassification
from all_functions import classify_sentences
from rag import get_system_prompt, generate_response

http_client = httpx.Client(verify=False)
client = Groq(api_key = '',  http_client=http_client)

model_name = './all-mpnet-base-v2'
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}

embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs )

vector_store = FAISS.load_local("sent_index", embeddings, allow_dangerous_deserialization=True)

def main(paragraph):
    model_save_path = './ade_classifier_model-pytorch-default-v1'
    tokenizer = BertTokenizer.from_pretrained(model_save_path)
    model = BertForSequenceClassification.from_pretrained(model_save_path)

    classified_sentences = classify_sentences(paragraph, model, tokenizer)
    ade_related_sentences = [sentence for sentence, label in classified_sentences if label == 1]

    all_drugs = []
    all_ades = []
    all_relations = []
    
    output_str=''
    for sen in ade_related_sentences:
        system_prompt = get_system_prompt(sen, vector_store)
        res = generate_response(client, sen, system_prompt)
        res = res[res.index('[') : res.rfind(']')+1]
        
        try:
            json_data = json.loads(res)
        except Exception as e:
            print(f"Error: {e}")

        drugs = []
        ades = []
        relations = []

        for elm in json_data:
            for entity in elm['entities']:
                if entity['label'] == 'Drug':
                    drugs.append(entity['str'])
                elif entity['label'] == 'ADE':
                    ades.append(entity['str'])

            for relation in elm['relations']:
                if relation['label'] == 'causing':
                    relations.append(list(relation.values()))

        all_drugs.extend(drugs)
        all_relations.extend(relations)
        
        res_str = sen + '\n'

        res_str = res_str + 'Drugs:\n'
        for drug in drugs:
            res_str = res_str + '    ' + drug + '\n'

        res_str = res_str + '\nADE:\n'
        for ade in ades:
            res_str = res_str + '    ' + ade + '\n'

        res_str = res_str + '\nRelations (Drug-ADE):\n'
        for relation in relations:
            res_str = res_str + '    ' + relation[1] + '    ->    ' + relation[0] + '\n'

        output_str= output_str + "\n\n\n\n" + res_str
    
    return output_str.strip(), all_drugs, all_relations

