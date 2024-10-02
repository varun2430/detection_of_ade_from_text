import json
import httpx
from groq import Groq
from rag import generate_response


if __name__ == '__main__':
    http_client = httpx.Client(verify=False)
    client = Groq(api_key = '',  http_client=http_client)

    system_prompt = """You are an advanced named entity recognition model designed to extract drug names, symptoms and adverse drug effects from text. Additionally, identify and categorize the relationships between these entities into two types: 'curing' (where a drug is taken to alleviate a symptom) and 'causing' (where a drug causes an adverse drug effect).
Respond with the output in the following JSON format only and do not give any additional text or explanation:
[
    {{
        "text": "<input_text>",
        "entities": [
            {{
                "label": "Drug",
                "str": "<drug_name>"
            }},
            {{
                "label": "Symptom",
                "str": "<symptom_name>"
            }},
            {{
                "label": "ADE",
                "str": "<adverse_effect_name>"
            }}
        ],
        "relations": [
            {{
                "Symptom": "<symptom_name>",
                "Drug": "<drug_name>",
                "label": "curing"
            }},
            {{
                "ADE": "<adverse_effect_name>",
                "Drug": "<drug_name>",
                "label": "causing"
            }}
        ]
    }}
]
"""


    test_files = ['./dataset/n2c2/test/100035.json', './dataset/n2c2/test/103761.json']

    correct_entities = 0
    total_predicted_entities = 0
    total_actual_entities = 0

    correct_relations = 0
    total_predicted_relations = 0
    total_actual_relations = 0

    correct_drug_entities = 0
    total_predicted_drug_entities = 0
    total_actual_drug_entities = 0

    correct_symptom_entities = 0
    total_predicted_symptom_entities = 0
    total_actual_symptom_entities = 0

    correct_ade_entities = 0
    total_predicted_ade_entities = 0
    total_actual_ade_entities = 0


    for filename in test_files:
        try:
            with open(filename, 'r') as json_file:
                data = json.load(json_file)
                
                for elm in data:
                    if elm['entities'] == []:
                        continue
                        
                    query = elm['text']
                    entities = elm['entities']
                    relations = elm['relations']
                                        
                    actual_entities = {entity['str'] for entity in entities}
                    actual_relations = {tuple(relation.values()) for relation in relations}
                    
                    actual_drug_entities = {entity['str'] for entity in entities if entity['label'] == 'Drug'}
                    actual_symptom_entities = {entity['str'] for entity in entities if entity['label'] == 'Symptom'}
                    actual_ade_entities = {entity['str'] for entity in entities if entity['label'] == 'ADE'}
                    
                    res = generate_response(client, query, system_prompt)
                    res = res[res.index('[') : res.rfind(']')+1]
                    
                    try:
                        predicted_data = json.loads(res)
                        
                        predicted_entities = {entity['str'] for entity in predicted_data[0]['entities']}
                        predicted_relations = {tuple(relation.values()) for relation in predicted_data[0]['relations']}
                        
                        predicted_drug_entities = {entity['str'] for entity in predicted_data[0]['entities'] if entity['label'] == 'Drug'}
                        predicted_symptom_entities = {entity['str'] for entity in predicted_data[0]['entities'] if entity['label'] == 'Symptom'}
                        predicted_ade_entities = {entity['str'] for entity in predicted_data[0]['entities'] if entity['label'] == 'ADE'}
                        
                        
                        correct_entities += len(predicted_entities.intersection(actual_entities))
                        total_predicted_entities += len(predicted_entities)
                        total_actual_entities += len(actual_entities)

                        correct_relations += len(predicted_relations.intersection(actual_relations))
                        total_predicted_relations += len(predicted_relations)
                        total_actual_relations += len(actual_relations)
                        
                        correct_drug_entities += len(predicted_drug_entities.intersection(actual_drug_entities))
                        total_predicted_drug_entities += len(predicted_drug_entities)
                        total_actual_drug_entities += len(actual_drug_entities)
                        
                        correct_symptom_entities += len(predicted_symptom_entities.intersection(actual_symptom_entities))
                        total_predicted_symptom_entities += len(predicted_symptom_entities)
                        total_actual_symptom_entities += len(actual_symptom_entities)
                        
                        correct_ade_entities += len(predicted_ade_entities.intersection(actual_ade_entities))
                        total_predicted_ade_entities += len(predicted_ade_entities)
                        total_actual_ade_entities += len(actual_ade_entities)
                    except Exception as e:
                        print(f"Error: {e}")
                                    
        except Exception as e:
            print(f"Error reading {filename}: {e}")


    print('\n\n\n\nEntities:')
    print(f'correct_entities: {correct_entities}')
    print(f'total_predicted_entities: {total_predicted_entities}')
    print(f'total_actual_entities: {total_actual_entities}')

    precision = correct_entities / total_predicted_entities if total_predicted_entities > 0 else 0
    recall = correct_entities / total_actual_entities if total_actual_entities > 0 else 0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

    print(f'precision: {precision}')
    print(f'recall: {recall}')
    print(f'f1: {f1}', '\n\n')



    print('Relations:')
    print(f'correct_relations: {correct_relations}')
    print(f'total_predicted_relations: {total_predicted_relations}')
    print(f'total_actual_relations: {total_actual_relations}')

    precision = correct_relations / total_predicted_relations if total_predicted_relations > 0 else 0
    recall = correct_relations / total_actual_relations if total_actual_relations > 0 else 0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

    print(f'precision: {precision}')
    print(f'recall: {recall}')
    print(f'f1: {f1}', '\n\n')



    print('Drugs:')
    print(f'correct_drug_entities: {correct_drug_entities}')
    print(f'total_predicted_drug_entities: {total_predicted_drug_entities}')
    print(f'total_actual_drug_entities: {total_actual_drug_entities}')

    precision = correct_drug_entities / total_predicted_drug_entities if total_predicted_drug_entities > 0 else 0
    recall = correct_drug_entities / total_actual_drug_entities if total_actual_drug_entities > 0 else 0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

    print(f'precision: {precision}')
    print(f'recall: {recall}')
    print(f'f1: {f1}', '\n\n')



    print('Symptom:')
    print(f'correct_symptom_entities: {correct_symptom_entities}')
    print(f'total_predicted_symptom_entities: {total_predicted_symptom_entities}')
    print(f'total_actual_symptom_entities: {total_actual_symptom_entities}')

    precision = correct_symptom_entities / total_predicted_symptom_entities if total_predicted_symptom_entities > 0 else 0
    recall = correct_symptom_entities / total_actual_symptom_entities if total_actual_symptom_entities > 0 else 0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

    print(f'precision: {precision}')
    print(f'recall: {recall}')
    print(f'f1: {f1}', '\n\n')



    print('ADE:')
    print(f'correct_ade_entities: {correct_ade_entities}')
    print(f'total_predicted_ade_entities: {total_predicted_ade_entities}')
    print(f'total_actual_ade_entities: {total_actual_ade_entities}')

    precision = correct_ade_entities / total_predicted_ade_entities if total_predicted_ade_entities > 0 else 0
    recall = correct_ade_entities / total_actual_ade_entities if total_actual_ade_entities > 0 else 0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

    print(f'precision: {precision}')
    print(f'recall: {recall}')
    print(f'f1: {f1}')