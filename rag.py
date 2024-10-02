import json


n2c2_json_dir = './dataset/n2c2/train'
ade_corpus_json = './dataset/ade_corpus/ade_corpus_relations.json'

def get_sent_data(vs_result):
    sentence = vs_result.page_content
    source = vs_result.metadata['source']
    file_name = ade_corpus_json

    if source != 'ade_corpus_relations.json':
        file_id = source.split('/')[-1].split('.')[0]
        file_name = f'{n2c2_json_dir}/{file_id}.json'
    
    try:
        with open(file_name, 'r') as file:
            data = json.load(file)
            
            for elm in data:
                if sentence == elm['text']:
                    return elm['entities'], elm['relations']
                    break
    
    except Exception as e:
        print(f"An error occurred while reading {file_name}: {e}")


def get_system_prompt(query, vector_store):
    results = vector_store.similarity_search(query, k=5)
    
    output = []
    for result in results:
        entities, relations = get_sent_data(result)
        formated_res = ([{
            "text": result.page_content,
            "entities": entities,
            "relations": relations } ] )
        
        output.append(json.dumps(formated_res, indent=4))
        
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


Examples:
1. Input Text: {}
   Expected Output: 
   {}

2. Input Text: {}
   Expected Output: 
   {}

3. Input Text: {}
   Expected Output: 
   {}

4. Input Text: {}
   Expected Output: 
   {}

5. Input Text: {}
   Expected Output: 
   {}
""".format(
    results[0].page_content, output[0], 
    results[1].page_content, output[1], 
    results[2].page_content, output[2],
    results[3].page_content, output[3],
    results[4].page_content, output[4] )
    
    return system_prompt


def generate_response(client, query, system_prompt):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": f'Here is the text from which you need to extract entities and relations:\n "{query}"',
            }
        ],
        model="llama3-8b-8192" )

    return chat_completion.choices[0].message.content
