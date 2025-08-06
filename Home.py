# App Imports
import streamlit as st
import pandas as pd
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Dict, Any, Set, Tuple
import pickle
import re


# Functions

@st.cache_resource
def load_models():
    # Model A
    ner_pipeline = pipeline("ner", model = "./model5ver2_pubmedbert_ner_model", 
                            tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"))
    # Model B
    model_b = AutoModelForSequenceClassification.from_pretrained(
        './pubmedbert_drug_classifier_results',
        id2label={0:'generic_name', 1: 'brand_name'},
    )
    clf_pipeline = pipeline("text-classification", model=model_b, top_k=None,
                            tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"))
    # Load English words
    with open('english_words.pkl', 'rb') as file:
        english_words = pickle.load(file)
    
    return {'ner':ner_pipeline, 
            'classifier':clf_pipeline,
            'english_words':english_words
            }


def preprocess_and_extract_flavors(strings: List[str], flavor_words: set):
    """Removes flavor words from strings and extracts them."""
    pattern = re.compile(
        r'\s*[\(\[]?\b(' + '|'.join(re.escape(f) for f in flavor_words) + r')\b[\)\]]?\s*',
        flags=re.IGNORECASE
    )
    cleaned_strings = []
    extracted_flavors = []
    for s in strings:
        found = pattern.findall(s)
        extracted_flavors.append([f.lower() for f in found])
        cleaned_s = pattern.sub(' ', s).strip()
        cleaned_s = re.sub('nan', ' ', cleaned_s, flags=re.IGNORECASE).strip() # remove "nan"
        cleaned_strings.append(cleaned_s)
    return cleaned_strings, extracted_flavors


def preprocess_strings_case_insensitive(string_list: list[str]) -> list[str]:
    """
    Processes a list of strings by splitting each string by whitespace and
    removing case-insensitive duplicate words, while preserving the case of
    the first occurrence.

    This implementation is efficient and works across all Python versions.

    Args:
        string_list: A list of strings to preprocess.

    Returns:
        A new list of strings with case-insensitive duplicates removed.
    """
    processed_list = []
    for text in string_list:
        words = text.split()
        seen_lower = set()
        unique_words = []
        for word in words:
            # Convert word to lowercase for the check.
            lower_word = word.lower()
            # If the lowercase version hasn't been seen, it's a new unique word.
            if lower_word not in seen_lower:
                # Add the lowercase version to the set to track it.
                seen_lower.add(lower_word)
                # Add the original, cased word to the result list.
                unique_words.append(word)
        processed_list.append(" ".join(unique_words))
    return processed_list


def get_word_level_predictions(pipeline, texts):
    """
    Runs an NER pipeline on a list of texts and converts subword predictions
    to word-level predictions using the "first subword wins" strategy.

    Args:
        pipeline: A Hugging Face NER pipeline object.
        texts (list): A list of strings to process.

    Returns:
        list: A list of lists, where each inner list contains word-level
              prediction dictionaries for a text.
    """
    raw_predictions = pipeline(texts)

    all_word_predictions = []
    for text, predictions in zip(texts, raw_predictions):
        word_predictions = []
        # Keep track of word indices that have already been assigned a label
        processed_word_indices = set()

        for entity in predictions:
            current_word = text[entity['start']:entity['end']]
            word_start_char_index = text.rfind(' ', 0, entity['start']) + 1

            if word_start_char_index not in processed_word_indices:
                word_end_char_index = text.find(' ', entity['start'])
                if word_end_char_index == -1:
                    word_end_char_index = len(text)
                
                full_word = text[word_start_char_index:word_end_char_index]

                if not entity['word'].startswith('##'):
                    # This is the start of a new word.
                    # We will now find all subsequent subwords for this word.
                    full_word_entity = {
                        'label': entity['entity'],
                        'score': entity['score'],
                        'start': entity['start'],
                        'end': entity['end']
                    }
                    
                    # Look ahead for subwords of this word
                    next_idx = predictions.index(entity) + 1
                    while next_idx < len(predictions) and predictions[next_idx]['word'].startswith('##'):
                        full_word_entity['end'] = predictions[next_idx]['end']
                        # Optionally, average the scores
                        full_word_entity['score'] = (full_word_entity['score'] + predictions[next_idx]['score']) / 2
                        next_idx += 1
                    
                    full_word_entity['text'] = text[full_word_entity['start']:full_word_entity['end']]
                    word_predictions.append(full_word_entity)

        all_word_predictions.append(word_predictions)
    
    return all_word_predictions


def merge_ner_entities(entities):
    """
    Merges contiguous or adjacent entities from a Hugging Face NER pipeline output.

    Args:
        entities (list): A list of dictionaries, where each dictionary represents
                         an entity prediction from a model. Expected keys are
                         'label', 'score', 'start', 'end', and 'text'.
                         The labels should follow the BIO scheme (e.g., 'B-PER', 'I-PER').

    Returns:
        list: A new list of merged entity dictionaries. Each dictionary has
              'label', 'score' (averaged), 'start', 'end', and 'text'.
    """
    if not entities:
        return []

    merged_entities = []
    
    for entity in entities:
        # The B/I prefix is removed to get the base entity type (e.g., 'dosage_strength')
        # This handles cases like 'B-PER', 'I-LOC', etc.
        current_label = entity['label'].split('-')[-1]
        
        # Check if this entity can be merged with the previous one
        if (merged_entities and 
            merged_entities[-1]['label'] == current_label and 
            entity['label'].startswith('I-') and
            merged_entities[-1]['end'] == entity['start']):
            
            # Merge with the previous entity
            last_entity = merged_entities[-1]
            last_entity['end'] = entity['end']
            last_entity['text'] += entity['text']
            # Store scores to average them later
            last_entity['scores'].append(entity['score'])
            last_entity['score'] = np.mean(last_entity['scores'])

        else:
            # Start a new entity
            # The B/I prefix is removed for the final label
            new_entity = {
                'label': current_label,
                'score': entity['score'],
                'start': entity['start'],
                'end': entity['end'],
                'text': entity['text'],
                # Store scores in a list for potential future merges
                'scores': [entity['score']]
            }
            merged_entities.append(new_entity)

    # Clean up the final list by removing the temporary 'scores' key
    for entity in merged_entities:
        del entity['scores']

    return merged_entities


def clean_ner_predictions(pipeline, list_of_entries):
    """
    Cleans the raw output from the NER model.
    Example: a real function might merge sub-word tokens.
    """

    results = get_word_level_predictions(pipeline, list_of_entries)
    output = []

    for result in results:
        merged = merge_ner_entities(result)
        label = []
        for j in merged:
            j_label = 'dosage_pack' if j['label'] == 'packaging_form' else j['label']
        output.append(merged)

    return output


def process_and_update_names(df: pd.DataFrame, candidate_indices: List, model_b_classifier_pipeline) -> pd.DataFrame:
    """
    Efficiently re-classifies names and updates scores based on new rules.

    This function identifies rows with missing scores, runs the model in a single
    batch on all unique names, and then applies separate, vectorized logic for
    rows with one name versus rows with two names.

    Args:
        df: The input DataFrame. It must contain the columns 'generic_name',
            'brand_name', 'score_new_generic_name', and 'score_new_brand_name'.
        candidate_indices: List of all candidates for split-point scoring.
        model_b_classifier_pipeline: The classifier model pipeline.

    Returns:
        The DataFrame with 'generic_name_checked', 'brand_name_checked',
        'score_new_generic_name', and 'score_new_brand_name' updated according to the rules.
    """
    
    ### 1. PREPARATION & BATCH PREDICTION ###

    # Define the mask for rows that need processing.
    # This targets rows where at least one of the new scores is not yet filled.
    mask_base = ~df.index.isin(candidate_indices)

    if not mask_base.any():
        print("No rows need processing. Skipping.")
        return df

    # Collect all unique, non-null strings from the original name columns for the targeted rows.
    # This is the set of all texts we need the model to analyze.
    texts_to_process = list(pd.unique(df.loc[mask_base, ['generic_name', 'brand_name']].values.ravel('K')))
    texts_to_process = [text for text in texts_to_process if pd.notna(text)]

    if not texts_to_process:
        print("No valid text found in rows to update. Skipping.")
        return df

    print(f"Processing {len(texts_to_process)} unique names in a single batch...")

    # Run the expensive model call only ONCE.
    batch_results = model_b_classifier_pipeline(texts_to_process)

    # Create a lookup dictionary that stores the BEST classification and its score for each text.
    # This is more useful than keeping all scores.
    classification_lookup: Dict[str, Tuple[str, float]] = {}
    for text, result_list in zip(texts_to_process, batch_results):
        if result_list and isinstance(result_list, list):
            # Find the dictionary with the highest score
            best_pred = max(result_list, key=lambda item: item['score'])
            classification_lookup[text] = (best_pred['label'], best_pred['score'])
        else:
            classification_lookup[text] = (np.nan, np.nan) # Handle empty results


    df[['generic_name_checked', 'brand_name_checked']] = df[['generic_name_checked', 'brand_name_checked']].astype('object')
    
    ### 2. LOGIC FOR SINGLE-NAME ROWS ###

    # Create a sub-mask for rows that have exactly one name (generic OR brand, but not both).
    gn_present = df['generic_name'].notna()
    bn_present = df['brand_name'].notna()
    mask_single = mask_base & (gn_present ^ bn_present) # XOR is perfect for this

    if mask_single.any():
        print("Applying logic to single-name rows...")
        # Coalesce the two name columns into one, containing the single name for each row.
        single_names = df.loc[mask_single, 'generic_name'].fillna(df.loc[mask_single, 'brand_name'])

        # Map the names to our lookup to get their predicted labels and scores.
        pred_labels = single_names.map(lambda x: classification_lookup.get(x, (np.nan, np.nan))[0])
        pred_scores = single_names.map(lambda x: classification_lookup.get(x, (np.nan, np.nan))[1])

        # Use np.where to place the names and scores in the correct columns based on the predicted label.
        # This is a fully vectorized operation, which is very fast.
        is_generic = (pred_labels == 'generic_name')
        df.loc[mask_single, 'generic_name_checked'] = np.where(is_generic, single_names, np.nan)
        df.loc[mask_single, 'score_new_generic_name'] = np.where(is_generic, pred_scores, np.nan)

        df.loc[mask_single, 'brand_name_checked'] = np.where(~is_generic, single_names, np.nan)
        df.loc[mask_single, 'score_new_brand_name'] = np.where(~is_generic, pred_scores, np.nan)

    ### 3. LOGIC FOR DUAL-NAME ROWS ###

    # Create a sub-mask for rows that have BOTH a generic and brand name.
    mask_dual = mask_base & gn_present & bn_present
    
    if mask_dual.any():
        print("Applying logic to dual-name rows...")
        # For this more complex, conditional logic, using .apply() on the subset of rows is the
        # clearest and most maintainable approach. It's still efficient because the expensive
        # model call is already done, and this only runs on a small slice of the data.
        
        def process_dual_name_row(row):
            gn, bn = row['generic_name'], row['brand_name']
            
            # Get pre-computed predictions from our lookup
            gn_label, gn_score = classification_lookup.get(gn, (np.nan, np.nan))
            bn_label, bn_score = classification_lookup.get(bn, (np.nan, np.nan))

            # Initialize outputs
            res = {
                'generic_name_checked': np.nan, 'brand_name_checked': np.nan,
                'score_new_generic_name': np.nan, 'score_new_brand_name': np.nan
            }

            # Case 1: Both classified as 'generic_name' -> Concatenate
            if gn_label == 'generic_name' and bn_label == 'generic_name':
                res['generic_name_checked'] = f"{gn} {bn}"
                res['score_new_generic_name'] = max(gn_score, bn_score) # Use the higher confidence score
            # Case 2: Both classified as 'brand_name' -> Concatenate
            elif gn_label == 'brand_name' and bn_label == 'brand_name':
                res['brand_name_checked'] = f"{gn} {bn}"
                res['score_new_brand_name'] = max(gn_score, bn_score)
            # Case 3: All other combinations -> Assign to respective classified slots
            else:
                if gn_label == 'generic_name':
                    res['generic_name_checked'] = gn
                    res['score_new_generic_name'] = gn_score
                elif gn_label == 'brand_name':
                    res['brand_name_checked'] = gn
                    res['score_new_brand_name'] = gn_score
                
                if bn_label == 'generic_name':
                    res['generic_name_checked'] = bn
                    res['score_new_generic_name'] = bn_score
                elif bn_label == 'brand_name':
                    res['brand_name_checked'] = bn
                    res['score_new_brand_name'] = bn_score

            return pd.Series(res)

        # Apply the function and update the DataFrame for the dual-name rows.
        dual_results = df.loc[mask_dual].apply(process_dual_name_row, axis=1)
        df.update(dual_results)

    return df


def process_drug_data(
    input_list: List,
    model_a_ner_pipeline,
    model_b_classifier_pipeline,
    english_words: Set[str],
    # split_decision_threshold: float = 1.2,
    # column_name: str = 0, # Preserved as per your function signature
    confidence_threshold: float = 0.95
) -> pd.DataFrame:
    """
    Processes a list of drug strings to extract, validate, and refine entities.
    This version is modified for performance and feature enhancement based on the provided structure.
    """

    # 1. PREPROCESSING: Extract flavors BEFORE running Model A.
    print('Preprocessing data...')
    
    with open('flavors.pkl', 'rb') as f:
        flavor_words = pickle.load(f)

    input_list_no_duplicates = preprocess_strings_case_insensitive(input_list)
    processed_strings, extracted_flavors = preprocess_and_extract_flavors(input_list_no_duplicates, flavor_words)
    
    df = pd.DataFrame({
        'input_string': input_list,
        'processed_string': processed_strings # This will be passed to Model A
    })
    
    # 2. Generate entity predictions with Model A (using the processed strings)
    print('Generating entity predictions with Model A')
    # This assumes your clean_ner_predictions function takes the pipeline and list, and returns predictions
    # If clean_ner_predictions is a separate step, you would do:
    # raw_predictions = model_a_ner_pipeline(df['processed_string'].tolist())
    # cleaned_predictions = clean_ner_predictions(raw_predictions)
    # For now, we follow your provided calling signature:
    cleaned_predictions = clean_ner_predictions(model_a_ner_pipeline, df['processed_string'].tolist()) # Simplified for clarity
    df['cleaned_ner'] = [p for p in cleaned_predictions]

    # Ensure all required columns exist, including the new 'flavor' column
    all_labels = [
        "generic_name", "brand_name", "dosage_strength", "dosage_form",
        "dosage_pack", "volume", "total_individual_units", "flavor"
    ]

    # 3. Create new columns for each label (your original code)
    def extract_entities(ner_list: List[Dict]) -> Dict[str, str]:
        entities = {}
        scores = {}
        for entity in ner_list:
            # Using .get() for safety against missing keys
            label = entity.get('entity_group', entity.get('label'))
            text = entity.get('word', entity.get('text'))
            if label and text:
                if label in entities:
                    entities[label] += " " + text
                else:
                    entities[label] = text

                if label not in scores:
                    scores[label] = [entity.get('score',0)]
                else:
                    scores[label].append(entity.get('score',0))

        # Generate scores for all entities
        for label in all_labels:
            if label in scores:
                entities[f'score_orig_{label}'] = sum(scores[label]) / len(scores[label])
            else:
                entities[f'score_orig_{label}'] = np.nan

        return entities

    # def is_confident(ner_list: List[Dict]) -> bool:
    #     if not ner_list:
    #         return True # Or False, depending on desired behavior for no entities
    #     return all(entity.get('score', 0) > confidence_threshold for entity in ner_list)

    print('Creating columns from predicted entities')
    entity_df = pd.json_normalize(df['cleaned_ner'].apply(extract_entities))
    df = df.join(entity_df)
    # df['is_confident'] = df['cleaned_ner'].apply(is_confident)


    df['flavor'] = ["; ".join(f) if f else np.nan for f in extracted_flavors] # Add extracted flavors
    
    for label in all_labels:
        if label not in df.columns:
            df[label] = np.nan

    # 4. Flag names found in the English dictionary (your original code)
    print('Flagging generic and brand names that are in the English dictionary')
    for label in ["generic_name", "brand_name"]:
        flag_col = f"{label}_in_english_dictionary"
        if label in df.columns:
            df[flag_col] = df[label].apply(
                lambda x: x.lower() in english_words if isinstance(x, str) else False
            )
        else:
            df[flag_col] = False

    # 5. Refine 'generic_name' and 'brand_name' with BATCHED Model B
    print('Applying model B refinement logic')
    texts_to_classify = set()
    candidate_indices = []

    for idx, row in df.iterrows():
        gn, bn = row.get('generic_name'), row.get('brand_name')
        candidate_text = None
        if pd.notna(gn) and pd.isna(bn) and " " in gn:
            candidate_text = gn
        elif pd.notna(bn) and pd.isna(gn) and " " in bn:
            candidate_text = bn
        elif pd.notna(gn) and pd.notna(bn):
            candidate_text = f"{gn} {bn}"

        if candidate_text:
            candidate_indices.append(idx)
            texts_to_classify.add(candidate_text)
            words = candidate_text.split()
            if len(words) > 1:
                for i in range(1, len(words)):
                    texts_to_classify.add(" ".join(words[:i]))
                    texts_to_classify.add(" ".join(words[i:]))

    # Run Model B only if there are candidates
    if texts_to_classify:
        # Single batched call to Model B
        results = model_b_classifier_pipeline(list(texts_to_classify))
        scores_lookup = {
            text: {p['label']: p['score'] for p in res}
            for text, res in zip(list(texts_to_classify), results)
        }

    # Initialize refined columns with original values
    df['generic_name_checked'] = np.nan
    df['brand_name_checked'] = np.nan
    df['score_new_generic_name'] = np.nan
    df['score_new_brand_name'] = np.nan

    df[['generic_name_checked', 'brand_name_checked']] = df[['generic_name_checked', 'brand_name_checked']].astype('object')

    # Apply refinement logic using the pre-computed scores

    
    for idx in candidate_indices:
        row = df.loc[idx]
        gn, bn = row.get('generic_name'), row.get('brand_name')
        
        # Reconstruct the candidate_text for this row
        candidate_text = None
        if pd.notna(gn) and pd.isna(bn) and " " in gn: candidate_text = gn
        elif pd.notna(bn) and pd.isna(gn) and " " in bn: candidate_text = bn
        elif pd.notna(gn) and pd.notna(bn): candidate_text = f"{gn} {bn}"
        
        if not candidate_text: continue

        words = candidate_text.split()
        best_split = {'score': 0.0, 'gn': None, 'bn': None}
        no_split_scores = scores_lookup.get(candidate_text)
        no_split_score_max = max(no_split_scores.values())

        best_generic_score, best_brand_score = 0, 0
        for i in range(1, len(words)):
            part1, part2 = " ".join(words[:i]), " ".join(words[i:])
            scores1 = scores_lookup.get(part1, {})
            scores2 = scores_lookup.get(part2, {})

            p_gn1, p_bn1 = scores1.get('generic_name', 0), scores1.get('brand_name', 0)
            p_gn2, p_bn2 = scores2.get('generic_name', 0), scores2.get('brand_name', 0)

            score_a = p_gn1 * p_bn2
            if score_a > best_split['score']: 
                best_split.update({'score': score_a, 'gn': part1, 'bn': part2})
                best_generic_score, best_brand_score = p_gn1, p_bn2

            score_b = p_bn1 * p_gn2
            if score_b > best_split['score']: 
                best_split.update({'score': score_b, 'gn': part2, 'bn': part1})
                best_generic_score, best_brand_score = p_gn2, p_bn1

        # Decision Making
        # if best_split['score'] > split_decision_threshold * no_split_score_max:
        if best_generic_score > confidence_threshold and best_brand_score > confidence_threshold:
            df.loc[idx, 'generic_name_checked'] = best_split['gn']
            df.loc[idx, 'brand_name_checked'] = best_split['bn']

            df.loc[idx, 'score_new_generic_name'] = best_generic_score
            df.loc[idx, 'score_new_brand_name'] = best_brand_score
        else:
            if no_split_scores.get('generic_name') > no_split_scores.get('brand_name'):
                df.loc[idx, 'generic_name_checked'] = candidate_text
                df.loc[idx, 'brand_name_checked'] = np.nan
                df.loc[idx, 'score_new_generic_name'] = no_split_scores.get('generic_name')
            else:
                df.loc[idx, 'generic_name_checked'] = np.nan
                df.loc[idx, 'brand_name_checked'] = candidate_text
                df.loc[idx, 'score_new_brand_name'] = no_split_scores.get('brand_name')

    # Use model B for the rest of strings that are not candidates.
    print('Processing non-candidate strings using model B...')
    df = process_and_update_names(df, candidate_indices, model_b_classifier_pipeline)

    # Final cleanup of unnecessary columns
    df.drop(columns=['cleaned_ner', 'processed_string'], inplace=True, errors='ignore')
  
    return df


# --- Streamlit App UI (remains mostly the same) ---

st.title("💊 AC Health Pharma Parser")


with st.spinner("Loading AI models... This may take a moment on first run."):
    model = load_models()
st.success("✅ Models loaded successfully!")

st.write("---")
input_method = st.radio(
    "Choose your input method:", ("Enter Text", "Upload a File"),
    horizontal=True, label_visibility="collapsed"
)

if input_method == "Enter Text":
    st.subheader("Enter Text to Process")
    text_input = st.text_area(
        "Enter one drug description per line:",
        "Paracetamol 500mg Tablet (Biogesic)\nAmoxicillin Trihydrate 250mg Cap",
        height=150, label_visibility="collapsed"
    )
    if st.button("Process Text", use_container_width=True, type="primary"):
        if text_input:
            input_data = [line.strip() for line in text_input.split('\n') if line.strip()]
            with st.spinner("Analyzing text with..."):
                df_output = process_drug_data(input_data, model['ner'], model['classifier'], model['english_words'], confidence_threshold=0.90)
            st.dataframe(df_output)
        else:
            st.warning("Please enter some text to process.")

else: # Upload a File
    st.subheader("Upload a CSV or Excel File")
    uploaded_file = st.file_uploader(
        "The first column of the file should contain the text to be processed.",
        type=['csv', 'xlsx'], label_visibility="collapsed"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
            input_col = df.columns[0]
            st.info(f"Processing column: **`{input_col}`**")
            input_data = df[input_col].dropna().astype(str).tolist()

            with st.spinner(f"Analyzing {len(input_data)} rows..."):
                # Call the new main function
                df_output = process_drug_data(input_data, model['ner'], model['classifier'], model['english_words'], confidence_threshold=0.90)
            st.dataframe(df_output)

            @st.cache_data
            def convert_df_to_csv(df_to_convert):
                return df_to_convert.to_csv(index=False)
            csv_output = convert_df_to_csv(df_output)
            st.download_button(
               label="📥 Download results as CSV",
               data=csv_output,
               file_name='processed_drug_entities.csv',
               mime='text/csv', use_container_width=True
            )
        except Exception as e:
            st.error(f"An error occurred while processing the file: {e}")
