import os
import pickle
from tqdm import tqdm
from collections import defaultdict
from contextlib import redirect_stdout
from Knowledge import Knowledge
from Tools import Tools
from DirectoriesUtil import Dicrectories
import numpy as np

knowledge_directory = Dicrectories.knowledge
max_id = 39999 

target_similarity=defaultdict(list)
clause_weight_threshold = 0
number_of_examples = 2000
accumulation = 25
clause_drop_p = 0.0
factor = 80
clauses = int(factor*20/(1.0 - clause_drop_p))
T = factor*40
s = 5.0
epochs = 25

knowledge = Knowledge(
    clause_weight_threshold, 
    number_of_examples, 
    accumulation, 
    clause_drop_p, 
    factor, 
    T, 
    s, 
    epochs)

def preprocess_text(text):
    return text
vectorizer_X = Tools.read_pickle_data("vectorizer_X.pickle")
number_of_features = vectorizer_X.get_feature_names_out().shape[0]
X_train = Tools.read_pickle_data("X.pickle")

all_ids = set(range(max_id+1))
existing_ids = set()

# loop through all files in folder
for filename in os.listdir(knowledge_directory):
    # check if file matches pattern
    if filename.endswith('.pkl') and filename[:-4].isdigit():
        existing_ids.add(int(filename[:-4]))

# print missing ids
missing_ids = all_ids - existing_ids
target_words=[]
output_active = np.empty(len(missing_ids), dtype=np.uint32)
i = 0
for id in sorted(missing_ids):
    word = vectorizer_X.get_feature_names_out()[id]
    output_active[i] = id
    target_words.append(i)
    # print(id)
    i = i + 1
    
print("Loading missing IDs")
print("")
print("Epochs: %d" % epochs)
print("Example: %d" % number_of_examples)
print("Target words: %d" % len(target_words))
print("Accumulation: %d" % accumulation)
print("No of features: %d" % number_of_features)
output_active_list = output_active
total_training_time = 0

words_progress_bar = tqdm(total=len(output_active), desc="Running Words")
for tw in output_active_list:
    knowledge_filepath = os.path.join(knowledge_directory , str(tw) + '.pkl')
    if os.path.exists(knowledge_filepath):
        print("\nTW file exists: %s" % vectorizer_X.get_feature_names_out()[tw])
        with open(knowledge_filepath, 'rb') as f:
            target_word_clauses = pickle.load(f)
        training_time = 0
    else:
        print("\nTW run: %s" % vectorizer_X.get_feature_names_out()[tw])
        training_time, target_word_clauses = knowledge.generate(X_train, tw)
        
    total_training_time = total_training_time + training_time
    clauses_progress_bar = tqdm(total=len(target_word_clauses), desc="Running Clauses")
    for clause in target_word_clauses:
        # weight = clause[0]
        related_literals = clause[1]
        # feature_progress_bar = tqdm(total=len(related_literals), desc="Running Features")
        for literal in related_literals:
            knowledge_filepath = os.path.join(knowledge_directory , str(literal) + '.pkl')
            if os.path.exists(knowledge_filepath):
                pass
                # print("Feature file exists: %s" % vectorizer_X.get_feature_names_out()[literal])
            else:
                print("Feature run: %s" % vectorizer_X.get_feature_names_out()[literal])
                training_time, inner_target_word_clauses = knowledge.generate(X_train, literal)
                total_training_time = total_training_time + training_time
            # feature_progress_bar.update(1)
        # feature_progress_bar.close()
        clauses_progress_bar.update(1)
    clauses_progress_bar.close()

    words_progress_bar.update(1)
words_progress_bar.close()                
Tools.print_training_time(total_training_time)