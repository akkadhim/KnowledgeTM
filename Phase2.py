import os
import numpy as np
from contextlib import redirect_stdout
from tqdm import tqdm
from time import time
from collections import defaultdict
from tmu.models.autoencoder.autoencoder import TMAutoEncoder
from sklearn.metrics.pairwise import cosine_similarity
from Evaluation import Evaluation
from Tools import Tools
from DirectoriesUtil import Dicrectories

target_similarity=defaultdict(list)
clause_weight_threshold = 0
clause_drop_p = 0.0
factor = 20
clauses = 80
T = factor*40
s = 5.0
epochs = 100
number_of_examples = 1000
accumulation = 10
sub_accumulation = 10
top_max_clauses1 = 0
top_max_clauses2 = 0
with_clause_update = False
max_spearman = 0.9
true_weight = 0.7
false_weight = 1 - true_weight
neg_length = 50

eval = Evaluation()
def preprocess_text(text):
    return text
vectorizer_X = Tools.read_pickle_data("vectorizer_X.pickle")
feature_names = vectorizer_X.get_feature_names_out()
number_of_features = vectorizer_X.get_feature_names_out().shape[0]

for dataset_name in os.listdir(Dicrectories.datasets):
    if dataset_name == 'rg-65':
        current_folder_path = os.path.join(Dicrectories.datasets, dataset_name)
        if os.path.isdir(current_folder_path):
            files_start_name = os.path.join(current_folder_path, dataset_name)

            pair_list = Tools.get_dataset_pairs(files_start_name)
            output_active, target_words = Tools.get_dataset_targets(files_start_name,pair_list, vectorizer_X)
            
            result_filepath = Dicrectories.test(dataset_name,"all_phase2")
            with open(result_filepath, 'w') as file, redirect_stdout(file):
                tm = TMAutoEncoder(clauses, T, s, output_active, max_included_literals=3, accumulation=accumulation, feature_negation=False, platform='CPU', output_balancing=0.5)
                total_training = 0
                print("Epochs: %d" % epochs)
                print("Target words: %d" % len(target_words))
                print("No of features: %d" % number_of_features)
                print("Clauses: %d" % clauses)
                print("with_clause_update: %s" % with_clause_update)
                print("Examples: %d" % number_of_examples)
                print("Accumulation: %d" % accumulation)
                print("Sub Accumulation: %d" % sub_accumulation)
                print("true_weight: %f" % true_weight)
                print("false_weight: %f" % false_weight)
                print("top_max_clauses1: %d" % top_max_clauses1)
                print("top_max_clauses2: %d\n" % top_max_clauses2)
                
                epochs_progress_bar = tqdm(total=epochs, desc="Running Epochs")
                for e in range(epochs):
                    print("\nEpoch #: %d" % e)
                    start_training = time()
                    tm.knowledge_fit(
                        number_of_examples = number_of_examples,
                        number_of_features = number_of_features,
                        sub_accumulation = sub_accumulation,
                        top_max_clauses1 = top_max_clauses1,
                        top_max_clauses2 = top_max_clauses2,
                        neg_length = neg_length,
                        with_clause_update = with_clause_update,
                        true_weight = true_weight,
                        false_weight = false_weight,
                        print_c = False
                        )
                    stop_training = time()
                    epoch_time = stop_training - start_training
                    Tools.print_training_time(epoch_time)
                    total_training = total_training + epoch_time

                    profile = np.empty((len(target_words), clauses))
                    for i in range(len(target_words)):
                        weights = tm.get_weights(i)
                        profile[i,:] = np.where(weights >= clause_weight_threshold, weights, 0)
                    similarity = cosine_similarity(profile)
                    for i in range(len(target_words)):
                        sorted_index = np.argsort(-1*similarity[i,:])
                        for j in range(1, len(target_words)):
                            target_similarity[(target_words[i], target_words[sorted_index[j]])]  = similarity[i,sorted_index[j]]
                    spearman = eval.calculate(target_similarity,pair_list)
                    if spearman > max_spearman:
                        break
                    epochs_progress_bar.update(1)
                epochs_progress_bar.close()

                print("\n=====================================\nClauses\n=====================================")
                for j in range(clauses):
                    print("Clause #%-2d " % (j), end=' ')
                    for tw in range(len(target_words)):
                        print("%s:W%-5d " % (target_words[tw], tm.get_weight(tw, j)), end='| ')
                    l = [] 
                    number_of_literals = 0 
                    for k in range(tm.clause_bank.number_of_literals):
                        if tm.get_ta_action(j, k) == 1:
                            number_of_literals = number_of_literals + 1
                            if k < tm.clause_bank.number_of_features:
                                l.append("%s(%d)" % (feature_names[k], tm.clause_bank.get_ta_state(j, k)))
                            else:
                                l.append("¬%s(%d)" % (feature_names[k-tm.clause_bank.number_of_features], tm.clause_bank.get_ta_state(j, k)))
                    print(": No of features:%-6d" % (number_of_literals), end=" ==> ")
                    try:
                        print(" - ".join(l))
                    except UnicodeEncodeError:
                        print(" exception ")
                
                print("\n=====================================\nWord Similarity\n=====================================")
                max_word_length = len(max(target_words, key=len))
                list_of_words = []
                target_words_with_min_max = []
                for i in range(len(target_words)):
                    row_of_similarity = []
                    sorted_index = np.argsort(-1*similarity[i,:])
                    min_similarity = 1.0
                    max_similarity = 0.0
                    word_similarity = []
                    for j in range(1, len(target_words)):
                        target_similarity[(target_words[i], target_words[sorted_index[j]])]  = similarity[i,sorted_index[j]]
                        row_of_similarity.append(target_words[sorted_index[j]])
                        word_similarity.append("{:<{}}({:.2f})  ".format(target_words[sorted_index[j]], max_word_length, similarity[i, sorted_index[j]]))
                        if(min_similarity > similarity[i,sorted_index[j]]):
                            min_similarity = similarity[i,sorted_index[j]]
                        if(max_similarity < similarity[i,sorted_index[j]]):
                            max_similarity = similarity[i,sorted_index[j]]
                
                    output_line = f"{target_words[i]:<{max_word_length}}: Min:{min_similarity:.2f}, Max:{max_similarity:.2f}"
                    print(output_line, end='     ==> ')
                    print(word_similarity)
                    list_of_words.append(row_of_similarity)
                    target_words_with_min_max.append(output_line)

                Tools.print_training_time(total_training)
                
            eval = ResultHelper.get_file_max_spearman(result_filepath)
            dir_name, old_file_name = os.path.split(result_filepath)
            new_file_path = os.path.join(dir_name, "{:.2f}".format(eval) + "_"  + old_file_name)
            os.rename(result_filepath, new_file_path)