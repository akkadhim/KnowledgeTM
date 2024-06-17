import codecs
import pickle
import numpy as np
from functools import lru_cache
import codecs
from DirectoriesUtil import Dicrectories

class Tools:
    @staticmethod
    def get_dataset_pairs(base_path):
        path = base_path + ".csv"
        fread_simlex=codecs.open(path, 'r', 'utf-8')
        pair_list = []
        line_number = 0
        for line in fread_simlex:
            if line_number > 0:
                tokens = line.split(',')
                word_i = tokens[1].lower()
                word_j = tokens[2].lower()
                score = float(tokens[3].replace('\n', ''))
                pair_list.append( ((word_i, word_j), score) )
            line_number += 1
        return pair_list

    @staticmethod
    def get_dataset_words(base_path):
        pair_list = Tools.get_dataset_pairs(base_path)
        words=[]
        for (x,y) in pair_list:
            (word1, word2) = x
            if word1 not in words:
                words.append(word1)
            if word2 not in words:
                words.append(word2)
        return words
        
    @staticmethod
    def get_dataset_targets(base_path, vectorizer_X, pair_list = None):
        if pair_list == None:
            pair_list = Tools.get_dataset_pairs(base_path)
        word1 = []
        word2 = []
        for (x,y) in pair_list:
            (w1, w2) = x
            word1.append(w1)
            word2.append(w2)
                
        word_total= list(set(word1 + word2))
        print("Dataset words count: ", len(word_total))
        target_words=[]
        for i in word_total:
            if i in vectorizer_X.vocabulary_ :
                file_path = Dicrectories.pickle_by_id(Dicrectories.knowledge , str(vectorizer_X.vocabulary_[i]))
                if Dicrectories.pickle_exist(file_path):
                    target_words.append(i)
        output_active = np.empty(len(target_words), dtype=np.uint32)
        for i in range(len(target_words)):
            target_word = target_words[i]
            target_id = vectorizer_X.vocabulary_[target_word]
            output_active[i] = target_id
        return output_active, target_words
        
    @staticmethod    
    @lru_cache(maxsize=None)
    def read_pickle_data(path):
        if Dicrectories.pickle_exist(path):
            with open(path, "rb") as saved:
                try:
                    return pickle.load(saved)
                except (pickle.UnpicklingError, EOFError):
                    print("Error: The file could not be unpickled: ",path)
                    return []
        else:
            print("Error: The file does not exist: ",path)
            return []
    
    @staticmethod
    def print_training_time(seconds):
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60
        print(f"Training duration: {hours} hours, {minutes} minutes, {seconds} seconds")
