# This script contains helper functions to help debugging
import pickle
picDir = "pickle_file/"

#
# @Fucntion: This function helps to save pickle such that
#            we can test the script step by step
#
# @Params: Data      -> data you would like to save as a pickle
#          file_name -> specify the file_name
#
def save_as_pickle(data,file_name):
    with open(picDir+file_name+".p",'wb') as file:
        pickle.dump(data,file)

#
# @Fucntion: This function was used to read pickle
#
# @Params: specify the file_name you would like to read
#
def read_from_pickle(file_name):
    with open(picDir+file_name+".p",'r') as file:
        data = pickle.load(file)
    return data

