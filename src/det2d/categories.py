from types import SimpleNamespace
import json
from .keys import Keys

def read_categories(path: str) -> SimpleNamespace:
    '''
    Read category indices from a cats.json file
    
    Inputs:
    - path: path to the categories
    
    Outputs:
    - Namespace with category indices
    '''
    categories_dict = read_category_details(path)
    categories_dict = {key: index for index, key in enumerate(categories_dict.keys())}
    categories_namespace = SimpleNamespace(**categories_dict)
    
    return categories_namespace

def read_category_keypoints(path: str, category: int) -> SimpleNamespace:
    '''
    Read all keypoints for a category from a cats.json file

    Inputs:
    - path: path to the categories
    - category: integer representing the wanted category, obtained from read_categories()
    
    Outputs:
    - Namespace with keypoint indices from the category
    '''
    categories_dict = read_category_details(path)
    keypoints_dict = {key: index for index, key in enumerate(list(categories_dict.values())[category][Keys.keypoints])}
    keypoints_namespace = SimpleNamespace(**keypoints_dict)
    
    return keypoints_namespace

def read_category_details(path: str) -> dict:
    '''
    Read category dicts from a cats.json file
    
    Inputs:
    - path: path to the categories
    
    Outputs:
    - dict corresponding to the json structure in path
    '''
    with open(path, 'r') as f: categories_dict = json.load(f)
    return categories_dict