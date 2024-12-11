import os
import json
def main():
    print('Hello Danny!')
    results_folder = 'results'
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    file_path = os.path.join(results_folder, 'auc_results.json')
    with open(file_path, 'w') as f:
        json.dump({"hel danny": 1, "hi dan": 2}, f, indent=4)
    

if __name__ == "__main__":
    main()