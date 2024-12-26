def print_intermediate(
        accuracies_test, 
        accuracies_val=None, 
        task='anomaly_unsupervised',
        task_params=None,
        ):
     
    if task == 'anomaly_unsupervised':
        dyads_to_omit = task_params['dyads_to_omit']
        print('Anomaly Vals')
        if dyads_to_omit is not None:
            print('Link Vals')
            print(f'{best_prior_link_auc= },{iter_best_prior_link= },{count_not_improved_prior_link= }')
                                    
        print('Anomaly Vals')
        print(f"last vanilla auc: {accuracies_test['vanilla_star'][-1]}") 
        print(f"last prior auc {accuracies_test['prior'][-1]}")
        print(f"last prior star auc {accuracies_test['prior_star'][-1]= }")
