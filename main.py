if __name__ == '__main__':
    def train_model():
        import death

        model_select = {
            'light_gbm': death.light_gbm_model,
            'knn': death.knn_model,
            'svm': death.svm_model,
            'logistic_regression': death.logistic_regression_model,
            'decision_tree': death.decision_tree_model,
            'random_forest': death.random_forest_model,
        }
        list_model = []
        print('(light_gbm|knn|svm|logistic_regression|decision_tree|random_forest|*)')
        while len(list_model) < 6:
            model_name = input('select model:')
            if model_name in list(model_select.keys()) or model_name == "*":
                if model_name in list_model:
                    print('duplicat model  "Please select other models"')
                    print('-' * 10)
                    continue
                if model_name == '*':
                    list_model.clear()
                    print(list(model_select.keys()))
                    list_model.extend(list(model_select.keys()))
                    break
                if model_name not in list_model:
                    list_model.append(model_name)
                print('if you dont want to continue please write "ese"')
            elif model_name == 'ese':
                break
            else:
                print('-' * 10)
                print('please write the names of the models correctly')
                print('select model')
                print('(light_gbm|knn|svm|logistic_regression|decision_tree|random_forest|all)')
        if list_model:
            print('/' * 10)
            print('start train models')
            for i in list_model:
                model_select.get(i)()
            print('done')


    def run_report():
        import death.report


    def run():
        dic = {
            'train_model': train_model,
            'report': run_report
        }
        print('run (train_model|report|both)')
        run_ = input('select run:')
        if run_ == 'both':
            for i in dic.values():
                i()
        else:
            x = dic.get(run_)
            if x:
                x()


    run()
