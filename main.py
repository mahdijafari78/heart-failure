if __name__ == '__main__':
    def train_model(run_file):
        if run_file == 'disease':
            import disease as file
        else:
            import death as file

        model_select = {
            'light_gbm': file.light_gbm_model,
            'knn': file.knn_model,
            'svm': file.svm_model,
            'logistic_regression': file.logistic_regression_model,
            'decision_tree': file.decision_tree_model,
            'random_forest': file.random_forest_model,
            'ada_boost_model': file.ada_boost_model,
        }
        list_model = []
        print_name = '|'.join(model_select.keys()) + '|*'
        print(print_name)
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
                print(print_name)
        if list_model:
            print('/' * 10)
            print('start train models')
            for i in list_model:
                model_select.get(i)()
            print('done')


    def run_report(run_file):
        if run_file == 'disease':
            import disease.report
        else:
            import death.report


    def run():
        dic = {
            'train_model': train_model,
            'report': run_report
        }
        select_dic = ['disease', 'death']
        print('disease,death')
        select_file = input('select file:')
        if select_file not in select_dic:
            print('Choose the right file')
            return None
        print('run (train_model|report|both)')
        run_ = input('select run:')
        if run_ == 'both':
            for i in dic.values():
                i(select_file)
        else:
            x = dic.get(run_)
            if x:
                x(select_file)


    run()
