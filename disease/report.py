from library import ReportModel
from death import data_set

report_model = ReportModel('disease/pickle', data_set.x_test,data_set.y_test ,'disease')
report_model()
