from library import ReportModel
from death import data_set

report_model = ReportModel('death/pickle', data_set.x_test,data_set.y_test ,'death')
report_model()
