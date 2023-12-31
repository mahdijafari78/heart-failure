from library import ReportModel
from death import data_set

report_model = ReportModel('death/pickle/predict', data_set.y_test, 'death')
report_model()
