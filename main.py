from eppy import modeleditor
from eppy.modeleditor import IDF

iddfile = 'C:/EnergyPlusV9-4-0/Energy+.idd'
fname1 = './Building.idf'
epwfile = './Vancouver - 2021 AMY.epw'

IDF.setiddname(iddfile)
idf1 = IDF(fname1, epwfile)

idf1.run()