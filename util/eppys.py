from eppy.modeleditor import IDF
from .energyplus_utils import modify_idf

IDF.setiddname("C:/EnergyPlusV9-4-0/Energy+.idd")
idf = IDF('Building.idf', 'Vancouver - 2021 AMY.epw')

parameters = {
    "external_wall": "R35 - Wall",
    "roof": "R55 - Roof",
    "window": "SageGlass Climatop Blue No Tint - 1008",
    "infiltration_ach": 0.78,
    "wwr_east": 0.36,
    "wwr_west": 0.30,
    "wwr_north": 0.20,
    "wwr_south": 0.25,
    "wall_solar_absorptance": 0.15,
    "roof_solar_absorptance": 0.37,
}

idf = modify_idf(idf, parameters)
idf.saveas('./IDF/Building_modified.idf')
# idf.run()


