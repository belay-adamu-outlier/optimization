from eppy.modeleditor import IDF
from .energyplus_utils import (
    rename_shading_elements_by_orientation,
    assign_shading_transmittance_schedules,
    apply_shading_schedules_to_idf
)


def main():
    """Quick manual test for shading rename/schedule assignment."""
    IDF.setiddname('C:/EnergyPlusV9-4-0/Energy+.idd')
    idf = IDF('Building_Shading_cleaned_specific_synced.idf', 'Vancouver - 2021 AMY.epw')

    # Rename shading elements by orientation and type
    renamed = rename_shading_elements_by_orientation(idf)
    print('Renamed shading elements: %d' % len(renamed))

    # Assign schedules per orientation/type (skip north as required)
    assigned = assign_shading_transmittance_schedules(idf)
    schedule = [0] * 144
    idf =apply_shading_schedules_to_idf(idf, schedule)
    idf.saveas('Building_Shading_Renamed.idf')
    print('Assigned schedules: %d' % assigned)

    # Show a small sample for verification
    shown = 0
    for key in (
        'SHADING:BUILDING:DETAILED',
        'SHADING:ZONE:DETAILED',
        'SHADING:SITE:DETAILED',
        'Shading:Building:Detailed',
        'Shading:Zone:Detailed',
        'Shading:Site:Detailed',
    ):
        for shade in idf.idfobjects.get(key, []):
            name = getattr(shade, 'Name', '')
            sched = getattr(shade, 'Transmittance_Schedule_Name', '')
            print('%s -> %s' % (name, sched))
            shown += 1
            if shown >= 8:
                break
        if shown >= 8:
            break


if __name__ == '__main__':
    main()

