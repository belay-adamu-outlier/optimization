"""
Test script for extract_building_parameters function
"""

from simulation_utils import extract_building_parameters, initialize_idf


def test_extract_building_parameters():
    """Test the extract_building_parameters function"""
    
    # File paths
    path_to_idf = "Building.idf"
    path_to_idd = "C:/EnergyPlusV9-4-0/Energy+.idd"  # You'll need to update this path
    weather_file = "Vancouver - 2021 AMY.epw"
    
    try:
        # Initialize IDF
        print("Initializing IDF...")
        idf = initialize_idf(path_to_idf, path_to_idd, weather_file)
        print("IDF initialized successfully!")
        
        # Define parameters to extract - focusing only on constructions
        parameters_to_extract = [
            "CONSTRUCTION"  # All constructions
        ]
        
        print(f"\nExtracting parameters: {parameters_to_extract}")
        
        # Extract building parameters
        extracted_params = extract_building_parameters(parameters_to_extract, idf)
        
        # Print results
        print("\nExtraction Results:")
        print("=" * 50)
        
        for param_name, objects in extracted_params.items():
            print(f"\n{param_name}:")
            if objects:
                print(f"  Found {len(objects)} object(s)")
                
                # Special handling for CONSTRUCTION objects
                if param_name == "CONSTRUCTION":
                    print("  Construction Details:")
                    for i, construction in enumerate(objects):
                        print(f"    {i+1}. {construction.Name}:")
                        # Print all layers in the construction
                        for field_name in construction.fieldnames:
                            if field_name.startswith("Layer_"):
                                layer_value = getattr(construction, field_name, "N/A")
                                if layer_value and layer_value != "":
                                    print(f"      {field_name}: {layer_value}")
                
                # For other objects, show first object as example
                else:
                    if len(objects) > 0:
                        first_obj = objects[0]
                        print(f"  First object fields:")
                        for field_name in first_obj.fieldnames:
                            value = getattr(first_obj, field_name, "N/A")
                            print(f"    {field_name}: {value}")
            else:
                print("  No objects found")
        
        print(f"\nTotal parameters extracted: {len(extracted_params)}")
        
        # Summary of construction types found
        if "CONSTRUCTION" in extracted_params and extracted_params["CONSTRUCTION"]:
            print("\nConstruction Summary:")
            print("-" * 30)
            constructions = extracted_params["CONSTRUCTION"]
            construction_names = [obj.Name for obj in constructions]
            print(f"Total constructions found: {len(constructions)}")
            print("Construction names:")
            for name in construction_names:
                print(f"  - {name}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure the file paths are correct and EnergyPlus is properly installed.")


if __name__ == "__main__":
    print("Testing extract_building_parameters function")
    print("=" * 50)
    test_extract_building_parameters()
