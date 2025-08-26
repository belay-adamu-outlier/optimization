def get_shading_templates(min_x, min_y, max_x, max_y, min_z, max_z):
    south_overhang = {
        'Transmittance_Schedule_Name': 'Off',
        'Number_of_Vertices': 4,
        
        'Vertex_1_Xcoordinate': min_x,
        'Vertex_1_Ycoordinate': min_y,
        'Vertex_1_Zcoordinate': max_z,
        
        'Vertex_2_Xcoordinate': min_x,
        # 'Vertex_2_Ycoordinate': min_y - overhang_dim['length'],
        'Vertex_2_Zcoordinate': max_z,
        
        'Vertex_3_Xcoordinate': max_x,
        # 'Vertex_3_Ycoordinate': min_y - overhang_dim['length'],
        'Vertex_3_Zcoordinate': max_z,
        
        'Vertex_4_Xcoordinate': max_x,
        'Vertex_4_Ycoordinate': min_y,
        'Vertex_4_Zcoordinate': max_z,
    }

    south_right_fin = {
        'Transmittance_Schedule_Name': 'Off',
        'Number_of_Vertices': 4,
        
        'Vertex_1_Xcoordinate': max_x,
        # 'Vertex_1_Ycoordinate': min_y - left_fin_dim['length'],
        'Vertex_1_Zcoordinate': max_z,
        
        'Vertex_2_Xcoordinate': max_x,
        'Vertex_2_Ycoordinate': min_y,
        'Vertex_2_Zcoordinate': max_z,
        
        'Vertex_3_Xcoordinate': max_x,
        'Vertex_3_Ycoordinate': min_y,
        'Vertex_3_Zcoordinate': min_z,
        
        'Vertex_4_Xcoordinate': max_x,
        # 'Vertex_4_Ycoordinate': min_y - left_fin_dim['length'],
        'Vertex_4_Zcoordinate': min_z,
    }

    south_left_fin= {
        'Transmittance_Schedule_Name': 'Off',
        'Number_of_Vertices': 4,
        
        'Vertex_1_Xcoordinate': min_x,
        'Vertex_1_Ycoordinate': min_y,
        'Vertex_1_Zcoordinate': min_z,
        
        'Vertex_2_Xcoordinate': min_x,
        'Vertex_2_Ycoordinate': min_y,
        'Vertex_2_Zcoordinate': max_z,
        
        'Vertex_3_Xcoordinate': min_x,
        # 'Vertex_3_Ycoordinate': min_y - left_fin_dim['length'],
        'Vertex_3_Zcoordinate': max_z,
        
        'Vertex_4_Xcoordinate': min_x,
        # 'Vertex_4_Ycoordinate': min_y - left_fin_dim['length'],
        'Vertex_4_Zcoordinate': min_z,
    }
    
    east_overhang = {
        'Transmittance_Schedule_Name': 'Off',
        'Number_of_Vertices': 4,
        
        'Vertex_1_Xcoordinate': min_x,
        'Vertex_1_Ycoordinate': min_y,
        'Vertex_1_Zcoordinate': max_z,
        
        'Vertex_2_Xcoordinate': min_x,
        'Vertex_2_Ycoordinate': max_y,
        'Vertex_2_Zcoordinate': max_z,
        
        # 'Vertex_3_Xcoordinate': min_x + overhang_dim['length'],
        'Vertex_3_Ycoordinate': max_y,
        'Vertex_3_Zcoordinate': max_z,
        
        # 'Vertex_4_Xcoordinate': min_x + overhang_dim['length'],
        'Vertex_4_Ycoordinate': min_y,
        'Vertex_4_Zcoordinate': max_z,
    }
    
    east_fin = {
        'Transmittance_Schedule_Name': 'Off',
        'Number_of_Vertices': 4,
        
        'Vertex_1_Xcoordinate': max_x,
        'Vertex_1_Ycoordinate': min_y,
        'Vertex_1_Zcoordinate': min_z,
        
        'Vertex_2_Xcoordinate': min_x,
        'Vertex_2_Ycoordinate': min_y,
        'Vertex_2_Zcoordinate': max_z,
        
        # 'Vertex_3_Xcoordinate': min_x + fin_dim['length'],
        'Vertex_3_Ycoordinate': min_y,
        'Vertex_3_Zcoordinate': max_z,
        
        # 'Vertex_4_Xcoordinate': min_x + fin_dim['length'],
        'Vertex_4_Ycoordinate': min_y,
        'Vertex_4_Zcoordinate': min_z,
    }
    
    west_overhang = {
        'Transmittance_Schedule_Name': 'Off',
        'Number_of_Vertices': 4,
        
        'Vertex_1_Xcoordinate': min_x,
        'Vertex_1_Ycoordinate': min_y,
        'Vertex_1_Zcoordinate': max_z,
        
        # 'Vertex_2_Xcoordinate': min_x - overhang_dim['length'],
        'Vertex_2_Ycoordinate': max_y,
        'Vertex_2_Zcoordinate': max_z,
        
        # 'Vertex_3_Xcoordinate': max_x - overhang_dim['length'],
        'Vertex_3_Ycoordinate': max_y,
        'Vertex_3_Zcoordinate': max_z,
        
        'Vertex_4_Xcoordinate': min_x,
        'Vertex_4_Ycoordinate': max_y,
        'Vertex_4_Zcoordinate': max_z,
    }
    
    west_fin = {
        'Transmittance_Schedule_Name': 'Off',
        'Number_of_Vertices': 4,
        
        'Vertex_1_Xcoordinate': min_x,
        'Vertex_1_Ycoordinate': min_y,
        'Vertex_1_Zcoordinate': min_z,
        
        # 'Vertex_2_Xcoordinate': min_x - fin_dim['length'],
        'Vertex_2_Ycoordinate': min_y,
        'Vertex_2_Zcoordinate': max_z,
        
        # 'Vertex_3_Xcoordinate': min_x - fin_dim['length'],
        'Vertex_3_Ycoordinate': min_y,
        'Vertex_3_Zcoordinate': max_z,
        
        'Vertex_4_Xcoordinate': min_x,
        'Vertex_4_Ycoordinate': min_y,
        'Vertex_4_Zcoordinate': max_z,
    }

    return {
        'south_overhang': south_overhang,
        'south_right_fin': south_right_fin,
        'south_left_fin': south_left_fin,
        'east_overhang': east_overhang,
        'east_fin': east_fin,
        'west_overhang': west_overhang,
        'west_fin': west_fin,
    }