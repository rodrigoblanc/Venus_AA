include("Constantes.jl")

function recortar(area, imagen)
    
    # El radio se redondea hacia arriba para abarcar más area de la que abarcariamos de poder considerar los flotantes
    image_size = size(imagen)
    rounded_radius = Int(ceil(area[3])) * K_MULTIPLIER
    
    start_x = Int(max(1, area[1] - rounded_radius))
    end_x = Int(min(area[1]+rounded_radius, image_size[1]))
    start_y = Int(max(1, area[2] - rounded_radius))
    end_y = Int(min(area[2]+rounded_radius, image_size[2]))
    
    return imagen[start_x:end_x, start_y:end_y]
end