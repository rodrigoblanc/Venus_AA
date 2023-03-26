include("Constantes.jl")
include("personal.jl")

function ccd(relativeLocation::String)
    cd(mycd*relativeLocation)
end

function recortar(coords, imagen)

    #imshow(imagen)
    
    # El radio se redondea hacia arriba para abarcar más area de la que abarcariamos de poder considerar los flotantes
    image_size = size(imagen)
    rounded_radius = Int(ceil(coords[3])) * K_MULTIPLIER
    
    start_x = Int(max(1, coords[1] - rounded_radius))
    end_x = Int(min(coords[1]+rounded_radius, image_size[1]))
    start_y = Int(max(1, coords[2] - rounded_radius))
    end_y = Int(min(coords[2]+rounded_radius, image_size[2]))
    
    println("recortar-> Recortando:\t"*string(start_x)*","*string(start_y)*")-----"*"("*string(end_x)*","*string(end_y)*")")

    # Las imágenes están traspuestas, habrá que recortar acorde a ello y trasponer
    # el resultado
    #= #! transpose warning
        This operation is intended for linear algebra usage
        - for general data manipulation see [permutedims](@ref Base.permutedims),
        which is non-recursive
    =#
    imagen2 = imagen[start_y:end_y, start_x:end_x]
    #imagen2 = transpose(imagen2)
    return imagen2
end