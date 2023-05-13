############################# LOADING HIT & MISS ###############################

# Carga de Hit 'n Miss

hitImages = loadFolderImages(hit_path)

missImages = loadFolderImages(miss_path)

hitMiniCuts = []

missMiniCuts = []

for hitImage in hitImages
    push!(hitMiniCuts, recortar2(hitImage))
end

for missImage in missImages
    push!(missMiniCuts, recortar2(missImage))
end

println(hitMiniCuts[1])

# Crear el directorio si no existe
if !isdir(hit_path3)
    mkdir(hit_path3)
end

global cont = 1
for (fourImage) in hitMiniCuts
    for image in fourImage
        # Guardar la imagen en el directorio
        name = "recorte"*string(cont)*"_aprox3.png"
        println("saving \""*name*"\"")
        save(joinpath(hit_path3, name), image)
        global cont = cont+1
    end
end

# Crear el directorio si no existe
if !isdir(miss_path3)
    mkdir(miss_path3)
end

global cont = 1
for (fourImage) in missMiniCuts
    for image in fourImage
        # Guardar la imagen en el directorio
        name = "recorte"*string(cont)*"_aprox3.png"
        println("saving \""*name*"\"")
        save(joinpath(miss_path3, name), image)
        global cont = cont+1
    end
end
