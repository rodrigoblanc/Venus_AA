############################# LOADING HIT & MISS ###############################

# Carga de Hit 'n Miss

hitImages = loadFolderImages(hit_path)

missImages = loadFolderImages(miss_path)

hitMiniCuts = []

missMiniCuts = []

for hitImage in hitImages
    push!(hitMiniCuts, recortar(hitImage, 0.5))
end

for missImage in missImages
    push!(missMiniCuts, recortar(missImage, 0.5))
end


# Crear el directorio si no existe
if !isdir(hit_path2)
    mkdir(hit_path2)
end
global cont = 1
for (image) in hitMiniCuts
    # Guardar la imagen en el directorio
    name = "recorte"*string(cont)*"_aprox2.png"
    println("saving \""*name*"\"")
    save(joinpath(hit_path2, name), image)
    global cont = cont+1
end

# Crear el directorio si no existe
if !isdir(miss_path2)
    mkdir(miss_path2)
end
global cont = 1
for (image) in missMiniCuts
    # Guardar la imagen en el directorio
    name = "recorte"*string(cont)*"_aprox2.png"
    println("saving \""*name*"\"")
    save(joinpath(miss_path2, name), image)
    global cont = cont+1
end

