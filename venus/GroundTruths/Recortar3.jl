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


# Crear el directorio si no existe
if !isdir(hit_path3)
    mkdir(hit_path3)
end
global cont = 1
for (image) in hitMiniCuts
    for i in image
        # Guardar la imagen en el directorio
        name = "recorte"*string(cont)*"_aprox3.png"
        println("saving \""*name*"\"")
        save(joinpath(hit_path3, name), i)
        global cont = cont+1
    end
end

# Crear el directorio si no existe
if !isdir(miss_path3)
    mkdir(miss_path3)
end
global cont = 1
for (image) in missMiniCuts
    for i in image
        # Guardar la imagen en el directorio
        name = "recorte"*string(cont)*"_aprox3.png"
        println("saving \""*name*"\"")
        save(joinpath(miss_path3, name), i)
        global cont = cont+1
    end
end
