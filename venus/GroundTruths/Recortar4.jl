# Los recortes relacionados con los "hit" se obtienen como de costumbre, con
# include(Recortar1..3.jl)

missImages = loadFolderImages(cuts_path)

# aprox2

missMiniCuts = []

for missImage in missImages
    push!(missMiniCuts, recortar(missImage, 0.5))
end


# Crear el directorio si no existe
if !isdir(miss_path2_4)
    mkdir(miss_path2_4)
end

global cont = 1
for (image) in missMiniCuts
        # Guardar la imagen en el directorio
        name = "recorte"*string(cont)*"_aprox2.png"
        println("saving \""*name*"\"")
        save(joinpath(miss_path2_4, name), image)
        global cont = cont+1
end


# aprox3

missMiniCuts = []

for missImage in missImages
    push!(missMiniCuts, recortar2(missImage))
end

# Crear el directorio si no existe
if !isdir(miss_path3_4)
    mkdir(miss_path3_4)
end

global cont = 1
for (fourImage) in missMiniCuts
    for image in fourImage
        # Guardar la imagen en el directorio
        name = "recorte"*string(cont)*"_aprox3.png"
        println("saving \""*name*"\"")
        save(joinpath(miss_path3_4, name), image)
        global cont = cont+1
    end
end

