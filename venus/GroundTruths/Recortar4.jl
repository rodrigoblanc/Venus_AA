#Para recortar las imagenes de la primera aproximacion en 4

positive_images2 = []
negative_images2 = []

for i=1:size_data
    for j=1:(size(negative_dataset[i], 1))
        println(negative_dataset[i][j])
        recorte = negative_dataset[i][j][2:4]
        push!(negative_images2, recortar2(recorte, matrix[i]))
    end
end

for i=1:size_data
    for j=1:(size(positive_dataset[i], 1))
        println(positive_dataset[i][j])
        recorte = positive_dataset[i][j][2:4]
        push!(positive_images2, recortar2(recorte, matrix[i]))
    end
end


# Crear el directorio si no existe
if !isdir(hit_path3)
    mkdir(hit_path3)
end
global cont = 1
for (image) in positive_images2
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
for (image) in negative_images2
    for i in image
        # Guardar la imagen en el directorio
        name = "recorte"*string(cont)*"_aprox3.png"
        println("saving \""*name*"\"")
        save(joinpath(miss_path3, name), i)
        global cont = cont+1
    end
end
