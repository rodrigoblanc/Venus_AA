K_MULTIPLIER = 0.5

positive_images1 = []
negative_images1 = []

for i=1:size_data
    for j=1:(size(negative_dataset[i], 1))
        println(negative_dataset[i][j])
        recorte = negative_dataset[i][j][2:4]
        push!(negative_images1, recortar(recorte, matrix[i]))
    end
end

for i=1:size_data
    for j=1:(size(positive_dataset[i], 1))
        println(positive_dataset[i][j])
        recorte = positive_dataset[i][j][2:4]
        push!(positive_images1, recortar(recorte, matrix[i]))
    end
end


# Crear el directorio si no existe
if !isdir(hit_path2)
    mkdir(hit_path2)
end
global cont = 1
for (image) in positive_images1
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
for (image) in negative_images1
    # Guardar la imagen en el directorio
    name = "recorte"*string(cont)*"_aprox2.png"
    println("saving \""*name*"\"")
    save(joinpath(miss_path2, name), image)
    global cont = cont+1
end
